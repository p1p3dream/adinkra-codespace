#!/usr/bin/env python3
"""Import validated semantic proposals into the isolated Gates pilot graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2


CORPUS_ID = "gates_literature_pilot"


def stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(parts)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def db_node_type(semantic_type: str) -> str:
    if semantic_type in {"paper", "claim", "result"}:
        return semantic_type
    return "concept"


@dataclass
class ResolvedEntity:
    node_id: str
    node_type: str
    canonical_key: str
    name: str
    semantic_type: str
    existing: bool = False


def load_proposals(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_paper(cur: Any, key: str) -> tuple[ResolvedEntity, str]:
    cur.execute(
        """SELECT n.node_id,n.node_type,n.canonical_key,n.name,p.paper_id
           FROM gates_pilot_nodes n
           JOIN gates_pilot_papers p ON p.corpus_id=n.corpus_id AND p.stable_identifier=n.canonical_key
           WHERE n.corpus_id=%s AND n.node_type='paper' AND n.canonical_key=%s""",
        (CORPUS_ID, key),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"unknown pilot paper {key}")
    return ResolvedEntity(row[0], row[1], row[2], row[3], "paper", True), row[4]


def resolve_entity(cur: Any, entity: dict[str, Any]) -> ResolvedEntity:
    semantic_type = str(entity["type"])
    key = str(entity["key"])
    if semantic_type == "paper":
        return resolve_paper(cur, key)[0]
    node_type = db_node_type(semantic_type)
    canonical_key = f"{semantic_type}:{key.removeprefix(semantic_type + ':')}"
    cur.execute(
        """SELECT node_id,node_type,canonical_key,name FROM gates_pilot_nodes
           WHERE corpus_id=%s AND node_type=%s AND canonical_key=%s""",
        (CORPUS_ID, node_type, canonical_key),
    )
    row = cur.fetchone()
    if row:
        return ResolvedEntity(row[0], row[1], row[2], row[3], semantic_type, True)
    return ResolvedEntity(
        stable_id("semnode", CORPUS_ID, node_type, canonical_key), node_type,
        canonical_key, str(entity["name"]), semantic_type, False,
    )


def validate_db_evidence(cur: Any, proposal: dict[str, Any]) -> tuple[str, str]:
    evidence = proposal["evidence"]
    paper_key = "arxiv:" + str(evidence["paper_id"]).removeprefix("arxiv:")
    _, paper_id = resolve_paper(cur, paper_key)
    cur.execute(
        """SELECT content,page_start FROM gates_pilot_chunks
           WHERE corpus_id=%s AND chunk_id=%s AND paper_id=%s""",
        (CORPUS_ID, evidence["chunk_id"], paper_id),
    )
    row = cur.fetchone()
    if not row or int(row[1] or 0) != int(evidence["page_number"]):
        raise ValueError(f"invalid database evidence for {proposal['proposal_id']}")
    if normalized_text(str(evidence.get("excerpt") or "")) not in normalized_text(str(row[0])):
        raise ValueError(f"excerpt mismatch for {proposal['proposal_id']}")
    if proposal.get("basis") != "explicit_text" or proposal.get("review_status") != "pending":
        raise ValueError(f"invalid review controls for {proposal['proposal_id']}")
    return paper_id, str(evidence["chunk_id"])


def build_summary(cur: Any, proposals: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = {}
    edges = {}
    for proposal in proposals:
        source = resolve_entity(cur, proposal["source"])
        target = resolve_entity(cur, proposal["target"])
        nodes[source.node_id] = source
        nodes[target.node_id] = target
        edge_id = stable_id("semedge", CORPUS_ID, source.node_id, target.node_id, proposal["relationship"])
        edges[edge_id] = (source, target, proposal["relationship"])
        validate_db_evidence(cur, proposal)
    return {
        "proposals": len(proposals),
        "distinct_nodes_touched": len(nodes),
        "new_nodes": sum(not n.existing for n in nodes.values()),
        "distinct_edges": len(edges),
        "relationships": dict(sorted(Counter(p["relationship"] for p in proposals).items())),
        "semantic_entity_types": dict(sorted(Counter(
            entity["type"] for p in proposals for entity in (p["source"], p["target"])
        ).items())),
    }


def apply(cur: Any, proposals: list[dict[str, Any]]) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (CORPUS_ID + ":semantic",))
    for proposal in proposals:
        source = resolve_entity(cur, proposal["source"])
        target = resolve_entity(cur, proposal["target"])
        paper_id, chunk_id = validate_db_evidence(cur, proposal)
        for entity in (source, target):
            if entity.semantic_type == "paper":
                continue
            properties = json.dumps({"semantic_type": entity.semantic_type}, sort_keys=True)
            cur.execute(
                """INSERT INTO gates_pilot_nodes
                   (corpus_id,node_id,node_type,canonical_key,name,properties)
                   VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT (corpus_id,node_id) DO UPDATE SET
                     name=EXCLUDED.name,
                     properties=gates_pilot_nodes.properties || EXCLUDED.properties,
                     updated_at=now()""",
                (CORPUS_ID, entity.node_id, entity.node_type, entity.canonical_key, entity.name, properties),
            )
            evidence_id = stable_id("semnodeev", CORPUS_ID, entity.node_id, proposal["proposal_id"])
            ev = proposal["evidence"]
            cur.execute(
                """INSERT INTO gates_pilot_node_evidence
                   (corpus_id,evidence_id,node_id,paper_id,chunk_id,source_kind,locator,excerpt,
                    extraction_method,confidence,properties)
                   VALUES (%s,%s,%s,%s,%s,'semantic_proposal',%s,%s,'structured_literature_review',%s,%s::jsonb)
                   ON CONFLICT (corpus_id,evidence_id) DO UPDATE SET
                     locator=EXCLUDED.locator,excerpt=EXCLUDED.excerpt,
                     confidence=EXCLUDED.confidence,properties=EXCLUDED.properties""",
                (CORPUS_ID, evidence_id, entity.node_id, paper_id, chunk_id,
                 f"physical PDF page {ev['page_number']}; {ev.get('section') or 'section unavailable'}",
                 ev["excerpt"], proposal["confidence"], json.dumps({"proposal_id": proposal["proposal_id"]})),
            )
        edge_id = stable_id("semedge", CORPUS_ID, source.node_id, target.node_id, proposal["relationship"])
        properties = json.dumps({
            "source_semantic_type": source.semantic_type,
            "target_semantic_type": target.semantic_type,
        }, sort_keys=True)
        cur.execute(
            """INSERT INTO gates_pilot_edges
               (corpus_id,edge_id,src_node_id,dst_node_id,relationship,description,basis,
                review_status,confidence,properties)
               VALUES (%s,%s,%s,%s,%s,%s,'explicit_text','pending',%s,%s::jsonb)
               ON CONFLICT (corpus_id,edge_id) DO UPDATE SET
                 description=EXCLUDED.description,
                 review_status=CASE WHEN gates_pilot_edges.review_status IN ('accepted','rejected')
                   THEN gates_pilot_edges.review_status ELSE 'pending' END,
                 confidence=GREATEST(gates_pilot_edges.confidence,EXCLUDED.confidence),
                 properties=gates_pilot_edges.properties || EXCLUDED.properties,
                 updated_at=now()""",
            (CORPUS_ID, edge_id, source.node_id, target.node_id, proposal["relationship"],
             proposal.get("notes"), proposal["confidence"], properties),
        )
        evidence_id = stable_id("semedgeev", CORPUS_ID, edge_id, proposal["proposal_id"])
        ev = proposal["evidence"]
        cur.execute(
            """INSERT INTO gates_pilot_edge_evidence
               (corpus_id,evidence_id,edge_id,paper_id,chunk_id,source_kind,locator,excerpt,
                extraction_method,confidence,properties)
               VALUES (%s,%s,%s,%s,%s,'semantic_proposal',%s,%s,'structured_literature_review',%s,%s::jsonb)
               ON CONFLICT (corpus_id,evidence_id) DO UPDATE SET
                 locator=EXCLUDED.locator,excerpt=EXCLUDED.excerpt,
                 confidence=EXCLUDED.confidence,properties=EXCLUDED.properties""",
            (CORPUS_ID, evidence_id, edge_id, paper_id, chunk_id,
             f"physical PDF page {ev['page_number']}; {ev.get('section') or 'section unavailable'}",
             ev["excerpt"], proposal["confidence"],
             json.dumps({"proposal_id": proposal["proposal_id"], "notes": proposal.get("notes")})),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    proposals = load_proposals(args.input)
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            summary = build_summary(cur, proposals)
            if args.apply:
                apply(cur, proposals)
        if args.apply:
            conn.commit()
        else:
            conn.rollback()
        summary["mode"] = "apply" if args.apply else "dry-run"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
