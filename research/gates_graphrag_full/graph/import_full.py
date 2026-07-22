#!/usr/bin/env python3
"""Build or import the isolated full-corpus Gates literature graph.

Dry-run is the default. ``--apply`` is the only mode that opens a database.
The importer is additive and uses only ``gates_full_*`` tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "metadata" / "manifest.json"
DEFAULT_SHARDS = ROOT / "extraction" / "shards"
DEFAULT_CITATIONS = ROOT / "citations" / "citations.jsonl"
DEFAULT_UNRESOLVED = ROOT / "citations" / "unresolved.jsonl"
DEFAULT_SEMANTIC = ROOT / "semantic" / "proposals.jsonl"
MIGRATIONS = Path(__file__).resolve().parent / "migrations"
ALLOWED_BASES = {"bibliographic", "explicit_text", "automated_inference", "manual"}
ALLOWED_REVIEWS = {"observed", "pending", "accepted", "rejected"}


class InputError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{sha_text(chr(31).join(map(str, parts)))[:24]}"


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def norm_identifier(kind: str, value: str) -> str:
    value = str(value).strip()
    if kind == "arxiv":
        value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value, flags=re.I)
        value = re.sub(r"v\d+$", "", value.removesuffix(".pdf"), flags=re.I)
    elif kind == "doi":
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
        value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    elif kind == "inspire":
        value = re.sub(r"^https?://inspirehep\.net/literature/", "", value, flags=re.I)
        value = value.removeprefix("inspire:")
    return value.casefold().strip().rstrip("/")


def validate_no_nul(value: Any, path: str) -> None:
    """Reject PostgreSQL-incompatible U+0000 without changing source data."""
    if isinstance(value, str):
        if "\x00" in value:
            raise InputError(f"U+0000 NUL character at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.<key:{key!r}>"
            validate_no_nul(key, key_path)
            child = f"{path}.{key}" if isinstance(key, str) and key.isidentifier() else f"{path}[{key!r}]"
            validate_no_nul(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_no_nul(item, f"{path}[{index}]")


def validate_plan_strings(plan: "Plan") -> None:
    validate_no_nul(plan.corpus_id, "plan.corpus_id")
    validate_no_nul(plan.manifest_sha256, "plan.manifest_sha256")
    validate_no_nul(plan.warnings, "plan.warnings")
    for collection_name in (
        "node_evidence", "edge_evidence", "sources", "papers", "identifiers",
        "artifacts", "chunks", "nodes", "edges",
    ):
        collection = getattr(plan, collection_name)
        for record_key, record in collection.items():
            validate_no_nul(record_key, f"plan.{collection_name}.<record-key>")
            try:
                validate_no_nul(record, f"plan.{collection_name}[{record_key!r}]")
            except InputError as exc:
                properties = record.get("properties", {}) if isinstance(record, dict) else {}
                source_path = properties.get("source_path") if isinstance(properties, dict) else None
                source_line = properties.get("source_line") if isinstance(properties, dict) else None
                if source_path:
                    locator = f"{source_path}:{source_line}" if source_line else str(source_path)
                    raise InputError(f"{exc}; source {locator}") from exc
                raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise InputError(f"{path}:{line_no}: JSONL row is not an object")
        row["_source_path"] = str(path.resolve())
        row["_source_line"] = line_no
        rows.append(row)
    return rows


@dataclass
class Plan:
    corpus_id: str
    manifest_sha256: str
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)
    identifiers: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    edge_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, Any]:
        return {
            "papers": len(self.papers),
            "corpus_papers": sum(not x["is_external_stub"] for x in self.papers.values()),
            "external_stubs": sum(x["is_external_stub"] for x in self.papers.values()),
            "identifiers": len(self.identifiers),
            "artifacts": len(self.artifacts),
            "chunks": len(self.chunks),
            "nodes": len(self.nodes),
            "nodes_by_type": dict(sorted(Counter(x["node_type"] for x in self.nodes.values()).items())),
            "edges": len(self.edges),
            "edges_by_relationship": dict(sorted(Counter(x["relationship"] for x in self.edges.values()).items())),
            "edges_by_review_status": dict(sorted(Counter(x["review_status"] for x in self.edges.values()).items())),
            "node_evidence": len(self.node_evidence),
            "edge_evidence": len(self.edge_evidence),
            "sources": len(self.sources),
            "warnings": len(self.warnings),
        }

    def digest(self) -> str:
        payload = {
            name: sorted(getattr(self, name).values(), key=lambda x: canonical_json(x))
            for name in ("sources", "papers", "identifiers", "artifacts", "chunks", "nodes", "edges", "node_evidence", "edge_evidence")
        }
        return sha_text(canonical_json(payload))


def add_source(plan: Plan, path: Path, role: str, count: int) -> None:
    if not path.exists():
        return
    sid = stable_id("source", plan.corpus_id, role, str(path.resolve()))
    plan.sources[sid] = {
        "source_id": sid, "source_role": role, "local_path": str(path.resolve()),
        "sha256": sha_file(path), "record_count": count, "properties": {},
    }


def add_node(plan: Plan, node_type: str, key: str, name: str, description: str | None = None,
             properties: dict[str, Any] | None = None) -> dict[str, Any]:
    node_type = node_type.casefold().strip()
    key = key.casefold().strip()
    if not node_type or not key or not name.strip():
        raise InputError("node requires type, canonical key, and name")
    nid = stable_id("node", plan.corpus_id, node_type, key)
    candidate = {"node_id": nid, "node_type": node_type, "canonical_key": key,
                 "name": name.strip(), "description": description, "properties": properties or {}}
    current = plan.nodes.get(nid)
    if not current or len(description or "") > len(current.get("description") or ""):
        plan.nodes[nid] = candidate
    return plan.nodes[nid]


def add_edge(plan: Plan, src: dict[str, Any], dst: dict[str, Any], relationship: str,
             basis: str, review: str, confidence: float | None = None,
             description: str | None = None, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    relationship = re.sub(r"[^A-Z0-9]+", "_", relationship.upper()).strip("_")
    if basis not in ALLOWED_BASES or review not in ALLOWED_REVIEWS:
        raise InputError(f"invalid edge controls: {basis}/{review}")
    eid = stable_id("edge", plan.corpus_id, src["node_id"], dst["node_id"], relationship)
    candidate = {"edge_id": eid, "src_node_id": src["node_id"], "dst_node_id": dst["node_id"],
                 "relationship": relationship, "description": description, "basis": basis,
                 "review_status": review, "confidence": confidence, "properties": properties or {}}
    current = plan.edges.get(eid)
    if not current or (current["review_status"] == "pending" and review in {"accepted", "observed"}):
        plan.edges[eid] = candidate
    return plan.edges[eid]


def add_evidence(plan: Plan, target: dict[str, Any], target_kind: str, paper_id: str,
                 chunk_id: str | None, source_kind: str, locator: str | None,
                 excerpt: str | None, method: str, confidence: float | None,
                 properties: dict[str, Any] | None = None) -> None:
    target_id = target[f"{target_kind}_id"]
    evid = stable_id("evidence", plan.corpus_id, target_kind, target_id, paper_id,
                     chunk_id or "", locator or "", excerpt or "", method)
    row = {"evidence_id": evid, f"{target_kind}_id": target_id, "paper_id": paper_id,
           "chunk_id": chunk_id, "source_kind": source_kind, "locator": locator,
           "excerpt": excerpt, "extraction_method": method, "confidence": confidence,
           "properties": properties or {}}
    getattr(plan, f"{target_kind}_evidence")[evid] = row


def upsert_sql(table: str, columns: list[str], primary_key: str) -> str:
    """Return an idempotent upsert with review-decision protection for edges."""
    all_columns = ["corpus_id"] + columns
    update_columns = [column for column in columns if column not in primary_key.split(",")]
    assignments = []
    for column in update_columns:
        if table == "gates_full_edges" and column == "review_status":
            assignments.append(
                "review_status=CASE "
                "WHEN gates_full_edges.review_status IN ('accepted','rejected') "
                "AND EXCLUDED.review_status='pending' "
                "THEN gates_full_edges.review_status "
                "ELSE EXCLUDED.review_status END"
            )
        else:
            assignments.append(f"{column}=EXCLUDED.{column}")
    return (
        f"INSERT INTO {table} ({','.join(all_columns)}) VALUES %s "
        f"ON CONFLICT (corpus_id,{primary_key}) DO UPDATE SET {','.join(assignments)}"
    )


def paper_aliases(record: dict[str, Any]) -> list[tuple[str, str]]:
    identifiers = record.get("identifiers", {})
    found = []
    for kind in ("arxiv", "doi", "inspire", "report_number"):
        for value in identifiers.get(kind, []) or []:
            found.append((kind, norm_identifier(kind, value)))
    found.append(("paper_id", str(record["paper_id"]).casefold()))
    found.append(("canonical", str(identifiers.get("canonical") or record["paper_id"]).casefold()))
    return sorted(set(found))


def build_plan(manifest_path: Path, shard_dir: Path, citations_path: Path,
               unresolved_path: Path, semantic_paths: Iterable[Path]) -> Plan:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("papers"), list):
        raise InputError("full manifest must be an object with a papers list")
    corpus_id = str(manifest.get("corpus_id") or "gates_graphrag_full")
    plan = Plan(corpus_id, sha_file(manifest_path))
    add_source(plan, manifest_path, "manifest", len(manifest["papers"]))
    alias_owner: dict[tuple[str, str], str] = {}

    for record in manifest["papers"]:
        pid = str(record["paper_id"])
        ids = record.get("identifiers", {})
        stable = str(ids.get("canonical") or pid)
        full = record.get("full_text", {})
        plan.papers[pid] = {
            "paper_id": pid, "stable_identifier": stable, "title": str(record["title"]),
            "publication_year": record.get("year"), "document_type": record.get("document_type"),
            "is_external_stub": False, "full_text_status": str(full.get("status") or "metadata_only"),
            "properties": {"authors": record.get("authors", []), "pilot": record.get("pilot", {}),
                           "record_flags": record.get("record_flags", []), "source_record": record.get("source_record", {})},
        }
        for kind, value in paper_aliases(record):
            key = (kind, value)
            if kind != "report_number" and key in alias_owner and alias_owner[key] != pid:
                raise InputError(f"identifier collision {key}")
            if kind != "report_number":
                alias_owner[key] = pid
            plan.identifiers[(kind, value, pid)] = {"identifier_type": kind, "identifier_value": value,
                                     "paper_id": pid, "is_canonical": value == stable.casefold(), "properties": {}}
        pnode = add_node(plan, "paper", stable, str(record["title"]), properties={"paper_id": pid, "external": False})
        add_evidence(plan, pnode, "node", pid, None, "manifest", f"manifest record {record.get('corpus_order')}",
                     None, "verified_manifest", 1.0, {"source_path": str(manifest_path.resolve())})
        for author_name in record.get("authors", []) or []:
            author = add_node(plan, "author", norm_text(str(author_name)).casefold(), str(author_name))
            add_evidence(plan, author, "node", pid, None, "manifest", None, None, "verified_manifest", 1.0)
            edge = add_edge(plan, pnode, author, "AUTHORED_BY", "bibliographic", "observed", 1.0)
            add_evidence(plan, edge, "edge", pid, None, "manifest", None, None, "verified_manifest", 1.0)
        if full.get("canonical_path"):
            aid = stable_id("artifact", plan.corpus_id, pid, "canonical_pdf", full.get("sha256"))
            plan.artifacts[aid] = {
                "artifact_id": aid, "paper_id": pid, "artifact_type": "pdf",
                "local_path": full.get("canonical_path"), "source_url": full.get("source_url"),
                "sha256": full.get("sha256"), "byte_count": full.get("bytes"), "page_count": full.get("pages"),
                "is_canonical": True, "properties": {"source": full.get("source"), "hash_verification": full.get("hash_verification")},
            }
        for alt in full.get("alternate_local_artifacts", []) or []:
            aid = stable_id("artifact", plan.corpus_id, pid, alt.get("path"), alt.get("sha256"))
            plan.artifacts[aid] = {
                "artifact_id": aid, "paper_id": pid, "artifact_type": "pdf_copy",
                "local_path": alt.get("path"), "source_url": None, "sha256": alt.get("sha256"),
                "byte_count": alt.get("bytes"), "page_count": alt.get("pages"), "is_canonical": False,
                "properties": {k: v for k, v in alt.items() if k not in {"path", "sha256", "bytes", "pages"}},
            }

    def resolve(reference: Any) -> str | None:
        text = str(reference or "").casefold()
        if text in plan.papers:
            return text
        if ":" in text:
            kind, value = text.split(":", 1)
            if kind in {"arxiv", "inspire", "doi"}:
                return alias_owner.get((kind, norm_identifier(kind, value)))
        for kind in ("paper_id", "canonical", "arxiv", "inspire", "doi"):
            owner = alias_owner.get((kind, norm_identifier(kind, text)))
            if owner:
                return owner
        return None

    shard_paths = sorted(shard_dir.glob("*.jsonl")) if shard_dir.exists() else []
    for path in shard_paths:
        rows = read_jsonl(path)
        add_source(plan, path, "full_text_shard", len(rows))
        for row in rows:
            pid = resolve(row.get("paper_id")) or resolve(f"arxiv:{row.get('arxiv_id')}") or resolve(f"inspire:{row.get('inspire_id')}")
            if not pid:
                raise InputError(f"{path}: unresolved paper {row.get('paper_id')}")
            text = str(row.get("text") or row.get("content") or "")
            if not text:
                continue
            cid = str(row.get("chunk_id") or stable_id("chunk", plan.corpus_id, pid, sha_text(text)))
            page = row.get("page_number")
            plan.chunks[cid] = {
                "chunk_id": cid, "paper_id": pid, "page_start": page, "page_end": page,
                "section_title": row.get("section_heading"), "chunk_index": int(row.get("chunk_index", 0)),
                "content": text, "content_sha256": sha_text(text),
                "properties": {k: v for k, v in row.items() if k not in {"text", "content", "_source_path", "_source_line"}},
            }

    citation_rows = read_jsonl(citations_path)
    add_source(plan, citations_path, "resolved_citations", len(citation_rows))
    chunks_by_page: dict[tuple[str, int], list[str]] = {}
    for chunk in plan.chunks.values():
        if chunk["page_start"] is not None:
            chunks_by_page.setdefault((chunk["paper_id"], int(chunk["page_start"])), []).append(chunk["chunk_id"])

    def citation_chunk(paper_id: str, page: Any, excerpt: Any) -> str | None:
        if page is None:
            return None
        candidates = chunks_by_page.get((paper_id, int(page)), [])
        needle = norm_text(str(excerpt or ""))
        for cid in candidates:
            if needle and needle in norm_text(plan.chunks[cid]["content"]):
                return cid
        return candidates[0] if len(candidates) == 1 else None

    for row in citation_rows:
        src_pid, dst_pid = resolve(row.get("source_paper_id")), resolve(row.get("target_paper_id"))
        if not src_pid or not dst_pid:
            plan.warnings.append(f"unresolved internal citation {row.get('citation_id')}")
            continue
        src = add_node(plan, "paper", plan.papers[src_pid]["stable_identifier"], plan.papers[src_pid]["title"])
        dst = add_node(plan, "paper", plan.papers[dst_pid]["stable_identifier"], plan.papers[dst_pid]["title"])
        exact = str(row.get("review_status", "")).startswith("accepted_exact")
        edge = add_edge(plan, src, dst, "CITES", "bibliographic", "accepted" if exact else "pending",
                        float(row.get("confidence")) if row.get("confidence") is not None else None,
                        properties={"resolution_method": row.get("resolution_method")})
        locator = f"physical PDF page {row.get('physical_page')}; reference {row.get('reference_label')}"
        cid = citation_chunk(src_pid, row.get("physical_page"), row.get("excerpt"))
        add_evidence(plan, edge, "edge", src_pid, cid, "reference_entry", locator, row.get("excerpt"),
                     str(row.get("extraction_method") or "reference_parser"), row.get("confidence"),
                     {"citation_id": row.get("citation_id"), "source_path": row.get("_source_path"), "source_line": row.get("_source_line")})

    unresolved_rows = read_jsonl(unresolved_path)
    add_source(plan, unresolved_path, "external_citation_stubs", len(unresolved_rows))
    for row in unresolved_rows:
        src_pid = resolve(row.get("source_paper_id"))
        if not src_pid:
            plan.warnings.append(f"unresolved external citation source {row.get('stub_id')}")
            continue
        stub = str(row.get("stub_id") or stable_id("external-citation", row.get("excerpt")))
        if stub not in plan.papers:
            title = str(row.get("title_candidate") or row.get("excerpt") or stub).strip().rstrip(",")
            plan.papers[stub] = {"paper_id": stub, "stable_identifier": stub, "title": title,
                "publication_year": int(row["years"][0]) if row.get("years") else None,
                "document_type": "external_citation_stub", "is_external_stub": True,
                "full_text_status": "unavailable", "properties": {"identifiers": row.get("identifiers", {})}}
            for kind, values in (row.get("identifiers") or {}).items():
                for value in values:
                    key = (kind, norm_identifier(kind, value))
                    if (kind, key[1], stub) not in plan.identifiers:
                        plan.identifiers[(kind, key[1], stub)] = {"identifier_type": kind, "identifier_value": key[1],
                                                 "paper_id": stub, "is_canonical": False, "properties": {"external": True}}
            add_node(plan, "paper", stub, title, properties={"paper_id": stub, "external": True})
        src = add_node(plan, "paper", plan.papers[src_pid]["stable_identifier"], plan.papers[src_pid]["title"])
        dst = add_node(plan, "paper", stub, plan.papers[stub]["title"])
        locator = f"physical PDF page {row.get('physical_page')}; reference {row.get('reference_label')}"
        cid = citation_chunk(src_pid, row.get("physical_page"), row.get("excerpt"))
        add_evidence(plan, dst, "node", src_pid, cid, "reference_entry", locator, row.get("excerpt"),
                     str(row.get("extraction_method") or "reference_parser"), None,
                     {"stub_id": stub, "source_path": row.get("_source_path"), "source_line": row.get("_source_line")})
        edge = add_edge(plan, src, dst, "CITES", "bibliographic", "pending", None,
                        properties={"resolution_method": "external_stub"})
        add_evidence(plan, edge, "edge", src_pid, cid, "reference_entry", locator, row.get("excerpt"),
                     str(row.get("extraction_method") or "reference_parser"), None,
                     {"stub_id": stub, "source_path": row.get("_source_path"), "source_line": row.get("_source_line")})

    for semantic_path in semantic_paths:
        proposals = read_jsonl(semantic_path)
        add_source(plan, semantic_path, "semantic_proposals", len(proposals))
        for proposal in proposals:
            evidence = proposal.get("evidence") or {}
            pid = resolve(evidence.get("paper_id")) or resolve(f"arxiv:{evidence.get('paper_id')}")
            cid = evidence.get("chunk_id")
            if not pid or cid not in plan.chunks:
                raise InputError(f"{semantic_path}: invalid proposal evidence {proposal.get('proposal_id')}")
            excerpt = str(evidence.get("excerpt") or "")
            if norm_text(excerpt) not in norm_text(plan.chunks[cid]["content"]):
                raise InputError(f"{semantic_path}: excerpt mismatch {proposal.get('proposal_id')}")
            if proposal.get("review_status") != "pending" or proposal.get("basis") != "explicit_text":
                raise InputError(f"{semantic_path}: semantic proposal must be explicit_text/pending")
            entities = []
            for item in (proposal["source"], proposal["target"]):
                etype, key, name = str(item["type"]), str(item["key"]), str(item["name"])
                if etype == "paper":
                    ep = resolve(key)
                    if not ep:
                        raise InputError(f"unknown semantic paper {key}")
                    entities.append(add_node(plan, "paper", plan.papers[ep]["stable_identifier"], plan.papers[ep]["title"]))
                else:
                    entities.append(add_node(plan, etype, key, name, properties={"semantic_type": etype}))
            edge = add_edge(plan, entities[0], entities[1], proposal["relationship"], "explicit_text", "pending",
                            float(proposal["confidence"]), proposal.get("notes"),
                            {"proposal_id": proposal.get("proposal_id")})
            locator = f"physical PDF page {evidence.get('page_number')}; {evidence.get('section') or 'section unavailable'}"
            add_evidence(plan, edge, "edge", pid, cid, "semantic_proposal", locator, excerpt,
                         "structured_literature_review", float(proposal["confidence"]),
                         {"proposal_id": proposal.get("proposal_id"), "source_path": proposal.get("_source_path"),
                          "source_line": proposal.get("_source_line")})
            for entity in entities:
                if entity["node_type"] != "paper":
                    add_evidence(plan, entity, "node", pid, cid, "semantic_proposal", locator, excerpt,
                                 "structured_literature_review", float(proposal["confidence"]),
                                 {"proposal_id": proposal.get("proposal_id")})

    # Referential and evidence invariants are checked before a dry-run can succeed.
    for edge in plan.edges.values():
        if edge["src_node_id"] not in plan.nodes or edge["dst_node_id"] not in plan.nodes:
            raise InputError(f"dangling edge {edge['edge_id']}")
    for ev in plan.edge_evidence.values():
        if ev["edge_id"] not in plan.edges or ev["paper_id"] not in plan.papers:
            raise InputError(f"dangling edge evidence {ev['evidence_id']}")
    for ev in plan.node_evidence.values():
        if ev["node_id"] not in plan.nodes or ev["paper_id"] not in plan.papers:
            raise InputError(f"dangling node evidence {ev['evidence_id']}")
    validate_plan_strings(plan)
    return plan


def apply_plan(plan: Plan, dsn: str) -> None:
    try:
        import psycopg2
        from psycopg2.extras import Json, execute_values
    except ImportError as exc:
        raise RuntimeError("psycopg2 is required only for --apply") from exc
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (plan.corpus_id + ":full-import",))
            cur.execute("""CREATE TABLE IF NOT EXISTS gates_full_schema_migrations (
                migration_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
            for migration in sorted(MIGRATIONS.glob("*.sql")):
                migration_sha = sha_file(migration)
                cur.execute("SELECT sha256 FROM gates_full_schema_migrations WHERE migration_id=%s", (migration.name,))
                existing = cur.fetchone()
                if existing and existing[0] != migration_sha:
                    raise RuntimeError(f"applied migration changed: {migration.name}")
                if existing:
                    continue
                cur.execute(migration.read_text(encoding="utf-8"))
                cur.execute("""INSERT INTO gates_full_schema_migrations(migration_id,sha256)
                    VALUES (%s,%s) ON CONFLICT (migration_id) DO NOTHING""", (migration.name,migration_sha))
            cur.execute("""INSERT INTO gates_full_corpora
                (corpus_id,schema_version,description,manifest_sha256,plan_sha256,properties)
                VALUES (%s,'1.0','Full S. James Gates publication corpus',%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id) DO UPDATE SET manifest_sha256=EXCLUDED.manifest_sha256,
                plan_sha256=EXCLUDED.plan_sha256,properties=EXCLUDED.properties,updated_at=now()""",
                (plan.corpus_id, plan.manifest_sha256, plan.digest(), canonical_json({"counts": plan.counts()})))

            specs = [
                ("gates_full_ingest_sources", plan.sources, ["source_id","source_role","local_path","sha256","record_count","properties"]),
                ("gates_full_papers", plan.papers, ["paper_id","stable_identifier","title","publication_year","document_type","is_external_stub","full_text_status","properties"]),
                ("gates_full_identifiers", plan.identifiers, ["identifier_type","identifier_value","paper_id","is_canonical","properties"]),
                ("gates_full_artifacts", plan.artifacts, ["artifact_id","paper_id","artifact_type","local_path","source_url","sha256","byte_count","page_count","is_canonical","properties"]),
                ("gates_full_chunks", plan.chunks, ["chunk_id","paper_id","page_start","page_end","section_title","chunk_index","content","content_sha256","properties"]),
                ("gates_full_nodes", plan.nodes, ["node_id","node_type","canonical_key","name","description","properties"]),
                ("gates_full_edges", plan.edges, ["edge_id","src_node_id","dst_node_id","relationship","description","basis","review_status","confidence","properties"]),
                ("gates_full_node_evidence", plan.node_evidence, ["evidence_id","node_id","paper_id","chunk_id","source_kind","locator","excerpt","extraction_method","confidence","properties"]),
                ("gates_full_edge_evidence", plan.edge_evidence, ["evidence_id","edge_id","paper_id","chunk_id","source_kind","locator","excerpt","extraction_method","confidence","properties"]),
            ]
            pk = {"gates_full_ingest_sources":"source_id", "gates_full_papers":"paper_id",
                  "gates_full_identifiers":"identifier_type,identifier_value,paper_id", "gates_full_artifacts":"artifact_id",
                  "gates_full_chunks":"chunk_id", "gates_full_nodes":"node_id", "gates_full_edges":"edge_id",
                  "gates_full_node_evidence":"evidence_id", "gates_full_edge_evidence":"evidence_id"}
            for table, mapping, columns in specs:
                if not mapping:
                    continue
                values = []
                for row in mapping.values():
                    values.append((plan.corpus_id,) + tuple(Json(row[c]) if c == "properties" else row.get(c) for c in columns))
                sql = upsert_sql(table, columns, pk[table])
                execute_values(cur, sql, values, page_size=1000, template=None)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--unresolved", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument("--semantic", type=Path, action="append", default=[])
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    semantics = args.semantic or ([DEFAULT_SEMANTIC] if DEFAULT_SEMANTIC.exists() else [])
    plan = build_plan(args.manifest, args.shard_dir, args.citations, args.unresolved, semantics)
    result = {"mode": "apply" if args.apply else "dry-run", "corpus_id": plan.corpus_id,
              "manifest_sha256": plan.manifest_sha256, "plan_sha256": plan.digest(),
              "counts": plan.counts(), "warnings": plan.warnings[:50]}
    if args.snapshot:
        args.snapshot.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.apply:
        if not args.dsn:
            parser.error("--dsn or GATES_GRAPHRAG_DSN is required with --apply")
        apply_plan(plan, args.dsn)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
