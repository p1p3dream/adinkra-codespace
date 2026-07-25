#!/usr/bin/env python3
"""Find a semantic entity and display its evidence-backed graph neighborhood."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2

SESSION_SCRIPTS = Path.home() / ".claude/skills/session-assistant/scripts"
sys.path.insert(0, str(SESSION_SCRIPTS))
from embeddings import embed_text  # noqa: E402


def vector_literal(text: str) -> str:
    return "[" + ",".join(format(value, ".9g") for value in embed_text(text)) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--corpus-id", default="gates_literature_pilot")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--edge-limit", type=int, default=40)
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    vector = vector_literal(args.query)
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT node_id,name,node_type,coalesce(properties->>'semantic_type',node_type),
                          1-(description_embedding <=> %s::vector) AS similarity
                   FROM gates_pilot_nodes
                   WHERE corpus_id=%s AND description_embedding IS NOT NULL
                   ORDER BY description_embedding <=> %s::vector LIMIT %s""",
                (vector, args.corpus_id, vector, args.candidates),
            )
            candidates = cur.fetchall()
            if not candidates:
                print("No matching entities.")
                return 1
            print("Candidates:")
            for index, (_, name, _, semantic_type, similarity) in enumerate(candidates, 1):
                print(f"  {index}. {similarity:.4f}  [{semantic_type}] {name}")
            node_id, name, _, semantic_type, _ = candidates[0]
            print(f"\nNeighborhood for [{semantic_type}] {name}:\n")
            cur.execute(
                """SELECT CASE WHEN e.src_node_id=%s THEN 'out' ELSE 'in' END AS direction,
                          e.relationship,o.name,coalesce(o.properties->>'semantic_type',o.node_type),
                          e.review_status,ev.locator,ev.excerpt
                   FROM gates_pilot_edges e
                   JOIN gates_pilot_nodes o ON o.corpus_id=e.corpus_id AND o.node_id=
                     CASE WHEN e.src_node_id=%s THEN e.dst_node_id ELSE e.src_node_id END
                   LEFT JOIN LATERAL (
                     SELECT locator,excerpt FROM gates_pilot_edge_evidence x
                     WHERE x.corpus_id=e.corpus_id AND x.edge_id=e.edge_id
                     ORDER BY x.evidence_id LIMIT 1
                   ) ev ON true
                   WHERE e.corpus_id=%s AND (e.src_node_id=%s OR e.dst_node_id=%s)
                   ORDER BY e.relationship,o.name LIMIT %s""",
                (node_id, node_id, args.corpus_id, node_id, node_id, args.edge_limit),
            )
            for direction, relation, other, other_type, status, locator, excerpt in cur.fetchall():
                arrow = "->" if direction == "out" else "<-"
                print(f"  {arrow} {relation} [{status}] [{other_type}] {other}")
                if locator:
                    print(f"     {locator}")
                if excerpt:
                    print(f"     {excerpt[:300]}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
