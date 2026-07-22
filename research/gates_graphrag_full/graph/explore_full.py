#!/usr/bin/env python3
"""Traverse evidence-backed relationships around one full-corpus node."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any


IDENTIFIER_TYPES = {"arxiv", "doi", "inspire"}


def identifier_lookup(value: str) -> tuple[str | None, str | None]:
    """Parse a canonical prefixed identifier for gates_full_identifiers."""
    text = value.strip()
    if ":" not in text:
        return None, None
    kind, identifier = text.split(":", 1)
    kind = kind.casefold().strip()
    if kind not in IDENTIFIER_TYPES:
        return None, None
    identifier = identifier.strip().casefold()
    if kind == "arxiv":
        identifier = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", identifier)
        identifier = re.sub(r"v\d+$", "", identifier.removesuffix(".pdf"))
    elif kind == "doi":
        identifier = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", identifier)
        identifier = identifier.removeprefix("doi:")
    elif kind == "inspire":
        identifier = re.sub(r"^https?://inspirehep\.net/literature/", "", identifier)
        identifier = identifier.removeprefix("inspire:").rstrip("/")
    return kind, identifier or None


def resolution_query() -> str:
    return """WITH direct AS (
              SELECT n.node_id,n.node_type,n.canonical_key,n.name,0 AS match_priority
              FROM gates_full_nodes n
              WHERE n.corpus_id=%s AND (n.node_id=%s OR n.canonical_key=%s)
            ), by_identifier AS (
              SELECT n.node_id,n.node_type,n.canonical_key,n.name,1 AS match_priority
              FROM gates_full_identifiers i
              JOIN gates_full_papers p
                ON p.corpus_id=i.corpus_id AND p.paper_id=i.paper_id
              JOIN gates_full_nodes n
                ON n.corpus_id=p.corpus_id AND n.node_type='paper'
               AND (n.properties->>'paper_id'=p.paper_id OR n.canonical_key=p.stable_identifier)
              WHERE i.corpus_id=%s AND i.identifier_type=%s AND i.identifier_value=%s
            ), candidates AS (
              SELECT * FROM direct UNION SELECT * FROM by_identifier
            )
            SELECT node_id,node_type,canonical_key,name
            FROM candidates
            ORDER BY match_priority,node_id
            LIMIT 1"""


def traversal_query() -> str:
    return """WITH RECURSIVE walk
              (current_node_id,depth,node_path,edge_id,from_node_id,to_node_id) AS (
              SELECT %s::text,0,ARRAY[%s::text],NULL::text,NULL::text,NULL::text
              UNION ALL
              SELECT neighbor.node_id,w.depth+1,w.node_path || neighbor.node_id,
                     e.edge_id,w.current_node_id,neighbor.node_id
              FROM walk w
              JOIN gates_full_edges e
                ON e.corpus_id=%s
               AND (e.src_node_id=w.current_node_id OR e.dst_node_id=w.current_node_id)
              CROSS JOIN LATERAL (
                SELECT CASE WHEN e.src_node_id=w.current_node_id
                            THEN e.dst_node_id ELSE e.src_node_id END AS node_id
              ) neighbor
              WHERE w.depth < %s
                AND (%s OR e.review_status IN ('observed','accepted'))
                AND NOT neighbor.node_id=ANY(w.node_path)
            ), ranked AS (
              SELECT depth,edge_id,from_node_id,to_node_id,
                     row_number() OVER (
                       PARTITION BY edge_id ORDER BY depth,from_node_id,to_node_id
                     ) AS occurrence_rank
              FROM walk WHERE edge_id IS NOT NULL
            )
            SELECT r.depth,e.edge_id,r.from_node_id,source.node_type,source.canonical_key,source.name,
                   r.to_node_id,neighbor.node_type,neighbor.canonical_key,neighbor.name,
                   e.relationship,e.review_status,
                   CASE WHEN e.src_node_id=r.from_node_id THEN 'outgoing' ELSE 'incoming' END,
                   evidence.locator,evidence.excerpt
            FROM ranked r
            JOIN gates_full_edges e
              ON e.corpus_id=%s AND e.edge_id=r.edge_id
            JOIN gates_full_nodes source
              ON source.corpus_id=e.corpus_id AND source.node_id=r.from_node_id
            JOIN gates_full_nodes neighbor
              ON neighbor.corpus_id=e.corpus_id AND neighbor.node_id=r.to_node_id
            LEFT JOIN LATERAL (
              SELECT ev.locator,ev.excerpt
              FROM gates_full_edge_evidence ev
              WHERE ev.corpus_id=e.corpus_id AND ev.edge_id=e.edge_id
              ORDER BY (ev.locator IS NULL),(ev.excerpt IS NULL),ev.evidence_id
              LIMIT 1
            ) evidence ON true
            WHERE r.occurrence_rank=1
            ORDER BY r.depth,e.relationship,source.canonical_key,neighbor.canonical_key,e.edge_id
            LIMIT %s"""


@dataclass(frozen=True)
class TraversedEdge:
    depth: int
    edge_id: str
    from_node_id: str
    from_type: str
    from_key: str
    from_name: str
    neighbor_node_id: str
    neighbor_type: str
    neighbor_key: str
    neighbor_name: str
    relationship: str
    review_status: str
    direction: str
    locator: str | None
    excerpt: str | None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "TraversedEdge":
        return cls(*row)


def short_text(value: str | None, limit: int = 240) -> str:
    if not value:
        return "[excerpt unavailable]"
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def format_edge(edge: TraversedEdge) -> str:
    relation = f"{edge.relationship} | {edge.review_status}"
    if edge.direction == "outgoing":
        line = (
            f"{edge.depth}  {edge.from_name} [{edge.from_key}] "
            f"-[{relation}]-> {edge.neighbor_name} [{edge.neighbor_key}]"
        )
    else:
        line = (
            f"{edge.depth}  {edge.from_name} [{edge.from_key}] "
            f"<-[{relation}]- {edge.neighbor_name} [{edge.neighbor_key}]"
        )
    locator = edge.locator or "[physical locator unavailable]"
    return f"{line}\n   evidence: {locator}\n   excerpt: {short_text(edge.excerpt)}"


def resolve_node(cur: Any, corpus_id: str, requested: str) -> tuple[str, str, str, str] | None:
    kind, identifier = identifier_lookup(requested)
    cur.execute(
        resolution_query(),
        (corpus_id, requested, requested, corpus_id, kind, identifier),
    )
    return cur.fetchone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", help="node_id, canonical_key, or arxiv:/doi:/inspire: identifier")
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--corpus-id", default="gates_graphrag_full")
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    import psycopg2
    conn = psycopg2.connect(args.dsn)
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            node = resolve_node(cur, args.corpus_id, args.node)
            if not node:
                print(f"No node found for {args.node!r}")
                return 2
            node_id, node_type, canonical_key, name = node
            print(f"start  {node_type}  {canonical_key}  {name}")
            cur.execute(
                traversal_query(),
                (node_id,node_id,args.corpus_id,args.depth,args.include_pending,
                 args.corpus_id,args.limit),
            )
            rows = cur.fetchall()
            if not rows:
                print("[no traversable relationships under the selected review filter]")
                return 0
            for row in rows:
                print(format_edge(TraversedEdge.from_row(row)))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
