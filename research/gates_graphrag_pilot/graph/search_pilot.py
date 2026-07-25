#!/usr/bin/env python3
"""Hybrid vector and full-text search over the Gates literature pilot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2

SESSION_SCRIPTS = Path.home() / ".claude/skills/session-assistant/scripts"
sys.path.insert(0, str(SESSION_SCRIPTS))
from embeddings import embed_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--corpus-id", default="gates_literature_pilot")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    vector = "[" + ",".join(format(value, ".9g") for value in embed_text(args.query)) + "]"
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.stable_identifier, p.title, c.page_start, c.section_title,
                          1 - (c.content_embedding <=> %s::vector) AS similarity,
                          left(regexp_replace(c.content, E'[\\n\\r]+', ' ', 'g'), 360)
                   FROM gates_pilot_chunks c
                   JOIN gates_pilot_papers p USING (corpus_id, paper_id)
                   WHERE c.corpus_id=%s AND c.content_embedding IS NOT NULL
                   ORDER BY c.content_embedding <=> %s::vector
                   LIMIT %s""",
                (vector, args.corpus_id, vector, args.limit),
            )
            for stable_id, title, page, section, similarity, excerpt in cur.fetchall():
                print(f"{similarity:.4f}  {stable_id}  page {page}  {section or '[section unavailable]'}")
                print(f"  {title}\n  {excerpt}\n")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
