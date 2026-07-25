#!/usr/bin/env python3
"""Populate vector embeddings for the isolated Gates literature pilot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch

SESSION_SCRIPTS = Path.home() / ".claude/skills/session-assistant/scripts"
sys.path.insert(0, str(SESSION_SCRIPTS))
from embeddings import DIMS, embed_texts  # noqa: E402


def vector_literal(values: list[float]) -> str:
    if len(values) != DIMS:
        raise ValueError(f"expected {DIMS} dimensions, got {len(values)}")
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def embed_chunks(dsn: str, corpus_id: str, batch_size: int) -> int:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT chunk_id, content FROM gates_pilot_chunks
                   WHERE corpus_id=%s AND content_embedding IS NULL
                   ORDER BY chunk_id""",
                (corpus_id,),
            )
            rows = cur.fetchall()
        completed = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embed_texts([text for _, text in batch], batch_size=batch_size)
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """UPDATE gates_pilot_chunks SET content_embedding=%s::vector, updated_at=now()
                       WHERE corpus_id=%s AND chunk_id=%s""",
                    [(vector_literal(vector), corpus_id, chunk_id) for (chunk_id, _), vector in zip(batch, vectors)],
                    page_size=batch_size,
                )
            conn.commit()
            completed += len(batch)
            print(f"embedded {completed}/{len(rows)} chunks", flush=True)
        return completed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def embed_nodes(dsn: str, corpus_id: str, batch_size: int) -> int:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT node_id, name || CASE WHEN description IS NULL THEN ''
                          ELSE E'\n' || description END
                   FROM gates_pilot_nodes
                   WHERE corpus_id=%s AND description_embedding IS NULL
                   ORDER BY node_id""",
                (corpus_id,),
            )
            rows = cur.fetchall()
        completed = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embed_texts([text for _, text in batch], batch_size=batch_size)
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """UPDATE gates_pilot_nodes SET description_embedding=%s::vector, updated_at=now()
                       WHERE corpus_id=%s AND node_id=%s""",
                    [(vector_literal(vector), corpus_id, node_id) for (node_id, _), vector in zip(batch, vectors)],
                    page_size=batch_size,
                )
            conn.commit()
            completed += len(batch)
            print(f"embedded {completed}/{len(rows)} nodes", flush=True)
        return completed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--corpus-id", default="gates_literature_pilot")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    chunk_count = embed_chunks(args.dsn, args.corpus_id, args.batch_size)
    node_count = embed_nodes(args.dsn, args.corpus_id, args.batch_size)
    print(f"embedded {chunk_count} new chunks and {node_count} new nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
