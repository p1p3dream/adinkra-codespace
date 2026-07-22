#!/usr/bin/env python3
"""Populate missing chunk and node embeddings for one isolated corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


SESSION_SCRIPTS = Path.home() / ".claude/skills/session-assistant/scripts"
sys.path.insert(0, str(SESSION_SCRIPTS))
import embeddings  # noqa: E402


EMBED_TARGETS = (
    ("gates_full_chunks", "chunk_id", "content", "content_embedding", "chunks_embedded"),
    ("gates_full_nodes", "node_id", "name || E'\\n' || coalesce(description,'')", "description_embedding", "nodes_embedded"),
)


def vector_literal(values: list[float]) -> str:
    if len(values) != 768:
        raise ValueError(f"expected 768 embedding dimensions, received {len(values)}")
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def missing_counts(cur: Any, corpus_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table, _, _, embedding_column, count_key in EMBED_TARGETS:
        cur.execute(
            f"SELECT count(*) FROM {table} WHERE corpus_id=%s AND {embedding_column} IS NULL",
            (corpus_id,),
        )
        counts[count_key.replace("_embedded", "_missing")] = int(cur.fetchone()[0])
    return counts


def embed_missing(
    conn: Any,
    corpus_id: str,
    batch_size: int,
    limit: int | None,
    embed_many: Callable[..., list[list[float]]] = embeddings.embed_texts,
    execute_batch_fn: Callable[..., Any] | None = None,
) -> dict[str, int]:
    """Embed missing rows, committing each selected database batch.

    ``limit`` applies across chunks and nodes together. A failed or malformed
    embedding response is rejected before its database batch is updated.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if execute_batch_fn is None:
        from psycopg2.extras import execute_batch as execute_batch_fn

    counts = {"chunks_embedded": 0, "nodes_embedded": 0}
    remaining = limit
    with conn.cursor() as cur:
        for table, id_column, text_sql, embedding_column, count_key in EMBED_TARGETS:
            while remaining is None or remaining > 0:
                take = min(batch_size, remaining) if remaining is not None else batch_size
                cur.execute(
                    f"SELECT {id_column},{text_sql} FROM {table} "
                    f"WHERE corpus_id=%s AND {embedding_column} IS NULL "
                    f"ORDER BY {id_column} LIMIT %s",
                    (corpus_id, take),
                )
                rows = cur.fetchall()
                if not rows:
                    break

                texts = [str(text) for _, text in rows]
                vectors = embed_many(texts, batch_size=len(texts))
                if len(vectors) != len(rows):
                    raise ValueError(
                        f"embedding response count {len(vectors)} does not match selected row count {len(rows)}"
                    )
                vector_values = [vector_literal(vector) for vector in vectors]
                update_sql = (
                    f"UPDATE {table} SET {embedding_column}=%s::vector,updated_at=now() "
                    f"WHERE corpus_id=%s AND {id_column}=%s AND {embedding_column} IS NULL"
                )
                params = [
                    (vector, corpus_id, object_id)
                    for vector, (object_id, _) in zip(vector_values, rows, strict=True)
                ]
                execute_batch_fn(cur, update_sql, params, page_size=len(params))
                conn.commit()
                counts[count_key] += len(rows)

                if remaining is not None:
                    remaining -= len(rows)
                    if remaining == 0:
                        break
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--corpus-id", default="gates_graphrag_full")
    parser.add_argument("--batch-size", type=positive_int, default=100)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    import psycopg2
    conn = psycopg2.connect(args.dsn)
    counts = {"chunks_missing": 0, "nodes_missing": 0, "chunks_embedded": 0, "nodes_embedded": 0}
    try:
        with conn.cursor() as cur:
            counts.update(missing_counts(cur, args.corpus_id))
        if args.apply:
            counts.update(embed_missing(conn, args.corpus_id, args.batch_size, args.limit))
        if not args.apply:
            conn.rollback()
        print(json.dumps({"mode": "apply" if args.apply else "dry-run", **counts}, sort_keys=True))
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
