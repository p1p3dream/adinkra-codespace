#!/usr/bin/env python3
"""Hybrid full-text and vector search over the isolated full corpus."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SESSION_SCRIPTS = Path.home() / ".claude/skills/session-assistant/scripts"
sys.path.insert(0, str(SESSION_SCRIPTS))
from embeddings import embed_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--corpus-id", default="gates_graphrag_full")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or GATES_GRAPHRAG_DSN is required")
    import psycopg2
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            if args.lexical_only:
                cur.execute("""SELECT p.stable_identifier,p.title,c.page_start,c.section_title,
                    ts_rank_cd(to_tsvector('english',c.content),websearch_to_tsquery('english',%s)) score,
                    left(regexp_replace(c.content,E'[\\n\\r]+',' ','g'),360)
                    FROM gates_full_chunks c JOIN gates_full_papers p USING(corpus_id,paper_id)
                    WHERE c.corpus_id=%s AND to_tsvector('english',c.content) @@ websearch_to_tsquery('english',%s)
                    ORDER BY score DESC,c.chunk_id LIMIT %s""", (args.query,args.corpus_id,args.query,args.limit))
            else:
                vector = "[" + ",".join(format(v, ".9g") for v in embed_text(args.query)) + "]"
                cur.execute("""WITH candidates AS (
                    SELECT c.chunk_id,p.stable_identifier,p.title,c.page_start,c.section_title,c.content,
                           row_number() OVER (ORDER BY c.content_embedding <=> %s::vector) vector_rank,
                           NULL::bigint lexical_rank
                    FROM gates_full_chunks c JOIN gates_full_papers p USING(corpus_id,paper_id)
                    WHERE c.corpus_id=%s AND c.content_embedding IS NOT NULL ORDER BY vector_rank LIMIT %s
                ), lexical AS (
                    SELECT c.chunk_id,p.stable_identifier,p.title,c.page_start,c.section_title,c.content,
                           NULL::bigint vector_rank,
                           row_number() OVER (ORDER BY ts_rank_cd(to_tsvector('english',c.content),websearch_to_tsquery('english',%s)) DESC) lexical_rank
                    FROM gates_full_chunks c JOIN gates_full_papers p USING(corpus_id,paper_id)
                    WHERE c.corpus_id=%s AND to_tsvector('english',c.content) @@ websearch_to_tsquery('english',%s) LIMIT %s
                ), fused AS (
                    SELECT chunk_id,max(stable_identifier) stable_identifier,max(title) title,max(page_start) page_start,
                           max(section_title) section_title,max(content) content,
                           sum(CASE WHEN vector_rank IS NULL THEN 0 ELSE 1.0/(60+vector_rank) END +
                               CASE WHEN lexical_rank IS NULL THEN 0 ELSE 1.0/(60+lexical_rank) END) score
                    FROM (SELECT * FROM candidates UNION ALL SELECT * FROM lexical) q GROUP BY chunk_id)
                SELECT stable_identifier,title,page_start,section_title,score,
                       left(regexp_replace(content,E'[\\n\\r]+',' ','g'),360)
                FROM fused ORDER BY score DESC,chunk_id LIMIT %s""",
                (vector,args.corpus_id,args.limit*4,args.query,args.corpus_id,args.query,args.limit*4,args.limit))
            for stable,title,page,section,score,excerpt in cur.fetchall():
                print(f"{float(score):.5f}  {stable}  page {page}  {section or '[section unavailable]'}")
                print(f"  {title}\n  {excerpt}\n")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
