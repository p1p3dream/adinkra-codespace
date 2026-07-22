# Gates full-corpus literature graph

This layer assembles the verified 295-work metadata set, full text from all 166
local publication PDFs, internal and external citation records, and optional
evidence-backed semantic proposals. It is isolated from the focused pilot.

## Validate and preview

```bash
python3 research/gates_graphrag_full/graph/validate_full.py
python3 research/gates_graphrag_full/graph/import_full.py
python3 -m unittest discover -s research/gates_graphrag_full/graph/tests -v
```

Dry-run is the default. It builds the complete deterministic plan, checks
references and evidence, prints counts, and never connects to PostgreSQL.

## Apply after review

```bash
python3 research/gates_graphrag_full/graph/import_full.py \
  --dsn "$GATES_GRAPHRAG_DSN" --apply

python3 research/gates_graphrag_full/graph/embed_full.py \
  --dsn "$GATES_GRAPHRAG_DSN" --apply
```

The embedding pass selects missing rows in database batches, submits each
selected batch through one `embed_texts` call, validates cardinality and all 768
dimensions, updates with a batched statement, and commits that batch. A restart
continues from rows whose embedding is still null. `--limit` applies across
chunks and nodes together.

The importer applies `migrations/*.sql` in lexical order, records migration
checksums, takes a corpus-specific transaction lock, and upserts deterministic
keys. It never deletes rows. Changed applied migrations stop the import.

Semantic proposals may be added or rerun as a separate idempotent phase:

```bash
python3 research/gates_graphrag_full/graph/import_full.py \
  --semantic research/gates_graphrag_full/semantic/proposals.jsonl \
  --dsn "$GATES_GRAPHRAG_DSN" --apply
```

## Retrieve and traverse

```bash
python3 research/gates_graphrag_full/graph/search_full.py \
  "permutahedron adinkra" --dsn "$GATES_GRAPHRAG_DSN"

python3 research/gates_graphrag_full/graph/explore_full.py \
  arxiv:2012.13308 --depth 2 --dsn "$GATES_GRAPHRAG_DSN"
```

Search uses reciprocal-rank fusion of vector and PostgreSQL full-text results.
`--lexical-only` avoids the embedding provider. Traversal excludes pending
relationships unless `--include-pending` is explicit. The explorer accepts a
node ID, raw canonical key, or `arxiv:`, `doi:`, and `inspire:` identifier. It
prints each traversed edge with its direction, relationship, review state,
neighbor, physical evidence locator, and a bounded evidence excerpt.

See [INPUT_CONTRACTS.md](INPUT_CONTRACTS.md) for accepted inputs, review-state
rules, and provenance requirements. See
[COUNT_RECONCILIATION.md](COUNT_RECONCILIATION.md) for the 166/170 PDF-artifact
distinction and the citation occurrence-to-edge calculation.
