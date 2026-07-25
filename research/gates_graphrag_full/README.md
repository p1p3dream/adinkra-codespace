# Gates literature graph

This corpus contains 295 publication records attributed to S. James Gates,
Jr. Local full text is available for 166 works. The other 129 records remain
metadata-only unless cited works supply enough information for a structured
external stub.

## Current graph

| Record | Count |
|---|---:|
| Corpus publications | 295 |
| Local PDFs used for extraction | 166 |
| Page-anchored text chunks | 10,153 |
| Resolved citation occurrences | 1,165 |
| Distinct internal citation edges | 1,131 |
| External citation stubs | 2,694 |
| Semantic proposals | 166 |
| Graph nodes | 3,407 |
| Graph edges | 4,827 |

All nodes and edges have provenance records. Citation edges resolved by
identifier are accepted, authorship is observed, and title-resolved citations
and semantic proposals remain pending review.

The semantic pass is intentionally narrow. It supplies one passage-backed
proposal for each full-text paper. It is a coverage baseline, not an
exhaustive account of every claim, method, construction, or comparison in the
corpus.

## Packages

- `metadata/`: canonical work manifest and curation record
- `extraction/`: deterministic PDF extraction with physical-page anchors
- `citations/`: bibliography parsing, resolution, external stubs, and review queue
- `semantic/`: conservative relationship proposals and entity aliases
- `graph/`: isolated PostgreSQL/pgvector schema, import, search, and traversal

Each package contains its own validation and reproduction instructions.

## Validate

```bash
python3 research/gates_graphrag_full/metadata/validate_manifest.py
python3 research/gates_graphrag_full/extraction/validate_extraction.py
python3 research/gates_graphrag_full/citations/validate.py \
  --manifest research/gates_graphrag_full/metadata/manifest.json \
  --artifact-dir research/gates_graphrag_full/citations
python3 research/gates_graphrag_full/semantic/validate_semantic.py
python3 research/gates_graphrag_full/graph/validate_full.py
```

## Search and traverse

Set `GATES_GRAPHRAG_DSN` to the isolated PostgreSQL/pgvector database, then:

```bash
python3 research/gates_graphrag_full/graph/search_full.py \
  "permutahedron adinkra"

python3 research/gates_graphrag_full/graph/explore_full.py \
  arxiv:2012.13308 --depth 1 --include-pending
```

Pending relationships are excluded from traversal unless
`--include-pending` is supplied.
