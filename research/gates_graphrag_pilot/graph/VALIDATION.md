# Pilot validation

Validation date: 2026-07-20

The nine-paper pilot has been extracted, imported and vectorized in the local
PostgreSQL vector store under `gates_literature_pilot`.

## Imported records

| Record | Count |
|---|---:|
| papers and PDF artifacts | 9 each |
| alternate identifiers | 24 |
| page-anchored chunks | 855 |
| chunks with 768-dimensional vectors | 855 |
| total graph nodes | 218 |
| nodes with vectors | 218 |
| authorship edges | 35 |
| within-pilot citation edges | 13 |
| series-membership edges | 9 |
| semantic edges pending review | 233 |
| total graph edges | 290 |
| distinct relationship types | 40 |
| node-evidence records | 309 |
| edge-evidence records | 290 |
| import warnings | 0 |

All 218 nodes and all 290 edges have provenance records. The nine confirmed
series memberships, including arXiv:2012.13308, are `accepted`. Authorship and
exact-identifier citation records are `observed`.
All 233 semantic relationships remain `pending`.

The extraction output is deterministic. Its SHA-256 digest after invalid PDF
U+0000 characters were replaced with U+FFFD is:

`ba5e6f478209c87f28d15bb48197489688c31a85659275af580a518de55fc6dd`

The manifest SHA-256 digest is:

`d1d2c69f537faff441d6c40229d526e890a7836dccb75a5e937b6843b1c1a848`

## Retrieval check

A vector query for how permutahedra enter Adinkra constructions returned
arXiv:2304.09830 first, followed by the permutahedron section of
arXiv:2012.13308. The first result directly discusses the new mapping from
Adinkra graphs to a permutahedron decorated by colored nodes.

A graph-neighborhood query for `hopping operator` resolves the canonical
concept first and returns its `INTRODUCES`, `DEFINES`, `CONSTRUCTS`, and
`GENERATED_BY` relationships with physical-page excerpts.

## Isolation

The existing shared `graphrag_*` tables do not carry corpus identifiers and
cannot safely isolate this literature. The pilot therefore uses dedicated
`gates_pilot_*` tables. No shared graph rows were changed.

## Tests

Seventeen integration tests pass. They cover deterministic extraction, physical-page and
section provenance, invalid PDF character handling, stable-identifier
deduplication, graph provenance, review status, citation stubs, dry-run safety
semantic-proposal validation and the absence of destructive or shared-graph
statements in the schema. Each work package also has its own deterministic
validation.

## Deliberate boundary

The semantic graph is populated but not promoted to accepted fact. Its 233
relationships are pending because passage-level support establishes what the
papers state, not independent correctness. Review can accept, reject or narrow
individual edges without changing the underlying source chunks.
