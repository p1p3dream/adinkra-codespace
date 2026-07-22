# Gates literature GraphRAG pilot

This pilot indexes nine priority papers in an isolated, evidence-backed
literature graph.

- `metadata/`: verified paper manifest and curation record
- `extraction/`: deterministic, page-anchored PDF extraction
- `graph/`: isolated schema, importer, citation enrichment, vectorization and
  search commands

The local database currently contains 9 papers, 855 vectorized chunks, 218
vectorized graph nodes and 290 provenance-backed edges. The 233 semantic edges
remain pending review. See `graph/VALIDATION.md` for the recorded validation
results and `graph/README.md` for reproduction steps.
