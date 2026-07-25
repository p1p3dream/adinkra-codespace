# Full-corpus semantic proposals

This directory contains a conservative semantic pass over the 166 Gates papers
with verified local PDFs. It emits one pending proposal per paper. The remaining
129 metadata-only records receive no semantic proposal because no full text is
available locally.

## Artifacts

- `proposals.jsonl`: 166 relationships with page and chunk evidence
- `nodes.jsonl`: source and target entities referenced by the proposals
- `ENTITY_ALIASES.json`: identifier aliases and inherited pilot aliases when a
  canonical pilot key is present
- `COVERAGE.json`: paper, method, relationship and fallback counts
- `VALIDATION.json`: provenance, coverage and deterministic-rebuild checks
- `METHOD.md`: selection rules and confidence policy
- `REVIEW.md`: review state and known limitations

## Rebuild and validate

```bash
python3 research/gates_graphrag_full/semantic/build_semantic.py
python3 research/gates_graphrag_full/semantic/validate_semantic.py
python3 -m unittest discover research/gates_graphrag_full/semantic/tests
```

All proposals remain `pending`. This pass does not accept claims, resolve
scientific disputes or write to a database.
