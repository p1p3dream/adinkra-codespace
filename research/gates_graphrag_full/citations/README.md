# Gates full-corpus citation extraction

This directory contains a conservative citation graph derived from the 166
locally available PDFs in the verified 295-record Gates publication manifest.

Only numbered entries found after an explicit `References`, `Bibliography`, or
equivalent heading are eligible. Prose mentions and vector similarity are not
citations. Resolution uses exact arXiv, DOI, or INSPIRE identifiers before
normalized title containment. Exact identifier matches are accepted. Title
matches remain pending review.

PostgreSQL cannot store U+0000 in text values. Any U+0000 emitted by a PDF text
layer is deterministically replaced with U+FFFD. Replacement counts are recorded
in `metrics.json`, and validation rejects U+0000 in parsed or raw artifacts.

## Reproduce

```bash
python3 research/gates_graphrag_full/citations/extract_citations.py \
  --manifest research/gates_graphrag_full/metadata/manifest.json \
  --output-dir research/gates_graphrag_full/citations

python3 research/gates_graphrag_full/citations/validate.py \
  --manifest research/gates_graphrag_full/metadata/manifest.json \
  --artifact-dir research/gates_graphrag_full/citations

python3 -m unittest research/gates_graphrag_full/citations/test_extract_citations.py
```

## Artifacts

- `citations.jsonl`: citations resolved to the 295-record corpus
- `unresolved.jsonl`: external citation stubs with sufficient bibliographic data
- `aliases.json`: exact identifier and normalized-title resolution indexes
- `metrics.json`: corpus-wide and per-paper extraction metrics
- `VALIDATION.json`: machine-readable validation report
- `REVIEW.md`: methods, results, review queue, and limitations
