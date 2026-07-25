# Full Gates corpus PDF extraction

This directory contains the deterministic evidence extraction for every local
PDF recorded as `downloaded` in the verified Gates publication manifest. It
wraps the pilot extractor without changing its page anchoring, literal-heading
or mathematical-text policies.

## Rebuild

Install the pilot extractor requirements, then run:

```bash
python3 -m pip install -r research/gates_graphrag_pilot/extraction/requirements.txt
research/gates_graphrag_full/extraction/rebuild.sh
```

The build reads the canonical
`research/gates_graphrag_full/metadata/manifest.json`, whose verified paths
resolve into `~/Documents/S_James_Gates_Publications/pdfs/`. It never modifies
the manifest or PDFs and does not write to a database. A source that fails is
tried twice, recorded in the failure report and left as a failed index record.
It is never omitted silently. The builder also accepts the original source CSV
through `--manifest`.

## Outputs

- `shards/*.jsonl`: one canonical JSONL shard per downloaded paper
- `extraction_index.jsonl`: one success or failure record per input paper
- `quality_report.json`: per-paper and corpus page, chunk, word, anchor and
  invalid-character measurements
- `retry_failure_report.json`: retry policy, recovered inputs and failures
- `input_coverage.json`: explicit accounting of downloaded and metadata-only
  manifest records
- `determinism_report.json`: full repeat-extraction byte comparison
- `validation_report.json`: independent index, shard, anchor, count and hash
  consistency checks

Paper identifiers use the first manifest arXiv ID exactly, including old-style
identifiers such as `hep-th/9709104`. Papers without an arXiv ID use
`inspire:<INSPIRE ID>`. File names replace `/` and `:` with `__`; the index is
the authority mapping a paper identifier to its shard.

Every chunk retains its physical page number and label, line interval, bounding
box, supported literal section heading, PDF SHA-256 and extractor provenance.
No OCR, equation repair, dehyphenation, header removal or inferred section name
is introduced.
