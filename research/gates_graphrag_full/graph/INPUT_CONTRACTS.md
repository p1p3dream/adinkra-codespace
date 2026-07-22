# Full-corpus graph input contracts

## Metadata

`../metadata/manifest.json` is authoritative. It must be a JSON object with
`corpus_id` and `papers`. Each paper requires `paper_id`, `title`,
`identifiers`, and `full_text`. The current verified package contains 295 work
records, 166 canonical local PDFs, and 129 metadata-only records.

The importer preserves one work-level paper record. arXiv, DOI, INSPIRE, and
report-number values are aliases. Report numbers are not assumed globally
unique. Exact alternate PDF copies remain artifacts of the same work.

## Full text

`../extraction/shards/*.jsonl` uses `gates-pdf-chunk-v1`. Each row must identify
its paper by a manifest alias and carry page, chunk, text, and extraction
provenance. Chunk IDs are retained. Text is not repaired by the graph layer.

## Citations

`../citations/citations.jsonl` uses `gates-citation-v1`. Exact arXiv and DOI
resolutions enter as accepted `CITES` edges. Normalized-title resolutions stay
pending. `../citations/unresolved.jsonl` becomes external paper stubs with
pending `CITES` edges. Every citation edge retains the reference entry, physical
PDF page, parser method, source file, and source line. When the matching page
chunk is unambiguous, it is linked directly.

## Semantic proposals

Zero or more `--semantic` JSONL files may use the pilot proposal contract. Every
proposal must be `explicit_text`, remain `pending`, name an existing chunk and
physical page, and provide an excerpt present in that chunk after whitespace
normalization. Missing or mismatched evidence stops the import.

## Database boundary

Dry-run is the default and does not connect to PostgreSQL. `--apply` is required
for schema migration or row writes. All storage names begin with `gates_full_`.
Every corpus row carries `corpus_id`. Imports use deterministic keys and upserts;
they do not delete rows or change the focused pilot.

Edge upserts protect completed review decisions. If an existing edge is
`accepted` or `rejected`, a later incoming `pending` record does not change that
status. An explicitly incoming `accepted` or `rejected` record may change the
stored decision. Other edge fields and evidence continue to update normally.

Before any database connection, the completed plan is recursively checked for
U+0000 in every string, record key, and nested JSON property. A NUL reports the
planned collection, record, and field path and stops the run. The graph layer
does not remove or replace it; the producing artifact must be corrected.
