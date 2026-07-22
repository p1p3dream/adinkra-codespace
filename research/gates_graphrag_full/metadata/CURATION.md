# Gates literature full-corpus curation

## Scope

This manifest preserves all 295 publication records in the curated S. James
Gates Jr. collection. It does not add publications or remove records from the
source collection.

| Record class | Count |
|---|---:|
| Publication records | 295 |
| Verified local publication PDFs | 166 |
| Metadata-only records | 129 |
| Focused-pilot papers | 9 |
| Canonical PDF pages | 4,934 |

`manifest.json` is the canonical work-level manifest. `manifest.csv` is its
deterministic flattened form. Source order is retained in `corpus_order`.

## Sources and verification

The build used only local curated materials:

- `/Users/brandon/Documents/S_James_Gates_Publications/MANIFEST.json`
- `/Users/brandon/Documents/S_James_Gates_Publications/MISSING_FULL_TEXT.csv`
- the 166 files under `S_James_Gates_Publications/pdfs/`
- the focused-pilot manifest for its nine membership labels
- the collection retrieval script for the documented false-author exclusions

For every canonical publication PDF, the build recomputed the SHA-256 digest,
file size, and page count. All 166 recomputed digests match the source manifest.
All files have PDF headers. The 166 canonical files also have 166 distinct
digests.

No web metadata was required for this pass.

## Identifier policy

The work-level `paper_id` is the first arXiv identifier when one exists. This
includes old-style identifiers such as `hep-th/9709104`. A record without an
arXiv identifier uses `inspire:<INSPIRE ID>`.

Each record retains:

- every arXiv identifier supplied by the collection manifest;
- every distinct DOI;
- its INSPIRE identifier and URL;
- every distinct report number;
- generated arXiv and DOI URLs.

Identifiers are aliases of a publication record, not separate paper nodes.
The record with INSPIRE ID `1728169` has two arXiv artifact aliases,
`1812.05097` and `1904.02328`. Both are retained. The first remains canonical
because that is their order in the source manifest.

## Pilot papers

The following nine records are marked with `pilot.included = true`:

1. `1911.00807`
2. `2002.08502`
3. `2006.03609`
4. `2007.07390`
5. `2012.13308`
6. `2012.14015`
7. `2304.09830`
8. `2311.06842`
9. `2407.09334`

All nine have verified canonical local PDFs. Their pilot order and series labels
are retained from the focused-pilot manifest.

## Artifact and duplicate handling

The publication directory contains 166 canonical PDFs. Four additional files
in `Gates_specifically_called_out/` are exact byte copies of canonical files:

- `2012.14015`
- `2304.09830`
- `2311.06842`
- `2012.13308`

They are recorded as alternate local artifacts and are not counted as separate
publications or full-text files. Their hashes match their canonical copies.

The two Gates CV PDFs under `metadata/` are collection-support documents. They
are inventoried but are not treated as publication full text.

No duplicate publication records and no duplicate canonical PDF hashes were
found. The source field `duplicate_pdf_of` is empty for all 295 records.

## Normalization findings

These are source-format findings, not publication discrepancies:

| Finding | Count | Treatment |
|---|---:|---|
| Source records with a repeated DOI value | 31 | Kept once in the normalized DOI alias array and flagged on the record |
| Records with more than one arXiv alias | 1 | Retained both aliases and flagged on the record |
| Exact duplicate noncanonical PDF copies | 4 | Attached to the canonical record as alternate artifacts |
| Metadata-support PDFs outside the publication directory | 2 | Inventoried and excluded from publication full text |

There were no differences in record count, title, source order, full-text
status, canonical file hash, or pilot membership between the relevant source
data and the generated manifest.

## False-author exclusions

INSPIRE records `2077897` and `2947909` are not members of this corpus. The
collection retrieval script identifies them as University of Southampton
photonics papers incorrectly linked to the primary Gates author profile. The
full manifest records both exclusions and their reason. Neither identifier
appears among the 295 publication records.

## Reproduction and validation

From the repository root:

```sh
python3 research/gates_graphrag_full/metadata/build_manifest.py
python3 research/gates_graphrag_full/metadata/validate_manifest.py
```

Validation performs 27 checks covering record preservation, identifier
uniqueness, file integrity, source missing-file agreement, pilot membership,
artifact copies, false-author exclusions, and JSON/CSV agreement. All 27 checks
pass. Results are in `VALIDATION.json`.

Two consecutive builds produced identical output hashes:

- `manifest.json`: `63fe5c34a1b355ace0d83197ef00dd17e62481d5610ad8b1000fe0520b6c514b`
- `manifest.csv`: `6cb32ba72ef66dc50805aa3aace2c26411a23f40858955fe736d5256365d785b`
