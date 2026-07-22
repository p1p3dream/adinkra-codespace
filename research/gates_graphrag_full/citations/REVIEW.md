# Gates full-corpus citation graph review

## Result

- Source manifest: `research/gates_graphrag_full/metadata/manifest.json`
- Manifest records: 295 verified publications
- PDFs processed: 166 of 166 locally available files
- PDF extraction errors: 0
- Physical PDF pages examined: 4,934
- Papers with explicit reference headings: 164
- Papers with accepted reference sections: 165
- Numbered reference entries extracted: 4,578
- Citation occurrences resolved to the 295-record corpus: 1,165
- Distinct source-target citation pairs: 1,131
- Source papers with at least one resolved corpus citation: 126
- External unresolved stubs retained: 2,694
- Entries omitted for insufficient bibliographic evidence: 794
- PostgreSQL-forbidden U+0000 replacements: 16 in one reference entry

Validation passes with no errors. A second complete run produced identical
artifact hashes.

## Text safety

One PDF text layer emitted 16 U+0000 characters in a bibliography entry. U+0000
cannot be stored in PostgreSQL text. The extractor replaces each occurrence
with U+FFFD before writing evidence or title candidates. The replacement does
not change identifier or title resolution. Validation checks parsed strings and
raw artifact bytes and rejects any remaining U+0000.

## Resolution evidence

| Method | Occurrences | Confidence | Review status |
|---|---:|---:|---|
| Exact arXiv identifier | 746 | 1.00 | accepted exact identifier |
| Exact DOI identifier | 268 | 1.00 | accepted exact identifier |
| Unique normalized-title containment | 151 | 0.95 | pending title review |

Identifier resolution is limited to exact normalized identifiers in the source
manifest. Title resolution requires the complete normalized corpus title to
occur contiguously in a bibliography entry and requires that title to identify
one corpus record. No approximate-title or embedding match creates a citation.

The 151 pending rows are the review queue. They can be selected from
`citations.jsonl` by `review_status == "pending_title_review"`. Each row carries
the source paper, target paper, physical PDF page, bibliography label, evidence
excerpt, matched title and confidence.

## Reference extraction

The parser accepts numbered entries only after an explicit `References`,
`Bibliography`, `Notes and References`, or `References and Notes` heading. Text
blocks are ordered by column before their lines are parsed so two-column
bibliographies do not become prose mentions.

One paper, INSPIRE 1821396, has no extractable reference heading. Its final
pages contain a continuous numbered bibliography from `[1]` through `[114]`.
The deterministic fallback accepted this section because labels 1 through 6
occur in sequence in the final 35 percent of the PDF.

The 568-page book *Superspace Or One Thousand and One Lessons in
Supersymmetry*, INSPIRE 195126, has no explicit or fallback reference section in
its extracted text. It was processed but supplied no citation records.

## External stubs

`unresolved.jsonl` is not a claim that every stub is a distinct publication. A
stub is retained only when the entry contains an exact external identifier or
at least three of these signals:

- year
- journal or publisher
- author pattern
- quoted title

The raw bibliography excerpt and all detected identifiers are preserved.
External identifiers were not resolved over the network. Stub deduplication
therefore remains future work.

## Deliberate exclusions

- Prose mentions are not citations.
- Text similarity is not a citation.
- Unnumbered text without sufficient bibliography structure is omitted.
- Self-edges are suppressed. One was encountered.
- The 129 records without local full text are target nodes only. Their outgoing
  references were not extracted.
- Approximate title matching is not used.
- No semantic relationship other than `CITES` is asserted here.

## Known limitations

- Extraction depends on each PDF's text layer and does not perform OCR.
- Mathematical symbols, ligatures, line-end hyphenation and printed page
  numbers are left as extracted in evidence excerpts.
- A reference label can contain several cited works. Each resolved work becomes
  a separate citation occurrence with the same label and page evidence.
- The 151 title-based resolutions require human review before acceptance.
- External stub identities and duplicate external works have not been resolved.

## Reproduction

The commands are documented in `README.md`. `VALIDATION.json` is the final
machine-readable check, and `metrics.json` contains per-paper extraction counts.
