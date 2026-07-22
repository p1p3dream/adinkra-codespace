# Gates pilot PDF extraction

This module produces deterministic, section-aware JSONL from source PDFs. It is
the evidence extraction stage only. It does not write to the vector store or any
database.

## Properties

- Preserves a supplied or inferred paper identifier.
- Anchors every chunk to a physical PDF page, extracted line range and bounding
  box.
- Uses literal headings found on the page. PDF outline entries may confirm that
  a printed line is a heading, but an outline title is never substituted for
  page text.
- Suppresses front-matter table-of-contents entries as section boundaries.
- Keeps extracted mathematical lines unchanged. It does not repair symbols,
  join line-ending hyphens, reinterpret equations or apply OCR.
- Replaces invalid embedded U+0000 characters with U+FFFD because PostgreSQL
  text fields cannot store U+0000. Each chunk records the replacement count.
- Records the PDF SHA-256, extraction backend and version, heading method and
  counting methods in every row.
- Emits keys and inputs in a stable order. It includes no run timestamp.

## Install

```bash
python3 -m pip install -r research/gates_graphrag_pilot/extraction/requirements.txt
```

## Extract PDFs

Paper IDs are inferred from arXiv identifiers in filenames when available:

```bash
python3 research/gates_graphrag_pilot/extraction/extract_papers.py \
  ~/Documents/S_James_Gates_Publications/pdfs/2020_2012.14015_*.pdf \
  ~/Documents/S_James_Gates_Publications/pdfs/2020_2007.07390_*.pdf \
  --output /tmp/gates-pilot-chunks.jsonl
```

Supply a stable identifier explicitly when a filename has no arXiv identifier:

```bash
python3 research/gates_graphrag_pilot/extraction/extract_papers.py \
  --paper 'inspire:12345=/path/to/source.pdf' \
  --output /tmp/chunks.jsonl
```

## Extract a manifest

CSV, JSON and JSONL are accepted. Recognized identifiers are `paper_id`, `id`,
`arxiv_id`, `arxiv_ids` and `inspire_id`. Recognized PDF fields are `pdf_path`,
`local_pdf_path`, `path`, `file` and `pdf_filename`. A relative
`pdf_filename` is checked relative to the manifest and then its adjacent
`pdfs/` directory. Manifest rows without a PDF field are skipped.

```bash
python3 research/gates_graphrag_pilot/extraction/extract_papers.py \
  --manifest /path/to/pilot-manifest.csv \
  --target-words 300 \
  --output /tmp/gates-pilot-chunks.jsonl
```

## Output fields

Each line is one JSON object containing:

- paper identity: `paper_id`, `arxiv_id`, `inspire_id`, `title`
- chunk identity: `chunk_id`, `chunk_index`, `page_chunk_index`
- page evidence: `page_number`, `page_label`, `page_line_start`,
  `page_line_end`, `bbox`
- section evidence: `section_heading`, `section_start_page`,
  `section_heading_source`
- content and counts: `text`, `word_count`, `token_count`
- reproducibility: `counting_provenance`, `extraction_provenance`

`token_count` is explicitly a Unicode lexical-unit count. It is not a count for
any model-specific tokenizer.

## Validation on local Gates PDFs

With the default 300-word target and PyMuPDF 1.27.2.3:

| arXiv | pages | chunks | extracted words | recovered section boundaries |
|---|---:|---:|---:|---:|
| 2012.14015 | 15 | 30 | 5,187 | 7 |
| 2007.07390 | 66 | 128 | 17,310 | 31 |

For 2012.14015, the recovered headings include `Introduction`, `4D, N = 1 SUSY
and the Permutahedron`, `Acknowledgments` and `References`. For 2007.07390, PDF
outline alignment recovers the dimension chapters and their printed
subheadings without treating its two Contents pages as article sections.

Run the tests with:

```bash
python3 -m pip install -r research/gates_graphrag_pilot/extraction/requirements-dev.txt
python3 -m pytest -q research/gates_graphrag_pilot/extraction/tests
```

## Limitations

- PDF text extraction is not mathematical typesetting recovery. Subscripts,
  superscripts, matrices, reading order and uncommon glyphs can remain damaged.
- There is no OCR fallback for image-only pages.
- Multi-column or unusually positioned text may follow the PDF's internal order
  rather than visual reading order.
- Repeated running headers, footers and page numbers are retained rather than
  removed without evidence.
- A heading absent from both page typography and the PDF outline remains
  unidentified. The extractor leaves the section null or carries forward the
  last supported heading instead of inventing one.
- A chunk boundary is a retrieval convenience. The PDF page and line anchors,
  not the chunk, remain the citation authority.
