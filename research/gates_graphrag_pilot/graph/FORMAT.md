# Pilot input contract

The importer accepts one manifest and one or more extracted JSONL files.

## Manifest

The manifest may be a JSON list, a JSON object with a `papers` list, or CSV.
Each paper should contain a title and at least one stable identifier:

```json
{
  "inspire_id": "1234567",
  "arxiv_ids": "2304.09830",
  "dois": "10.1000/example",
  "title": "Paper title",
  "year": "2023",
  "authors": "Gates, S.J.; Coauthor, A.",
  "pdf_filename": "2304.09830.pdf",
  "pdf_source_url": "https://arxiv.org/pdf/2304.09830",
  "sha256": "..."
}
```

The importer normalizes INSPIRE, DOI, and arXiv identifiers. Records sharing
any normalized identifier become one paper with multiple artifacts. The pilot
uses arXiv as its canonical paper key and keeps INSPIRE and DOI as aliases. If
none of those identifiers is available, a normalized-title hash is the
fallback.

## Extracted JSONL

Each line is associated with a source paper by `paper_id`, `stable_id`, or
`stable_identifier`. Use a normalized value such as `arxiv:2304.09830` or
`inspire:1234567`. The focused extractor also uses the bare arXiv identifier,
such as `2304.09830`, which the importer resolves through identifier aliases.

The section-aware extractor emits one flat chunk per line. The importer maps
`page_number` to the physical page range, `section_heading` to the section
title, and retains `page_label`, line anchors, bounding box, token-count
provenance, extraction backend and version, source path and hash, and the
mathematical-text policy in chunk properties.

```json
{
  "paper_id": "arxiv:2304.09830",
  "chunk_id": "2304.09830:results:1",
  "text": "Section-aware extracted text...",
  "section_heading": "Results",
  "page_number": 12,
  "page_label": "11",
  "page_line_start": 0,
  "page_line_end": 26,
  "bbox": [72.0, 90.0, 540.0, 710.0],
  "extraction_provenance": {
    "backend": "pdftotext",
    "backend_version": "...",
    "source_sha256": "...",
    "mathematical_text_policy": "..."
  },
  "concepts": [
    {
      "id": "hopping-operators",
      "name": "hopping operators",
      "description": "Description grounded in this passage",
      "excerpt": "Short supporting passage",
      "confidence": 0.94
    }
  ],
  "claims": [{"id": "claim-1", "text": "A claim stated in the paper"}],
  "results": [{"id": "result-1", "text": "A reported result"}],
  "series": [{"id": "permutahedron-atlas", "name": "Permutahedron atlas"}],
  "citations": [
    {
      "arxiv_id": "2012.14015",
      "locator": "reference 17",
      "extraction_method": "reference_parser"
    }
  ],
  "relationships": [
    {
      "source": "hopping-operators",
      "target": "result-1",
      "relationship": "USED_IN",
      "basis": "automated_inference",
      "excerpt": "Short supporting passage",
      "confidence": 0.82
    }
  ]
}
```

`source` and `target` refer to entity IDs or normalized names defined in the
same JSONL record. Unresolved relationships are skipped with a warning.

## Evidence and review rules

- Manifest authorship and parsed bibliography citations are `observed`.
- Text-extracted concept, claim, result, series, and arbitrary relationships
  default to `pending`.
- `automated_inference` edges cannot be marked `observed`.
- Human review may change a pending edge to `accepted` or `rejected` in a
  later review process.
- Every node and edge must have a provenance row identifying the source paper,
  extraction method, and, when available, chunk, page or section, and excerpt.
