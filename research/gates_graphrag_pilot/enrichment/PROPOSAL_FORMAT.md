# Semantic relationship proposal format

Each work package writes one JSON object per line to `proposals.jsonl`.

```json
{
  "proposal_id": "stable-work-package-id",
  "source": {
    "type": "paper",
    "key": "arxiv:2304.09830",
    "name": "N = 2 SUSY and the 7-Simplex"
  },
  "relationship": "INTRODUCES",
  "target": {
    "type": "concept",
    "key": "concept:hopping-operator",
    "name": "hopping operator"
  },
  "evidence": {
    "paper_id": "2304.09830",
    "chunk_id": "2304.09830:p0003:c000",
    "page_number": 3,
    "section": "Introduction",
    "excerpt": "A short verbatim passage supporting the relationship."
  },
  "basis": "explicit_text",
  "review_status": "pending",
  "confidence": 0.9,
  "notes": "Optional qualification."
}
```

## Requirements

- Use only relationships in `RELATIONSHIPS.md`.
- Use stable, singular, lowercase entity keys with a type prefix.
- Use `arxiv:<id>` for paper keys.
- Every proposal requires a physical PDF page, source chunk and short excerpt.
- The excerpt must occur in the named chunk after whitespace normalization.
- Preserve negative or qualified statements in `notes`.
- Extracted semantic proposals remain `pending`.
- Do not use `RELATED_TO`, `MENTIONS` or `PROVES`.
- Do not infer a relationship from vector similarity, title similarity or
  co-occurrence alone.
- Prefer one supported edge over several weak paraphrases of the same fact.
