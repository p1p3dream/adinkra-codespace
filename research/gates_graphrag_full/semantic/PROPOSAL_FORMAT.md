# Proposal contract

Each line of `proposals.jsonl` follows the pilot semantic proposal contract and
adds a required `method` field. Required fields are:

- stable `proposal_id`;
- typed `source` and `target` entities;
- controlled `relationship`;
- `basis` equal to `explicit_text`;
- `review_status` equal to `pending`;
- numeric `confidence`;
- `method` identifying the deterministic cue rule;
- evidence with `paper_id`, `chunk_id`, physical `page_number`, section and a
  verbatim excerpt.

The source excerpt must occur in the named chunk after Unicode and whitespace
normalization.
