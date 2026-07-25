# Review state

## Current result

- 295 metadata records
- 166 verified local PDFs
- 166 papers covered by semantic proposals
- one pending proposal per full-text paper
- 129 metadata-only records excluded from semantic extraction
- deterministic rebuild and provenance validation required to pass

## Limitations

- This is a high-precision first pass, not an exhaustive account of each paper.
- Every proposal requires human review before acceptance into a curated graph.
- Target names preserve the language available near the selected cue and may
  retain line-break artifacts from PDF text extraction.
- Equations are preserved as extracted and are not symbolically repaired.
- The pass does not infer unstated equivalence, causation, priority or paper
  genealogy.
- Citation relationships are produced by the separate citation pipeline.
- Two title-based subject proposals are lower-confidence fallbacks and should
  be reviewed first.
- Identifier aliases are conservative. No speculative synonym expansion is
  performed.

## Review order

1. Review the two title fallbacks.
2. Review passive-cue proposals.
3. Review active and special-cue proposals.
4. Accept, revise or reject each edge while retaining its source evidence.
