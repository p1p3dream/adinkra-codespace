# Count reconciliation

## PDF artifacts

The manifest contains 166 canonical publication PDFs. It also records four
exact-byte copies in `Gates_specifically_called_out`. These are alternate local
artifacts of the same four works, not additional papers or additional full-text
inputs.

| Artifact role | Count |
|---|---:|
| Canonical publication PDF | 166 |
| Exact called-out copy | 4 |
| Graph artifact rows | 170 |

Only the 166 canonical PDFs produce extraction shards and chunks.

## Citation edges and review states

The repaired citation input has 1,165 resolved internal citation occurrences:

| Resolution class | Occurrences | Distinct source-target pairs |
|---|---:|---:|
| Exact arXiv or DOI identifier | 1,014 | 995 |
| Normalized title containment | 151 | 140 |

Nineteen repeated exact occurrences and eleven repeated title occurrences
collapse under the deterministic `(corpus, source, target, CITES)` edge key.
Four source-target pairs occur in both resolution classes. Exact identifier
resolution takes precedence for those four pairs. The internal graph therefore
has 995 accepted edges and 136 title-only pending edges, for 1,131 distinct
internal citation edges.

The unresolved-citation input contains 2,694 external stubs. Each has one
distinct source-stub pair and remains pending. Final citation review counts are:

| Review state | Derivation | Edges |
|---|---|---:|
| Accepted | 995 exact-identifier pairs | 995 |
| Pending | 136 title-only pairs + 2,694 external stubs | 2,830 |
| Total `CITES` | 995 + 2,830 | 3,825 |

This is not an addition of raw occurrence counts. Repeated references collapse,
and the four exact/title overlaps have one accepted edge rather than conflicting
accepted and pending edges. The importer enforces this precedence regardless of
input order.
