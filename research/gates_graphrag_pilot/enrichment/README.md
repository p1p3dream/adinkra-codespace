# Semantic relationship enrichment

Five independent work packages reviewed all nine pilot papers:

| Package | Proposals |
|---|---:|
| paper genealogy | 25 |
| concepts and methods | 53 |
| mathematical constructions | 34 |
| claims and results | 41 |
| scope, groups and multiplets | 80 |
| **Total** | **233** |

The combined graph covers 37 semantic relationship types. Every proposal has
an arXiv paper, physical PDF page, exact chunk, matching excerpt, confidence
and pending review status. The combined validator reports zero errors and zero
duplicate evidence signatures.

## Files

- `PROPOSAL_FORMAT.md`: evidence contract
- `RELATIONSHIPS.md`: controlled vocabulary
- `CANONICAL_ENTITY_MAP.json`: reviewed cross-package aliases
- `validate_proposals.py`: shared evidence validator and deterministic merger
- `canonicalize_proposals.py`: cross-package entity normalization
- `import_proposals.py`: dry-run-first transactional importer
- `combined/proposals.jsonl`: canonical 233-proposal import artifact
- `combined/VALIDATION.json`: combined validation report
- each work-package directory: proposals, aliases, review notes and validation

## Status policy

All semantic edges are `pending`. A matching passage establishes that the
paper states or uses the relationship. It does not independently establish the
mathematical or physical correctness of that statement.

arXiv:2012.13308 is confirmed as the fourth paper in Gates's four-paper set.
Its within-paper relationships remain separately supported by its text.
