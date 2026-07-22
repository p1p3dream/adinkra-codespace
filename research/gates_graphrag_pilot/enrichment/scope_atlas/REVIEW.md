# Atlas, classification, algebra, and scope relationship review

## Scope

This work package examines the nine Gates GraphRAG pilot papers for explicit relationships involving research scope, groups, algebras, representations, supermultiplets, inputs, and outputs.

All 80 proposals have `review_status: pending`. None has been imported into the graph database.

## Coverage

| Paper | Proposals |
|---|---:|
| 1911.00807 | 6 |
| 2002.08502 | 6 |
| 2006.03609 | 8 |
| 2007.07390 | 8 |
| 2012.13308 | 8 |
| 2012.14015 | 8 |
| 2304.09830 | 13 |
| 2311.06842 | 10 |
| 2407.09334 | 13 |
| **Total** | **80** |

## Relationship counts

| Relationship | Count |
|---|---:|
| `APPLIES_TO` | 13 |
| `USES_GROUP` | 12 |
| `USES_ALGEBRA` | 7 |
| `DESCRIBES_REPRESENTATION` | 10 |
| `DESCRIBES_MULTIPLET` | 18 |
| `HAS_INPUT` | 10 |
| `HAS_OUTPUT` | 10 |

No additional `CATALOGS` or `CLASSIFIES` proposals were added. The construction package already records the supported atlas and classification relationships, including the 10D and 11D component catalogs, the lower-dimensional Adynkra library, HYMN classification, and the partition of the S8 permutahedron. Repeating those edges here would add no new scope information.

## Evidence policy

Every proposal includes:

- a physical PDF page number;
- an exact extracted `chunk_id`;
- a short excerpt present in that chunk after whitespace normalization;
- an explicit-text basis;
- a pending review status;
- a confidence value;
- a qualification where the source uses conjectural or prospective language.

Dimension, supercharge count, color count, permutation degree, and node or vertex count are recorded in proposal notes. They are not represented as relationships.

Run:

```bash
python3 research/gates_graphrag_pilot/enrichment/scope_atlas/build.py
python3 research/gates_graphrag_pilot/enrichment/scope_atlas/validate.py
```

The current deterministic result is recorded in `VALIDATION.json` and passes with no errors.

## Review decisions requiring attention

1. **1911.00807 prepotential scan:** `HAS_OUTPUT` is limited to candidate prepotential superfields. It does not assert that the scan establishes a prepotential formulation.
2. **1911.00807 Nordström supergravity:** the multiplet edge retains the paper's qualification that the examples are reducible and have finite field content.
3. **2006.03609 general superspace scope:** the all-superspaces edge has lower confidence because the abstract describes this as a suggestion while explicitly treating ten-dimensional cases.
4. **2007.07390 library output:** the library supports exploration of component-multiplet embeddings. The proposal does not assert exhaustive enumeration.
5. **2012.13308 permutahedral embedding:** the `APPLIES_TO` edge records a conjectural research scope, not an established embedding theorem.
6. **2012.13308 auxiliary-field problem:** the paper proposes an approach based on algorithms probing symmetric groups. The edge does not claim a solution.
7. **2012.14015 GR(8,8):** the representation edge records the authors' stated interpretation of its role in combining 4D N = 1 theories.
8. **2304.09830 supermultiplet terminology:** the paper explicitly uses “supermultiplet” for an L/R matrix collection. The recursive input and output entities retain that operational meaning.
9. **2311.06842 U(3):** the group edge applies to the block-diagonal L/R matrix reduction discussed on page 38, not to every unfolded construction in the paper.
10. **2407.09334 six multiplets:** the six `DESCRIBES_MULTIPLET` edges reproduce the paper's enumerated worked examples. They do not claim a complete classification of 4D N = 1 supermultiplets.
11. **2012.13308 corpus status:** it is confirmed as the fourth paper Gates intended, and its within-paper relationships are independently evidence-backed.

## Files

- `build.py`: deterministic proposal generator
- `proposals.jsonl`: 80 pending relationship proposals
- `ENTITY_ALIASES.json`: 30 canonical entities with literature aliases
- `validate.py`: deterministic validator
- `VALIDATION.json`: validation result and content hashes
- `REVIEW.md`: scope and review decisions
