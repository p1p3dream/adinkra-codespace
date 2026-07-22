# Claims and reported results review

## Scope

This work package records material claims, reported results, qualifications, stated evidentiary support, one derivation, and one explicit assumption across the nine papers in the Gates GraphRAG pilot.

All 41 proposals remain `pending`. Nothing in this directory has been imported into the graph database.

## Coverage

| Paper | Proposals |
|---|---:|
| 1911.00807 | 4 |
| 2002.08502 | 5 |
| 2006.03609 | 4 |
| 2007.07390 | 3 |
| 2012.13308 | 5 |
| 2012.14015 | 2 |
| 2304.09830 | 8 |
| 2311.06842 | 7 |
| 2407.09334 | 3 |
| **Total** | **41** |

## Relationship counts

| Relationship | Count |
|---|---:|
| `DERIVES` | 1 |
| `MAKES_CLAIM` | 11 |
| `QUALIFIES` | 7 |
| `REPORTS_RESULT` | 19 |
| `REQUIRES_ASSUMPTION` | 1 |
| `SUPPORTS` | 2 |

No `CONTRADICTS` proposal was added. The reviewed passages did not explicitly establish a contradiction between two identified claims or results. Tension, open problems, and limitations are represented with `QUALIFIES` instead.

No proposal uses `PROVES`.

## Evidence policy

Every proposal contains:

- a physical PDF page number;
- the exact extracted `chunk_id`;
- an excerpt present in that chunk after NFKC and whitespace normalization;
- an explicit-text basis;
- a pending review status;
- a confidence value;
- a note when the paper states a conjecture, condition, limitation, or unresolved problem.

Claims and results are separate graph nodes. A paper-to-claim or paper-to-result edge does not convert the authors' statement into an independently verified fact.

## Review decisions requiring attention

1. **1911.00807:** The ten-dimensional conformal-supergravity direction is recorded with both the scan offered in support and the stated possible obstruction above six dimensions. The scan does not establish the existence of a higher-dimensional conformal-supergravity theory.
2. **2002.08502:** The M-theory and prepotential statement is explicitly a conjecture. The added note in proof observes the missing `{55}` representation and narrows the scalar-superfield interpretation toward a semi-prepotential.
3. **2006.03609:** The tying-rule branching statement is explicitly a conjecture. The paper says it is not a replacement for a rigorous mathematical proof. The component construction is also explicitly reducible.
4. **2007.07390:** The algorithmic embedding capability is an asserted use of the libraries. This proposal does not assert that the pilot reproduced the algorithm for arbitrary spectra. Complete supersymmetry transformations still require supercovariant derivative operators.
5. **2012.13308:** The permutahedron embedding and Sudoku statements retain the authors' conjectural or interpretive wording. This is the confirmed fourth paper from Gates's supplied set, and the proposals depend on its text.
6. **2012.14015:** The two proposals record results stated in the abstract. The phrase "same class" is retained rather than expanded into a stronger equivalence claim.
7. **2304.09830:** The reported `G(3)` values are taken from the table in the evidence chunk. The formula for the special `S(2r)` cases remains a conjecture. The count of 30 pairings is limited to decomposable constructions and has a separate `REQUIRES_ASSUMPTION` edge for distinct pair members.
8. **2311.06842:** The reported `eχ(1) = eχ(2)` result is separated from the statement that the values depend on the field definitions. Extension to valise definitions and zero net nodal vorticity for all Adinkras remain conjectures.
9. **2407.09334:** Repeated representations are described as a necessary but insufficient indicator of reducibility. The paper separately states that complete reduction to an irreducible representation remains a challenge.

## Deterministic reproduction

Run:

```bash
python3 research/gates_graphrag_pilot/enrichment/claims_results/build.py
python3 research/gates_graphrag_pilot/enrichment/claims_results/validate.py
```

The current result is recorded in `VALIDATION.json` and passes with no errors.

## Files

- `proposals.jsonl`: 41 pending relationship proposals
- `ENTITY_ALIASES.json`: 11 canonical claim or result entities with literature aliases
- `build.py`: deterministic proposal generator with excerpt matching
- `validate.py`: deterministic schema, provenance, coverage, and excerpt validator
- `VALIDATION.json`: current validation result and content hashes
