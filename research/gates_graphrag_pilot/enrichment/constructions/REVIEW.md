# Mathematical construction relationship review

## Scope

This work package examines the nine papers in the Gates GraphRAG pilot for explicit mathematical construction relationships. It does not infer relationships from titles, co-occurrence, citation, or vector similarity.

All 34 proposals have `review_status: pending`. None has been imported into the graph database.

## Coverage

| Paper | Proposals |
|---|---:|
| 1911.00807 | 3 |
| 2002.08502 | 3 |
| 2006.03609 | 5 |
| 2007.07390 | 2 |
| 2012.13308 | 3 |
| 2012.14015 | 4 |
| 2304.09830 | 5 |
| 2311.06842 | 4 |
| 2407.09334 | 5 |
| **Total** | **34** |

## Relationship counts

| Relationship | Count |
|---|---:|
| `CATALOGS` | 3 |
| `CLASSIFIES` | 1 |
| `COMPUTES` | 3 |
| `CONSTRUCTS` | 4 |
| `DECOMPOSES_INTO` | 2 |
| `ENCODES` | 1 |
| `EQUIVALENT_TO` | 4 |
| `GENERATED_BY` | 3 |
| `ISOMORPHIC_TO` | 1 |
| `MAPS_TO` | 3 |
| `PARTITIONS_INTO` | 2 |
| `REDUCES_TO` | 1 |
| `REPRESENTS` | 6 |

No proposal uses `ENUMERATES`, `REALIZES`, `LIFTS_TO`, `EQUIVALENCE_CLASS_OF`, or `QUOTIENT_OF`. The reviewed passages did not support those predicates more precisely than the predicates used here.

## Evidence policy

Each proposal includes:

- a physical PDF page number;
- the exact extracted `chunk_id`;
- a short excerpt present in that chunk after whitespace normalization;
- an explicit-text basis;
- a pending review status;
- a confidence value and, where needed, a qualification.

The validator checks all of these properties against `/tmp/gates-graphrag-pilot/chunks-enriched.jsonl`. It also checks controlled vocabulary, entity-key prefixes, proposal-ID uniqueness, pilot-paper coverage, alias structure, and deterministic file hashes.

Run:

```bash
python3 research/gates_graphrag_pilot/enrichment/constructions/validate.py
```

The current result is recorded in `VALIDATION.json` and passes with no errors.

## Review decisions requiring attention

1. **2007.07390, proposal 2:** The abstract says the Adynkra libraries support exploration of component-multiplet embeddings. `CATALOGS` is plausible but may be broader than the authors' wording. Keep, narrow, or reject after examining the library tables.
2. **2012.14015, proposal 4:** `REPRESENTS` records the explicit dependence of Garden-algebra matrices on permutation elements. `ENCODES` may be preferred if the graph vocabulary distinguishes representation from parameterization.
3. **2311.06842, proposal 4:** The excerpt explicitly reduces 12 by 12 matrix multiplication to 4 by 4 multiplication. The target name specifies three blocks from the surrounding block-diagonal construction. Confirm that target wording during review.
4. **2007.07390, proposal 1:** `CONSTRUCTS` records the paper's presentation of the libraries. This does not assert that the paper computes every possible embedding.
5. **2012.14015, proposal 2:** `EQUIVALENT_TO` is restricted to membership in the same HYMN class. It is not an unrestricted equivalence of the two supermultiplets.
6. **2012.13308:** This is the confirmed fourth paper from Gates's supplied set. Its relationships are independently evidence-backed within the paper.

## Files

- `proposals.jsonl`: 34 pending relationship proposals
- `ENTITY_ALIASES.json`: 11 canonical entities with literature aliases
- `validate.py`: deterministic validator
- `VALIDATION.json`: current validation result and content hashes
