# Concepts and methods enrichment review

## Scope

This work package proposes evidence-backed relationships of the following types:

- `INTRODUCES`
- `DEFINES`
- `STUDIES`
- `USES`
- `DEPENDS_ON`
- `ASSUMES`

No `MENTIONS`, `RELATED_TO`, or inferred similarity relationships are included. All proposals remain `pending`.

## Results

- 53 relationship proposals
- 44 canonical concept, method, invariant, algorithm, and assumption targets
- 9 papers covered
- 53 excerpts verified against the specified chunks after whitespace normalization
- 0 duplicate source-relationship-target triples
- 0 unsupported excerpts

### Relationships

| Relationship | Count |
|---|---:|
| `ASSUMES` | 1 |
| `DEFINES` | 15 |
| `INTRODUCES` | 12 |
| `STUDIES` | 9 |
| `USES` | 16 |
| `DEPENDS_ON` | 0 |

No `DEPENDS_ON` proposal was made because the reviewed passages did not state a dependency with sufficient precision. Citations, method use, and conceptual continuity were not converted into dependencies.

### Paper coverage

| arXiv | Proposals |
|---|---:|
| 1911.00807 | 5 |
| 2002.08502 | 6 |
| 2006.03609 | 6 |
| 2007.07390 | 6 |
| 2012.13308 | 4 |
| 2012.14015 | 3 |
| 2304.09830 | 6 |
| 2311.06842 | 9 |
| 2407.09334 | 8 |

## Canonicalization decisions

- `Adinkra` and `adynkra` remain distinct concepts.
- `unfolded Adinkra` and `infinite unfolded Adinkra` remain distinct concepts.
- The two net-centric quantities `eχ(1)` and `eχ(2)` remain separate invariants.
- `Adynkra genome` is distinct from the algorithm used to construct one.
- `adynkrafield` is distinct from the overlap construction used to obtain one from an Adynkra genome.
- `ten-dimensional Adinkra graph` is distinct from the broader method called higher-dimensional Adinkra technology.
- Height Yielding Matrix Number, HYMN, HYMNs, HYMN value, and HYMN values resolve to one invariant.
- Hopping operator, hopper, left hopping operator, and right hopping operator resolve to one concept at this stage. A later review can split left and right operators if their algebraic roles require separate nodes.
- Line-break hyphens in several excerpts are preserved because the evidence must remain verbatim with respect to extracted PDF text.

## Review priorities

1. arXiv:2012.13308 is confirmed as the fourth paper from Gates's supplied set. Its four proposals are independently supported by that paper.
2. Review whether left and right hopping operators should be distinct concept nodes rather than aliases of the common hopping-operator concept.
3. Review whether the `INTRODUCES` relationship for higher-dimensional Adinkra technology should instead be limited to `STUDIES`. The cited text says the chapter introduces the technology.
4. Review the scope of the `USES` relationship between arXiv:2012.14015 and Height Yielding Matrix Numbers. The evidence directly states that HYMN selects candidate combinations, but this does not by itself establish completeness.
5. Preserve the qualification on the nilpotent-level-parameter proposal. The source calls nilpotency a postulate, so the relationship is `ASSUMES`, not `DEFINES` or `REPORTS_RESULT`.

## Rebuild and validation

Run from the repository root after producing `/tmp/gates-graphrag-pilot/chunks-enriched.jsonl`:

```bash
python3 research/gates_graphrag_pilot/enrichment/concepts_methods/build.py
python3 research/gates_graphrag_pilot/enrichment/concepts_methods/validate.py
```

Expected validation summary:

```text
PASS: 53 proposals
PASS: 44 canonical targets
PASS: all excerpts verified across 9 papers
relationships: ASSUMES=1, DEFINES=15, INTRODUCES=12, STUDIES=9, USES=16
papers: 1911.00807=5, 2002.08502=6, 2006.03609=6, 2007.07390=6, 2012.13308=4, 2012.14015=3, 2304.09830=6, 2311.06842=9, 2407.09334=8
```
