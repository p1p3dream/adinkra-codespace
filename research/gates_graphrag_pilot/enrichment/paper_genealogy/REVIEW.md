# Paper genealogy review

## Scope and method

All nine pilot papers were reviewed for explicit statements of paper-to-paper
lineage. Candidate passages were located in introductions, methodology sections,
conclusions, and passages that cited another pilot paper by reference number.
Reference numbers were resolved against each paper's bibliography.

A proposal was retained only when a physical PDF page and one extracted chunk
supported one of the controlled genealogy relationships. Citations alone were
not restated. Every retained relationship remains `pending` for review.

## Retained proposals

The file contains 25 proposals:

| Relationship | Count | Interpretation used |
|---|---:|---|
| `EXTENDS` | 8 | The later paper explicitly continues a stated program, scope, or construction. |
| `USES_RESULT_FROM` | 8 | The later paper uses an identified result, decomposition, classification, or numerical construction. |
| `REUSES_METHOD_FROM` | 6 | The later paper says it applies, follows, or adapts the earlier method. |
| `COMPARES_WITH` | 1 | The later paper explicitly compares its classification with the earlier paper's classification. |
| `CORRECTS` | 1 | The later paper explicitly corrects a rendering in the earlier paper. |
| `PRECEDES_IN_SERIES` | 1 | A later paper calls the identified earlier work part of the continuing series. |

### Adynkra sequence

- `2002.08502` extends the 10D analysis in `1911.00807` into 11D, applies the
  same adinkra-diagram technique, and reproduces a published 10D decomposition
  by projection.
- `2002.08502` explicitly says its Figure 2.1 corrects a previous rendering in
  `1911.00807`. The proposal is limited to that rendering.
- `2006.03609` carries lessons from the 11D analysis in `2002.08502` into an
  end-to-end 10D Dynkin-label-to-component-field construction. It also starts
  from a 10D adinkra published in `2002.08502`.
- `2007.07390` identifies `2006.03609` as part of the continuing series and uses
  methods from `2002.08502` and `2006.03609` in its lower-dimensional
  libraries. It also uses the earlier 10D and 11D decompositions from
  `1911.00807` and `2002.08502`.
- `2407.09334` collectively identifies `1911.00807`, `2002.08502`,
  `2006.03609`, and `2007.07390` as prior work on the path it continues. These
  four `EXTENDS` proposals share one collective passage and should be reviewed
  as a bundle. The paper separately says it adapts techniques from
  `2006.03609` and `2007.07390` to 4D, and uses the even-red-box result from
  `2006.03609`.

### Permutahedron sequence

- `2012.14015` uses the S4 permutahedron result from `2012.13308` and explicitly
  compares its split of six candidate representations with the earlier split
  of six S4 subsets.
- `2304.09830` uses the HYMN result from `2012.14015`, uses the magic-number
  construction from `2012.13308`, and explicitly develops the lower-degree
  permutahedron-face observation from `2012.14015` into higher-order
  constructions and face analysis.

## Rejected or narrowed interpretations

### Plain citations

All existing `CITES` edges were excluded. A bibliography entry or reference
number without a statement of use, continuation, correction, comparison, or
method transfer was insufficient.

### `GENERALIZES` and `SPECIALIZES`

No proposal uses these labels. Several papers move between dimensions or from
S4 to S8, but the text more precisely supports `EXTENDS`,
`REUSES_METHOD_FROM`, or `USES_RESULT_FROM`. Changing dimension alone was not
classified as a formal generalization or specialization.

### `VERSION_OF`

No pilot paper is a version of another pilot paper. arXiv revisions and
publisher artifacts belong to the same paper record and should be represented
as artifact metadata, not paper-to-paper semantic edges.

### Broad series ordering

The existing `PART_OF_SERIES` relationships already record curated series
membership. The review did not manufacture a complete chronological
`PRECEDES_IN_SERIES` chain from dates or citation order. Only the directly
supported `2006.03609` to `2007.07390` edge was retained.

### `2012.14015` and `2012.13308`

It is reasonable to describe the later S8 work as developing themes from the
S4 paper, but the strongest cited passages say that it uses and compares with
the S4 result. The proposals therefore use `USES_RESULT_FROM` and
`COMPARES_WITH`, not `GENERALIZES` or `EXTENDS`.

### `2304.09830` and `2012.13308`

The later paper extends the magic-number calculation to larger permutation
groups, but the passage that names `2012.13308` explicitly supports adoption
of its construction. The retained edge is `USES_RESULT_FROM`. A stronger
`EXTENDS` edge would require joining separate passages and was rejected.

### `2311.06842`

This paper explicitly calls unfolded Adinkras an extension of the original
Adinkra concept. It does not identify any of the other eight pilot papers as
that source, and it has no within-pilot citation. No paper-to-paper genealogy
edge was proposed. Its concept-level `EXTENDS` relationship belongs in the
concept or construction work package.

### `1911.00807`

This paper describes extensions of lower-dimensional programs and methods, but
those cited antecedents are outside the nine-paper pilot. It therefore appears
as the target of later genealogy proposals, not as the source of a new
within-pilot proposal.

## Validation

Run:

```bash
python3 research/gates_graphrag_pilot/enrichment/paper_genealogy/validate.py
```

The validator checks the controlled vocabulary, canonical paper titles and
identifiers, physical page and chunk agreement, exact excerpt containment after
whitespace normalization, pending status, confidence range, duplicate IDs,
and duplicate semantic edges. `VALIDATION.json` records hashes and counts.
