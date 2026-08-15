# Maxwell gauge enhancement recovered from worldline data

## Result

The Maxwell positive control is now recovered from its four-color worldline
linkage matrices rather than supplied with its four-dimensional spatial
linkages.

The search follows Section 5.5 of arXiv:0907.3605:

1. take the four raised-boson worldline linkage matrices;
2. examine signed boson and fermion frames;
3. designate three bosons as electric components and the fourth as the
   auxiliary field;
4. append three magnetic phantom fields using Eq. (5.6);
5. construct the spatial linkages using Eqs. (5.8)-(5.9);
6. build the bosonic and fermionic Omega matrices;
7. apply the Bianchi reshufflings in Eqs. (5.4)-(5.5); and
8. test the complete gauge-enhancement condition in Eq. (5.11).

No spatial linkage from the four-dimensional Maxwell source is passed into the
search. The independently transcribed source linkages are used only as a
positive-control comparison outside the search.

## Search space and normalization

There are 24 permutations and 16 sign choices for each four-component field
basis:

```text
384 signed boson frames x 384 signed fermion frames = 147,456 frame pairs
```

The charge-zero linkage is normalized to the fixed Majorana source frame before
the enhancement gate is evaluated. Exactly 384 frame pairs survive that
normalization for each input.

Three searches were run:

| Input | Frame pairs | Normalized | Eq. (5.11) passers |
|---|---:|---:|---:|
| Maxwell source basis | 147,456 | 384 | 8 |
| Maxwell, independently scrambled field basis | 147,456 | 384 | 8 |
| Chiral negative control | 147,456 | 384 | 0 |

The scrambled input uses nontrivial boson and fermion permutations and signs.
Recovering the same eight passing frames shows that the result does not depend
on the initial component ordering used for the test.

The eight witnesses are related field-frame descriptions. They are not eight
different four-dimensional Maxwell multiplets, and no uniqueness claim is
made.

## Independent cross-check

The JavaScript calculation separately constructs the Majorana gamma matrices,
the provisional phantom and spatial linkages, the Omega tensors, both Bianchi
reshufflings, and the entire 442,368-pair search. It reproduces all three Rust
counts.

## Reproduction

```sh
cargo run --release -- maxwell-worldline-search-build
cargo run --release -- maxwell-worldline-search-verify
cargo test --release maxwell_worldline_search
node scripts/test_maxwell_worldline_search.mjs
```

Artifact:

- `results/maxwell_worldline_search.json`
- SHA256: `67e06bf0410c0eeae75878439b1dc8fa4acdfa7a5ca1338791ebd7517add7c10`

## Consequence

The gauge-enhancement machinery now passes both required four-color controls:
it recovers Maxwell from worldline data and rejects the tested chiral shadow as
a Maxwell field-strength multiplet.

The next step is not a direct eight-color substitution. The published
construction treats a closed two-form field strength with three magnetic
phantoms. The eight-color systems first require a stated higher-dimensional
target, Lorentz representation, gauge-potential degree, field-strength degree,
and Bianchi complex. Once those are fixed, the same search architecture can be
generalized without inventing the missing physics from the valise matrices.

## Boundary

The search retains the published four-supercharge order and fixed Majorana
gamma basis. It varies signed component frames, not supercharge frames. The
chiral calculation is one negative control, not an exhaustive classification of
all four-color multiplets. Nothing here assigns a four-dimensional parent to
`VM1`, `VM2`, or `VM3`.
