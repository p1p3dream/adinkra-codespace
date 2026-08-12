# Maxwell classification of the embedded four-color blocks in the signed S8 recursion

## Question

Do the two signed four-color inputs retained by a closing eight-color recursive construction preserve information that is absent from its complete one-dimensional Garden class?

This is a finite test of the recursion in arXiv:2304.09830. It is not a complete eight-supercharge enhancement calculation.

## Method

The relative-color scan contains

- 30 ordered pairs of distinct four-color inputs;
- 24 relative color orders for the second input;
- 8 cyclic Boolean masks per ordered and aligned pair;
- 5,760 signed eight-color candidates in total;
- 24 candidates that satisfy the eight-color Garden algebra.

For colors 1 through 4, each recursive 8 by 8 matrix is block diagonal. The upper and lower 4 by 4 blocks are therefore separate four-color Garden representations. The calculation extracts both blocks from every closer rather than reconstructing them from their labels.

For each of the 48 extracted blocks it then:

1. verifies the four-color Garden algebra;
2. computes the integer chromocharacter class `chi0`;
3. searches all 384 signed boson frames and all 384 signed fermion frames;
4. applies the Maxwell phantom and Bianchi gauge-enhancement gate derived from arXiv:0907.3605;
5. records the ordered pair of results for the complete eight-color closer.

The Rust artifact is independently checked in JavaScript by extracting the blocks again from the serialized eight-color matrices, recomputing Garden closure and `chi0`, and rerunning the signed-frame search.

## Result

All 48 extracted blocks close under the four-color Garden algebra.

The 24 eight-color closers divide into two equal classes:

| ordered `chi0` pair | ordered Maxwell-gate pair | closers |
|---|---:|---:|
| `(-1, +1)` | `(pass, fail)` | 12 |
| `(+1, -1)` | `(fail, pass)` | 12 |

Thus:

- every closer pairs one `chi0=+1` block with one `chi0=-1` block;
- every closer contains exactly one block that passes the four-color Maxwell gate;
- on this library, the Maxwell result is an exact relabeling of the ordered `chi0` pair;
- both ordered classes contain constructions from named four-dimensional parents and constructions from sources with no stated four-dimensional parent;
- the published CT and CV anchors both have `(fail, pass)` and are not distinguished.

The frame search examines 7,077,888 frame pairs across the 48 embedded blocks. Combined with the 96-signing four-color atlas cross-check, that is 21,233,664 frame pairs. Including the Maxwell source, scrambled-source, and chiral controls, the independent validation examines 21,676,032 frame pairs.

## Interpretation

The opposite-`chi0` pairing is a real structural property of the 24 closers in this finite recursion scan. The four-color Maxwell gate does not refine it. It does not distinguish CT from CV, and it does not separate named four-dimensional parentage from the three mathematical Garden solutions.

This closes the embedded-Maxwell signature as a selection rule for this dataset. It remains useful as a consistency check and as a compact description of the closing pairs.

## Boundary

The first four colors preserve the two 4 by 4 blocks. The full eight-color representation is irreducible and does not split into those blocks. A property of either retained block is not, by itself, a higher-dimensional property of the complete eight-color system.

The calculation does not test spatial linkage matrices for an eight-color multiplet and does not assign a four-dimensional parent to VM1, VM2, or VM3.

## Reproduction

```bash
cargo run --release -- maxwell-s8-subalgebra-build \
  results/maxwell_s8_subalgebra_scan.json

cargo run --release -- maxwell-s8-subalgebra-verify

node scripts/test_maxwell_worldline_search.mjs
```

Primary artifact:

```text
results/maxwell_s8_subalgebra_scan.json
```

SHA256:

```text
c4fd1396d11c262f7bc0ef63d32501ca5ffadac5c80d885ac3c6ada1501a286a
```

## Next decision

Further progress requires information not contained in the retained four-color `chi0` or Maxwell class. The next useful discriminator must use data that couples the two blocks, such as the full signed eight-color holoraumy, the signed recursion convention, or an explicit higher-dimensional target supplied independently of the one-dimensional Garden representation.
