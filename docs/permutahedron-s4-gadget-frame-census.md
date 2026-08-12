# First-Gadget frame census for the six four-color quartets

## Question

The four-color work states that its six weighted representatives form an
orthonormal frame under the first Gadget. This suggests a concrete possible
selection rule, provided the stated frame reproduces and is rare:

> The distinguished Boolean assignments may be those that form an
> orthonormal first-Gadget frame.

The required test is whether that frame is unique or unusually rare among all
Garden-closing Boolean assignments on the same six ordered permutation
quartets.

## Howard-talk connection

`HowardTLK.v2.pdf` supplies the geometric base case used here:

- page 61 lists the six four-color quartets;
- page 64 displays the 24-node permutahedron;
- page 73 defines the hopping operators; and
- pages 77 and 80-81 build the hopping paths used to separate the solid into
  six four-node chains.

The talk fixes the permutation-side problem. The calculation below attaches
all closing Boolean assignments and asks whether the Gadget selects a special
joint signing.

Local source SHA256:

```text
3e83021c95d467d8a01ba1af20bc7e6481c07d7df328c5610adbd7688c4b454d
```

## Complete fixed-order library

For each of the six ordered quartets, the Rust calculation checks all
`16^4 = 65,536` Boolean-factor assignments. Exactly 256 satisfy the Garden
algebra. Thus the complete fixed-color, fixed-order library contains:

```text
6 x 65,536 = 393,216 Boolean assignments checked
6 x 256 = 1,536 Garden-closing signed representations
```

Each sector contains 128 representatives with `chi0 = +1` and 128 with
`chi0 = -1`. Vertex switching partitions the 256 representatives into two
classes of 128.

Every candidate has one of 16 distinct compatibility profiles against the
other five sectors. This compression is exact: two candidates share a type
only when their Gadget numerator against every candidate in every other
sector agrees.

## Source audit

The literal weighted representatives in Table 5 of arXiv:2408.09342 do not
reproduce the paper's stated six-by-six identity, even after replacing the
printed `VM2` support entry `3412` with `3421`. In units of `1/24`, their first
Gadget matrix is

```text
24   0   0   0   8   0
 0  24  -8   0   0   0
 0  -8  24   0   0   0
 0   0   0  24   0   0
 8   0   0   0  24   0
 0   0   0   0   0  24
```

Thus `CM` and `VM2` have cross-Gadget `+1/3`, while `TM` and `VM` have
cross-Gadget `-1/3`. This leaves at least one additional sign or convention
discrepancy between Table 5 and the stated orthonormality.

The first Appendix-B fiducial signing for each quartet in arXiv:1701.00304
does form a six-by-six identity. It is retained as a source-documented
orthonormal reference frame, but it is not presented as the literal Table 5
frame.

## Complete Gadget census

The program evaluates 983,040 cross-sector Gadget entries. At `N=4`, the
Gadget is stored as an integer numerator over 24. The only cross values are:

```text
-8/24 = -1/3
 0/24 = 0
+8/24 = +1/3
```

A six-frame is orthonormal when all 15 cross entries vanish. The complete
count is:

| Quantity | Count |
|---|---:|
| all fixed-order six-frames | 281,474,976,710,656 |
| orthonormal six-frames | 28,862,180,229,120 |
| fraction | `105/1024 = 10.25390625%` |
| classes after common vertex switching | 225,485,783,040 |

The source-documented Appendix-B reference frame is orthonormal, but it is not
rare in this library. More than one frame in ten is orthonormal. The literal
weighted Table 5 frame retains the discrepancy shown above.

The common vertex-switching action is free after removing its global
two-fold redundancy. Division by 128 therefore gives the stated orbit count.
Independent switching of each representation is not used because a
cross-Gadget is not invariant under independent basis changes. A frame is a
jointly aligned object.

Even an intentionally generous remaining relabeling group cannot rescue
uniqueness. After vertex switching, allow all common permutations of the four
bosons, four fermions, and four colors, together with eight effective common
color-sign choices. This group has at most

```text
24 x 24 x 24 x 8 = 110,592
```

elements. Dividing by this upper bound still leaves at least 2,038,898
classes. Some of those transformations do not preserve the fixed labeled
problem, so this is a conservative lower bound.

## Result

First-Gadget orthonormality does not provide a unique or rare Boolean-factor
selection rule for the six four-color quartets. It selects a large family of
joint conventions. Separately, the literal weighted Table 5 representatives
do not reproduce the paper's stated identity without an additional
correction or convention change.

This is a useful negative result. It removes Gadget orthonormality by itself
from the leading candidate list. Any viable selector must use additional
data, such as higher-dimensional parentage, a fixed lifting map, the printed
Boolean recursion, HYMN or chromocharacter information, or another invariant
that survives the relevant equivalences.

## Validation

```sh
cargo run --release -- perm-s4-gadget-frames-build
cargo run --release -- perm-s4-gadget-frames-verify
cargo test --bin adinkra-codespace permutahedron_s4_gadget_frames
node scripts/test_permutahedron_s4_gadget_frames.mjs
```

The independent JavaScript implementation reconstructs all 393,216 Boolean
assignments, recomputes the 1,536 closing systems and their exact Gadget
profiles, and independently recovers 28,862,180,229,120 orthonormal frames.

Artifacts:

- `data/permutahedron_s4_gadget_frames.json`
- `results/permutahedron_s4_gadget_frames_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `11d93f60718a71c64d0091c0cedb72b9ad078fa4d1c21850e8cbe29f8efe9d32` |
| validation | `df62ebc46c5c42a0df38f93be8f0aefa9eb8bb6a7feb1d26075f0cb0858f6212` |

## Boundary

This census is complete for fixed quartet order and fixed color labels. It
does not identify one-dimensional nodal equivalence with higher-dimensional
physical equivalence. It also does not test a selector that combines the
Gadget with higher-dimensional parentage or lifting constraints.
