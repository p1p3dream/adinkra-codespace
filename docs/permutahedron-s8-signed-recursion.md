# Signed Boolean recursion for the thirty ordered S4 pairs

## Question

Section 2.2 of arXiv:2304.09830v2 gives a procedure for adding Boolean factors
to the unsigned eight-color recursion:

1. concatenate the Boolean words of two signed four-color systems for colors
   one through four;
2. copy those words to colors five through eight;
3. flip one cyclic run of four neighboring bits;
4. accept the result if the Garden algebra closes;
5. otherwise advance to the next of the eight cyclic runs.

The paper works the chiral-tensor example. This calculation applies the same
fixed-color-order procedure to all `6 x 5 = 30` ordered pairs of distinct
signed four-color sets printed immediately before Sec. 2.1.

## Source inputs

The inputs are the printed `CM`, `TM`, `VM`, `VM1`, `VM2`, and `VM3` signed
one-line words. The Rust source stores both the signed words and their derived
Boolean factors. A permanent test rederives every absolute permutation and
Boolean factor from the signed words.

The `VM2` source uses `3421`. It does not use `3412`, which belongs to `VM3`.
All six input systems satisfy the four-color Garden algebra.

The construction uses:

- arXiv:2304.09830v2, PDF pp. 9-12, Eqs. (2.17)-(2.25) and Secs. 2.1-2.2;
- arXiv:2012.14015v7, Eqs. (5.1)-(5.6), for exact comparison with the six
  published eight-color fixtures.

## Exhaustive result under the printed order

The calculation tests all eight cyclic four-bit masks for every ordered pair.

| Quantity | Result |
|---|---:|
| Ordered distinct pairs | 30 |
| Masks per pair | 8 |
| Signed candidates | 240 |
| Exact dense Garden entries checked | 1,966,080 |
| Closing candidates | 16 |
| Nonclosing candidates | 224 |
| Ordered pairs with a closing mask | 8 |
| Ordered pairs without a closing mask | 22 |
| Exact closing matrix systems before quotienting | 16 |

The eight accepted ordered pairs are:

| First | Second | Closing mask starts |
|---|---|---|
| `CM` | `TM` | 2, 6 |
| `CM` | `VM` | 2, 6 |
| `TM` | `CM` | 2, 6 |
| `VM` | `CM` | 2, 6 |
| `VM1` | `VM2` | 2, 6 |
| `VM1` | `VM3` | 2, 6 |
| `VM2` | `VM1` | 2, 6 |
| `VM3` | `VM1` | 2, 6 |

Here start 2 flips positions `{2,3,4,5}` and start 6 flips
`{6,7,8,1}`. Their masks are bitwise complements. Consequently the two
accepted results for a fixed ordered pair differ by reversing the signs of
all four supercharges numbered five through eight. If supercharge sign
changes are included in the equivalence relation, the sixteen results reduce
to eight at this stage. No broader equivalence quotient is asserted here.

## Source anchor and convention issue

For `CM -> TM`, start 6 gives Boolean factors

```text
234, 76, 134, 32, 11, 173, 103, 193
```

and reproduces the paper's `CT` permutations and Boolean factors exactly.
This is a direct source anchor for the implementation.

No other output is an exact match for the other five signed eight-color
fixtures as printed. In particular, direct same-index `CM -> VM` recursion
does not reproduce the printed `CV` system. The printed `CV` fixture uses a
different relative color ordering and Boolean assignment. Therefore this
result is an exhaustive scan of the literal fixed source order, not a claim
that another published system cannot be obtained after color or field
relabeling.

This convention dependence must be resolved before assigning physical names
to the additional closing systems.

## Irreducibility and signed invariants

Every closing candidate has:

- HYMN trace zero;
- self-Gadget one;
- commutant dimension one;
- antisymmetric commutant dimension zero.

The scalar commutant certifies that each accepted `8|8` system is irreducible
as a one-dimensional eight-color Garden representation. It does not identify
a four-dimensional parent or separate physical multiplet types.

The full `16 x 16` Gadget matrix is stored in the data artifact. Its entries
are not simply zero or one. Testing orthonormal frames requires the signed
equivalence quotient and remains a separate calculation.

## Validation

Two implementations check the result:

1. Rust uses both sparse signed-permutation closure and a separate dense
   matrix calculation for both Garden relations.
2. The JavaScript audit reconstructs the recursion from an independent
   transcription of the signed source words and recomputes all 240 dense
   closure decisions without reading the Rust closure flags.

Run:

```sh
cargo run --release -- perm-s8-signed-recursion-build
cargo run --release -- perm-s8-signed-recursion-verify
cargo test --release --bin adinkra-codespace permutahedron_s8_signed_recursion
node scripts/test_permutahedron_s8_signed_recursion.mjs
```

Artifacts:

- `data/permutahedron_s8_signed_recursion.json`
- `results/permutahedron_s8_signed_recursion_validation.json`

| Artifact | SHA256 |
|---|---|
| data | `3ed8e5fca168c4e9867115e1afb1ff121cef276d611091636f0d057bdab88dff` |
| validation | `05858f02ab1bd2e8a037635f4d78b356dee11a8543528707d367571a865a4e1c` |

## Boundary and next gate

This closes the literal eight-mask scan for all thirty ordered pairs under the
printed source ordering. It does not yet:

- quotient the results by all boson, fermion, color, switching, block-swap,
  or duality equivalences;
- scan alternative relative color alignments;
- enumerate arbitrary Garden signings outside the paper's cyclic-flip ansatz;
- establish four-dimensional enhancement.

The next gate is therefore the canonical signed equivalence calculation. It
should first determine whether the eight accepted ordered-pair results reduce
beyond the proved complementary-mask identification, then test alternative
relative color alignments against the published `CV` fixture.
