# Side-L, m=3 slot-0 semantics for the CLS G-matrix CSP census

Companion dump: `/tmp/lever_a/slot0_items.json` (825 entries, engine order).
Source files (clean at HEAD `510d3bd`, branch `s4-permutahedron-disassembly`):
`/Users/brandon/code/adinkra-codespace/src/four_color/gmatrix_full.rs` and
`/Users/brandon/code/adinkra-codespace/src/four_color/gmatrix_csp.rs`.
All line numbers below refer to that committed state.

## (a) What integer entries of G a slot-0 value fixes

The census searches integer `G` in `{-1,0,1}^{12x12}` with `G^2 = A_L`. The engine
does not search G directly. It uses the block-diagonal eigenbasis change of `A_L`
(`build_coords`, gmatrix_full.rs:408):

- `A_L` is block diagonal with three 4x4 blocks `B_0, B_1, B_2`
  (`cls_a_blocks`, gmatrix_full.rs:368), each with `B^2 = 2B - 4I`, eigenvalues
  `lambda = 1 + sqrt(-3)` and its conjugate, each twice per block.
- `P` is the block-diagonal 12x12 matrix whose 4x4 block `b` is
  `P_b = [V_b | conj(V_b)]`, `V_b` a basis of the 2-dimensional lambda-eigenspace
  of `B_b`. Columns of the big `P`: `V_b` lands at columns `2b, 2b+1`, `conj(V_b)`
  at `2m+2b, 2m+2b+1` (gmatrix_full.rs:439-449).
- `g = ` top-left `2m x 2m` (6x6 at m=3) of `P^{-1} G P` (`g_of_int`,
  gmatrix_full.rs:465). Inverse map `G = P diag(g, conj(g)) P^{-1}` (`int_of_g`,
  gmatrix_full.rs:478). The search equation is `g^2 = lambda * I_6`.

The 6x6 matrix `g` is decomposed into a 3x3 grid of 2x2 K-blocks ("slots"), flat
position `p = bi*3 + bj` holding `g[2bi..2bi+2][2bj..2bj+2]`. Slot 0 is
`(bi,bj) = (0,0)`, i.e. `g[0..2][0..2]`. This is the source of the 9-bit class
support masks (one bit per slot position, `m*m = 9` at m=3, 4 at m=2).

Because `P` is block diagonal, `g`'s slot `(bi,bj)` determines exactly one 4x4
integer block of `G`: `block = P_bi [slot 0; 0 conj(slot)] P_bj^{-1}`
(`slot_int_of_g`, gmatrix_full.rs:507-527). So:

**A slot-0 value fixes exactly the leading 4x4 integer block `G[0..4][0..4]`
(rows 0-3, columns 0-3 of the 12x12 G), entries in `{-1,0,1}`, and nothing else.**
Every solution counted in shard `item = i` has that same leading 4x4 block. This
binding is enforced on every published shard: `shard_decode_value` recomputes
`want0 = slot_int_of_g(coords, 0, 0, slot0[item])` (gmatrix_csp.rs:1863) and
rejects the shard unless every class rep satisfies `rep[i][..4] == want0[i]`
(gmatrix_csp.rs:1914-1920). The map 2x2-K-slot -> 4x4-block is injective
(conjugation by the invertible `P_b0`; verified empirically in the dump: 825
alphabet values give 825 distinct blocks), so fixing the integer block and
fixing the K-slot are equivalent.

## (b) Integer encoding of a slot-0 value

Two exact encodings of the same value, both in the JSON dump per item:

1. `k_slot`: the engine-native 2x2 matrix over `K = Q(sqrt(-3))`. A `K` element
   (`struct K`, gmatrix_full.rs:58) is `(re + im*s)/den` with `s^2 = -3`, stored
   as a reduced `(re, im, den)` triple of i128: `den > 0` and
   `gcd(gcd(|re|,|im|), den) = 1` (`K::new`, gmatrix_full.rs:75-96; `raw_parts`
   at :180). In the JSON, `k_slot[u][w] = [re, im, den]`, `u,w in {0,1}`,
   row-major: `k_slot = [[g[0][0], g[0][1]], [g[1][0], g[1][1]]]`.
2. `slot0_int` (flat row-major 16 ints) and `entries` (4 rows of 4): the direct
   output of `slot_int_of_g(coords, 0, 0, &slot)`, i.e. the fixed block
   `G[0..4][0..4]`, all entries in `{-1,0,1}`. For the offline orbit computation
   this integer 4x4 block is the complete fixed data; the K-encoding is included
   for provenance.

There is no single scalar integer code for a slot value in the engine; positions
are compared by exact `K` equality and indexed by list position (below).

## (c) How the surviving set is computed and its ordering

Alphabet construction (`build_alphabets`, gmatrix_full.rs:547-604): for each
slot position `(bi,bj)`, take ALL integer `{-1,0,1}` intertwiners
`X B_j = B_i X` (`gmatrix::intertwiners`, complete by construction per the v1
bracket engine), embed each `X` at block position `(bi,bj)` of the 12x12 zero
matrix, push through `g_of_int`, and collect the resulting 2x2 K-slots into a
`BTreeSet<[[K;2];2]>`. The slot alphabet is that set as a `Vec`, so it is sorted
by the `Ord` of `[[K;2];2]`: lexicographic over the flattened row-major entries,
each compared by `K`'s total order `(den, re, im)` (gmatrix_full.rs:212-217;
NOT a field order, purely a deterministic sort key). For side L, m=3 the slot-0
alphabet has exactly 825 values; `build_shared` asserts all nine slot alphabets
have equal length (gmatrix_csp.rs:309-312).

Surviving set (`canonical_run`, gmatrix_csp.rs:2053-2078, mirrored in `solve` at
:1636-1637): start from `Search` with every domain the full bitset over the 825
slot values (`b_full(sh.slots[0].len())`; bit `i` of domain `p` = "slot p may
take alphabet value i", `b_set`/`b_test`/`b_each` at gmatrix_csp.rs:56-97);
seed all 27 "wide" constraints of `g^2 = lambda*I` (one per `(bi,bj,bk)` triple,
constraint index `ci = (bi*m+bj)*m + bk`, `wide_seed` at :1084); propagate
(`propagate` at :1364); then `items = b_each(&root.dom[0])`, which emits the set
bits of domain 0 in ascending index order.

Ordering: **item index `i` = position `i` in the sorted slot-0 alphabet list
`alph.slots[0]`** (equivalently `sh.slots[0]`). For side L, m=3 nothing is
pruned at the root: the dump run asserted `items.len() == 825` and
`items == [0,1,...,824]` exactly, matching the `items.json` manifest that
`run_shards` publishes (gmatrix_csp.rs:2178-2188) and slices
`items[start..end]` over (:2238-2243). Per item, the worker pins domain 0 to the
singleton `{x0}` (gmatrix_csp.rs:1544-1545 in `run_worker`) and searches the
remaining 8 slot positions.

## Provenance of the dump

Temporary test `dump_slot0_alphabet_L_m3` appended to `mod tests` in
`src/four_color/gmatrix_csp.rs`, run with
`CARGO_TARGET_DIR=target-csp cargo test --release dump_slot0_alphabet_L_m3 -- --nocapture`
(passed; assertions: alphabet length 825, 825 surviving items in order, 825
distinct 4x4 blocks, flat == nested rows, all entries trits). The test edit was
reverted immediately after; `git diff src/four_color/gmatrix_csp.rs` is empty.

## Sanity anchors (first/last five `slot0_int`, row-major 16 ints)

```
0   [-1,-1,-1, 1, -1, 1,-1,-1,  1,-1,-1,-1,  1, 1,-1, 1]
1   [-1, 0,-1, 0, -1, 0, 0,-1,  1,-1,-1,-1,  1, 1, 0, 0]
2   [-1,-1, 1,-1, -1,-1,-1, 1,  1,-1,-1,-1,  1,-1, 1, 1]
3   [-1, 1,-1,-1, -1,-1, 1,-1,  1,-1,-1,-1,  1, 1, 1,-1]
4   [-1,-1, 0, 0, -1, 0,-1, 0,  1,-1,-1,-1,  1, 0, 0, 1]
...
820 [ 1,-1, 1, 0,  1, 1, 0, 1, -1, 0, 1, 1,  0,-1,-1, 1]
821 [ 0, 1, 0, 0,  1, 0, 1,-1, -1, 0, 1, 1, -1, 0,-1,-1]
822 [ 1, 0, 1,-1,  1, 0, 1, 1, -1, 0, 1, 1,  0,-1, 0, 0]
823 [ 1, 0, 1, 1,  0, 1, 0, 0, -1, 0, 1, 1, -1, 0,-1, 1]
824 [ 1, 1, 1, 0,  0, 0, 1, 0, -1, 0, 1, 1, -1, 0, 0, 0]
```
