# BBBM nine-supercharge partial off-shell construction

## Source result

Baulieu, Berkovits, Bossard, and Martin construct ten-dimensional, (N=1)
super-Yang-Mills with nine of its sixteen supersymmetries closing off shell after
seven auxiliary scalars are added. The retained charges transform as
(1+8) under \(\mathrm{Spin}(7)\). Their component transformations and closure
algebra are Eqs. (22)-(24) of arXiv:0705.2002.

Before gauge reduction, the fields contain 17 bosonic variables and 16
fermionic variables:

```text
10 gauge-potential components + 7 auxiliary scalars
8 vector fermions + 1 scalar fermion + 7 antiselfdual-tensor fermions
```

Removing one gauge redundancy leaves 16 gauge-invariant bosonic degrees of
freedom and 16 fermionic degrees of freedom.

## Exact component verification

`src/bbbm_component.rs` encodes Eqs. (22)-(23) as sparse polynomial
differential operators at linearized abelian order. It checks Eq. (24) on all
33 raw component fields:

- \(\delta_0^2\);
- all eight \(\{\delta_0,\delta_i\}\) relations;
- all 36 \(\{\delta_i,\delta_j\}\) relations.

The calculation performs 1,485 field-relation checks. Every residual vanishes
after the explicit abelian gauge transformation is subtracted. No field
equation or integration by parts is used.

`src/bbbm_source_audit.rs` independently repeats the calculation with rational
arithmetic, a separately entered Cayley-form fixture, and all 28 projected
two-form components. Its results agree with the production calculation.

`src/bbbm_closure.rs` also verifies the reduced superspace derivative algebra
of Eqs. (33)-(34) on all \(2^9=512\) Grassmann monomials.

These are reproductions of the published nine-charge algebra. They are not a
new off-shell construction.

## Spin(7) structure

The implementation constructs the projector

\[
P^- = \frac14\left(I-\Omega\right)
\]

on the 28 independent antisymmetric index pairs. This is Eq. (14) after the
factor of two from the ordered-pair contraction is removed. The projector is
symmetric, idempotent, and has rank seven.

The paper defines \(\Omega\) through a normalized spinor but does not print a
component basis. The implementation therefore uses a stated octonion basis and
does not claim that its component signs are the authors' unpublished basis.

## One-dimensional reduction

`src/bbbm_worldline.rs` derives the one-dimensional transformations from Eqs.
(22)-(23). It sets the spatial derivatives to zero, normalizes
\(\partial_+=\partial_-=D\), and quotients the two light-cone potentials to the
gauge-invariant combination

\[
B=A_- - A_+.
\]

The resulting local representation has the engineering-height distribution

\[
(9,16,7),
\]

with nine gauge-field bosons, sixteen fermions, and seven auxiliary bosons.
Its polynomial differential linkage matrices satisfy the (N=9) Garden
relations coefficient by coefficient.

Formally writing \(G_a=Dg_a\) produces constant \(16\times16\) signed-permutation
matrices. Their stabilizer code is the \([9,4]\) extended-Hamming code with a
trivial ninth coordinate, after the explicit color map
\(\delta_1,\ldots,\delta_8\mapsto0,\ldots,7\) and \(\delta_0\mapsto8\).

This formal node lowering is not a local equivalence over \(\mathbb Z[D]\):

- recovering \(g_a\) from \(G_a\) requires \(D^{-1}\);
- seven auxiliary integration constants remain in \(\ker D\);
- temporal gauge fixing also requires an inverse derivative and leaves a zero
  mode.

The formal valise comparison is therefore valid only after boundary conditions
or zero-mode removal are specified.

## Generic valise and gadget calculations

`src/bbbm.rs` builds every dashing class of the generic minimal (N=9),
(16|16\) valise with the same \([9,4]\) chromotopology. It verifies the Garden
algebra for all 16 dashing classes.

`src/bbbm_holoraumy.rs` computes the associated generic holoraumy and gadget
invariants. An independent dense implementation reproduces them:

- each holoraumy matrix is traceless and antisymmetric;
- each squared holoraumy matrix equals \(-I_{16}\);
- each self-gadget equals one;
- distinct-dashing cross-gadgets take the values \(1/8\), \(1/6\), and \(7/24\);
- the code has weight enumerator \(\{0:1,4:14,8:1\}\) and automorphism-group
  order 1,344.

The all-dashing survey concerns the generic scaffold. The BBBM reduction fixes
a formal chromotopology but has not been matched to every generic dashing.

## The seven discarded charges

The paper sets the antiselfdual tensor parameter \(\nu^{ij}\) to zero and does
not print twisted component transformations for the other seven charges. Their
non-closure functions are therefore not present in Eqs. (22)-(24).

They may be reconstructed from the parent spinor transformations in Eqs.
(2)-(6), but that requires gamma-matrix conventions, a nonzero
\(\nu^{ij}\), the full twist, and a separate on-shell closure calculation. No
such reconstruction is claimed here.

Equation (3) is the transformation of \(G_a\), not an equation of motion. The
auxiliary equation from the action is algebraic, while a fermionic non-closure
term must be compared with the Dirac equation derived from the action.

## Reproducibility

```bash
cargo run --release -- bbbm
cargo run --release -- bbbm-holoraumy
cargo run --release -- bbbm-closure
cargo test
```

Primary reference: L. Baulieu, N. Berkovits, G. Bossard, and A. Martin,
"Ten-dimensional super-Yang-Mills with nine off-shell supersymmetries,"
arXiv:0705.2002, *Physics Letters B* 658 (2008) 249-254.
