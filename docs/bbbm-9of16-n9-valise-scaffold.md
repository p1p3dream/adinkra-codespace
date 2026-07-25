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

## Linearized component verification

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

## Full nonabelian verification

`src/bbbm_nonabelian.rs` expands the printed transformations into an exact free
associative differential superalgebra. Adjoint-valued fields remain
noncommuting, ordinary derivatives use the full Leibniz rule, and the
supersymmetry variations are odd graded derivations. The implementation uses

\[
D_\mu X=\partial_\mu X+[A_\mu,X],
\]

\[
F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu+[A_\mu,A_\nu],
\]

with

\[
\delta_{\mathrm{gauge}}(\lambda)A_\mu=-D_\mu\lambda,
\qquad
\delta_{\mathrm{gauge}}(\lambda)X=[\lambda,X].
\]

All 1,485 component instances of Eq. (24) have zero canonical residual. The
calculation uses no equations of motion, integration by parts, trace
cyclicity, numerical sampling, or commutativity assumption.

`src/bbbm_nonabelian_crosscheck.rs` repeats the calculation with a separately
written free-word engine. A deliberate sign mutation produces nonzero
residuals. `src/bbbm_nonabelian_source_audit.rs` independently checks the
gauge conventions, graded signs, Jacobi identity, covariant-derivative
commutator, and Bianchi identity against the original arXiv TeX.

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

## The other seven charges

The supersymmetry parameter decomposes under \(Spin(7)\) as
\(\mathbf{16}=\mathbf{1}+\mathbf{8}+\mathbf{7}\). The last term is the
antiselfdual tensor parameter \(\nu^{ij}\). BBBM's covariant linear solution of
the auxiliary-spinor constraints sets \(\nu^{ij}=0\), leaving the scalar and
vector charges.

This is not merely an omitted component table. A transformation on the
independent auxiliaries requires a linear assignment \(v_a(\nu)\). The
constraints in Eqs. (15)-(17) provide no such extension with nonzero
\(\nu^{ij}\). Individual generalized transformations can be chosen only with
an auxiliary \(SO(7)\) ambiguity, and those choices do not define a linear
sixteen-charge off-shell algebra.

After eliminating the auxiliaries, the ordinary transformations of all sixteen
supercharges are defined on the ten gauge potentials and sixteen gaugino
components. `src/bbbm_sixteen_onshell.rs` evaluates all 136 charge pairs in the
noncommutative differential algebra:

- 3,536 component relations are checked;
- 2,366 of those relations involve at least one charge from the
  seven-dimensional tensor subspace;
- all 1,360 gauge-potential relations close modulo translations and gauge
  transformations;
- all 2,176 gaugino relations factor through
  \(\mathcal E_\Psi=\sigma^\mu D_\mu\Psi\);
- 1,120 gaugino component relations have a nonzero Dirac multiplier in the
  chosen sparse spinor basis;
- no residual remains after the Dirac factors are removed.

The calculation uses a covariant spinor-component charge basis. It does not
yet construct the explicit intertwiner from that basis to BBBM's scalar,
vector, and antiselfdual tensor charge basis.

`src/bbbm_sixteen_onshell_crosscheck.rs` independently verifies the Clifford
and closure coefficient identities and detects a one-entry bivector mutation.
`src/bbbm_sixteen_source_audit.rs` records the source boundary and the
\(1+8+7\) projector ranks. This is an on-shell calculation on 26 fields. It is
not a sixteen-charge extension of BBBM's 33-field auxiliary multiplet.

## Reproducibility

```bash
cargo run --release -- bbbm
cargo run --release -- bbbm-holoraumy
cargo run --release -- bbbm-closure
cargo run --release -- bbbm-nonabelian
cargo run --release -- bbbm-sixteen-onshell
cargo test
```

Primary reference: L. Baulieu, N. Berkovits, G. Bossard, and A. Martin,
"Ten-dimensional super-Yang-Mills with nine off-shell supersymmetries,"
arXiv:0705.2002, *Physics Letters B* 658 (2008) 249-254.

Additional references:

- N. Berkovits, "A Ten-Dimensional Super-Yang-Mills Action with Off-Shell
  Supersymmetry," arXiv:hep-th/9308128, *Physics Letters B* 318 (1993) 104-106.
- J. M. Evans, "Supersymmetry Algebras and Lorentz Invariance for d=10
  Super-Yang-Mills," arXiv:hep-th/9404190, *Physics Letters B* 334 (1994)
  105-112.
- S. J. Gates Jr., G. Hannon, R. X. Siew, and K. Stiffler,
  "Infinite-Dimensional Algebraic Spin(N) Structure in Extended/Higher
  Dimensional SUSY Holoraumy for Valise and On-Shell Supermultiplet
  Representations," arXiv:2010.06124, *JHEP* 05 (2022) 173.

The research consequences and ordered next steps are recorded in
[`from-bbbm-closure-to-adynkrafield-equations.md`](from-bbbm-closure-to-adynkrafield-equations.md).
