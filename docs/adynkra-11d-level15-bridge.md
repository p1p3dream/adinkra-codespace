# Eleven-dimensional level-15 bridge systems

## Question

The discussion following Eq. (2.7) of arXiv:2007.05097 requires the
vector-spinor semi-prepotential (H_\alpha{}^a(V)) to contain fifteen spinor
derivatives of a scalar superfield. The level-15 decomposition of
arXiv:2002.08502 contains

\[
2(00001)\oplus(10001)\oplus\text{other representations}.
\]

The target vector-spinor decomposes as

\[
(10000)\otimes(00001)=(00001)\oplus(10001).
\]

The leading symbol therefore has three Lorentz-equivariant channels: two
spinor channels and one gamma-traceless vector-spinor channel.

## Exact Rust construction

The verifier constructs the level-15 source as

\[
\bigwedge^{15}S,\qquad
S=\{(\pm1,\pm1,\pm1,\pm1,\pm1)\},
\]

in doubled (B_5) weight coordinates. For each target highest weight it:

1. enumerates every 15-element spinor-weight wedge with that weight;
2. applies each of the five simple-root raising operators, including the
   exterior-algebra sign;
3. constructs the resulting sparse integer homogeneous system; and
4. checks that every raising term lands in the enumerated source basis.

The resulting systems are:

| target | columns | rows | nonzero integer entries | published multiplicity |
|---|---:|---:|---:|---:|
| `(00001)` | 591,810 | 1,943,600 | 7,412,645 | 2 |
| `(10001)` | 388,720 | 1,174,806 | 4,551,287 | 1 |

The first four raising blocks of `(00001)` each have 388,720 rows and
1,255,260 nonzero entries. Its fifth block has 388,720 rows and 2,391,605
entries. For `(10001)`, the five blocks have row counts 166,158, 252,162,
252,162, 252,162, and 252,162.

These are the exact equations whose kernels supply the two `(00001)` highest
weight vectors and the one `(10001)` highest-weight vector. The published
multiplicities predict kernel dimensions two and one. They do not replace an
explicit kernel calculation.

All three highest-weight kernels have now been extracted and verified exactly:

| target | vector | coefficients | nonzero | range | raising rows with nonzero residual |
|---|---:|---:|---:|---:|---:|
| `(00001)` | 1 | 591,810 | 374,246 | -660 to 3,960 | 0 of 1,943,600 |
| `(00001)` | 2 | 591,810 | 6,435 | -1 to 1 | 0 of 1,943,600 |
| `(10001)` | 1 | 388,720 | 260,267 | -1,320 to 1,320 | 0 of 1,174,806 |

Each integer vector has greatest common divisor one. The artifacts are stored
under `data/eleven_dimensional_bridge/` and read directly by the Rust verifier.
The independent numerical calculation finds two zero eigenvalues for
`(00001)`, followed by a gap of approximately 0.07677. For `(10001)` it finds
one zero eigenvalue, followed by a gap of approximately 0.06730. The exact
raising-equation checks, not those floating-point values, are the acceptance
test for the stored vectors.

The verifier also applies every simple-root lowering operator. For each
`(00001)` vector, only the fifth lowering is nonzero. For `(10001)`, only the
first and fifth lowerings are nonzero. Applying the same lowering a second
time gives zero in every nontrivial case. These are the exact Dynkin strings
specified by labels `00001` and `10001`.

## Generic bridge

Before normalization, the most general leading symbol is

\[
H_\alpha{}^a(V)=
a\,I_{32}^{(1)}(D^{15}V)+
b\,I_{32}^{(2)}(D^{15}V)+
c\,I_{320}(D^{15}V).
\]

Here $I_{32}^{(1)}$, $I_{32}^{(2)}$, and $I_{320}$ denote kernel vectors of
the two exact sparse systems above, followed by the target projectors already
verified in `src/eleven_dimensional_clifford.rs`. All three highest-weight
vectors are now exact. Their covariant descendants remain.

## Current boundary

This pass makes the source-embedding problem explicit and reproducible and
supplies all three exact highest-weight vectors. It does not yet provide their
full covariant descendant sets. Consequently:

- the three coefficients have not yet been substituted into the torsion
  constraints of Eq. (2.7);
- no surviving coefficient combination has been identified; and
- generic-momentum gauge reducibility and a complete curvature complex remain
  open. The scalar-divergence kernel test has identified the direct two-form
  and five-form channels.

## Primary-source dependency for Eq. (2.7)

Equation (2.7) does not by itself print the linearized torsion as an operator
on $H_\alpha{}^a$. Its surrounding text points to Gates and Nishino,
hep-th/0101037, for the linearized calculation. That paper supplies the needed
starting formulas:

- Eq. (24), the linearized spinorial frame;
- Eq. (25), the induced vectorial frame;
- Eqs. (26)-(29), the linearized anholonomy coefficients; and
- the subsequent conventional-constraint analysis relating the holonomy
  superfields and spin connections to $H_\alpha{}^a$.

Therefore the next calculation is not a bare substitution into the displayed
lines of Eq. (2.7). It is an exact implementation of the linearized frame and
anholonomy formulas, elimination of the conventional fields, and then
evaluation of the Eq. (2.7) projections on the three bridge channels.

The independent Python program rebuilds the same matrices and numerically
checks the small eigenvalues of $A^T A$. It was also used to propose integer
kernel bases. It is a cross-check only. Rust independently reads all three
integer artifacts and verifies every sparse raising equation exactly.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

```bash
cargo run --release -- adynkra-11d-bridge-verify \
  > results/adynkra_11d_level15_bridge_validation.json
cargo test eleven_dimensional_bridge

python3 scripts/crosscheck_11d_level15_bridge.py --label 10001 \
  --integer-artifact-directory data/eleven_dimensional_bridge
python3 scripts/crosscheck_11d_level15_bridge.py --label 00001 \
  --integer-artifact-directory data/eleven_dimensional_bridge
```

Primary implementation: `src/eleven_dimensional_bridge.rs`

Independent numerical cross-check:
`scripts/crosscheck_11d_level15_bridge.py`

## References

1. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Adinkra Foundation of Component
   Decomposition and the Scan for Superconformal Multiplets in 11D, N = 1
   Superspace," [arXiv:2002.08502](https://arxiv.org/abs/2002.08502).
2. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Weyl Covariance, and Proposals
   for Superconformal Prepotentials in 10D Superspaces,"
   [arXiv:2007.05097](https://arxiv.org/abs/2007.05097).
3. S. J. Gates Jr. and H. Nishino, "Deliberations on 11-D Superspace for the
   M-Theory Effective Action,"
   [arXiv:hep-th/0101037](https://arxiv.org/abs/hep-th/0101037).
