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

| target | vector | coefficients | nonzero | squared norm | range | raising rows with nonzero residual |
|---|---:|---:|---:|---:|---:|---:|
| `(00001)` | 1 | 591,810 | 374,246 | 426,254,400 | -660 to 3,960 | 0 of 1,943,600 |
| `(00001)` | 2 | 591,810 | 6,435 | 6,435 | -1 to 1 | 0 of 1,943,600 |
| `(10001)` | 1 | 388,720 | 260,267 | 245,044,800 | -1,320 to 1,320 | 0 of 1,174,806 |

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

The two complete spinor intertwiners are now generated. Each produces all 32
spinor weights. The verifier checks all 48 nonzero simple-root lowering edges,
including 17 states reached by more than one lowering path. Both source copies
have zero path mismatches. The first copy has support 374,246 at every weight;
the second has support 6,435 at every weight.

The `(10001)` target module is now generated independently inside
$(10000)\otimes(00001)$. Exact rational Chevalley lowering from its highest
weight produces dimension 320, with 192 distinct weights and 752 nonzero
lowering actions. Its weight multiplicities are 160 weights of multiplicity
one and 32 weights of multiplicity five. This is the complete target-side
gamma-traceless vector-spinor module inside the 352-dimensional tensor
product.

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
vectors are now exact. Both 32-component spinor descendant systems are exact.
The 320-component gamma-traceless vector-spinor source descendant system is
also exact.

## Complete `(10001)` source intertwiner

The verifier lowers the primitive `(10001)` source vector through the complete
320-dimensional target module. The layer-adapted target basis has 192 weights.
Across its 320 states, the verifier evaluates all 1,600 simple-root actions:

- 776 actions are nonzero and 824 vanish;
- 319 nonzero actions discover a new independent state;
- 457 nonzero actions reproduce a target-space linear relation; and
- every one of those 457 relations holds on the level-15 source vectors with
  zero exact residual.

Every target-zero action also vanishes on the source. Source-state supports
range from 260,267 to 555,664 exterior basis terms, and the largest absolute
integer coefficient is 18,480. The count of nonzero actions depends on the
basis selected inside the 32 weights of multiplicity five. The invariant
result is that all 320 states are generated and the complete simple-root
action agrees between source and target.

## Level-16 derivative channels

A further spinor derivative gives the multiplicity-free tensor product

\[
(00001)\mathbin{\otimes}(10001)=
(00002)\oplus(00010)\oplus(00100)\oplus(01000)\oplus(10000)
\oplus(10002)\oplus(10010)\oplus(10100)\oplus(11000)\oplus(20000).
\]

Its dimension is $32\times320=10{,}240$. Comparison with the scalar
superfield at level 16 removes two candidate channels:

| channel | dimension | highest-weight domain | level-16 multiplicity | exterior image |
|---|---:|---:|---:|---|
| `(00002)` | 462 | 10 | 1 | nonzero |
| `(00010)` | 330 | 18 | 2 | nonzero |
| `(00100)` | 165 | 32 | 2 | nonzero |
| `(01000)` | 55 | 56 | 0 | zero |
| `(10000)` | 11 | 96 | 1 | nonzero |
| `(10002)` | 4,290 | 1 | 3 | nonzero |
| `(10010)` | 3,003 | 2 | 1 | nonzero |
| `(10100)` | 1,430 | 4 | 1 | nonzero |
| `(11000)` | 429 | 8 | 0 | zero |
| `(20000)` | 65 | 16 | 1 | nonzero |

For each channel, the verifier constructs the unique highest-weight kernel in
the indicated domain and checks every raising equation with rational
arithmetic. It then applies exterior multiplication to the corresponding
level-15 source vectors. This is the zero-spacetime-momentum exterior symbol of
the sixteenth spinor derivative. The eight inventory-allowed channels have
nonzero images. The `(01000)` and `(11000)` images vanish, consistent with
their absence from the scalar level-16 inventory.

The nonzero results use three deterministic modular linear functionals. A
nonzero residue is a certificate that the underlying integer exterior image is
nonzero. All three residues vanish for each inventory-forbidden channel; their
vanishing is a cross-check, while the representation inventory supplies the
zero proof. The `(11000)` is the 429-dimensional hook left by the conventional
decomposition of the two-form-vector torsion, so its absence gives the
vanishing of the final Eq. (2.7) projection.

## Dimension-zero torsion sectors

Equations (38)-(40) of hep-th/0101037 identify the two dimension-zero torsion
field strengths as a two-form-vector and a five-form-vector. Their irreducible
decompositions account for six of the ten derivative channels.

For the two-form-vector,

\[
55\mathbin{\otimes}11=(10000)\oplus(00100)\oplus(11000),
\qquad 605=11+165+429.
\]

The 11-dimensional trace vector and 165-dimensional three-form have nonzero
exterior images and are removed by the conventional constraints in Eq. (40).
The remaining 429-dimensional hook has zero image. Thus the scalar bridge has
no surviving $X_{[2]}$ hook at the exterior symbol.

For the five-form-vector,

\[
462\mathbin{\otimes}11=(00010)\oplus(00002)\oplus(10002),
\qquad 5{,}082=330+462+4{,}290.
\]

The 330-dimensional trace four-form and 462-dimensional six-form have nonzero
images and are removed by the conventional constraints. The remaining
4,290-dimensional hook also has a nonzero image. The exterior symbol therefore
contains a candidate $X_{[5]}$ hook after the conventional projections.

The other four derivative channels have total dimension 4,553:

\[
(01000)\oplus(20000)\oplus(10100)\oplus(10010).
\]

Together, $605+5{,}082+4{,}553=10{,}240$. This identifies the representation
sectors in the dimension-zero torsion formulas. It does not impose the full
superspace Bianchi identities.

## Spacetime completion boundary

The flat superspace derivative algebra is

\[
\{D_\alpha,D_\beta\}=i(\Gamma^a)_{\alpha\beta}p_a.
\]

Consequently, a local Lorentz-covariant completion with the same engineering
order can mix eight derivative orders:

\[
D^{15}V,\ pD^{13}V,\ p^2D^{11}V,\ p^3D^9V,\
p^4D^7V,\ p^5D^5V,\ p^6D^3V,\ p^7DV.
\]

The present calculation fixes the first term. The cited papers state that the
bridge contains fifteen spinor derivatives but do not print coefficient
systems for the seven lower-spinor-level momentum terms. Those systems must be
constructed before claiming a generic-momentum bridge or a complete curvature
complex.

## Canonical computational normalization

The surviving `(10001)` highest-weight kernel is a primitive integer vector
$v$ with greatest common divisor one. Its first nonzero coefficient is
positive, fixing the remaining sign convention, and

\[
\langle v,v\rangle=245{,}044{,}800.
\]

The exact rank-one projector onto this source highest-weight line is therefore

\[
P_v(x)=v\,\frac{\langle v,x\rangle}{245{,}044{,}800}.
\]

The verifier checks $P_v(v)=v$ and the corresponding exact identity
$P_v^2=P_v$. This fixes a reproducible computational normalization for the
source line. It does not fix the overall scale of $c$.

At linear order the replacement

\[
V\longrightarrow\lambda V,
\qquad
c\longrightarrow c/\lambda
\]

leaves $\widehat H(V)$ unchanged for nonzero $\lambda$. Homogeneous torsion
constraints and Bianchi identities can reject the bridge or constrain relative
coefficients, but cannot remove this rescaling freedom. Fixing the remaining
scale requires a declared normalization of $V$ or matching a component of
$H_\alpha{}^a$ to an independently normalized graviton or gravitino field.

## Current boundary

This pass makes the source-embedding problem explicit and reproducible and
supplies all three exact highest-weight vectors and both complete 32-component
spinor intertwiners. It also materializes the complete 320-dimensional target
module, verifies its full source intertwiner, and fixes a canonical
computational normalization of the surviving source highest-weight line. It
also resolves the zero-spacetime-momentum exterior symbol on all ten level-16
candidate channels.
Consequently:

- the final two-form torsion projection has now been evaluated at the
  representation level and has rank zero on the three coefficients;
- Eq. (2.7) therefore does not select a unique combination of the three bridge
  channels; and
- component normalization of $V$, the seven lower-level momentum-completion
  systems, generic-momentum gauge reducibility, and the full superspace Bianchi
  complex remain open. The exterior symbol identifies the dimension-zero
  $X_{[2]}$ and $X_{[5]}$ representation sectors.

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

Equations (39)-(40) already state the representation-level elimination needed
for the final projection. A component-level reproduction of the complete
linearized frame remains useful as an independent check, but is not required
to determine the rank of this final constraint on the three bridge channels.

## Result of the final Eq. (2.7) projection

Equations (39)-(40) of hep-th/0101037 decompose the two-form-vector tensor
$X_{ab}{}^c$. Its 605 components split as

\[
55\mathbin{\otimes}11=11\oplus165\oplus429.
\]

The vector and three-form pieces are removed by conventional constraints. The
remaining traceless hook is the `(11000)` representation of dimension 429.
The exact level-16 scalar-superfield inventory contains zero copies of
`(11000)`. Therefore the final gamma-two torsion projection in Eq. (2.7)
vanishes on every Lorentz-equivariant map from the scalar superfield.

For the three-parameter bridge this gives a constraint matrix of rank zero:

\[
\dim\ker C_{\mathrm{Eq.\,(2.7)}}=3.
\]

All three coefficients $a$, $b$, and $c$ survive this projection. This confirms
the representation-content argument made after Eq. (2.7), but it also shows
that Eq. (2.7) does not determine the bridge coefficients. A unique bridge, if
one exists, requires an additional condition such as a gauge transformation,
a curvature normalization, a Bianchi identity, or matching to component
supergravity.

## Local gamma-trace quotient

Equations (2.2)-(2.3) of arXiv:2007.05097 define the gamma-traceless
semi-prepotential and its local spinor symmetry:

\[
\widehat H_\alpha{}^a
=H_\alpha{}^a-\frac1{11}(\Gamma^a\Gamma_b)_\alpha{}^\beta
H_\beta{}^b,
\qquad
H_\alpha{}^a\longrightarrow H_\alpha{}^a+(\Gamma^a)_\alpha{}^\beta
\Lambda_\beta.
\]

The exact Clifford verifier gives rank 32 for the gamma-trace image and rank
320 for its complement. The $a$ and $b$ maps both land in the `(00001)`
gamma-trace image. They vanish after applying the rank-320 projector. The $c$
map lands in `(10001)` and survives. Thus

\[
\widehat H_\alpha{}^a(V)
=c\,P_{320}I_{320}(D^{15}V),
\]

up to normalization. Eq. (2.7) alone leaves a three-dimensional coefficient
space, but the local gamma-trace quotient leaves one non-gamma-trace bridge
class. The remaining nonzero scale is equivalent to rescaling $V$ until a
component convention is supplied. The gauge complex of the underlying scalar
or spinor prepotential is not yet established.

## Inherited direct gauge channels

Under the formal relation $V=D^\alpha\Psi_\alpha$, the exact generic-momentum
calculation identifies the two-form and five-form spinor-parameter channels as
the direct kernel of $V$. Since the quotient bridge is linear in $V$, both
channels also leave

\[
\widehat H_\alpha{}^a(V)=c\,P_{320}I_{320}(D^{15}V)
\]

invariant. This supplies two inherited direct gauge channels for the quotient
bridge. It does not establish that $V=D\Psi$ is the fundamental 11D
prepotential relation, determine gauge-for-gauge reducibility, or fix the
component normalization of $V$.

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
