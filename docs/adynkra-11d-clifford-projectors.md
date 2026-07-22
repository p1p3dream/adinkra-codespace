# Eleven-dimensional Clifford and vector-spinor projectors

## Purpose

The fifteen-derivative bridge

\[
D^{15}V\longrightarrow H_\alpha{}^a
\]

requires explicit intertwiners for the gamma trace and gamma-traceless parts of
the 352-dimensional vector-spinor. This verifier constructs the target-side
Clifford data needed for those maps.

It does not yet construct the three embeddings from
\(\bigwedge^{15}\{32\}\) into those target sectors.

## Clifford system

The Rust implementation constructs eleven \(32\times32\) Euclidean gamma
matrices with Gaussian-rational entries and checks

\[
\Gamma_a\Gamma_b+\Gamma_b\Gamma_a=2\delta_{ab}\mathbf 1_{32}.
\]

All 123,904 matrix entries in the 121 anticommutators agree with the Clifford
relation.

The charge-conjugation matrix is

\[
C=\Gamma_2\Gamma_4\Gamma_6\Gamma_8\Gamma_{10}.
\]

The code verifies \(C^T=-C\) and

\[
C\Gamma_aC^{-1}=-\Gamma_a^T
\]

with zero residual entries.

## Spinor bilinears

Every antisymmetrized gamma product through degree five is checked. The
bilinear symmetries are

| Form degree | Dimension | \(C\Gamma^{[p]}\) symmetry |
|---:|---:|---|
| 0 | 1 | antisymmetric |
| 1 | 11 | symmetric |
| 2 | 55 | symmetric |
| 3 | 165 | antisymmetric |
| 4 | 330 | antisymmetric |
| 5 | 462 | symmetric |

Thus

\[
\operatorname{Sym}^2\{32\}=\{11\}\oplus\{55\}\oplus\{462\},
\qquad 528=11+55+462,
\]

and

\[
\bigwedge^2\{32\}=\{1\}\oplus\{165\}\oplus\{330\},
\qquad 496=1+165+330.
\]

The two dimensions sum to \(32^2=1024\), matching the six gauge-parameter
channels identified in the prepotential inventory.

## Vector-spinor projectors

For \(H_a{}^\alpha\), the gamma-trace projector is

\[
(P_{32})_{ab}=\frac1{11}\Gamma_a\Gamma_b,
\]

and the complementary projector is

\[
P_{320}=\delta_{ab}\mathbf 1_{32}-P_{32}.
\]

The verifier checks:

- \(P_{32}^2=P_{32}\);
- \(P_{320}^2=P_{320}\);
- \(P_{32}+P_{320}=1\);
- the complementary image is gamma traceless;
- \(\operatorname{rank}P_{32}=32\);
- \(\operatorname{rank}P_{320}=320\).

The projector-product check covers 123,904 entries and has zero residuals.
The gamma-tracelessness check covers 11,264 entries and has zero residuals.

## Direct gauge-channel test at zero momentum

For the direct first-derivative ansatz

\[
\delta\Psi_\alpha=(C\Gamma^{[p]})_\alpha{}^\beta
D_\beta\Lambda_{[p]},
\]

the induced scalar variation contains a contraction of
\(C\Gamma^{[p]}\) with two spinor derivatives. At zero spacetime momentum,
the spinor derivatives anticommute. The symmetric bilinear channels therefore
vanish in this contraction, while the antisymmetric channels do not vanish as
an algebraic identity.

| Result at zero momentum | Form degrees |
|---|---|
| contraction vanishes | 1, 2, 5 |
| not identically zero | 0, 3, 4 |

This separates the six direct channels into two classes. It does not select a
gauge transformation at generic momentum, where the spinor-derivative
anticommutator produces a spacetime derivative.

## Remaining work on the four requested stages

1. **Bridge intertwiners:** the target projectors are now explicit. The two
   embeddings of \(\{32\}\) and the one embedding of \(\{320\}\) inside
   \(\bigwedge^{15}\{32\}\) remain to be constructed.
2. **Torsion constraints:** Eq. (2.7) of arXiv:2007.05097 cannot be evaluated
   on the three coefficients until those source embeddings are explicit.
3. **Surviving combination:** no coefficient combination has yet been selected.
4. **Gauge complex:** the six direct channels are now split at zero momentum,
   but generic-momentum reducibility and curvature kernels remain open.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

```bash
cargo run --release -- adynkra-11d-clifford-verify \
  > results/adynkra_11d_clifford_projectors_validation.json
cargo test eleven_dimensional_clifford
```

Implementation: `src/eleven_dimensional_clifford.rs`

## References

1. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Adinkra Foundation of Component
   Decomposition and the Scan for Superconformal Multiplets in 11D, N = 1
   Superspace," [arXiv:2002.08502](https://arxiv.org/abs/2002.08502).
2. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Weyl Covariance, and Proposals
   for Superconformal Prepotentials in 10D Superspaces,"
   [arXiv:2007.05097](https://arxiv.org/abs/2007.05097).
