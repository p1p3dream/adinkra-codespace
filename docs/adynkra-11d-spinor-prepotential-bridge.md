# Direct eleven-dimensional spinor-prepotential bridge

## Result

The direct Lorentz-equivariant map space from the proposed spinor
prepotential \(\Psi_\alpha\) to the gamma-traceless vector-spinor
\(H_\alpha{}^a\) has been counted in Rust.

At sixteen spinor derivatives,

\[
 \mathrm{Hom}_{\mathrm{Spin}(1,10)}
 \left(\bigwedge^{16}S\otimes S,\,(10001)\right)
\]

has dimension 12. The map obtained by first forming the scalar
semi-prepotential

\[
 V=D^\alpha\Psi_\alpha
\]

and then applying the scalar-to-\(H\) bridge occupies one gamma-traceless
direction. The direct spinor source therefore has 11 additional directions
that do not factor through \(V\) at the level of Lorentz representations.

| Source irrep in \(\bigwedge^{16}S\) | Multiplicity | Map coefficients |
|---:|---:|---:|
| `(10000)` | 1 | 1 |
| `(20000)` | 1 | 1 |
| `(00100)` | 2 | 2 |
| `(00010)` | 2 | 2 |
| `(00002)` | 1 | 1 |
| `(10100)` | 1 | 1 |
| `(10010)` | 1 | 1 |
| `(10002)` | 3 | 3 |
| **Total** |  | **12** |

This is a representation-level result. It does not choose the 12
coefficients or construct their component Clebsch-Gordan tensors.

## Explicit construction status

The explicit intertwiner pass began with the sparse highest-weight systems
for the scalar irreps that generate the 12 leading channels and seven hook
channels. Rust now constructs the exact weight-space columns and all five
simple-root raising-block dimensions for each system.

| Systems | Scalar irreps | Weight-space columns | Raising rows | Expected kernels |
|---|---:|---:|---:|---:|
| Level-16 leading sources | 8 | 2,624,572 | 8,131,298 | 12 |
| Level-17 hook sources | 4 | 911,915 | 2,676,961 | 7 |

These are decomposed systems. They replace a single direct-product solve with
12 level-16 source embeddings and seven level-17 source embeddings.

All 19 integer highest-weight kernels are now constructed. The Rust verifier
rebuilds every simple-root raising equation and checks 15,429,636
kernel-row relations with integer arithmetic. Every residual is zero. It also
checks the complete highest-weight lowering string for every simple root.

The numerical sparse calculation is retained only as an independent method
for proposing the integer vectors. The stored vectors are accepted only after
the Rust verifier checks them exactly. The component Clebsch-Gordan coupling
on the target side is also constructed:

\[
 (11000)\longrightarrow(00001)\otimes(10001).
\]

Its highest-weight domain has dimension 8, its kernel has dimension 1, all
eight primitive coefficients are nonzero, and every raising residual is zero.

The first leading source coupling is also complete. The level-16 `(10000)`
source highest vector, tensored with the highest spinor weight, gives the
`(10001)` vector-spinor highest vector. Its stored source has 594,896 nonzero
integer coefficients and zero raising residuals. This completes one of the 12
leading couplings.

The remaining work is to construct the other 11 leading source couplings, the
seven hook source couplings, and the derivative matrix.

## The proposed hook cancellation is not the direct-spinor test

Equations (2.6) and (2.7) of arXiv:2007.05097 impose different constraint
sets. Equation (2.6) retains the dimension-zero two-form hook
\(X_{[ab]}{}^c\) in the 429-dimensional `(11000)` representation. The paper
then proposes the stronger Eq. (2.7), which sets that full gamma-two torsion
sector to zero, in connection with the scalar-superfield prepotential
candidate.

The scalar inventory supports that restriction: the scalar \(V\) has no
`(11000)` at level 16. The Added Note in Proof of arXiv:2002.08502 proposes
the spinor \(\Psi_\alpha\) precisely because the scalar inventory is
incomplete. It follows that imposing the scalar-motivated Eq. (2.7) hook
cancellation on the direct spinor route would discard a sector that the
conventional Eq. (2.6) constraints permit.

The direct spinor inventory contains seven hook directions at level 17:

| Source irrep in \(\bigwedge^{17}S\) | Multiplicity into `(11000)` |
|---:|---:|
| `(10001)` | 1 |
| `(01001)` | 2 |
| `(20001)` | 1 |
| `(11001)` | 3 |
| **Total** | **7** |

A hook component is therefore allowed under Eq. (2.6). Its value for any
selected linear combination of the 12 leading maps has not been computed.
The planned projection into `(11000)` using Eq. (2.7) closes here because it
was the wrong constraint for this candidate, not because the seven directions
were shown to vanish or survive.

## First momentum corrections

For a correction with one vector momentum and fourteen spinor derivatives,
the target-side tensor product is

\[
 (10000)\otimes(10001)
 = (00001)\oplus(01001)\oplus(10001)\oplus(20001).
\]

The complete representation-level map space has dimension 44:

| Intermediate irrep | Multiplicity in \(D^{14}\Psi\) | Map coefficients |
|---:|---:|---:|
| `(00001)` | 5 | 5 |
| `(01001)` | 18 | 18 |
| `(10001)` | 8 | 8 |
| `(20001)` | 13 | 13 |
| **Total** |  | **44** |

These corrections remain available in a future direct-spinor construction.
They are not required to cancel the seven hook directions solely on the
basis of Eq. (2.7).

## Gauge boundary

The spinor square gives six possible first-derivative parameter types:

\[
 (00000),\ (10000),\ (01000),\ (00100),\ (00010),\ (00002),
\]

with total dimension \(1+11+55+165+330+462=1024\). The cited papers do not
select a linear combination of these channels, give its coefficients,
specify its gauge-for-gauge relations, or print the induced transformation
of \(H_\alpha{}^a\).

Consequently, a gauge-compatible quotient cannot be determined from the
published premises. Imposing invariance under all six channels would add an
unsupported assumption. The next required input is a selected direct map
and a gauge law for \(\Psi_\alpha\), together with the induced transformation
of \(H_\alpha{}^a\).

## What is established

1. The direct leading map space has dimension 12.
2. The scalar-factorizing gamma-traceless subspace has dimension 1.
3. The direct nonfactorizing complement has dimension 11.
4. The level-17 `(11000)` hook space has dimension 7.
5. The first momentum-correction map space has dimension 44.
6. The Eq. (2.7) hook cancellation is not applicable to the direct spinor
   route on the stated published premises.
7. The gauge quotient is underdetermined by the cited sources.

No torsion solution, gauge quotient, action, or field equation is claimed.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

```bash
cargo run --release -- adynkra-11d-spinor-bridge-verify \
  > results/adynkra_11d_spinor_bridge_validation.json
cargo run --release -- adynkra-11d-spinor-kernel-verify \
  > results/adynkra_11d_spinor_bridge_kernel_validation.json
cargo test eleven_dimensional_spinor_bridge
```

Implementation:
`src/eleven_dimensional_spinor_bridge.rs` and
`src/eleven_dimensional_spinor_bridge_kernels.rs`

Independent numerical kernel proposal:

```bash
python3 scripts/crosscheck_11d_level15_bridge.py \
  --degree 16 --label 10002 --iterations 1200 --tolerance 1e-11 \
  --integer-artifact-directory data/eleven_dimensional_spinor_bridge
python3 scripts/crosscheck_11d_level15_bridge.py \
  --degree 17 --label 11001 --iterations 1800 --tolerance 1e-12 --seed 29 \
  --integer-artifact-directory data/eleven_dimensional_spinor_bridge
```

The same command covers each level-16 and level-17 source label listed above.

## References

1. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Adinkra Foundation of Component
   Decomposition and the Scan for Superconformal Multiplets in 11D, N = 1
   Superspace," JHEP 09 (2020) 089,
   [arXiv:2002.08502](https://arxiv.org/abs/2002.08502). See the discussion
   around Eq. (4.36) and the Added Note in Proof, Eqs. (6.1)-(6.3).
2. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Weyl Covariance, and Proposals
   for Superconformal Prepotentials in 10D Superspaces,"
   [arXiv:2007.05097](https://arxiv.org/abs/2007.05097). See Eqs. (2.6)-(2.7)
   and the discussion following Eq. (2.7).
