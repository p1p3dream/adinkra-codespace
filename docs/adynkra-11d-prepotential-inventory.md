# Eleven-dimensional prepotential-candidate inventories

## Result

The complete Lorentz-irrep inventory of the unconstrained real scalar
superfield \(V(x,\theta)\) in eleven dimensions has been reproduced from
Appendix F of Gates, Hu, and Mak, arXiv:2002.08502.

The Rust verifier checks all 33 Grassmann levels. The published levels 0-16
are transcribed as \(B_5=\mathfrak{so}(11)\) Dynkin labels. Levels 17-32 are
generated from the stated level reflection \(n\leftrightarrow32-n\).

| Check | Result |
|---|---:|
| Published levels transcribed | 17 |
| Reflected levels generated | 16 |
| Levels checked | 33 |
| Level-dimension mismatches | 0 |
| Reflection mismatches | 0 |
| Bosonic irreducible fields | 1,494 |
| Fermionic irreducible fields | 1,186 |
| Bosonic component dimension | \(2^{31}\) |
| Fermionic component dimension | \(2^{31}\) |

The field counts and parity dimensions agree with Table 4 and the text that
follows it. Here, "irreducible fields" counts Lorentz irreps with
multiplicity. "Component dimension" weights each occurrence by its Lorentz
representation dimension.

## Validation method

For every published Dynkin label \((a_1a_2a_3a_4a_5)\), the verifier computes
the \(B_5\) Weyl dimension over all positive roots using rational arithmetic.
It then checks

\[
 \sum_{\lambda}m_{n,\lambda}\,\dim(\lambda)
 = \binom{32}{n}
\]

at each level \(n\). All 33 identities pass. The largest identity is the
middle level,

\[
 \sum_{\lambda}m_{16,\lambda}\,\dim(\lambda)
 = \binom{32}{16}
 = 601{,}080{,}390.
\]

The implementation also checks standard \(SO(11)\) dimensions independently,
including

\[
\{1\},\{11\},\{55\},\{165\},\{330\},\{32\},\{65\},\{320\},\{462\}.
\]

Together with the source comparison, these checks detect any isolated missing
term, incorrect label, or incorrect multiplicity in the Appendix F
transcription.

## Supergravity representations

The reproduced inventory contains the representations identified in the
paper:

- level 16 contains one \((20000)\), dimension 65, the conformal graviton;
- level 16 contains two \((00100)\), dimension 165, three-form occurrences;
- levels 15 and 17 each contain one \((10001)\), dimension 320, conformal
  gravitino occurrence;
- level 16 contains no \((01000)\), dimension 55, two-form occurrence.

The last fact is the boundary stated in the paper's note in proof. The scalar
superfield contains the scalar, conformal-graviton, three-form, spinor, and
conformal-gravitino representations at the required middle levels, but it
lacks the 55-dimensional two-form required by the inverse-frame
decomposition. Gates, Hu, and Mak therefore describe \(V\) as a possible
semi-prepotential and conjecture a spinor superfield \(\Psi_\alpha\), related
by \(V=D^\alpha\Psi_\alpha\), as a prepotential candidate.

## Spinor-prepotential candidate

The verifier also constructs the complete Lorentz inventory of
\(\Psi_\alpha\). At level \(n\), this is

\[
 (00001)\otimes\bigwedge^n(00001).
\]

The 32-dimensional \((00001)\) representation is minuscule. The Rust code
therefore decomposes each scalar-inventory irrep by adding all 32 spinor
weights and retaining the dominant weights. The spinor-square case also reproduces the six channels printed in Eq. (E.2).
It performs two dimension checks:

1. every individual product satisfies
   \(\sum_\mu\dim(\mu)=32\dim(\lambda)\);
2. every complete level satisfies
   \(\sum_\mu m_{n,\mu}\dim(\mu)=32\binom{32}{n}\).

Both checks have zero residuals. The bosonic and fermionic component
dimensions of the spinor superfield are each \(2^{36}\).

The six middle-level multiplicities printed in Table 5 are reproduced:

| Level | Irrep | Dimension | Published | Computed |
|---:|---:|---:|---:|---:|
| 17 | `(00000)` | 1 | 2 | 2 |
| 17 | `(01000)` | 55 | 5 | 5 |
| 17 | `(20000)` | 65 | 2 | 2 |
| 17 | `(00100)` | 165 | 8 | 8 |
| 18 | `(00001)` | 32 | 5 | 5 |
| 18 | `(10001)` | 320 | 8 | 8 |

This confirms the representation-content statement behind the paper's
spinor-prepotential conjecture. It does not establish that the conjectured
field has the required gauge complex or dynamics.

## Source-defined bridge to the semi-prepotential

The prepotential notation is not uniform across the source papers. In
arXiv:2007.05097, the scalar conformal compensator is written as \(\Psi\) and
the vector-spinor semi-prepotential as \(H_\alpha{}^a\). In the note in proof
of arXiv:2002.08502, \(\Psi_\alpha\) instead denotes the conjectured
fundamental spinor prepotential. The verifier keeps these objects separate.

The discussion following Eq. (2.7) of arXiv:2007.05097 states that the
functional dependence \(H_\alpha{}^a(V)\) must involve fifteen spinor
derivatives. The target vector-spinor decomposes as

\[
 (10000)\otimes(00001)=(00001)\oplus(10001),
 \qquad 352=32+320.
\]

The level-15 scalar inventory contains

\[
 2(00001)\oplus(10001)\oplus\text{other irreps}.
\]

Consequently, the leading symbol of a Lorentz-equivariant map

\[
 D^{15}V\longrightarrow H_\alpha{}^a
\]

has three channels in the Dynkin-label representation convention: two into
the gamma trace and one into the gamma-traceless vector-spinor. Both target
irreps are present. Composing this with the conjectured relation
\(V=D^\alpha\Psi_\alpha\) gives a formal sixteen-spinor-derivative bridge from
\(\Psi_\alpha\) to \(H_\alpha{}^a\).

This reduces the scalar-factorizing leading-symbol problem to three
coefficients. The two cited papers do not give the three intertwiners or their
coefficients. Equation (2.7) is the strengthened constraint proposed in
connection with this scalar-superfield route.

The direct spinor-prepotential route is different. Its gamma-traceless
leading map space has dimension 12, only one direction factors through the
scalar \(V\), and its level-17 `(11000)` hook content is permitted by the
conventional constraints in Eq. (2.6). The direct route is documented in
[`adynkra-11d-spinor-prepotential-bridge.md`](adynkra-11d-spinor-prepotential-bridge.md).

## Gauge-parameter channel census

The introduction of arXiv:2007.05097 states a conjectural high-dimensional
rule: a supergravity prepotential transforms as one spinor derivative of a
Lorentz-compatible parameter superfield. For a spinor prepotential, the
possible parameter representations are the channels in

\[
 (00001)\otimes(00001)
 = (00000)\oplus(10000)\oplus(01000)\oplus(00100)
   \oplus(00010)\oplus(00002).
\]

These are the scalar, vector, two-form, three-form, four-form, and five-form,
with dimensions

\[
 1+11+55+165+330+462=1024=32^2.
\]

Each occurs once. This is a complete census of Lorentz-compatible
first-derivative parameter channels. It is not a claim that all six are gauge
symmetries. The two cited papers do not select a channel combination, give its
coefficients, or state its gauge-for-gauge reducibility.

## What this establishes

This completes the representation-inventory gate for applying the
validated four-dimensional Adynkrafield machinery in eleven dimensions:

1. every scalar-superfield Lorentz block is available in machine-readable
   Rust data;
2. every block dimension and multiplicity is checked against the full
   Grassmann expansion;
3. the middle-level supergravity candidates and the missing 55-dimensional
   representation are detected directly from the scalar inventory;
4. the full spinor-superfield inventory is derived by the minuscule tensor
   rule, and all Table 5 multiplicities are reproduced;
5. the fifteen-derivative bridge is reduced to three leading-symbol channels;
6. the six possible first-derivative gauge-parameter channels are enumerated;
7. explicit gamma-trace and gamma-traceless vector-spinor projectors are
   constructed;
8. the direct gauge ansatz is separated into three channels that vanish in
   the induced scalar variation at zero momentum and three that do not.

It does not solve the proposed eleven-dimensional torsion constraints for the
scalar-factorizing bridge, select the spinor prepotential's gauge
transformation, or derive an eleven-dimensional action or field equation.
Those are separate steps.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

```bash
cargo run --release -- adynkra-11d-prepotential-verify \
  > results/adynkra_11d_prepotential_inventory_validation.json
cargo test eleven_dimensional_prepotential
```

Implementation: `src/eleven_dimensional_prepotential.rs`

The Clifford construction and vector-spinor projectors are documented in
[`adynkra-11d-clifford-projectors.md`](adynkra-11d-clifford-projectors.md).

## References

1. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Adinkra Foundation of Component
   Decomposition and the Scan for Superconformal Multiplets in 11D, N = 1
   Superspace," JHEP 09 (2020) 089,
   [arXiv:2002.08502](https://arxiv.org/abs/2002.08502).
2. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, "Weyl Covariance, and Proposals
   for Superconformal Prepotentials in 10D Superspaces,"
   [arXiv:2007.05097](https://arxiv.org/abs/2007.05097).
