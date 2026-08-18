# 11D top-down exact program status

Status date: 2026-08-17

## Scope

This work attacks the linearized 11D superfield problem from the top down. It separates exact representation-theoretic and component-level certificates from the still-open physical superfield gauge-curvature complex.

Primary source anchors:

- Gates, Hu, and Mak, arXiv:2002.08502, especially the complementary-level inventory and spinor-prepotential proposal.
- Gates, Hu, and Mak, arXiv:2007.05097, a 10D Weyl/prepotential paper used here only for its review of the conjectured 11D semi-prepotential route and Eqs. (2.2), (2.3), (2.6), and (2.23), not as an 11D cohomology oracle.
- Howe, hep-th/9707184; Cederwall, arXiv:1001.0112; Cederwall, Nilsson, and Tsimpis, hep-th/0110069; and Tsimpis, hep-th/0407271 for the 11D on-shell, pure-spinor, and spinorial-cohomology boundaries.
- Gates, hep-th/0101037, Eqs. (24)-(29), (39)-(40), and (44).
- The linearized component transformations collected in arXiv:0903.0259.

The source files and PDFs used by the curvature scaffold are content-hashed in the generated artifacts.

## Completed exact gates

### Target-resolved composition API

The `(10001)` gamma-traceless vector-spinor target now has a deterministic exact 320-state basis and invariant dual inside the full `11 x 32 = 352` ambient space. Exact visitors emit the target vector-weight index, spinor-weight index, gauge-parameter component, exterior mask, optional momentum index, and Gaussian-rational coefficient.

The work list contains 72 zero-momentum jobs and 336 first-momentum jobs, for 408 total. All 336 first-momentum physical `F_X` jobs have been executed on the declared parameter and target slice. The aggregate artifact has SHA-256 `5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9`. A separate promotion manifest with SHA-256 `98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4` binds the complete six-by-56 checkpoint key set to 336 content hashes. The 72 zero-momentum streams are separately covered by their bounded artifact, so this does not claim one materialized artifact for all 408 jobs.

The invariant target dual is also certified in the serialized report: 102,400 Kronecker pairings and 619,520 Chevalley metric-invariance entries have zero exact residuals.

### Source-fixed curvature scaffold

The source-fixed module certifies:

- rank-320 `P_320` and rank-32 gamma-trace representative redundancy;
- the exact 429-dimensional `X_[2]` hook projector;
- the exact 4,290-dimensional `X_[5]` hook projector;
- the printed linearized coefficients for the precontracted `X`, `W`, and `J` terms.

The executable physical-curvature slice implements Eqs. (24)-(29), the conventional-constraint part of Eqs. (39)-(40), the exact 429-dimensional `X_[2]` and 4,290-dimensional `X_[5]` quotient sectors, and typed 2001 and 2021 `W` assemblies. Source audits fixed the two-spinor raising sign, Eq. (26) factorial normalization, Eq. (24) lower-coordinate injection, Eq. (39) signs, both spinorial and mixed-torsion connection signs, the named `p=5` inverse-Hodge orientation and normalization, and the 2021 `D J^(+)` coefficient `11i/128`. All bounded source gates pass. The current enriched physical-curvature envelope has SHA-256 `3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13`. The induced `J/T/W` quotient, physical `K`, complete `F`, and full `F A G_p` test remain open.

### Abstract B5 Clifford join

A basis-independent exact solve avoids the unresolved Cartesian phase join. Weight selection leaves 192 possible gamma coefficients. The 736 Chevalley raising/lowering equations have rank 191, so the gamma intertwiner is unique up to scale. Its normalized coefficients are integral signs.

The resulting gamma trace has rank 32 and annihilates all 320 deterministic target states exactly. The induced vector metric has rank 11. The invariant spinor bilinear is unique and antisymmetric. All 55 rank-two Clifford operators pass exact Chevalley intertwining. The abstract `X_[2]` hook projector on `Lambda^2 V tensor V` is idempotent, has rank and trace 429, and removes both trace and total exterior components.

### B5 target to Lorentzian Majorana join

The compact B5 target basis is now connected exactly to the Lorentzian Majorana basis through a complex intertwiner rather than phase guessing. All ten Chevalley generators and all 55 Lorentz generators intertwine with zero residual. The invariant bilinear matches with scale `2i`; all 352 ambient gamma-trace states and all 320 target states agree; and the mapped target has rank 320 with an explicit two-sided ambient inverse. Upper and lower vector variance is explicit. A purely real join is obstructed by the signature mismatch `(6,5)` versus `(10,1)`, while the maximal complex join is complete and ready for the physical-curvature adapter.

The physical adapter has an independent exact audit. All 352 ambient states and all 320 target states roundtrip with zero residual; wrong vector variance is rejected; exterior masks and all eleven momentum exponents are preserved; malformed legacy keys fail closed; and a target-stream record agrees term by term with direct Cartesian `F_X`, producing 44 `X_[2]` and 128 `X_[5]` terms with zero comparison residual.

### Leading abstract X_[2] gauge symbol

The bounded zero-momentum calculation completed all 72 jobs, covering 12 source columns independently in each of the six gauge degrees. Exact cyclic-vector and lowering-orbit certificates cover every parameter irrep. The 1,024 exact output functionals give rigorous projected-rank lower bounds `[1,4,3,5,2,2]` for gauge degrees zero through five, with `[8,6,4,5,3,2]` nonzero source columns.

Every individual channel projection remains below 12, so the per-channel full ranks remain open. Across the direct sum of all six channels, however, the projected rank is seven and its five candidate kernel relations vanish exactly on the complete target-resolved source streams before curvature projection. The matching lower and upper bounds prove exact joint rank seven and nullity five, with a serialized exact kernel basis and mutation control. This does not select a physical source combination and does not prove the momentum branch or full `F A G_p = 0` identity.

### Level-18 source kernels

The level-18 source inventory is complete:

- 16 irreducible labels;
- 42 of 42 exact highest-weight kernels;
- 27 kernels from certified Hodge lifts;
- 15 kernels from direct exact sparse solves;
- exact raising and lowering verification for every persisted kernel;
- source labels available for all 77 target-resolved incidences.

This is source readiness, not 77 completed embedded Clebsch-Gordan maps.

### Embedded level-18 maps

All 77 source-target maps have exact checkpoints and zero raising residuals. The final six `11001_from_11002` copies each have certified image rank 10,240 and distinct content hashes. Their typed incidence-space direct sum has dimension 439,904 and now supports exact six-parameter specialization, rank, kernel, image containment, and quotient dimension. Synthetic generic, special-locus, zero-map, positive-containment, and transverse negative controls pass. The physical target gauge quotient remains false until `K` supplies the actual block-to-channel routing and coefficients.

### Complete first-momentum parameter screens

The four channels with nonempty zero-momentum kernels have complete first-momentum screens. Gauge degree zero has exact functional rank 17 and nullity 38. Gauge degrees one, two, and five have been evaluated on every parameter component, respectively 11, 55, and 462 components, and each has rank 42 and nullity 3. All four functional kernels have zero projection onto their recorded zero-momentum leading kernels. Gauge degrees three and four have zero-dimensional zero-momentum kernels and are excluded before first momentum. This makes the bounded six-channel strict-source screen green. It is a negative control for the old ansatz, not a generic-momentum test of the new source-fixed physical curvature.

### First-momentum physical F_X obstruction

The target-resolved `F_X = (X_[2],X_[5])` run is complete on its declared slice: all 56 recorded operators in each of the six gauge channels, parameter component zero, and target highest-weight state 319. It was computed against the immutable v10 physical-curvature input snapshot with SHA-256 `c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f`, not against the later enriched physical envelope. The promotion manifest verifies the exact 336-key checkpoint set without depending on remote paths. The run emitted 1,014,543,703 target terms.

The normal CLI entry `adynkra-11d-first-momentum-fx-aggregate` performs a merge-only rebuild from the canonical six-by-56 checkpoint tree. It validates every checkpoint and all final report invariants, never launches operator computation, and leaves the requested output untouched if any checkpoint is missing, partial, or corrupt.

The channelwise joint rank/nullity bounds are `11/38`, `46/3`, `35/14`, `49/0`, `48/1`, and `45/4`. Across all six channels, the `X_[2]`, `X_[5]`, and joint systems each have rank 49 and nullity zero, with exactness certified by dimension saturation. Therefore this slice excludes the complete recorded coefficient space consisting of the five leading-kernel directions plus 44 correction directions.

This is a sharp bounded negative result, not the full gauge-curvature identity. The parameter and target projections are not complete, `F_X` omits the `J` and `W` sectors, higher momentum orders are absent, and full `F A G_p` remains false.

### Generic K and F A G_p decision engine

The coefficient engine now works over exact Gaussian rationals and formal polynomials in all eleven momentum variables. It keeps derivative masks, derivative order, lower-symbol coverage, target-basis provenance, and the six inequivalent gauge domains explicit. It supports exact unique-ray, family, zero-kernel, and no-solution verdicts, plus channelwise and joint-kernel intersection after a target gauge quotient. Both `D^17 Lambda` and `p D^15 Lambda` streams are ingestible. The pinned harness artifact has SHA-256 `11ec33c36d9536e17e617839cc8dbabc885b9d30bf13ff05a4d0dc5e6b9fe562`; it independently binds the current `3c31...` physical envelope and frozen `c308...` F_X input snapshot. The bounded physical `F_X` slice now excludes the recorded 49-vector coefficient space, but the generic physical verdict remains false because `K`, complete `F`, complete projections, and the physical quotient are not yet available.

### Higher-jet Lorentz quotient

The lifted conventional-constraint projector on traceless `D Delta` jets has ambient dimension 32,736, rank 30,976, and nullity 1,760. Exact Clifford orthogonality identifies that kernel with the `32 x 55` derivative-Lorentz image. A deterministic 30,976-dimensional quotient basis is therefore available. The direct local-Lorentz audit shows that Eq. (25) supplies `7/32` and the raw frame chain supplies `1/2`, giving the Eq. (26) coefficient `23/32` on all 1,760 columns with zero residual. After correcting the Eq. (28) spinor-index variance, the connection trace is uniformly `17/32`; the earlier boost/spatial split was an index-placement artifact.

The remaining `J^(1)` descent is a unique Lorentz-equivariant `Gamma_[2]` map with coefficient `109/1056`, rank 32, and nullity 1,728. It is nonzero, and no source-authorized subtraction removes it. The geometry-level first superspace jet is now complete through `D omega`, `D J^(1)`, `D J^(2)`, `D J^(+)`, mixed torsion, and both the 2001 and 2021 `W` conventions. Its report passes. The missing inputs are the compensator-eliminated `H_hat` jet, complete physical `F`, and full `F A G_p` composition.

### Lorentzian Majorana form and physical supersymmetry

The complex Euclidean B5 spinor has an explicit Lorentzian Majorana real form. In that basis all eleven gamma matrices are real signed permutations and the exact charge-conjugation identities pass.

At null momentum `p=(1,1,0,...,0)`, exact sparse maps join the physical SO(9) content:

- graviton 44;
- three-form 84;
- transverse gamma-traceless vector-spinor 128.

All 32 boson-to-fermion and fermion-to-boson maps are public and executable. All 528 unordered charge pairs close exactly on both sectors. The calculation checks 17,301,504 exact closure entries with zero residual. This is an on-shell light-cone realization, not an off-shell scalar-superfield decomposition.

### Lowest spinorial differential slice

The exact Majorana basis now supports the first source-backed slice of the spinorial bicomplex. The map `tau_0: Omega^(1,0) -> Omega^(0,2)` has rank 11, its canonical quotient has dimension 517, and the scalar-symbol identity `d_1^2 + tau_0 d_0 = 0` vanishes coefficientwise in all eleven formal momentum variables. This is an entry-level Bianchi and nilpotence certificate. It is not the relaxed `X_[2] + X_[5]` cohomology, the full Bianchi tower, or finite-auxiliary off-shell closure.

### Relaxed spinorial cohomology worklist

The relaxed torsion sector `X_[2](11000) + X_[5](10002)` has dimension 4,719 and a 151,008-dimensional first spinor jet. Exact tensor-product incidence gives three multiplicity-one projected Bianchi arrows into `(11001) + (10003)`. On the source-supported order-`l^3` physical reduction, the exact canonical normal-form complex has dimensions `1 -> 3 -> 1`, ranks one and one, zero composition residual, and one-dimensional middle cohomology. The physical A/B/C tensor coefficients and unrestricted component differential are not reconstructed, so the result is a source-backed reduced cohomology certificate rather than the full relaxed Bianchi complex.

### Target curvature, Bianchi, and equation complex

The complete free target-side complex is exact over `Q(i)[p_0,...,p_10]`. The graviton sector factors through the linearized Riemann tensor and Einstein operator, the three-form sector through `F_4`, and the gravitino sector through its curl and Rarita-Schwinger operator. Gauge-curvature, curvature-Bianchi, curvature-Euler, gauge-Euler, and Euler-Noether compositions vanish exactly in every sector, and coefficient mutations are detected. The null fiber reproduces `44+84|128` and agrees with the exact real light-cone supersymmetry maps. The adapter for a future physical `F` exists, but the source-to-curvature and source-to-equation maps remain false.

## Open gates

1. Finish the convention-fixed `H_hat -> torsion -> (W,X_[2],X_[5],J)` operator `F`.
2. Solve the physical `Psi_alpha -> H_hat` map `K` and its six channel coefficients.
3. Connect the 77 exact embeddings to `K` and compute the physical target gauge quotient.
4. Apply the full physical `F` to both `D^17 Lambda` and `p D^15 Lambda` target streams independently for all six gauge domains at generic polynomial momentum.
5. Build the curvature, Bianchi, and field-equation complex and match its physical quotient to `44+84|128`.
6. Extend the exact spinorial differential to the relaxed `X_[2] + X_[5]` torsion complex and the relevant physical cohomology.
7. Join the physical `44+84|128` supersymmetry maps to the covariant superfield complex.

## Current interpretation

The 11D program is no longer blocked by the absence of a target-resolved stream, a Lorentzian real form, the physical on-shell multiplet, or level-18 source kernels. The recorded 49-dimensional first-momentum ansatz is now exactly excluded on the declared physical `F_X` slice. The remaining obstruction is sharply localized: the complete physical superfield differential, generic gauge-curvature composition, and target quotient are not source-fixed or computationally complete.

No current artifact proves an off-shell 11D multiplet, an irreducible scalar-superfield decomposition, or a nonlinear supersymmetric extension of Einstein's equation.
