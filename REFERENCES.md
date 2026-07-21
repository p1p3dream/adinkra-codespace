# References and attribution

This project implements and *validates against* published results in the adinkra /
off-shell-supersymmetry literature. Every externally-sourced formula, validation
target, definition, or dataset used in the code is attributed below, with the
source paper and the file that uses it. Where a result is only a necessary
condition, conjectural, or experimental, the code says so explicitly (see the
relevant module doc-comments).

## Enumeration and codes (foundational)

- **Doran, Faux, Gates, Hübsch, Iga, Landweber, Miller**, *Topology Types of
  Adinkras and the Corresponding Representations of N-Extended Supersymmetry*,
  arXiv:0806.0050. Enumeration of code and Adinkra-topology equivalence classes.
  Used by: `code.rs`, `canonical.rs`, `search.rs`.
- **Doran, Faux, Gates, Hübsch, Iga, Landweber**, *Relating Doubly-Even
  Error-Correcting Codes, Graphs, and Irreducible Representations of N-Extended
  Supersymmetry*, arXiv:0806.0051. Doubly-even codes and Adinkra
  chromotopologies.
- **Doran, Faux, Gates, Hübsch, Iga, Landweber**, *Adinkras for Clifford Algebras,
  and Worldline Supermultiplets*, arXiv:0811.3410; **Faux, Gates**, *Adinkras: A
  Graphical Technology for Supersymmetric Representation Theory*, hep-th/0408004.
  GR(d,N) Garden algebra `L_I R_J + L_J R_I = 2 δ_IJ I`; L/R signed-permutation
  matrices. Used by: `lr_matrix.rs`, `signed_perm.rs`.

## Heights / rankings / dashings

- **Y. X. Zhang**, *Adinkras for Mathematicians*, arXiv:1111.6055 (Trans. AMS).
  Definition of a ranking (height function), valise, and the ranking-counting
  recursion (§6). Used by: `ranking.rs` (definition, `enumerate`, valise), and the
  N=4 ranking count cross-check.
- **Doran, Iga, Landweber**, *Cubical Cohomology of Adinkras*, arXiv:1207.6806.
  Dashings classified by H¹ (the 2^k classes), distinct from rankings. Used by:
  `dashing.rs`, and the ranking-vs-dashing orthogonality noted in `ranking.rs`.
- Adinkra height-function / "hanging gardens" context: arXiv:2410.11137. Used by:
  `ranking.rs` (the `|h(u)−h(v)|=1` hanging-gardens constraint).

## Holoraumy, gadget, χ₀ (chromocharacter)

- **Gates, Hübsch, et al.**, *A Lorentz Covariant Holoraumy-Induced "Gadget" From
  Minimal Off-Shell 4D N=1 Supermultiplets*, arXiv:1508.07546 (JHEP 11(2015)113).
  Gadget definition `G = −2/(N(N−1)dmin) Σ Tr(Ṽ Ṽ′)` and normalization; the N=4
  gadget values. Used by: `holoraumy.rs` (gadget), `decompose.rs` (dense gadget),
  `chromochar.rs`.
- **Gates et al.**, *Adinkra height-yielding / chromocharacter & χ₀* — arXiv:1712.07826
  and arXiv:1701.00304. The chromocharacter `Tr(L_I R_J L_K R_L) = 4[(n_c+n_t)(δδ
  −δδ+δδ) + χ₀ ε_IJKL]` and χ₀ = ±1 (cis/trans). Used by: `chromochar.rs`.
- **Gates et al.**, *Adinkra "color" confinement in exemplary off-shell 4D N=2
  representations*, arXiv:1405.0048 (JHEP 07(2014)051). χ₀ = 0 as a NECESSARY
  (not proven sufficient) condition for off-shell fusion; chiral=+1, vector=−1.
  Used by: `chromochar.rs` (the validated convention + the honest "necessary, not
  sufficient" scope).

## Dimensional enhancement / worldsheet / off-shell

- **Gates, Hübsch**, *On Dimensional Extension of Supersymmetry: From Worldlines to
  Worldsheets*, arXiv:1104.0722. The worldsheet bow-tie / height-weighted spin-sum
  obstruction (Thm 2.1/2.2, Cor 2.2). The paper describes evasion of the
  obstruction as conjectured to be sufficient. Used by: `filters.rs`
  (`worldsheet_spin_sum`, the certificate verifier, and the weight-2 condition).
- **Faux, Iga, Landweber**, *Dimensional Enhancement via Supersymmetry*,
  arXiv:0907.3605. The Ω = 0 enhancement obstruction (N=4; not yet generalized to
  N=16). Referenced for the (deferred) 4D non-gauge filter.
- **Calkins, Gates, Gates, McPeak**, *Is It Possible To Embed A 4D N=4 SUSY Vector
  Multiplet Within A Completely Off-Shell Adinkra Hologram?*, arXiv:1402.5765
  (JHEP 05(2014)057). Off-shell N=4 as a linear-algebra closure problem; explicitly
  isolates the paired-auxiliary-spinor premise and asks what changes if it is
  removed. Used by: `sr_hole.rs` (the minimal unpaired arithmetic audit).
- **Gates et al.**, *Think Different: ... the SUSY Auxiliary Field Problem*,
  arXiv:1502.04164 (JHEP 04(2015)056). The counting/Diophantine framing of the
  off-shell no-go.
- **Siegel, Roček**, *On off-shell supermultiplets*, Phys. Lett. B 105 (1981) 275.
  The auxiliary-field counting argument. Its contradiction uses the additional
  conventional premise that auxiliary spinors occur in pairs; it is not an
  algebra-only exclusion of every finite representation. Used by: `sr_hole.rs`.
- **Arunseangroj, Bedessem, Gates, Yerger**, *Adinkras & Genomics in Sixteen Color
  Systems (I)*, arXiv:2503.13797 (2025). N=16 k=8 D16/E8×E8 distance-spectrum
  validation; "naive off-shell route closed for 4D N=4 Maxwell." Used as the
  validation target for the k=8 gadget classification (`decompose.rs`).
- **Baulieu, Berkovits, Bossard, Martin**, *Ten-dimensional super-Yang-Mills with
  nine off-shell supersymmetries*, arXiv:0705.2002 (Phys. Lett. B 658 (2008) 249).
  10D SYM: SO(1,1)×Spin(7), 1+8 close off-shell, 7 obstructed (the "9 of 16"
  result). Calibration target for the (designed) Closure-Defect classifier.

## 10D Clifford / supergravity data

- **Plefka, Waldron**, *Asymptotic Supergraviton States in Matrix Theory*,
  arXiv:hep-th/9801093. The SO(9) supergraviton state content is
  `44 + 84 | 128`, with symmetric-traceless tensor, three-form, and
  gamma-traceless vector-spinor sectors. Used as the identification target for
  `scripts/check_sr_spin9_decomposition.py`; the decomposition itself is
  verified there by exact Cartan characteristic polynomials.

- **Jacob Cigliano, Bergen Dahl, S. James Gates, Jr.**, *10D Supergravity Numerical
  Data Sets for L & R Matrices*, arXiv:2512.12157v1; data repo
  **github.com/mcmulaz/Super-Sym** ("Garden
  Algebra" Mathematica file, commit `8c8df92`). The split SO(1,9) sigma relation
  `σ^μ σ̃^ν + σ^ν σ̃^μ = 2 η^μν I_16` and the explicit 82×176 / 176×82 L/R matrices +
  non-closure tensor E_IJ. Used by: `lorentz.rs` (split-sigma relation),
  `tendim_generate.rs`, `tendim_data.rs`, and `data/tendim_10d_lr.json`.
  **Provenance note:** our JSON is generated in Rust from a transcription of the
  authors' Garden Algebra source at commit
  `8c8df92`, not a literal download. It satisfies the bosonic Garden relation to
  ~1.7e-12 in float (exactly 0 in exact arithmetic). The fermionic non-closure
  remnant E_IJ is nonzero (computed here, NOT a paper-reported figure). This is
  evidence of algebraic plausibility. Rust verifies all 136 bosonic relations
  exactly over `Q(sqrt(2))`. The NumPy and SymPy ports remain independent
  cross-checks and match the Rust artifact entrywise. None executes Wolfram
  Language or establishes the authors' intended convention.
  **Paper/source inconsistency:** the audit checks all 13 displayed equations in
  Eqs. 6.0.5-6.0.6. Four equations and seven of 26 terms match directly. No fixed
  shared permutation of the displayed spinor labels reconciles the sampled
  equations with the generator. Appendix A's
  printed sigma basis supports the generator's `Q_1 h_11 = 2 ψ_1(16)`, not the
  displayed `ψ_1(6)`. The executable source uses `1/16 MixedLeft`, while its nearby
  comment says `1/8`. The Rust generator and both cross-checks agree; the intended L rows and
  assembled R coefficients still require author confirmation. See PROVENANCE.md §6 and run
  `cargo run --release -- tendim-reproduce`.
  **License / citation posture:** the upstream repo carries no LICENSE file. The
  local artifact is a cited academic transcription of published generative formulas,
  not a copied upstream matrix file.
  Full hashes, conventions, regen command, and the mismatch status are recorded in
  **PROVENANCE.md**.
- Octonionic SO(9) Clifford construction (Fano-plane imaginary-octonion
  left-multiplications) and even-dimensional recursive Clifford construction:
  standard; cf. arXiv:2205.09509 (*Lecture note on Clifford algebra*) and the
  Brink-Schwarz-Scherk 10D SYM construction. Used by: `lorentz.rs`, `sr_hole.rs`.

## Scope honesty (what is NOT claimed)

- The χ₀-derived N=16 invariant `Q` (`chromochar.rs`) is a basis-independent
  *classification coordinate / conjectural screen*, not a proven necessary condition
  for N=16 off-shell liftability.
- The non-valise height-aware L/R is only partially implemented (`height_signature`
  is correct; the square sign-flipped R was removed as incorrect — the correct
  construction is rectangular, matching the 82×176 dataset, and is future work).
- The `lorentz::assemble_and_check` `e_norm` is an EXPERIMENTAL residual, not a
  calibrated off-shell/on-shell certificate.
- No positive off-shell lift of 4D N=4 / 10D N=1 is claimed. The existing
  Siegel-Rocek argument excludes its stated conventional paired-auxiliary ansatz,
  not every finite algebraic representation. `sr_hole.rs` tests a narrower exact
  minimal adinkraic embedding ansatz and labels that boundary explicitly.
