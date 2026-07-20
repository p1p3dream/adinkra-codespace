# BBBM 9-of-16 partial off-shell: the N=9 valise scaffold

## Summary

Baulieu, Berkovits, Bossard, and Martin (arXiv:0705.2002, "Ten-dimensional
super-Yang-Mills with nine off-shell supersymmetries") close 9 of the 16
supersymmetries of 10D, N=1 super-Yang-Mills off-shell by adding 7 auxiliary
scalars. The remaining 7 supercharges close only on-shell (modulo the equations
of motion). This is the "partial off-shell" loophole around the Siegel-Rocek
no-go: do not demand that all 16 close, settle for 9.

Reduced to the worldline, the off-shell field content is

    16 bosons  = 9 gauge components + 7 auxiliary scalars
    16 fermions (gaugino)

under N = 9 supersymmetries. That is exactly the minimal N=9 valise, GR(16,9),
since the minimal representation dimension is d_min(9) = 16.

## What this scaffold does

`src/bbbm.rs` (subcommand `bbbm`) builds the minimal N=9 valise from its maximal
doubly-even code, the [8,4] extended Hamming code padded with a trivial ninth
coordinate (k = 4, so d = 2^(9-1-4) = 16), through the codebase's tested
`Chromotopology` -> `DashingEnumerator` -> `AdinkraRep` machinery, and verifies
the Garden algebra

    L_I R_J + L_J R_I = 2 delta_IJ I_16

exactly, for every dashing class.

Result (`bbbm` output): N = 9, d = 16, 16 bosons (9 gauge + 7 auxiliary) and 16
fermions, 16 dashing classes, Garden algebra verified for all of them.

## Honest scope

The minimal N=9 valise is unique up to adinkra equivalence, so BBBM's off-shell
content must realize this object. Establishing that it exists and closes is the
first, byte-reproducible step, and it is built through already-tested machinery,
so there is no hand-rolled-physics risk.

This step was essentially guaranteed to pass (the minimal N=9 valise closing the
Garden algebra is a known fact). Its value is that the object is now reproducible
inside the apparatus and is the correct scaffold for the two follow-on routes.

This scaffold does NOT:

- reduce the specific BBBM SUSY transformation rules of arXiv:0705.2002 (that
  requires the paper's explicit field variations);
- compute the non-closure functions of the remaining 7 supercharges, the
  off-shell-sector "equation of motion".

## Next routes

- Route A (in-apparatus, low-risk): compute the holoraumy / gadget invariants of
  the N=9 valise with the existing `holoraumy` machinery, and locate its
  chromotopology in the N=9 code family (code class, automorphism group, weight
  enumerator). Tells us the code/gadget structure of the BBBM shadow.
- Route B (BBBM-specific, higher value): extract the arXiv:0705.2002
  transformation rules, build the actual 9 supercharge matrices, and compute the
  non-closure of the unclosed 7. See the Route B verdict below.

## Route A results (done)

`src/bbbm_holoraumy.rs` (subcommand `bbbm-holoraumy`) computes the holoraumy and
gadget invariants of the N=9 valise, and `src/bbbm.rs::crosscheck` re-derives
them and the code invariants through a fully independent dense-matrix path.
Both agree exactly. Findings, for all 16 dashing classes:

- Holoraumy is purely antisymmetric and traceless: every V_IJ and Vtilde_IJ has
  trace 0, V_IJ = -V_JI, and V_IJ^2 = -I_16.
- Self-gadget G[R,R] = 1.0 for every dashing class, matching d/d_min = 16/16
  (irreducible).
- The self-gadget is dashing-invariant, but the cross-gadget between distinct
  dashings takes exactly three values: 1/8, 1/6, 7/24. So the 16 dashings carry
  a non-trivial gadget inner-product structure (never orthogonal here).
- The [9,4] code: weight enumerator {0:1, 4:14, 8:1} (identical to extended
  Hamming[8,4]), self-orthogonal, |Aut| = 1344 = |AGL(3,2)|, classical d_min 4;
  k = 4 is maximal (a doubly-even length-9 code has k <= 4). The 9th coordinate
  is a trivial (support-free) color; mathematically the code is Hamming[8,4] plus
  a trivial coordinate, though the codebase's support-only decomposition routines
  report it as indecomposable.

These characterize the GENERIC minimal N=9 valise. They are not BBBM-specific
(the BBBM twist is not built here); they are the code/gadget structure of the
N=9 shadow.

## Route B verdict (retired as an equation route)

Two independent agents (one from the paper's equations, one adversarial) found
Route B does not yield a new off-shell equation, for concrete reasons:

- The 7 supercharges that close only on-shell are never written in
  arXiv:0705.2002; the paper explicitly discards the antiselfdual tensor charge
  and calls the invariance under the residual 7 "accidental." There are no
  transformation rules and no non-closure tensor for them to extract; they would
  have to be reconstructed, so any computed answer is a function of solver
  choices, not of BBBM.
- "Non-closure of the 7 = equation of motion" is a category slip. The
  non-closure of on-shell charges is proportional to the fermionic equation of
  motion, which is an input (the BBBM Lagrangian), not a new output. The
  off-shell sector (the 9) has no equation of motion, which is the point of
  making it off-shell; the on-shell sector (the 7) has the ordinary equation of
  motion one started with.
- The 9-vs-7 split is a 10D SO(1,1)xSpin(7) twist phenomenon (needs the spatial
  derivatives, the field strength, and the nonabelian connection). A 1D valise is
  free, abelian, and closes all its charges by construction, so the worldline
  reduction trivializes exactly the structure Route B wanted to interrogate.

What was done instead (the honest, positive half). `src/bbbm_closure.rs`
(subcommand `bbbm-closure`) verifies exactly, in the reduced 9-theta superspace
(not the trivializing 1D valise), that the 9 = 1 + 8 supercharges close
off-shell: the flat covariant derivatives of Eq. (33), built as exact operators
on the 2^9 = 512-dimensional Grassmann module, satisfy Eq. (34) with every
anticommutator a pure translation (max_residual_terms = 0), and the component
delta_0^2 = d_+ + gauge(A_+) (Eq. 22-24) closes with zero equation-of-motion
residue. It also builds the exact Spin(7) antiselfdual projector (rank 7,
symmetric, idempotent) that the 7 tensor charges live in. This is BBBM's
9-off-shell closure metabolized into the codebase apparatus for the first time,
a translation, not a new equation.

The 7 antiselfdual tensor charges are reported (not fabricated) as
underspecified by arXiv:0705.2002: the paper drops them (nu^{ij} = 0, Eq. 17) and
prints no delta^-_{ij} transformation laws, so their explicit non-closure/EOM
functions are not extractable from the printed equations. This is asserted by the
test `seven_charges_reported_underspecified_not_fabricated`.

## Context

The graph RAG over the Gates corpus (295 papers, 166 full text) finds BBBM cited
in zero of them: the adinkra program has never metabolized this result. That is
why the route is attractive: a rigorous existing theorem, it drops straight into
the d=16 valise apparatus, and it is untouched.

Primary reference: L. Baulieu, N. Berkovits, G. Bossard, A. Martin,
"Ten-dimensional super-Yang-Mills with nine off-shell supersymmetries,"
arXiv:0705.2002, Phys. Lett. B658 (2008) 249.
