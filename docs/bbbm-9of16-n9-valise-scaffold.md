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
  non-closure of the unclosed 7. The non-closure functions are the
  off-shell-sector equation of motion, the equation-shaped object.

## Context

The graph RAG over the Gates corpus (295 papers, 166 full text) finds BBBM cited
in zero of them: the adinkra program has never metabolized this result. That is
why the route is attractive: a rigorous existing theorem, it drops straight into
the d=16 valise apparatus, and it is untouched.

Primary reference: L. Baulieu, N. Berkovits, G. Bossard, A. Martin,
"Ten-dimensional super-Yang-Mills with nine off-shell supersymmetries,"
arXiv:0705.2002, Phys. Lett. B658 (2008) 249.
