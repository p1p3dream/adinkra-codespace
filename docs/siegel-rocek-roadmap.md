# Siegel-Rocek auxiliary-field investigation roadmap

## Objective and standard of proof

The objective is to determine whether the conventional auxiliary-field counting
argument misses a finite, local off-shell realization of sixteen
supersymmetries, and, if so, to identify the precise assumption that fails.

Each stage has a stated acceptance criterion. Advancement requires an analytic
proof or independently checkable data establishing existence. Numerical
results may be used to select candidates.

Rust is the primary implementation language. Validated symbolic calculations belong in the
Rust library and test suite. Python may be used for exploratory searches,
independent symbolic checks, and figure generation, but a result does not become
a final validated result until it has a reproducible Rust implementation or an
independently checked result consumed by Rust.

The current result has the following scope:

> The tested `128|128` valise does not contain the physical fields under either
> the Spin(9) covariance condition or the direct four-dimensional compact
> subgroup condition. A different four-dimensional nonvalise, gauge, or
> auxiliary complex has not been
> excluded.

In particular, neither obstruction must be presented as a no-go theorem for all
finite 4D N=4 gauge or auxiliary complexes.

## Completed calculations

### 1. Minimal size and time-reduced target transformations

Dropping the paired-auxiliary-spinor premise changes the counting motivation
from

```text
16 + 32 n = 128 m
```

to

```text
16 + 16 q = 128 m.
```

The first candidate has `m=1`, `q=7`, hence `128|128`. This is only a size
motivation. The equality `112 = 7 x 16` does not prove that seven admissible
auxiliary spinors exist.

Nine explicit real symmetric SO(9) gamma matrices give the linkage matrices for
the one-dimensional reduction of 10D SYM. The bosonic Garden relation holds
identically, while the expected
on-shell fermionic remnant has rank seven for each diagonal charge pair.

**Result:** The matrices reproduce the required bosonic relation. They do not
define an off-shell supermultiplet.

### 2. Literal and mixed embeddings

Both self-dual length-16 chromotopologies and all 256 dashing classes of each
were tested.

* No literal nine-boson coordinate subblock exists.
* The standard no-leakage embedding has only the zero solution even with
  arbitrary real field mixing.

**Result:** None of the 512 tested minimal valise embeddings has a solution.

### 3. Projected one-dimensional linkage system

The quotient system introduced injections and projections

```text
J_B : R^9  -> R^128       P_B : R^128 -> R^9
J_F : R^16 -> R^128       P_F : R^128 -> R^16
```

with left-inverse identities and projected one-dimensional linkage equations.
For E8 x E8, a coordinate boson injection and rank-16 fermion projection
satisfy one set of linkage equations. Coordinate and uniform right inverses for
the fermion projection fail. With unrestricted real injection and projection
maps, a Banach fixed-point calculation establishes a real solution of all 2,304
linkage equations.

**Result:** The projected one-dimensional linkage system has a real solution.

### 4. Spin(9) equivariance

The induced Spin(9) action on the full fields was constructed symbolically. The
quadratic Casimirs are

```text
physical vector:     8
physical spinor:     9
full bosons:        18
full fermions:      18.
```

The mismatched scalar Casimirs force every Spin(9)-equivariant injection or
projection between the physical and full spaces to vanish.

**Result:** No Spin(9)-equivariant linear retraction exists between the physical
representation and the tested valise representation.

### 5. Time node raising and local spatial derivatives

The Casimir obstruction persists coefficient by coefficient over `R[D]` and
`R[D,D^-1]`. Over `R[D,p_1,...,p_9]`, evaluation at zero spatial momentum
reduces a local polynomial retraction to the excluded time-derivative
retraction.

Gauge shifts `delta A_i = p_i epsilon` and positive-spatial-degree Bianchi
corrections also vanish at `p=0`, so they cannot satisfy the retraction identity on
the same module.

**Result:** Ordinary node raising and the stated gauge and Bianchi corrections
do not satisfy the required identities on the tested potential representation.

**Not excluded:** Added representations, algebraic maps that remain nonzero at
zero momentum,
rectangular complexes, or nonlocal removal of the spatial zero mode.

### 6. Spin(9) identity of the valise

Joint Cartan characters prove

```text
bosons:   Sym^2_0(9) + Lambda^3(9) = 44 + 84
fermions: (9 tensor 16) - 16 = 128.
```

The fermions form the gamma-traceless vector-spinor. The projected
one-dimensional solution was therefore found inside a representation with
`44+84|128` Spin(9)
content, not inside a module containing the SYM `9|16` as Spin(9) summands.

**Result:** The decomposition is verified and accounts for the absence of the
required equivariant maps.

### 7. Exterior gauge complex and first enlargement

The abelian exterior complex

```text
Lambda^0 -> Lambda^1 -> Lambda^2 -> Lambda^3
```

has `d^2=0` and generic ranks `1,9,36`. Its spatial Spin(9) content is

```text
gauge parameter:  1
potential:        1 + 9
field strength:   9 + 36
Bianchi:          36 + 84.
```

At total momentum zero, every derivative-built differential vanishes. Thus a
chain-homotopy identity `P J - I = d H + H d` reduces to the ordinary linear retraction
already excluded by the Casimir obstruction.

The next direct-sum size containing the tested representation is `256|256`.
Direct sums of the representation and its parity reverse contain only `44`,
`84`, and `128`; none contains
a bosonic `9` together with a fermionic `16`.

**Result:** The derivative-only exterior complex and the three tested
`256|256` direct sums do not contain the required physical representation.

**Scope:** This does not classify every possible `256|256` nonvalise or
gauge-extended representation.

### 8. Direct four-dimensional branching of the tested valise

The validated Spin(9) field action was restricted to

```text
Spin(3) x Spin(6) ~= SU(2) x SU(4)_R.
```

This is the compact symmetry relevant to the spatial vector and six scalars of the
4D N=4 vector multiplet:

```text
9 -> (3,1) + (1,6).
```

The restricted generators are skew, the two subgroup actions commute,
and their quadratic Casimirs commute. Restricted characters and joint
Casimir eigenspaces give the bosonic branching

```text
44 -> (5,1) + (3,6) + (1,20') + (1,1)
84 -> (1,1) + (3,6) + (3,15) + (1,20).
```

After complexification, the fermions branch as

```text
128 -> (4,4) + (4,4bar)
     + (2,4) + (2,4bar)
     + (2,20) + (2,20bar).
```

The target multiplicities are

```text
required spatial vector (3,1):      multiplet dimension 0, required 3
required six scalars (1,6):         multiplet dimension 0, required 6
real gaugino 16:
  (2,4) + (2,4bar):                 multiplet dimension 16, required 16.
```

Thus the real gaugino has multiplicity one, but neither required physical
bosonic representation occurs. Compact-group complete reducibility forbids a
`Spin(3) x Spin(6)`-equivariant linear retraction onto the full `9|16` physical
representation.

**Result:** The tested `128|128` valise has no ordinary direct
four-dimensional linear retraction onto the physical field representation.

**Scope:** This does not exclude a changed nonvalise, gauge-extended, or
algebraic auxiliary complex whose zero-momentum maps alter the representation.

## Current calculation

### 9. Direct 4D algebraic gauge and auxiliary complex

Construct the zero-momentum complex first, before adding derivative linkages.
Candidate terms may include gauge parameters, `A_0`, compensators,
Stueckelberg-type fields, Bianchi multipliers, and algebraic auxiliary bosons or
fermions. Every proposed algebraic differential must respect
`Spin(3) x Spin(6)` and engineering parity.

**Required result:** The cohomology contains the physical 4D N=4 vector
supermultiplet, and the equivariant chain-retraction or homotopy identity
remains valid at zero
momentum.

**Rejection condition:** Cohomology has the wrong physical representations, or every
allowed algebraic map leaves the zero-momentum retraction identities
unsatisfied.

## Subsequent calculations

### 10. Missing representation inventory and minimum size

If the first complex fails, compute the missing irreducible
representations rather than guessing another adinkra size. Determine the
minimum bosonic and fermionic dimensions, whether equal raw dimensions are
required, and whether the structure must be rectangular across engineering
levels.

**Required result:** A finite representation inventory satisfies covariance,
cohomology, grading, and off-shell boson/fermion balance constraints.

**Rejection condition:** The constraints force an infinite tower, nonlocality, or an
incompatible dimension or grading assignment.

### 11. Smallest nonvalise or rectangular linkage system

Build the smallest candidate implied by the inventory. Assign engineering
heights and solve the one-dimensional Garden and linkage equations together with the
algebraic complex compatibility conditions. The existing solution of the
projected one-dimensional equations may be used for comparison, but it is not
a required subrepresentation.

**Required result:** One-dimensional closure and chain compatibility, with the
physical multiplet in cohomology and no equations of motion used.

**Rejection condition:** The polynomial system is inconsistent, or all
solutions collapse the physical cohomology or introduce forbidden propagating
content.

### 12. Full 4D spatial closure

Add the three spatial derivative linkages and verify every supercharge
anticommutator. Each remainder must be classified explicitly as a translation,
gauge transformation, a term in the image of the complex differential, or an
identity.

**Required result:** All sixteen supercharges close at finite differential
order without equations of motion or gauge fixing.

**Rejection condition:** Any remainder requires the Maxwell equation, gaugino equation,
an auxiliary equation of motion, a momentum restriction, or deletion of a zero
mode.

### 13. Lorentz and R-symmetry covariance

Construct rotations and boosts for `Spin(1,3)`, together with the required
`SU(4)_R` action, on fields, supercharges, gauge parameters, auxiliaries, and
differentials.

**Required result:** All Lie-algebra commutators and covariance identities hold
and are compatible with the closure maps.

**Rejection condition:** Closure works only in a preferred frame, violates the required
R-symmetry, or cannot support consistent boosts.

### 14. Off-shell, finite, and local verification

Verify every simplification used in the closure proof. The construction must
have finitely many component fields and finite-order local transformations,
with no inverse spatial derivatives, hidden boundary conditions, or infinite
auxiliary tower.

**Required result:** Every closure remainder belongs to the admitted algebraic or
gauge complex, and no dynamical equation or nonlocal operator is used.

**Rejection condition:** Closure is on shell, nonlocal, zero-momentum dependent, or requires
infinitely many auxiliaries.

### 15. Invariant local action

Construct a quadratic abelian action and test invariance under all sixteen
supercharges. Check gauge invariance, Lorentz covariance, kinetic signs,
algebraicity of auxiliaries, and recovery of ordinary N=4 SYM after auxiliary
elimination.

**Required result:** The local action is invariant up to a total derivative,
has the correct physical spectrum, and introduces no propagating ghosts.

**Rejection condition:** No nondegenerate invariant pairing exists, an auxiliary becomes
an unwanted propagating mode, or invariance requires equations of motion.

### 16. Counting-premise assessment

Only after closure and the action conditions should the result be compared directly
with the Siegel-Rocek assumptions. Candidate failure points include paired
auxiliary-spinor counting, treatment of pure-gauge fields, reducible or
constrained auxiliaries, rectangular complexes, and assumptions about manifest
covariance.

**Required result:** One published premise is isolated and the construction gives a
concrete counterexample to the conclusion that depends on it.

**Rejection condition:** The candidate relaxes locality, finiteness, off-shell closure,
or Lorentz covariance and therefore does not address the counting argument.

### 17. Nonabelian extension

Promote derivatives to covariant derivatives, add commutator terms, and solve
closure and action invariance order by order in the coupling. The free abelian
result addresses the representation question but is not the complete
Yang-Mills result.

**Required result:** Nonabelian closure and an invariant local action without
new equations of motion or an uncontrolled auxiliary tower.

**Rejection condition:** The deformation is obstructed at any finite order or forces
the construction on shell.

## Decision tree

```text
direct four-dimensional branching of tested 128|128 valise
  -> physical subrepresentation and quotient absent
  -> build direct four-dimensional algebraic gauge and auxiliary complex at zero momentum

validated algebraic complex at zero momentum
  -> one-dimensional linkage equations
  -> full spatial closure
  -> Spin(1,3) x SU(4)_R covariance
  -> finite/local/off-shell verification
  -> invariant abelian action
  -> identify the specific counting premise
  -> nonabelian deformation
```

## Separate cases

Inverse spatial derivatives, removal of the zero mode, or boundary-condition
dependent constructions may be useful comparisons, but they are not the
primary research direction. They weaken locality or change the original finite auxiliary-field
problem and should be labeled separately if pursued.

Extension to ten dimensions remains an optional target. A direct four-dimensional
solution need not satisfy the Spin(9) condition unless ten-dimensional origin is
made part of the claim.
