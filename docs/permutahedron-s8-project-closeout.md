# S8 permutahedron and higher-dimensional parentage closeout

## Final question

Can the S8 adjacent-transposition Cayley graph, its R8 coset hypergraph, or its
one-dimensional signed Garden data canonically recover the physically relevant
higher-dimensional parent structure?

## Final answer

No. The calculations establish the R8 structure exactly, but also prove that
the graph and worldline data retain a symmetry that prevents a canonical
physical selection.

## Closed results

### Spectral structure

- The S4 targeted calculation recovers the six V4 cosets exactly with
  `ARI = 1.0`.
- Naive S8 extremal-eigenvector clustering gives `ARI = 0.000553` because the
  R8 information lies in interior spectral components.
- Every one of the 30 discovered R8 conjugates defines an equitable partition
  of S8 into 5,040 octets.
- Exact left multiplication maps the standard partition bijectively to every
  conjugate partition while preserving all adjacent-transposition edges.
- The 30 quotient graphs therefore form one isomorphism class.
- Any polynomial in the adjacency or Laplacian operator, including Chebyshev
  filtering and polynomial-filtered Lanczos, preserves this ambiguity.

Targeting the interior spectrum may recover an R8-type partition, but cannot
canonically choose the standard member of the 30-partition orbit. Label-seeded
coset priming and quotient lifting were not retained as discovery evidence
because they assume the target partition they are intended to recover.

### Hypergraph structure

- The exact unsigned compatibility equations discover 30 order-eight identity
  subgroups and 151,200 distinct octets.
- Every permutation belongs to 30 octets, one from each subgroup family.
- Every unsigned octet is Garden-signable with rank 45 and nullity 19.
- A minimum four-for-four mixed trade disproves uniqueness of the subgroup
  exact covers.

### Signed worldline structure

- All valid signings collapse to one unlabeled-color signed graph class.
- The complete unrestricted recursion checks 221,184 candidates and finds 64
  closers on 32 supports, with two realizations per support.
- Embedded Maxwell data reproduce the four-color `chi0` distinction but do not
  identify S8 parentage.
- Normalizer orbit correlates with source categories only in the fixed
  published basis.
- Common node relabeling reaches all 5,040 family-zero supports and all 20
  normalizer orbits, proving the orbit label is not intrinsic.

### Higher-dimensional controls

- CT and CV pass their complete sourced four-dimensional component closure,
  gauge-residue, and worldline-reduction gates.
- Their spatial and gauge fingerprints differ even though their worldline
  representations lie in the same unsigned family and signed graph class.
- O is a valid one-dimensional diadem construction without a claimed
  four-dimensional component parent in the audited sources.
- VM1, VM2, and VM3 are sourced as mathematical Garden solution sectors, not
  as independent four-dimensional component fixtures.
- No independent physical holdout was found in the audited corpus.

## Exact spectral no-go certificate

The final audit verifies:

| Quantity | Result |
|---|---:|
| S8 vertices | 40,320 |
| undirected Cayley edges | 141,120 |
| R8 conjugate families | 30 |
| cosets per family | 5,040 |
| equitability violations | 0 |
| directed adjacency intertwining checks | 8,467,200 |
| vertex-partition transport checks | 1,209,600 |
| partition transports failing | 0 |
| quotient isomorphism classes | 1 |

The automorphism certificate is stronger than another numerical clustering
run. It proves that graph-only spectral processing cannot break the relevant
30-fold ambiguity, regardless of which adjacency-polynomial filter is used.

## Machine-readable artifacts

- `results/permutahedron_s8_spectral_identifiability.json`
- `results/permutahedron_s4_spectral_probe_closeout.txt`
- `results/permutahedron_s8_spectral_probe_closeout.txt`
- `results/permutahedron_s8_unrestricted_recursion.json`
- `results/permutahedron_s8_orbit_leakage.json`
- `results/permutahedron_s8_source_fixture_audit.json`
- `results/permutahedron_hypergraph_recursion_maxwell_bridge.json`
- `results/permutahedron_hypergraph_higher_dimensional_gate_validation.json`
- `results/permutahedron_hypergraph_signed_equivalence_validation.json`
- `results/permutahedron_hypergraph_resolution_validation.json`

## Reproduction

```sh
cargo run --release -- perm-spectral-probe s4
cargo run --release -- perm-spectral-probe s8
cargo run --release -- perm-s8-spectral-identifiability-build
cargo run --release -- perm-s8-unrestricted-recursion-verify
cargo run --release -- perm-s8-orbit-leakage-verify
cargo run --release -- perm-s8-source-fixture-audit-verify
cargo run --release -- perm-hypergraph-higher-dimensional-verify
```

## Publication-level conclusion

The S8 worldline system contains exact R8 coset organization, but the relevant
structure occurs as a symmetry orbit rather than a canonical physical label.
Unsigned support, Garden signability, signed graph equivalence, embedded
four-color Maxwell data, normalizer orbit, and graph-only spectral filters all
fail as intrinsic higher-dimensional parentage selectors. The sourced CT and
CV controls show directly that distinguishing spatial and gauge information is
discarded by the worldline quotient.

Further physical discrimination requires genuinely new external input:
complete Lorentz representations, spatial linkage, gauge and Bianchi data,
and an independently specified field-to-node reduction map. Until such a
fixture is available, additional clustering or learned selection would fit the
symmetry rather than resolve it.

## Post-closeout central-charge result

The subsequent vector-tensor program does not reopen graph-only selection. It
strengthens the boundary.

- The published `TV` system has eight exact one-central-charge sign branches.
- All 25 printed one-charge branches form one enriched signed-node/color class.
- That class transports to every one of the 151,200 R8 supports.
- The ordinary Garden class also transports to every one of those supports.

Thus each unsigned support admits both central rank zero and central rank one.
The higher-dimensional information exists in the signed central operator, but
it is erased by projection to the unsigned graph. See
`docs/vector-tensor-central-charge-completion.md`.

## Scalar-tensor holdout audit

The proposed scalar-tensor holdout from arXiv:2412.16527 was taken through an
exact rigid-tangent preflight. A regular fully supersymmetric expansion exists
only on the nonzero scalar patch with `X=0`. In the canonical frame
`xi_i=(1,0)`, the composite connection removes the longitudinal phase and
combines the radial scalar derivative with the dual tensor strength.

After the central-U(1) and tensor gauge quotients, the tangent is 8+8 with the
field roles

```text
xi_2 complex -> CT chiral scalars
X complex -> CT chiral auxiliaries
Re(xi_1) -> CT tensor scalar
B_mu_nu -> CT tensor potential
psi, theta -> crossed CT fermions
```

This is the CT coupling pattern, not an independent S8 holdout. The nonlinear
conformal multiplet is still distinct away from the tangent, but that
information is erased by a regular linear worldline reduction. See
`docs/scalar-tensor-holdout-boundary.md` and
`results/scalar_tensor_tangent.json`.

The result strengthens the project boundary: new 4D parent information must be
carried by Lorentz type, spatial linkage, gauge residues, nonlinear composite
connections, or genuine central extensions. Another ordinary 8x8 signed
worldline matrix is not by itself an independent higher-dimensional control.
