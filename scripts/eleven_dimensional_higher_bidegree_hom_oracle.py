#!/usr/bin/env python3
"""Exact B5 character oracle for the Hhat (2,1) and (0,2) slices."""

from __future__ import annotations

import itertools
import json
import hashlib
from collections import Counter
from pathlib import Path


Weight = tuple[int, int, int, int, int]
Character = Counter[Weight]
RHO: Weight = (9, 7, 5, 3, 1)  # twice the B5 Weyl vector
ROOT = Path(__file__).resolve().parents[1]
DG4_PROJECTOR_ARTIFACT = ROOT / "results/adynkra_11d_dg4_casimir_projectors.json"
D02_00001_GENERATOR_ARTIFACT = ROOT / "results/adynkra_11d_d02_00001_source_generator.json"


def add(left: Weight, right: Weight) -> Weight:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def convolve(left: Character, right: Character) -> Character:
    output: Character = Counter()
    for left_weight, left_multiplicity in left.items():
        for right_weight, right_multiplicity in right.items():
            output[add(left_weight, right_weight)] += (
                left_multiplicity * right_multiplicity
            )
    return +output


def exterior_character(basis_weights: list[Weight], degree: int) -> Character:
    return Counter(
        tuple(map(sum, zip(*choice))) if choice else (0, 0, 0, 0, 0)
        for choice in itertools.combinations(basis_weights, degree)
    )


def symmetric_character(basis_weights: list[Weight], degree: int) -> Character:
    return Counter(
        tuple(map(sum, zip(*choice))) if choice else (0, 0, 0, 0, 0)
        for choice in itertools.combinations_with_replacement(basis_weights, degree)
    )


def highest_weight(label: str) -> Weight:
    digits = [int(value) for value in label]
    assert len(digits) == 5
    return tuple(
        2 * sum(digits[index:4]) + digits[4] for index in range(5)
    )  # type: ignore[return-value]


def dominant_label(weight: Weight) -> str | None:
    if any(weight[index] < weight[index + 1] for index in range(4)) or weight[4] < 0:
        return None
    if any((weight[index] - weight[index + 1]) % 2 for index in range(4)):
        return None
    labels = [
        (weight[index] - weight[index + 1]) // 2 for index in range(4)
    ] + [weight[4]]
    return "".join(str(value) for value in labels)


def tensor_spinor_labels(label: str) -> list[str]:
    source = highest_weight(label)
    output = {
        target
        for spinor_weight in SPINOR_BASIS
        if (target := dominant_label(add(source, spinor_weight))) is not None
    }
    return sorted(output)


def permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(5)
        for right in range(left + 1, 5)
    )
    return -1 if inversions % 2 else 1


SIGNED_WEYL_RHO: list[tuple[Weight, int]] = []
for permutation in itertools.permutations(range(5)):
    permuted = tuple(RHO[permutation[index]] for index in range(5))
    base_sign = permutation_sign(permutation)
    for signs in itertools.product((1, -1), repeat=5):
        value = tuple(signs[index] * permuted[index] for index in range(5))
        sign = base_sign * (-1 if signs.count(-1) % 2 else 1)
        SIGNED_WEYL_RHO.append((value, sign))


def irreducible_multiplicity(character: Character, label: str) -> int:
    """Coefficient of the B5 Weyl character by the alternant formula."""

    shifted = add(highest_weight(label), RHO)
    value = 0
    for weyl_rho, sign in SIGNED_WEYL_RHO:
        probe = tuple(shifted[index] - weyl_rho[index] for index in range(5))
        value += sign * character.get(probe, 0)
    assert value >= 0
    return value


SPINOR_BASIS: list[Weight] = list(itertools.product((1, -1), repeat=5))
VECTOR_BASIS: list[Weight] = [(0, 0, 0, 0, 0)]
for axis in range(5):
    for sign in (-2, 2):
        weight = [0] * 5
        weight[axis] = sign
        VECTOR_BASIS.append(tuple(weight))  # type: ignore[arg-type]

S: Character = Counter(SPINOR_BASIS)
V: Character = Counter(VECTOR_BASIS)
T: Character = convolve(V, S)
T.subtract(S)
T = +T  # V tensor S minus its gamma-trace S

E2S = exterior_character(SPINOR_BASIS, 2)
SYM2V = symmetric_character(VECTOR_BASIS, 2)
A3 = exterior_character(VECTOR_BASIS, 3)
G4 = exterior_character(VECTOR_BASIS, 4)

S_T_DECOMPOSITION = [
    "00002",
    "00010",
    "00100",
    "01000",
    "10000",
    "10002",
    "10010",
    "10100",
    "11000",
    "20000",
]
D_A3_TARGETS = ["00001", "00101", "01001", "10001"]
D_G4_TARGETS = ["00001", "00011", "00101", "01001", "10001"]


def target_multiplicities(character: Character, labels: list[str]) -> dict[str, int]:
    return {label: irreducible_multiplicity(character, label) for label in labels}


def main() -> None:
    projector_bytes = DG4_PROJECTOR_ARTIFACT.read_bytes()
    projector = json.loads(projector_bytes)
    assert projector["passed_canary"] is True
    assert projector["exhaustive_projector_ranks_constructed"] is True
    assert projector["exhaustive_projector_ranks"] == [32, 5280, 3520, 1408, 320]
    d02_00001 = json.loads(D02_00001_GENERATOR_ARTIFACT.read_bytes())
    assert d02_00001["passed"] is True
    assert d02_00001["generator_operator_rank"] == 32
    assert d02_00001["target_casimir_eigen_residual_entries"] == 0
    assert d02_00001["target_projector_residual_entries"] == 0
    known = target_multiplicities(convolve(S, T), S_T_DECOMPOSITION)
    assert set(known.values()) == {1}
    assert tensor_spinor_labels("00100") == D_A3_TARGETS
    assert tensor_spinor_labels("00010") == D_G4_TARGETS
    assert set(target_multiplicities(convolve(S, A3), D_A3_TARGETS).values()) == {1}
    assert set(target_multiplicities(convolve(S, G4), D_G4_TARGETS).values()) == {1}

    d21 = convolve(convolve(E2S, V), T)
    d02 = convolve(SYM2V, T)
    d21_d_a3 = target_multiplicities(d21, D_A3_TARGETS)
    d21_d_g4 = target_multiplicities(d21, D_G4_TARGETS)
    d02_d_a3 = target_multiplicities(d02, D_A3_TARGETS)
    d02_d_g4 = target_multiplicities(d02, D_G4_TARGETS)

    # A direct bosonic A3/G4 target has trivial Spin-center character. Hhat
    # and every even-D jet have the opposite character, so these must vanish.
    direct = {
        "d2_p1_to_a3": irreducible_multiplicity(d21, "00100"),
        "d2_p1_to_g4": irreducible_multiplicity(d21, "00010"),
        "d0_p2_to_a3": irreducible_multiplicity(d02, "00100"),
        "d0_p2_to_g4": irreducible_multiplicity(d02, "00010"),
    }
    assert set(direct.values()) == {0}

    # Factor the d2,p1 domain as (S outer tensor V) acting on each unique
    # form projection in S inner tensor Hhat. This enumerates every formal
    # compensator pullback through Psi_[p].
    form_pullbacks: dict[str, dict[str, int]] = {}
    for degree in range(6):
        form = exterior_character(VECTOR_BASIS, degree)
        downstream = convolve(convolve(S, V), form)
        d_a3 = target_multiplicities(downstream, D_A3_TARGETS)
        d_g4 = target_multiplicities(downstream, D_G4_TARGETS)
        form_pullbacks[str(degree)] = {
            "d_a3": sum(d_a3.values()),
            "d_g4": sum(d_g4.values()),
            "pullback_multiplicity_in_S_tensor_Hhat": irreducible_multiplicity(
                convolve(S, T),
                ["00000", "10000", "01000", "00100", "00010", "00002"][
                    degree
                ],
            ),
        }

    all_form_g4 = sum(form_pullbacks[str(p)]["d_g4"] for p in range(1, 6))
    eq40_form_g4 = sum(form_pullbacks[str(p)]["d_g4"] for p in (1, 3, 4, 5))
    total_d21_d_g4 = sum(d21_d_g4.values())
    assert all_form_g4 == 51
    assert eq40_form_g4 == 43
    assert total_d21_d_g4 == 52

    report = {
        "schema_version": "adynkra-11d-higher-bidegree-hom-oracle-v3",
        "weight_convention": "twice-orthonormal B5 weights; exact integer character convolution and Weyl alternant extraction",
        "validation": {
            "S_tensor_Hhat_known_irreps": known,
            "S_tensor_Hhat_dimension": sum(convolve(S, T).values()),
            "exterior2_spinor_dimension": sum(E2S.values()),
            "symmetric2_vector_dimension": sum(SYM2V.values()),
            "D_A3_target_dimension": sum(convolve(S, A3).values()),
            "D_G4_target_dimension": sum(convolve(S, G4).values()),
        },
        "direct_a3_g4": direct,
        "descendant_targets": {
            "D_A3_irreps": D_A3_TARGETS,
            "D_G4_irreps": D_G4_TARGETS,
            "d2_p1_D_A3_by_irrep": d21_d_a3,
            "d2_p1_D_A3_total": sum(d21_d_a3.values()),
            "d2_p1_D_G4_by_irrep": d21_d_g4,
            "d2_p1_D_G4_total": total_d21_d_g4,
            "d0_p2_D_A3_by_irrep": d02_d_a3,
            "d0_p2_D_A3_total": sum(d02_d_a3.values()),
            "d0_p2_D_G4_by_irrep": d02_d_g4,
            "d0_p2_D_G4_total": sum(d02_d_g4.values()),
        },
        "source_graph_pullbacks_d2_p1": {
            "through_form_degree": form_pullbacks,
            "ordered_D_form_factorization_columns_p1_through_p5_before_outer_inner_antisymmetrization": all_form_g4,
            "ordered_D_eq40_p1_p3_p4_p5_columns_before_outer_inner_antisymmetrization": eq40_form_g4,
            "ordered_D_p2_local_lorentz_columns_before_outer_inner_antisymmetrization": form_pullbacks["2"]["d_g4"],
            "antisymmetrized_form_factorization_rank": None,
            "rank_blocker": "the Lambda2 S projector recouples the two spinors across the form and hook intermediate bases; the ordered coefficient count is not the rank after PBW antisymmetrization",
        },
        "source_graph_pullbacks_d0_p2": {
            "bosonic_form_pullback_count": 0,
            "reason": "an even-D jet of spinorial Hhat cannot map to a bosonic form compensator",
            "spinor_intermediate_multiplicity": d02_d_g4["00001"],
        },
        "fixed_witness_support": {
            "row": "source0/output0/exterior-mask-0x00010001/p_1",
            "d2_p1_raw_Hom_can_populate": True,
            "d2_p1_evidence": "the corrected teleparallel operator is itself an equivariant member of this 52-dimensional Hom space and has exact coefficient 1/1280 on the row",
            "d0_p2_can_populate": False,
            "d0_p2_reason": "every d0,p2 map has total momentum degree two, while the pinned row has total momentum degree one",
            "canonical_target_irrep_for_witness": None,
            "target_irrep_status": "exact exhaustive Cartesian Casimir projectors exist for all five D G4 irreps; a single Cartesian row is not itself assigned one canonical irrep",
        },
        "dg4_cartesian_projector_dependency": {
            "path": str(DG4_PROJECTOR_ARTIFACT.relative_to(ROOT)),
            "artifact_sha256": hashlib.sha256(projector_bytes).hexdigest(),
            "module_source_sha256": projector["module_source_sha256"],
            "cartesian_basis_sha256": projector["cartesian_basis_sha256"],
            "casimir_operator_sha256": projector["casimir_operator_sha256"],
            "projector_polynomials_sha256": projector["projector_polynomials_sha256"],
            "exhaustive_projector_ranks": projector["exhaustive_projector_ranks"],
        },
        "d0_p2_00001_generator_dependency": {
            "path": str(D02_00001_GENERATOR_ARTIFACT.relative_to(ROOT)),
            "semantic_stream_sha256": d02_00001["stream_sha256"],
            "source_basis_sha256": d02_00001["source_basis_sha256"],
            "emitted_nonzero_rows": d02_00001["emitted_nonzero_rows"],
            "generator_operator_rank": d02_00001["generator_operator_rank"],
            "target_casimir_eigen_residual_entries": d02_00001["target_casimir_eigen_residual_entries"],
            "target_projector_residual_entries": d02_00001["target_projector_residual_entries"],
            "artifact_hash_intentionally_omitted": "the generator report already binds this Hom inventory artifact; copying its artifact hash here would create a circular digest dependency",
        },
        "boundary": "Raw descendant Hom counts do not impose superspace integrability, Bianchi descent, source gauge quotient, Eq. 40 coefficients, or engineering-degree completeness.",
        "passed": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
