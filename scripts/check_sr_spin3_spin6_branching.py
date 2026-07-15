#!/usr/bin/env python3
"""Exact Spin(3) x Spin(6) branching audit of the fixed 128|128 module.

The subgroup acts on the Spin(9) vector as 9 = (3,1) + (1,6).  This
script constructs its induced full-field generators exactly, verifies the
restricted Cartan character, and computes exact joint quadratic-Casimir
eigenspace dimensions.  The latter directly test whether the physical 4D
N=4 bosons and gauginos occur as compact-group subrepresentations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import sympy as sp

from check_sr_so9_equivariance import dense_signed_perm
from check_sr_spin9_decomposition import (
    actual_charpoly,
    coefficient_digest,
    expected_charpoly,
)


SubgroupWeight = tuple[int, int, int, int]
So6Weight = tuple[int, int, int]


def add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right))


def counter_dimension(weights: Counter[tuple[int, ...]]) -> int:
    return sum(weights.values())


def spin3_irrep(highest_weight: int) -> Counter[tuple[int]]:
    return Counter({(weight,): 1 for weight in range(-highest_weight, highest_weight + 1, 4)})


def so6_vector() -> Counter[So6Weight]:
    weights: Counter[So6Weight] = Counter()
    for axis in range(3):
        for sign in (-1, 1):
            weight = [0, 0, 0]
            weight[axis] = 4 * sign
            weights[tuple(weight)] += 1
    return weights


def so6_spinor(chirality: int) -> Counter[So6Weight]:
    return Counter(
        {
            tuple(2 * sign for sign in signs): 1
            for signs in product((-1, 1), repeat=3)
            if signs[0] * signs[1] * signs[2] == chirality
        }
    )


def tensor_weights(
    left: Counter[tuple[int, ...]], right: Counter[tuple[int, ...]]
) -> Counter[tuple[int, ...]]:
    output: Counter[tuple[int, ...]] = Counter()
    for left_weight, left_multiplicity in left.items():
        for right_weight, right_multiplicity in right.items():
            output[add(left_weight, right_weight)] += (
                left_multiplicity * right_multiplicity
            )
    return output


def exterior_weights(
    source: Counter[tuple[int, ...]], degree: int
) -> Counter[tuple[int, ...]]:
    expanded = [weight for weight, multiplicity in source.items() for _ in range(multiplicity)]
    output: Counter[tuple[int, ...]] = Counter()
    zero = tuple(0 for _ in expanded[0])
    for indices in combinations(range(len(expanded)), degree):
        weight = zero
        for index in indices:
            weight = add(weight, expanded[index])
        output[weight] += 1
    return output


def symmetric_square_traceless(
    source: Counter[tuple[int, ...]],
) -> Counter[tuple[int, ...]]:
    expanded = [weight for weight, multiplicity in source.items() for _ in range(multiplicity)]
    output: Counter[tuple[int, ...]] = Counter()
    for left in range(len(expanded)):
        for right in range(left, len(expanded)):
            output[add(expanded[left], expanded[right])] += 1
    output[tuple(0 for _ in expanded[0])] -= 1
    return output


def product_group_weights(
    spin3: Counter[tuple[int]], so6: Counter[So6Weight]
) -> Counter[SubgroupWeight]:
    output: Counter[SubgroupWeight] = Counter()
    for (spin3_weight,), spin3_multiplicity in spin3.items():
        for so6_weight, so6_multiplicity in so6.items():
            output[(spin3_weight, *so6_weight)] += (
                spin3_multiplicity * so6_multiplicity
            )
    return output


def expected_branches() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    one3 = spin3_irrep(0)
    two3 = spin3_irrep(2)
    three3 = spin3_irrep(4)
    four3 = spin3_irrep(6)
    five3 = spin3_irrep(8)
    one6: Counter[So6Weight] = Counter({(0, 0, 0): 1})
    six6 = so6_vector()
    four6 = so6_spinor(1)
    fourbar6 = so6_spinor(-1)
    fifteen6 = exterior_weights(six6, 2)
    twenty_prime6 = symmetric_square_traceless(six6)
    twenty_real6 = exterior_weights(six6, 3)
    twenty6 = tensor_weights(six6, four6) - fourbar6
    twentybar6 = tensor_weights(six6, fourbar6) - four6

    assert counter_dimension(fifteen6) == 15
    assert counter_dimension(twenty_prime6) == 20
    assert counter_dimension(twenty_real6) == 20
    assert counter_dimension(twenty6) == 20
    assert counter_dimension(twentybar6) == 20

    bosons = [
        {"source": "44", "irrep": "(5,1)", "weights": product_group_weights(five3, one6)},
        {"source": "44", "irrep": "(3,6)", "weights": product_group_weights(three3, six6)},
        {"source": "44", "irrep": "(1,20')", "weights": product_group_weights(one3, twenty_prime6)},
        {"source": "44", "irrep": "(1,1)", "weights": product_group_weights(one3, one6)},
        {"source": "84", "irrep": "(1,1)", "weights": product_group_weights(one3, one6)},
        {"source": "84", "irrep": "(3,6)", "weights": product_group_weights(three3, six6)},
        {"source": "84", "irrep": "(3,15)", "weights": product_group_weights(three3, fifteen6)},
        {"source": "84", "irrep": "(1,20)", "weights": product_group_weights(one3, twenty_real6)},
    ]
    fermions = [
        {"source": "128", "irrep": "(4,4)", "weights": product_group_weights(four3, four6)},
        {"source": "128", "irrep": "(4,4bar)", "weights": product_group_weights(four3, fourbar6)},
        {"source": "128", "irrep": "(2,4)", "weights": product_group_weights(two3, four6)},
        {"source": "128", "irrep": "(2,4bar)", "weights": product_group_weights(two3, fourbar6)},
        {"source": "128", "irrep": "(2,20)", "weights": product_group_weights(two3, twenty6)},
        {"source": "128", "irrep": "(2,20bar)", "weights": product_group_weights(two3, twentybar6)},
    ]
    return bosons, fermions


def combine_branch_weights(branches: list[dict[str, object]]) -> Counter[SubgroupWeight]:
    output: Counter[SubgroupWeight] = Counter()
    for branch in branches:
        output += branch["weights"]  # type: ignore[operator]
    return output


def construct_generators(
    input_path: Path,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    report = json.loads(input_path.read_text())
    topology = next(
        item for item in report["topologies"] if item["auxiliary_projection"].get("witness")
    )
    data = topology["auxiliary_projection"]["witness"]["joint_section_data"]
    full_l = [
        dense_signed_perm(perm, sign)
        for perm, sign in zip(data["l_perm"], data["l_sign"])
    ]
    gamma = [
        dense_signed_perm(perm, sign)
        for perm, sign in zip(data["gamma_perm"], data["gamma_sign"])
    ]
    bosons: dict[tuple[int, int], np.ndarray] = {}
    fermions: dict[tuple[int, int], np.ndarray] = {}
    for i in range(9):
        for j in range(i + 1, 9):
            spin2 = gamma[i] @ gamma[j]
            boson4 = np.zeros((128, 128), dtype=np.int64)
            fermion4 = np.zeros((128, 128), dtype=np.int64)
            for alpha in range(16):
                for beta in range(alpha + 1, 16):
                    coefficient = int(spin2[alpha, beta])
                    if coefficient:
                        boson4 += coefficient * (full_l[alpha] @ full_l[beta].T)
                        fermion4 += coefficient * (full_l[alpha].T @ full_l[beta])
            bosons[(i, j)] = boson4
            fermions[(i, j)] = fermion4
    return bosons, fermions


def exact_joint_nullity(
    first: np.ndarray, first_eigenvalue: int, second: np.ndarray, second_eigenvalue: int
) -> int:
    identity = np.eye(first.shape[0], dtype=np.int64)
    stacked = np.vstack(
        (first - first_eigenvalue * identity, second - second_eigenvalue * identity)
    )
    matrix = sp.Matrix(stacked.tolist())
    rank = sp.polys.matrices.DomainMatrix.from_Matrix(matrix).rank()
    return first.shape[0] - rank


def casimir_spectrum(
    spin3_casimir16: np.ndarray,
    spin6_casimir16: np.ndarray,
    expected: list[tuple[int, int, int, str]],
) -> list[dict[str, object]]:
    output = []
    for spin3_value, spin6_value, dimension, content in expected:
        actual_dimension = exact_joint_nullity(
            spin3_casimir16, spin3_value, spin6_casimir16, spin6_value
        )
        output.append(
            {
                "spin3_casimir_scaled_by_16": spin3_value,
                "spin6_casimir_scaled_by_16": spin6_value,
                "expected_dimension": dimension,
                "actual_exact_dimension": actual_dimension,
                "content": content,
                "matches": actual_dimension == dimension,
            }
        )
    return output


def branch_summary(branches: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "source": branch["source"],
            "irrep": branch["irrep"],
            "dimension": counter_dimension(branch["weights"]),  # type: ignore[arg-type]
        }
        for branch in branches
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/sr_hole_minimal.json")
    parser.add_argument(
        "--spin9-audit", default="results/sr_spin9_decomposition_audit.json"
    )
    parser.add_argument(
        "--output", default="results/sr_spin3_spin6_branching_audit.json"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    spin9_path = Path(args.spin9_audit)
    spin9_audit = json.loads(spin9_path.read_text())
    if not spin9_audit.get("gate_passed"):
        raise SystemExit("The prerequisite exact Spin(9) decomposition did not pass")

    boson_generators, fermion_generators = construct_generators(input_path)
    spin3_labels = list(combinations(range(3), 2))
    spin6_labels = list(combinations(range(3, 9), 2))
    cartan_labels = [(0, 1), (3, 4), (5, 6), (7, 8)]

    cross_subgroups_commute = all(
        np.array_equal(
            first[(i, j)] @ second[(k, l)], second[(k, l)] @ first[(i, j)]
        )
        for first in (boson_generators, fermion_generators)
        for second in (first,)
        for i, j in spin3_labels
        for k, l in spin6_labels
    )
    subgroup_generators_are_skew = all(
        np.array_equal(matrix + matrix.T, np.zeros_like(matrix))
        for generators in (boson_generators, fermion_generators)
        for label, matrix in generators.items()
        if label in spin3_labels or label in spin6_labels
    )

    boson_branches, fermion_branches = expected_branches()
    boson_weights = combine_branch_weights(boson_branches)
    fermion_weights = combine_branch_weights(fermion_branches)
    separating = (1, 32, 32**2, 32**3)

    character_audits = {}
    for name, generators, weights in (
        ("bosons", boson_generators, boson_weights),
        ("fermions", fermion_generators, fermion_weights),
    ):
        combined = sum(
            coefficient * generators[label]
            for coefficient, label in zip(separating, cartan_labels)
        )
        actual = actual_charpoly(combined)
        expected = expected_charpoly(weights, separating)
        character_audits[name] = {
            "dimension": counter_dimension(weights),
            "distinct_restricted_weights": len(weights),
            "separating_coefficients": list(separating),
            "exact_characteristic_polynomial_matches": actual == expected,
            "coefficient_sha256": coefficient_digest(actual),
        }

    def subgroup_casimirs(
        generators: dict[tuple[int, int], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            -sum(generators[label] @ generators[label] for label in spin3_labels),
            -sum(generators[label] @ generators[label] for label in spin6_labels),
        )

    boson_c3, boson_c6 = subgroup_casimirs(boson_generators)
    fermion_c3, fermion_c6 = subgroup_casimirs(fermion_generators)
    casimirs_commute = all(
        np.array_equal(c3 @ c6, c6 @ c3)
        for c3, c6 in ((boson_c3, boson_c6), (fermion_c3, fermion_c6))
    )

    boson_spectrum = casimir_spectrum(
        boson_c3,
        boson_c6,
        [
            (96, 0, 5, "(5,1)"),
            (32, 80, 36, "two copies of (3,6)"),
            (0, 192, 20, "(1,20')"),
            (0, 0, 2, "two copies of (1,1)"),
            (32, 128, 45, "(3,15)"),
            (0, 144, 20, "(1,20)"),
        ],
    )
    fermion_spectrum = casimir_spectrum(
        fermion_c3,
        fermion_c6,
        [
            (60, 60, 32, "(4,4) + (4,4bar)"),
            (12, 60, 16, "(2,4) + (2,4bar)"),
            (12, 156, 80, "(2,20) + (2,20bar)"),
        ],
    )

    physical_boson_tests = {
        "spatial_vector_(3,1)": {
            "joint_casimir_scaled_by_16": [32, 0],
            "exact_multiplet_space_dimension": exact_joint_nullity(
                boson_c3, 32, boson_c6, 0
            ),
            "required_dimension": 3,
        },
        "six_scalars_(1,6)": {
            "joint_casimir_scaled_by_16": [0, 80],
            "exact_multiplet_space_dimension": exact_joint_nullity(
                boson_c3, 0, boson_c6, 80
            ),
            "required_dimension": 6,
        },
    }
    physical_fermion_test = {
        "real_gaugino_16_complexification": "(2,4) + (2,4bar)",
        "joint_casimir_scaled_by_16": [12, 60],
        "exact_multiplet_space_dimension": exact_joint_nullity(
            fermion_c3, 12, fermion_c6, 60
        ),
        "required_dimension": 16,
    }
    bosonic_target_absent = all(
        test["exact_multiplet_space_dimension"] == 0
        for test in physical_boson_tests.values()
    )
    fermionic_target_present = (
        physical_fermion_test["exact_multiplet_space_dimension"]
        == physical_fermion_test["required_dimension"]
    )
    all_exact_checks_pass = bool(
        cross_subgroups_commute
        and subgroup_generators_are_skew
        and casimirs_commute
        and all(
            audit["exact_characteristic_polynomial_matches"]
            for audit in character_audits.values()
        )
        and all(item["matches"] for item in boson_spectrum + fermion_spectrum)
        and bosonic_target_absent
        and fermionic_target_present
    )

    output = {
        "source": args.input,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "prerequisite_spin9_audit": args.spin9_audit,
        "prerequisite_spin9_audit_sha256": hashlib.sha256(
            spin9_path.read_bytes()
        ).hexdigest(),
        "subgroup_embedding": "R^9 = R^3 + R^6, with indices 0..2 and 3..8",
        "normalization": "Field generators and quadratic Casimirs are scaled by 4 and 16 respectively",
        "cross_subgroups_commute_exactly": cross_subgroups_commute,
        "subgroup_generators_are_skew_exactly": subgroup_generators_are_skew,
        "subgroup_casimirs_commute_exactly": casimirs_commute,
        "branching": {
            "bosons": branch_summary(boson_branches),
            "fermions_after_complexification": branch_summary(fermion_branches),
        },
        "restricted_cartan_character_audit": character_audits,
        "joint_casimir_eigenspaces": {
            "bosons": boson_spectrum,
            "fermions": fermion_spectrum,
        },
        "direct_4d_target_test": {
            "physical_bosons": physical_boson_tests,
            "physical_fermions": physical_fermion_test,
            "physical_bosonic_representation_absent": bosonic_target_absent,
            "physical_gaugino_representation_present_once": fermionic_target_present,
            "strict_compact_group_retract_possible": False,
            "reason": (
                "The real gaugino 16 occurs once, but neither the spatial vector "
                "(3,1) nor the six scalars (1,6) occur in the bosonic 128. "
                "Compact-group complete reducibility therefore forbids a strict "
                "Spin(3) x Spin(6)-equivariant retract onto the full 9|16 target."
            ),
        },
        "scope": (
            "This excludes a strict direct-4D retract of the fixed valise field "
            "representation. It does not exclude a nonvalise gauge or auxiliary "
            "complex whose algebraic zero-momentum maps change the representation."
        ),
        "gate_passed": all_exact_checks_pass,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not all_exact_checks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
