#!/usr/bin/env python3
"""Exact Cartan-character audit of the fixed 128|128 Spin(9) module.

The four commuting rotations M_01, M_23, M_45, M_67 form a Cartan
subalgebra.  Their induced full-field matrices are stored with scale four.
Exact characteristic polynomials of each generator and of a separating
base-32 linear combination are compared with the weight multisets of

  Sym^2_0(9) + Lambda^3(9)       (44 + 84 bosons),
  (9 tensor 16) - 16             (128 gamma-traceless vector-spinor).

Every candidate weight coordinate lies in [-8,8] in the scale-four
normalization, so base 32 makes the linear combination injective on the full
weight box.  Since the Cartan matrices commute and are skew, the individual
coordinate bounds plus the separating characteristic polynomial determine the
joint weight multiset exactly.
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


Weight = tuple[int, int, int, int]


def add(left: Weight, right: Weight) -> Weight:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def vector_weights() -> list[Weight]:
    weights: list[Weight] = [(0, 0, 0, 0)]
    for axis in range(4):
        for sign in (-1, 1):
            weight = [0, 0, 0, 0]
            weight[axis] = 4 * sign
            weights.append(tuple(weight))
    return weights


def spinor_weights() -> list[Weight]:
    return [tuple(2 * sign for sign in signs) for signs in product((-1, 1), repeat=4)]


def expected_weights() -> tuple[Counter[Weight], Counter[Weight]]:
    vector = vector_weights()
    spinor = spinor_weights()

    symmetric_square: Counter[Weight] = Counter()
    for left in range(len(vector)):
        for right in range(left, len(vector)):
            symmetric_square[add(vector[left], vector[right])] += 1
    symmetric_square[(0, 0, 0, 0)] -= 1  # remove the invariant trace

    three_form: Counter[Weight] = Counter()
    for indices in combinations(range(len(vector)), 3):
        weight = (0, 0, 0, 0)
        for index in indices:
            weight = add(weight, vector[index])
        three_form[weight] += 1

    bosons = symmetric_square + three_form

    vector_spinor: Counter[Weight] = Counter()
    for vector_weight in vector:
        for spinor_weight in spinor:
            vector_spinor[add(vector_weight, spinor_weight)] += 1
    fermions = vector_spinor - Counter(spinor)
    return bosons, fermions


def expected_charpoly(weights: Counter[Weight], coefficients: tuple[int, ...]) -> list[int]:
    encoded: Counter[int] = Counter()
    for weight, multiplicity in weights.items():
        value = sum(
            coefficient * coordinate
            for coefficient, coordinate in zip(coefficients, weight)
        )
        encoded[value] += multiplicity
    x = sp.Symbol("x")
    polynomial = sp.Poly(x ** encoded.get(0, 0), x, domain=sp.ZZ)
    for eigenvalue in sorted(value for value in encoded if value > 0):
        positive = encoded[eigenvalue]
        negative = encoded.get(-eigenvalue, 0)
        assert positive == negative
        polynomial *= sp.Poly((x * x + eigenvalue * eigenvalue) ** positive, x, domain=sp.ZZ)
    return [int(value) for value in polynomial.all_coeffs()]


def actual_charpoly(matrix: np.ndarray) -> list[int]:
    return [int(value) for value in sp.Matrix(matrix.tolist()).charpoly().all_coeffs()]


def coefficient_digest(coefficients: list[int]) -> str:
    payload = ",".join(str(value) for value in coefficients).encode()
    return hashlib.sha256(payload).hexdigest()


def construct_cartan(input_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

    boson_cartan: list[np.ndarray] = []
    fermion_cartan: list[np.ndarray] = []
    for i, j in ((0, 1), (2, 3), (4, 5), (6, 7)):
        spin2 = gamma[i] @ gamma[j]
        boson4 = np.zeros((128, 128), dtype=np.int64)
        fermion4 = np.zeros((128, 128), dtype=np.int64)
        for alpha in range(16):
            for beta in range(alpha + 1, 16):
                coefficient = int(spin2[alpha, beta])
                if coefficient:
                    boson4 += coefficient * (full_l[alpha] @ full_l[beta].T)
                    fermion4 += coefficient * (full_l[alpha].T @ full_l[beta])
        boson_cartan.append(boson4)
        fermion_cartan.append(fermion4)
    return boson_cartan, fermion_cartan


def audit_character(
    cartan: list[np.ndarray], weights: Counter[Weight], separating: tuple[int, ...]
) -> dict[str, object]:
    coordinate_checks = []
    all_match = True
    for axis, matrix in enumerate(cartan):
        coefficients = tuple(1 if index == axis else 0 for index in range(4))
        expected = expected_charpoly(weights, coefficients)
        actual = actual_charpoly(matrix)
        matches = actual == expected
        all_match &= matches
        coordinate_checks.append(
            {
                "axis": axis,
                "matches": matches,
                "degree": len(actual) - 1,
                "coefficient_sha256": coefficient_digest(actual),
            }
        )

    combined = sum(
        coefficient * matrix for coefficient, matrix in zip(separating, cartan)
    )
    expected = expected_charpoly(weights, separating)
    actual = actual_charpoly(combined)
    separating_matches = actual == expected
    all_match &= separating_matches
    return {
        "dimension": sum(weights.values()),
        "distinct_joint_weights": len(weights),
        "maximum_scaled_weight_coordinate": max(
            abs(coordinate) for weight in weights for coordinate in weight
        ),
        "coordinate_characteristic_polynomials": coordinate_checks,
        "separating_characteristic_polynomial": {
            "coefficients": list(separating),
            "matches": separating_matches,
            "degree": len(actual) - 1,
            "coefficient_sha256": coefficient_digest(actual),
        },
        "joint_weight_character_matches_exactly": all_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/sr_hole_minimal.json")
    parser.add_argument("--output", default="results/sr_spin9_decomposition_audit.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    boson_cartan, fermion_cartan = construct_cartan(input_path)
    cartan_commutes = all(
        np.array_equal(left @ right, right @ left)
        for cartan in (boson_cartan, fermion_cartan)
        for left, right in combinations(cartan, 2)
    )
    cartan_is_skew = all(
        np.array_equal(matrix + matrix.T, np.zeros_like(matrix))
        for matrix in boson_cartan + fermion_cartan
    )

    boson_weights, fermion_weights = expected_weights()
    separating = (1, 32, 32**2, 32**3)
    boson_audit = audit_character(boson_cartan, boson_weights, separating)
    fermion_audit = audit_character(fermion_cartan, fermion_weights, separating)
    gate_passed = bool(
        cartan_commutes
        and cartan_is_skew
        and boson_audit["joint_weight_character_matches_exactly"]
        and fermion_audit["joint_weight_character_matches_exactly"]
    )
    output = {
        "source": args.input,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "normalization": "Cartan matrices and weights are scaled by four",
        "cartan_labels": ["M_01", "M_23", "M_45", "M_67"],
        "cartan_matrices_commute_exactly": cartan_commutes,
        "cartan_matrices_are_skew_exactly": cartan_is_skew,
        "separation_argument": (
            "Each exact coordinate characteristic polynomial bounds every scaled "
            "weight coordinate by eight. Base 32 is injective on that integer "
            "box, and the exact separating characteristic polynomial therefore "
            "determines the joint weight multiset."
        ),
        "bosons": {
            "claimed_decomposition": "Sym^2_0(9) + Lambda^3(9) = 44 + 84",
            "highest_weights_scaled_by_four": [[8, 0, 0, 0], [4, 4, 4, 0]],
            **boson_audit,
        },
        "fermions": {
            "claimed_decomposition": "(9 tensor 16) - 16 = 128",
            "highest_weight_scaled_by_four": [6, 2, 2, 2],
            **fermion_audit,
        },
        "identified_massless_little_group_content": (
            "44 + 84 | 128, the Spin(9) representation content familiar from "
            "the eleven-dimensional supergravity multiplet"
        ),
        "gate_passed": gate_passed,
        "conclusion": (
            "The fixed Garden module is exactly 44+84 on bosons and the "
            "gamma-traceless vector-spinor 128 on fermions as a Spin(9) module."
            if gate_passed
            else "The proposed Spin(9) decomposition was not established."
        ),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
