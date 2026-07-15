#!/usr/bin/env python3
"""A posteriori existence certificate for the dense joint section.

The affine coordinates stored by ``search_sr_joint_section.py`` are treated as
exact dyadic rationals.  A rank-revealing QR selects a square 2304-variable
subsystem, the remaining 559 variables are frozen, and a numerical inverse B
defines the fixed-point map

    T(x) = x - B F(x).

The residual F(x0) is evaluated exactly with Python integers at the common
scale 2^-2148.  The inverse and defect norms are bounded under the standard
IEEE-754 round-to-nearest model.  The final contraction and self-map tests use
only exact rational arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg


SCALE_EXP = 1074
SCALE = 1 << SCALE_EXP
PRODUCT_SCALE_EXP = 2 * SCALE_EXP


def scaled_dyadic(value: float) -> int:
    """Return the exact integer value of ``value * 2^1074``."""
    numerator, denominator = float(value).as_integer_ratio()
    shift = SCALE_EXP - (denominator.bit_length() - 1)
    assert denominator == 1 << (denominator.bit_length() - 1)
    assert shift >= 0
    return numerator << shift


def gamma(count: int) -> float:
    unit_roundoff = 2.0**-53
    product = count * unit_roundoff
    assert product < 1.0
    return float(np.nextafter(product / (1.0 - product), np.inf))


def upward_product(left: float, right: float) -> float:
    estimate = left * right
    return float(np.nextafter(estimate / (1.0 - gamma(1)), np.inf))


def upward_sum(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Upper-bound a nonnegative floating sum in the IEEE model."""
    count = values.size if axis is None else values.shape[axis]
    estimate = np.sum(values, axis=axis)
    upper = estimate / (1.0 - gamma(max(1, count - 1)))
    return np.nextafter(upper, np.inf)


def signed_permutation(
    perm: list[int], sign: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(perm, dtype=np.int64), np.asarray(sign, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/sr_hole_minimal.json")
    parser.add_argument(
        "--candidate", default="results/sr_joint_section_candidate.npz"
    )
    parser.add_argument("--output", default="results/sr_joint_root_audit.json")
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text())
    topology = next(
        t for t in report["topologies"] if t["auxiliary_projection"].get("witness")
    )
    witness = topology["auxiliary_projection"]["witness"]
    data = witness["joint_section_data"]
    l_data = [
        signed_permutation(perm, sign)
        for perm, sign in zip(data["l_perm"], data["l_sign"])
    ]
    gamma_data = [
        signed_permutation(perm, sign)
        for perm, sign in zip(data["gamma_perm"], data["gamma_sign"])
    ]

    candidate = np.load(args.candidate)
    zb = np.asarray(candidate["boson_affine"], dtype=np.float64)
    zf = np.asarray(candidate["fermion_affine"], dtype=np.float64)
    complement = np.asarray(candidate["boson_complement"], dtype=np.int64)
    assert zb.shape == (9, 119)
    assert zf.shape == (112, 16)
    assert complement.shape == (119,)

    fibers: list[list[tuple[int, int]]] = [[] for _ in range(16)]
    pf_sign = np.zeros((128, 16), dtype=np.int64)
    for entry in witness["projection"]:
        beta = entry["physical_beta"]
        full_f = entry["full_fermion"]
        sign = entry["sign"]
        fibers[beta].append((full_f, sign))
        pf_sign[full_f, beta] = sign
    assert all(len(fiber) == 8 for fiber in fibers)

    # Exact dyadic affine center.
    zb_int = [[scaled_dyadic(zb[i, k]) for k in range(119)] for i in range(9)]
    zf_int = [
        [scaled_dyadic(zf[col, beta]) for beta in range(16)]
        for col in range(112)
    ]
    pb_int = [[0 for _ in range(128)] for _ in range(9)]
    physical_bosons = witness["boson_by_vector"]
    for i, full_b in enumerate(physical_bosons):
        pb_int[i][full_b] = SCALE
    for i in range(9):
        for k, full_b in enumerate(complement):
            pb_int[i][int(full_b)] = zb_int[i][k]

    jf_int = [
        [int(pf_sign[full_f, beta]) * (SCALE // 8) for beta in range(16)]
        for full_f in range(128)
    ]
    nf_columns: list[tuple[int, int, int]] = []
    col = 0
    for fiber in fibers:
        f0, s0 = fiber[0]
        for fk, sk in fiber[1:]:
            first_coefficient = -sk // s0
            nf_columns.append((f0, fk, first_coefficient))
            for beta in range(16):
                jf_int[fk][beta] += zf_int[col][beta]
                jf_int[f0][beta] += first_coefficient * zf_int[col][beta]
            col += 1
    assert col == 112

    targets = np.zeros((16, 9, 16), dtype=np.int64)
    for vector, (perm, sign) in enumerate(gamma_data):
        for charge in range(16):
            targets[charge, vector, int(perm[charge])] = int(sign[charge])

    # Exact residual at denominator 2^2148.
    max_residual_numerator = 0
    for charge, (perm, sign) in enumerate(l_data):
        for i in range(9):
            for beta in range(16):
                value = -int(targets[charge, i, beta]) * (SCALE * SCALE)
                for full_b in range(128):
                    value += (
                        pb_int[i][full_b]
                        * int(sign[full_b])
                        * jf_int[int(perm[full_b])][beta]
                    )
                max_residual_numerator = max(max_residual_numerator, abs(value))

    exact_residual_below_1e14 = (
        max_residual_numerator * 10**14 < 1 << PRODUCT_SCALE_EXP
    )
    residual_float = float(
        Fraction(max_residual_numerator, 1 << PRODUCT_SCALE_EXP)
    )

    # Build the Jacobian by rounding each exact dyadic entry once.  Each row has
    # at most 119 + 112 nonzero entries.
    rows = 16 * 9 * 16
    columns = 9 * 119 + 112 * 16
    jacobian = np.zeros((rows, columns), dtype=np.float64)
    base = 9 * 119
    for charge, (perm, sign) in enumerate(l_data):
        inverse_perm = np.empty(128, dtype=np.int64)
        inverse_perm[perm] = np.arange(128, dtype=np.int64)
        for i in range(9):
            for beta in range(16):
                row = charge * 9 * 16 + i * 16 + beta
                for k, full_b_value in enumerate(complement):
                    full_b = int(full_b_value)
                    exact = int(sign[full_b]) * jf_int[int(perm[full_b])][beta]
                    jacobian[row, i * 119 + k] = float(Fraction(exact, SCALE))
                for nf_col, (f0, fk, first_coefficient) in enumerate(nf_columns):
                    b0 = int(inverse_perm[f0])
                    bk = int(inverse_perm[fk])
                    exact = (
                        pb_int[i][bk] * int(sign[bk])
                        + first_coefficient * pb_int[i][b0] * int(sign[b0])
                    )
                    jacobian[row, base + nf_col * 16 + beta] = float(
                        Fraction(exact, SCALE)
                    )

    _, r_factor, pivots = scipy.linalg.qr(
        jacobian, mode="economic", pivoting=True, check_finite=False
    )
    square = np.ascontiguousarray(jacobian[:, pivots[:rows]])
    inverse = scipy.linalg.inv(square, check_finite=False)
    product = inverse @ square

    # Norm of B, with the positive row sums bounded upward.
    inverse_row_sums = upward_sum(np.abs(inverse), axis=1)
    inverse_norm_upper = float(np.max(inverse_row_sums))

    # Rounding from exact dyadic J to its nearest binary64 representation.
    # Half an ulp bounds every scalar conversion.
    jacobian_rounding = 0.5 * np.abs(np.spacing(square))
    jacobian_rounding[square == 0.0] = 0.0
    jacobian_error_upper = float(
        np.max(upward_sum(jacobian_rounding, axis=1))
    )

    # Bound the BLAS matrix-product error by gamma_(2n) |B||J|, then add
    # B times the exact-to-binary64 Jacobian conversion error.
    square_row_sums = upward_sum(np.abs(square), axis=1)
    positive_product = np.abs(inverse) @ square_row_sums
    positive_product_upper = float(
        np.nextafter(
            np.max(positive_product) / (1.0 - gamma(2 * rows)), np.inf
        )
    )
    blas_product_error = upward_product(gamma(2 * rows), positive_product_upper)

    identity_defect = np.eye(rows, dtype=np.float64) - product
    defect_from_stored_product = float(
        np.max(upward_sum(np.abs(identity_defect), axis=1))
    )
    conversion_error = upward_product(inverse_norm_upper, jacobian_error_upper)
    defect_upper = float(
        upward_sum(
            np.asarray(
                [defect_from_stored_product, blas_product_error, conversion_error]
            )
        )
    )

    # The only nonzero second derivatives couple one Z_B coordinate to one
    # Z_F coordinate.  Count the absolute entries of E_B^T L_a N_F exactly.
    # The Jacobian Lipschitz bound counts this block in both Hessian directions.
    complement_set = {int(value) for value in complement}
    cross_counts = []
    for perm, _ in l_data:
        inverse_perm = np.empty(128, dtype=np.int64)
        inverse_perm[perm] = np.arange(128, dtype=np.int64)
        count = 0
        for f0, fk, _ in nf_columns:
            count += int(int(inverse_perm[f0]) in complement_set)
            count += int(int(inverse_perm[fk]) in complement_set)
        cross_counts.append(count)
    exact_jacobian_lipschitz_bound = 2 * max(cross_counts)
    assert exact_jacobian_lipschitz_bound <= 448

    # Coarse rational bounds deliberately leave several orders of magnitude of
    # margin over the floating audit.  H=448 follows from at most 224 absolute
    # cross coefficients per residual row, counted in both Hessian directions.
    chosen_f = Fraction(1, 10**14)
    chosen_m = Fraction(1024, 1)
    chosen_e = Fraction(1, 10**6)
    chosen_h = Fraction(448, 1)
    radius = Fraction(3, 10**11)
    eta = chosen_m * chosen_f
    contraction = chosen_e + chosen_m * chosen_h * radius
    self_map = eta + contraction * radius

    checks = {
        "exact_residual_below_1e-14": exact_residual_below_1e14,
        "inverse_norm_below_1024": inverse_norm_upper < float(chosen_m),
        "inverse_defect_below_1e-6": defect_upper < float(chosen_e),
        "contraction_constant_below_one": contraction < 1,
        "closed_ball_maps_strictly_inside_itself": self_map < radius,
    }
    passed = all(checks.values())

    output = {
        "source": args.input,
        "candidate": args.candidate,
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
        "catalog_index": topology["catalog_index"],
        "certificate": "square-subsystem Banach contraction",
        "arithmetic_model": (
            "exact dyadic residual and exact rational final inequalities; "
            "IEEE-754 round-to-nearest norm audit with gamma_n error bounds"
        ),
        "equations": rows,
        "affine_variables": columns,
        "solved_variables": rows,
        "frozen_variables": columns - rows,
        "minimum_qr_pivot": float(np.min(np.abs(np.diag(r_factor)))),
        "exact_center_residual": {
            "denominator_power_of_two": PRODUCT_SCALE_EXP,
            "numerator_bit_length": max_residual_numerator.bit_length(),
            "decimal_approximation": residual_float,
            "numerator_sha256": hashlib.sha256(
                str(max_residual_numerator).encode()
            ).hexdigest(),
        },
        "computed_upper_bounds": {
            "inverse_infinity_norm": inverse_norm_upper,
            "rounded_jacobian_error_infinity_norm": jacobian_error_upper,
            "blas_product_error_infinity_norm": blas_product_error,
            "stored_inverse_defect_infinity_norm": defect_from_stored_product,
            "exact_inverse_defect_infinity_norm": defect_upper,
            "exact_jacobian_lipschitz_bound": exact_jacobian_lipschitz_bound,
        },
        "adopted_exact_rational_bounds": {
            "residual_infinity_norm": "1/100000000000000",
            "inverse_infinity_norm": "1024",
            "inverse_defect_infinity_norm": "1/1000000",
            "jacobian_lipschitz_constant": "448",
            "ball_radius": "3/100000000000",
            "eta_upper": str(eta),
            "contraction_upper": str(contraction),
            "self_map_radius_upper": str(self_map),
        },
        "checks": checks,
        "existence_certificate_passed": passed,
        "conclusion": (
            "A real zero of all 2304 temporal linkage equations exists in the "
            "certified ball, with both affine inverse constraints exact."
            if passed
            else "The attempted a posteriori certificate did not pass."
        ),
        "pivot_columns": [int(value) for value in pivots[:rows]],
        "inverse_binary64_sha256": hashlib.sha256(inverse.tobytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "pivot_columns"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
