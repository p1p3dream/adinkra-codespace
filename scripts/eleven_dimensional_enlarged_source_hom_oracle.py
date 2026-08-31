#!/usr/bin/env python3
"""Exact low-bidegree Hom-count and fixed-column no-fill oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "projectors": ROOT / "results/adynkra_11d_clifford_projectors_validation.json",
    "lorentz_orbit": ROOT
    / "results/adynkra_11d_lorentz_holonomy_compensator_audit.json",
    "raw_g4": ROOT / "results/adynkra_11d_raw_three_channel_g4_bianchi.json",
    "normalization": ROOT
    / "results/adynkra_11d_right_c_full_chain_four_form_normalization.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def form_multiplicity(target: int, left: int, right: int) -> list[dict[str, int | str]]:
    hits: list[dict[str, int | str]] = []
    for contractions in range(min(left, right) + 1):
        degree = left + right - 2 * contractions
        if degree == target:
            hits.append(
                {
                    "derivative_form_degree": left,
                    "source_form_degree": right,
                    "contractions": contractions,
                    "route": "direct",
                }
            )
        if degree == 11 - target:
            hits.append(
                {
                    "derivative_form_degree": left,
                    "source_form_degree": right,
                    "contractions": contractions,
                    "route": "hodge",
                }
            )
    return hits


def two_spinor_inventory(target: int) -> list[list[dict[str, int | str]]]:
    return [
        sum((form_multiplicity(target, r, p) for r in (0, 3, 4)), [])
        for p in range(6)
    ]


def main() -> None:
    data = {name: json.loads(path.read_text()) for name, path in INPUTS.items()}
    projectors = data["projectors"]
    orbit = data["lorentz_orbit"]
    raw_g4 = data["raw_g4"]
    normalization = data["normalization"]

    a3_d2 = two_spinor_inventory(3)
    g4_d2 = two_spinor_inventory(4)
    a3_counts = [len(routes) for routes in a3_d2]
    g4_counts = [len(routes) for routes in g4_d2]

    checks = {
        "source_dimension_1376": 352 + sum((1, 11, 55, 165, 330, 462)) == 1376,
        "spinor_square_dimension_1024": sum((1, 11, 55, 165, 330, 462)) == 1024,
        "spinor_times_h_hat_dimension_10240": sum(
            (462, 330, 165, 55, 11, 4290, 3003, 1430, 429, 65)
        )
        == 10240,
        "gamma_trace_rank_32": projectors["gamma_trace_projector_rank"] == 32,
        "gamma_traceless_rank_320": projectors["gamma_traceless_projector_rank"]
        == 320,
        "only_p2_is_lorentz_stueckelberg": orbit["solved_orbit"]
        == "delta Psi=0; delta H_hat=0; delta Psi_[1,3,4,5]=0; delta Psi_[2]=Lambda_[2]",
        "h_hat_raw_g4_rank_3": raw_g4["g4_coefficient_rank_mod_prime"] == 3,
        "h_hat_bianchi_rank_2": raw_g4["bianchi_coefficient_rank_mod_prime"] == 2,
        "h_hat_closed_kernel_is_trace_ray": raw_g4["exact_kernel_basis"]
        == [[1, 0, 0]],
        "corrected_full_scan_has_zero_common_support": normalization[
            "common_support_rows"
        ]
        == 0,
        "fixed_witness_is_nonzero_target_only": normalization["first_exact_mismatch"]
        == {
            "source_ordinal": 0,
            "output_coordinate": 0,
            "exterior_spinor_mask": 65537,
            "momentum_exponents": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "candidate": "(0)+(0)i",
            "teleparallel": "(1/1280)+(0)i",
            "residual": "(-1/1280)+(0)i",
        },
        "a3_d2_counts": a3_counts == [1, 1, 1, 2, 2, 2],
        "g4_d2_counts": g4_counts == [1, 1, 1, 2, 3, 2],
    }
    passed = all(checks.values())
    report = {
        "schema_version": "adynkra-11d-enlarged-source-hom-oracle-v1",
        "input_sha256": {name: sha256(path) for name, path in INPUTS.items()},
        "source_coordinate_dimensions": {
            "h_full": 352,
            "h_gamma_traceless": 320,
            "h_gamma_trace_gauge": 32,
            "clifford_form_compensators": 1024,
            "raw_total": 1376,
            "current_quotiented_independent_h_hat_plus_scale": 321,
        },
        "hom_dimensions": {
            "d0_p0_forms_to_a3_g4": [1, 1],
            "d0_p1_forms_to_a3_g4": [2, 2],
            "d1_p0_full_h_to_a3_g4": [2, 2],
            "d1_p1_full_h_to_a3_g4": [5, 5],
            "d2_p0_forms_to_a3_g4": [sum(a3_counts), sum(g4_counts)],
            "d2_p0_a3_by_source_form_degree": a3_counts,
            "d2_p0_g4_by_source_form_degree": g4_counts,
        },
        "d2_p0_a3_basis": a3_d2,
        "d2_p0_g4_basis": g4_d2,
        "checks": checks,
        "independent_direct_sum_can_fill_fixed_h_hat_row": False,
        "reason": "source ordinal is part of the row key; independent irreducible source summands extend the operator by new columns and cannot change an existing H_hat column",
        "passed": passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
