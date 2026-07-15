#!/usr/bin/env python3
"""Local node-raising corollary of the SO(9) Casimir obstruction.

Ordinary nonvalise node raising replaces component fields by finite powers of
the time derivative D = d/dt. Spatial SO(9) rotations commute with D, so an
SO(9)-equivariant differential map is a matrix with coefficients in R[D] (or,
if inverse derivatives are admitted, R[D,D^-1]). The Casimir intertwining
equation applies coefficient by coefficient. Distinct scalar Casimirs therefore
force every coefficient to vanish over either characteristic-zero ring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs="?", default="results/sr_so9_equivariance_audit.json"
    )
    parser.add_argument("--output", default="results/sr_nonvalise_locality_audit.json")
    args = parser.parse_args()

    source_path = Path(args.input)
    source = json.loads(source_path.read_text())
    values = source["quadratic_casimir_eigenvalues"]
    maps = {
        "J_B_physical_vector_to_full_bosons": {
            "source_casimir": values["physical_vector"],
            "target_casimir": values["full_bosons"],
        },
        "P_B_full_bosons_to_physical_vector": {
            "source_casimir": values["full_bosons"],
            "target_casimir": values["physical_vector"],
        },
        "J_F_physical_spinor_to_full_fermions": {
            "source_casimir": values["physical_spinor"],
            "target_casimir": values["full_fermions"],
        },
        "P_F_full_fermions_to_physical_spinor": {
            "source_casimir": values["full_fermions"],
            "target_casimir": values["physical_spinor"],
        },
    }
    for data in maps.values():
        difference = data["target_casimir"] - data["source_casimir"]
        data["casimir_difference"] = difference
        data["difference_is_nonzero"] = difference != 0
        data["all_derivative_coefficients_forced_zero"] = difference != 0

    all_zero = all(
        data["all_derivative_coefficients_forced_zero"] for data in maps.values()
    )
    source_gate_passed = bool(source["casimir_forces_all_so9_retract_maps_zero"])
    output = {
        "source": args.input,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "assumptions": [
            "the full field spaces and SO(9) action are unchanged",
            "spatial SO(9) commutes with the time derivative D",
            "retract maps are matrices over a characteristic-zero derivative ring",
            "equivariance is strict rather than modulo gauge transformations",
        ],
        "rings_checked": [
            "R[D] (finite local differential operators)",
            "R[D,D^-1] (Laurent operators allowing inverse derivatives)",
        ],
        "coefficient_argument": (
            "For T(D)=sum_k T_k D^k, C_target T(D)=T(D) C_source gives "
            "(c_target-c_source) T_k=0 for every k. Each displayed difference "
            "is a nonzero real scalar and hence a unit in both rings."
        ),
        "maps": maps,
        "source_casimir_gate_passed": source_gate_passed,
        "all_equivariant_derivative_maps_forced_zero": all_zero,
        "ordinary_node_raising_can_remove_obstruction": not all_zero,
        "inverse_time_derivatives_can_remove_obstruction": not all_zero,
        "gate_passed": source_gate_passed and all_zero,
        "conclusion": (
            "Ordinary node raising, lowering, and even Laurent powers of the "
            "time derivative cannot cure the SO(9) Casimir mismatch while the "
            "128|128 field module is fixed. A surviving route must change the "
            "representation content or weaken strict equivariance through a "
            "gauge/Bianchi quotient."
            if source_gate_passed and all_zero
            else "The derivative-ring obstruction was not established."
        ),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
