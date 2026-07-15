#!/usr/bin/env python3
"""Zero-spatial-momentum obstruction for local spatial linkage maps.

Let p_1,...,p_9 denote spatial derivatives and D the time derivative.  A
local finite-order map is a matrix over R[D,p_1,...,p_9].  Evaluation at
p=0 is a ring homomorphism to R[D].  If the polynomial map is SO(9)
equivariant, its p-independent coefficient is an SO(9)-equivariant map over
R[D], which the preceding Casimir audit forces to vanish.

Consequently every local injection and projection vanishes at p=0, so their
composition cannot be the identity.  The same contradiction survives modulo
gauge or Bianchi corrections whose entries have positive spatial derivative
degree, since those corrections lie in (p_1,...,p_9) and also vanish at p=0.

The scope is deliberately narrow.  This does not cover a formulation that
changes the field module, uses field strengths with different zero-mode
relations, or localizes away from p=0 by admitting inverse spatial operators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs="?", default="results/sr_nonvalise_locality_audit.json"
    )
    parser.add_argument(
        "--so9-input", default="results/sr_so9_equivariance_audit.json"
    )
    parser.add_argument(
        "--output", default="results/sr_spatial_gauge_locality_audit.json"
    )
    args = parser.parse_args()

    time_path = Path(args.input)
    so9_path = Path(args.so9_input)
    time_audit = json.loads(time_path.read_text())
    so9_audit = json.loads(so9_path.read_text())

    time_gate_passed = bool(time_audit["gate_passed"])
    so9_gate_passed = bool(
        so9_audit["casimir_forces_all_so9_retract_maps_zero"]
    )
    map_zero_checks = {
        name: bool(data["all_derivative_coefficients_forced_zero"])
        for name, data in time_audit["maps"].items()
    }
    all_zero_at_spatial_origin = all(map_zero_checks.values())

    retracts = {
        "bosons": {
            "identity": "P_B(p,D) J_B(p,D) = I_9",
            "evaluated_left_side": "P_B(0,D) J_B(0,D) = 0",
            "evaluated_right_side": "I_9",
            "contradiction": all_zero_at_spatial_origin,
        },
        "fermions": {
            "identity": "P_F(p,D) J_F(p,D) = I_16",
            "evaluated_left_side": "P_F(0,D) J_F(0,D) = 0",
            "evaluated_right_side": "I_16",
            "contradiction": all_zero_at_spatial_origin,
        },
    }

    correction_classes = {
        "spatial_potential_gauge": {
            "form": "delta A_i = p_i epsilon",
            "contained_in_spatial_ideal": True,
            "vanishes_under_ev_p0": True,
        },
        "positive_spatial_degree_bianchi": {
            "form": "each correction coefficient belongs to (p_1,...,p_9)",
            "contained_in_spatial_ideal": True,
            "vanishes_under_ev_p0": True,
        },
        "fermion_identity": {
            "form": "no gauge correction is available for the gaugino identity",
            "contained_in_spatial_ideal": True,
            "vanishes_under_ev_p0": True,
        },
    }
    corrections_vanish = all(
        item["vanishes_under_ev_p0"] for item in correction_classes.values()
    )
    strict_retract_impossible = all(
        item["contradiction"] for item in retracts.values()
    )
    quotient_retract_impossible = strict_retract_impossible and corrections_vanish
    gate_passed = (
        time_gate_passed
        and so9_gate_passed
        and all_zero_at_spatial_origin
        and quotient_retract_impossible
    )

    output = {
        "sources": {
            "time_derivative_audit": args.input,
            "time_derivative_audit_sha256": sha256(time_path),
            "so9_audit": args.so9_input,
            "so9_audit_sha256": sha256(so9_path),
        },
        "ring": "R[D,p_1,...,p_9]",
        "spatial_ideal": "m=(p_1,...,p_9)",
        "evaluation_homomorphism": "ev_p0: R[D,p_1,...,p_9] -> R[D]",
        "assumptions": [
            "the full 128|128 field module and its SO(9) action are unchanged",
            "the retract maps are finite-order local differential operators",
            "SO(9) acts simultaneously on coefficients, fields, and spatial momenta",
            "the physical bosons are the nine spatial potentials A_i",
            "allowed gauge/Bianchi corrections vanish at zero spatial momentum",
        ],
        "coefficient_argument": (
            "The spatially constant coefficient T(0,D) of any equivariant "
            "T(p,D) is an equivariant map over R[D]. The time-derivative "
            "Casimir audit forces that coefficient to vanish for all four "
            "retract maps. Evaluation is multiplicative, so ev_p0(PJ)=0."
        ),
        "map_zero_checks": map_zero_checks,
        "all_retract_maps_vanish_at_spatial_origin": all_zero_at_spatial_origin,
        "strict_retracts": retracts,
        "strict_local_retract_possible": not strict_retract_impossible,
        "correction_classes": correction_classes,
        "modulo_gauge_bianchi_argument": (
            "If PJ=I+K with every entry of K in m, applying ev_p0 gives "
            "0=I. Spatial-potential gauge shifts p_i epsilon and all Bianchi "
            "corrections of positive spatial degree are in m."
        ),
        "local_retract_modulo_scoped_gauge_bianchi_possible": (
            not quotient_retract_impossible
        ),
        "not_excluded": [
            "changing from spatial potentials to a field-strength module",
            "adding gauge, Bianchi, auxiliary, or other SO(9) representations",
            "using a larger or genuinely rectangular Garden module",
            "localizing away from p=0 with inverse spatial operators or boundary conditions",
            "corrections that do not vanish at zero spatial momentum",
        ],
        "source_gates_passed": {
            "so9": so9_gate_passed,
            "time_derivative": time_gate_passed,
        },
        "gate_passed": gate_passed,
        "conclusion": (
            "Finite local spatial derivatives cannot produce a strict retract, "
            "or a retract modulo gauge/Bianchi corrections of positive spatial "
            "degree, on the same 128|128 potential module. The zero-momentum "
            "fiber retains the SO(9) Casimir obstruction."
            if gate_passed
            else "The zero-spatial-momentum obstruction was not established."
        ),
    }

    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
