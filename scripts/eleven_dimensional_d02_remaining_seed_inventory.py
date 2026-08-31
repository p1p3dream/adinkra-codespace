#!/usr/bin/env python3
"""Fail-closed seed inventory for the three remaining (0,2) D G4 maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "higher_hom": ROOT / "results/adynkra_11d_higher_bidegree_hom_inventory.json",
    "projectors": ROOT / "results/adynkra_11d_dg4_casimir_projectors.json",
    "first_generator": ROOT / "results/adynkra_11d_d02_00001_source_generator.json",
}
EXPECTED = {
    "higher_hom": "0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f",
    "projectors": "a616e996fb8b002473743840051df5792dfeef6d5b43c7fe378d8a9d0e2cab6d",
    "first_generator": "029beb189bbb8ad340d7f9f0b4f187e6bba992db9854a3f255f36a8711535b00",
}
PRIMES = [1_073_741_783, 1_073_741_723, 1_073_741_719]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    observed = {name: sha(path) for name, path in PATHS.items()}
    if observed != EXPECTED:
        raise SystemExit(f"dependency digest drift: expected={EXPECTED}, observed={observed}")
    hom = json.loads(PATHS["higher_hom"].read_text())
    projectors = json.loads(PATHS["projectors"].read_text())
    first = json.loads(PATHS["first_generator"].read_text())
    assert hom["descendant_targets"]["d0_p2_D_G4_by_irrep"] == {
        "00001": 1,
        "00011": 0,
        "00101": 0,
        "01001": 1,
        "10001": 2,
    }
    assert projectors["passed_canary"] and projectors["exhaustive_projector_ranks_constructed"]
    assert first["passed"] and first["generator_operator_rank"] == 32

    diagrams = [
        {
            "ordinal": 0,
            "name": "gamma4_slash_p_p_dot_h",
            "formula": "Gamma_[4] slash(p) (p_a H^a)",
            "status": "constructed_and_certified",
            "coefficient_column": 52,
            "target_seed": "00001",
            "stream_sha256": first["stream_sha256"],
        },
        {
            "ordinal": 1,
            "name": "p_wedge_gamma3_p_dot_h",
            "formula": "p_[a Gamma_bcd] (p_e H^e)",
            "status": "candidate_pending_equivariance_and_rref",
            "target_seed": "01001_or_10001",
        },
        {
            "ordinal": 2,
            "name": "p_square_gamma3_wedge_h",
            "formula": "(eta^ab p_a p_b) Gamma_[abc H_d]",
            "status": "candidate_pending_equivariance_and_rref",
            "target_seed": "01001_or_10001",
        },
        {
            "ordinal": 3,
            "name": "p_wedge_gamma2_slash_p_wedge_h",
            "formula": "p_[a Gamma_bc slash(p) H_d]",
            "status": "candidate_pending_equivariance_and_rref",
            "target_seed": "01001_or_10001",
        },
    ]
    report = {
        "schema_version": "adynkra-11d-d02-remaining-seed-inventory-v1",
        "dependencies": observed,
        "source": "Sym2(V*) tensor Hhat, tensor basis b_aa=e_a tensor e_a and b_ab=e_a tensor e_b+e_b tensor e_a",
        "target": "S tensor Lambda4(V*)",
        "multiplicity_inventory": hom["descendant_targets"]["d0_p2_D_G4_by_irrep"],
        "minimum_total_diagram_count": 4,
        "constructed_diagram_count": 1,
        "remaining_diagram_count": 3,
        "candidate_diagrams": diagrams,
        "polarization_convention": {
            "diagonal": "evaluate the two-momentum expression at (a,a) once",
            "off_diagonal": "sum the ordered (a,b) and (b,a) routes",
            "lorentz_action_scale": "2*m(input)/m(output), m(diagonal)=1 and m(off_diagonal)=2",
        },
        "projection_and_rref_gates": {
            "canonical_diagram_order": [item["name"] for item in diagrams],
            "canonical_sector_order": ["00001", "00011", "00101", "01001", "10001"],
            "required_equivariance_checks_per_diagram": 55 * 66 * 320,
            "required_equivariance_residual_entries": 0,
            "required_forbidden_sector_stream_entries": {"00011": 0, "00101": 0},
            "required_flattened_operator_ranks_by_sector": {
                "00001": 1,
                "00011": 0,
                "00101": 0,
                "01001": 1,
                "10001": 2,
            },
            "required_combined_rank": 4,
            "ordered_primes": PRIMES,
            "rref_rule": "project each full operator, flatten in canonical row order, and perform deterministic column RREF independently at all three primes",
            "characteristic_zero_rule": "full modular rank at one denominator-admissible prime proves independence over Q; all three primes are required operationally",
            "pivot_stability_rule": "record pivot diagram ordinals and first pivot row at each prime; require equal ranks, while pivot rows may differ",
            "exact_replay_rule": "replay every selected pivot row over Q and verify the lifted determinant is nonzero",
        },
        "mutations": [
            "revert symmetric-pair action to unconditional factor two",
            "omit the swapped off-diagonal momentum route",
            "remove the time metric from H_flat or p_square",
            "permute numeric four-form target masks into lexicographic order",
        ],
        "passed_inventory": True,
        "passed_remaining_generators": False,
        "boundary": "This pins a minimal four-diagram candidate family and exact RREF gates. Only the 00001 diagram is constructed. The 01001 and two 10001 generators are not certified until their projected full-operator ranks are 1 and 2 with zero equivariance and forbidden-sector residuals.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
