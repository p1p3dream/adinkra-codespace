#!/usr/bin/env python3
"""Exact engineering-degree and B5 character audit after the Eq. 40 no-go."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/adynkra_11d_eq40_source_exhaustion.json"
DEPENDENCIES = {
    "eq40_independent_a3_fiber": (
        ROOT / "results/adynkra_11d_eq40_independent_a3_fiber.json",
        "63cdc0edebfe62c1a9d279fa7d1df2d75cc66248a2d8fc513e0fce12147a57ee",
    ),
    "higher_bidegree_hom_inventory": (
        ROOT / "results/adynkra_11d_higher_bidegree_hom_inventory.json",
        "0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f",
    ),
    "clifford_projectors": (
        ROOT / "results/adynkra_11d_clifford_projectors_validation.json",
        "0f5efdec8c36c7449d27a80be9eab0cd8c39db970b72d9910ae2e6fd385686b2",
    ),
    "raw_three_channel_g4_bianchi": (
        ROOT / "results/adynkra_11d_raw_three_channel_g4_bianchi.json",
        "a0f18ddccaa0c526aa3c38af2ebef081efd1c66d5b6650cfa9f730112c00a6d1",
    ),
}
CHARACTER_ORACLE = ROOT / "scripts/eleven_dimensional_higher_bidegree_hom_oracle.py"
CHARACTER_ORACLE_SHA256 = (
    "7b0bf4a7f58a89903bd2c63d0fe9ebfcb61b8157ae3da635701b12ea3ea12b3c"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_character_oracle():
    if sha256(CHARACTER_ORACLE) != CHARACTER_ORACLE_SHA256:
        raise RuntimeError("higher-bidegree character oracle hash drifted")
    spec = importlib.util.spec_from_file_location("higher_hom", CHARACTER_ORACLE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load higher-bidegree character oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def character_sha256(character) -> str:
    digest = hashlib.sha256()
    digest.update(b"adynkra-b5-character-v1\0")
    for weight, multiplicity in sorted(character.items()):
        for coordinate in weight:
            digest.update(int(coordinate).to_bytes(4, "little", signed=True))
        digest.update(int(multiplicity).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def load_dependencies() -> tuple[dict[str, dict], dict[str, dict]]:
    values: dict[str, dict] = {}
    bindings: dict[str, dict] = {}
    for name, (path, expected) in DEPENDENCIES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"dependency {name} hash drifted: {actual}")
        values[name] = json.loads(path.read_text())
        bindings[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
        }
    return values, bindings


def build_report() -> dict:
    hom = load_character_oracle()
    dependencies, bindings = load_dependencies()

    s_hhat = hom.convolve(hom.S, hom.T)
    s_trace = hom.convolve(hom.S, hom.S)
    full_h = hom.convolve(hom.S, hom.convolve(hom.V, hom.S))
    exterior4_s = hom.exterior_character(hom.SPINOR_BASIS, 4)
    d40_hhat = hom.convolve(exterior4_s, hom.T)

    hhat_a3 = hom.irreducible_multiplicity(s_hhat, "00100")
    trace_a3 = hom.irreducible_multiplicity(s_trace, "00100")
    full_h_a3 = hom.irreducible_multiplicity(full_h, "00100")
    d40_d_a3 = hom.target_multiplicities(d40_hhat, hom.D_A3_TARGETS)
    d40_d_g4 = hom.target_multiplicities(d40_hhat, hom.D_G4_TARGETS)

    eq40 = dependencies["eq40_independent_a3_fiber"]
    higher = dependencies["higher_bidegree_hom_inventory"]
    clifford = dependencies["clifford_projectors"]
    bianchi = dependencies["raw_three_channel_g4_bianchi"]

    weight_one_bidegrees = [
        [d_degree, p_degree]
        for d_degree in range(2)
        for p_degree in range(2)
        if d_degree + 2 * p_degree == 1
    ]
    weight_four_bidegrees = [
        [d_degree, p_degree]
        for d_degree in range(5)
        for p_degree in range(3)
        if d_degree + 2 * p_degree == 4
    ]

    passed = (
        weight_one_bidegrees == [[1, 0]]
        and weight_four_bidegrees == [[0, 2], [2, 1], [4, 0]]
        and hhat_a3 == 1
        and trace_a3 == 1
        and full_h_a3 == 2
        and eq40["passed"] is True
        and eq40["physical_fiber_compatible_on_unrestricted_hhat_slice"] is False
        and eq40["d2_p1"]["off_image_slices"] == 3660
        and eq40["d2_p1"]["on_image_slices"] == 0
        and higher["passed"] is True
        and higher["descendant_targets"]["d2_p1_D_G4_total"] == 52
        and higher["descendant_targets"]["d0_p2_D_G4_total"] == 4
        and clifford["passed"] is True
        and clifford["gamma_trace_projector_rank"] == 32
        and clifford["gamma_traceless_projector_rank"] == 320
        and clifford["gamma_tracelessness_residual_entries"] == 0
        and bianchi["passed"] is True
        and bianchi["g4_coefficient_rank_mod_prime"] == 3
        and bianchi["bianchi_kernel_dimension"] == 1
        and bianchi["exact_kernel_basis"] == [[1, 0, 0]]
        and d40_d_a3 == {"00001": 3, "00101": 13, "01001": 13, "10001": 10}
        and d40_d_g4
        == {"00001": 3, "00011": 10, "00101": 13, "01001": 13, "10001": 10}
    )

    return {
        "schema_version": "adynkra-11d-eq40-source-exhaustion-v1",
        "engineering_filtration": {
            "weight_D": 1,
            "weight_p": 2,
            "required_Hhat_to_A3_potential_weight": 1,
            "nonnegative_local_weight_one_bidegrees": weight_one_bidegrees,
            "descendant_weight_four_PBW_bidegrees": weight_four_bidegrees,
        },
        "potential_hom_multiplicities": {
            "Hom_S_tensor_Hhat_to_A3": hhat_a3,
            "Hom_S_tensor_trace_spinor_to_A3": trace_a3,
            "Hom_S_tensor_full_H_to_A3": full_h_a3,
            "full_H_decomposition": "H = Hhat(10001) direct-sum Gamma(S)(00001)",
        },
        "full_H_trace_ray": {
            "formula": "tau=Gamma^a H_a; A3_trace is the unique S_D tensor tau to Lambda3 map",
            "factorization": "S_D tensor H -> S_D tensor S_trace -> Lambda3 V",
            "gamma_trace_image_rank": clifford["gamma_trace_projector_rank"],
            "gamma_traceless_projector_rank": clifford[
                "gamma_traceless_projector_rank"
            ],
            "gamma_trace_after_P320_residual_entries": clifford[
                "gamma_tracelessness_residual_entries"
            ],
            "vanishes_on_every_Hhat_input": True,
            "can_repair_an_existing_Hhat_row": False,
        },
        "unique_Hhat_ray": {
            "multiplicity": hhat_a3,
            "Bianchi_closed_raw_ray_dimension": bianchi["bianchi_kernel_dimension"],
            "Bianchi_kernel_basis_trace_exterior_hook": bianchi["exact_kernel_basis"],
            "physical_fiber_compatible_on_unrestricted_Hhat": eq40[
                "physical_fiber_compatible_on_unrestricted_hhat_slice"
            ],
            "d2_p1_off_image_slices": eq40["d2_p1"]["off_image_slices"],
            "exhausted_at_required_potential_weight": True,
        },
        "same_descendant_weight_audit": {
            "d2_p1_D_G4_Hom_dimension": higher["descendant_targets"][
                "d2_p1_D_G4_total"
            ],
            "d0_p2_D_G4_Hom_dimension": higher["descendant_targets"][
                "d0_p2_D_G4_total"
            ],
            "d4_p0_D_A3_by_irrep": d40_d_a3,
            "d4_p0_D_A3_total": sum(d40_d_a3.values()),
            "d4_p0_D_G4_by_irrep": d40_d_g4,
            "d4_p0_D_G4_total": sum(d40_d_g4.values()),
            "d4_p0_can_change_d2_p1_witness_without_source_relation": False,
            "reason": "PBW bidegrees are direct-sum row families; no audited differential source relation identifies D4p0 with D2p1",
        },
        "smallest_typed_source_restriction": {
            "direct_P320_Hhat_equals_component_gravitino_allowed": False,
            "direct_rejection_reason": "Hhat is a semi-prepotential and Eq. 25 constructs the physical component gravitino from D Delta and D Psi; a shared 320 irrep is not a source-to-target descendant map",
            "on_shell_diagnostic_chain": "Hhat -> corrected Eq40 Delta -> DDelta -> Eq25 physical psi -> curl -> Rarita Euler",
            "extra_outer_spinor_derivative_required": False,
            "expected_PBW_bidegrees": [[2, 1], [0, 2]],
            "status": "legitimate on-shell diagnostic after binding corrected right-C and horizontal Lorentz descent; not an off-shell Hhat source equation",
            "negative_gate": "form the complete residual functional across all 320 Hhat columns and test rank(C_RS) versus rank([C_RS;w])",
            "positive_gate": "require exact factorization of the full D2p1 plus D0p2 residual through C_RS; a D2p1-only equality is insufficient",
        },
        "character_hashes": {
            "S": character_sha256(hom.S),
            "V": character_sha256(hom.V),
            "Hhat": character_sha256(hom.T),
            "S_tensor_Hhat": character_sha256(s_hhat),
            "S_tensor_trace_spinor": character_sha256(s_trace),
            "exterior4_S": character_sha256(exterior4_s),
            "exterior4_S_tensor_Hhat": character_sha256(d40_hhat),
        },
        "dependencies": bindings
        | {
            "character_oracle": {
                "path": str(CHARACTER_ORACLE.relative_to(ROOT)),
                "sha256": CHARACTER_ORACLE_SHA256,
            }
        },
        "oracle_source_sha256": sha256(Path(__file__)),
        "passed": passed,
        "boundary": "Passing exhausts local polynomial Hhat-to-A3 potentials at the Eq. 40 engineering weight. It does not exhaust nonlocal inverse-momentum operators, higher-weight potentials, direct DG4 maps, unknown source differential quotients, or different physical source constructions. The Rarita chain is an on-shell diagnostic only and cannot be relabeled as an off-shell semi-prepotential constraint.",
    }


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report()
    if not report["passed"]:
        raise SystemExit("source-exhaustion gate failed")
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    write_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
