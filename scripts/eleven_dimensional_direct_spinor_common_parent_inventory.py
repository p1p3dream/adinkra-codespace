#!/usr/bin/env python3
"""Exact B5 inventory for local direct-spinor maps into Hhat at weight 16."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHARACTER_ORACLE = ROOT / "scripts/eleven_dimensional_higher_bidegree_hom_oracle.py"
PREPOTENTIAL_SOURCE = ROOT / "src/eleven_dimensional_prepotential.rs"
DERIVATIVE_ARTIFACT = ROOT / "results/adynkra_11d_level17_derivative_matrix.json"
SCALAR_NEGATIVE_CONTROL = ROOT / "results/adynkra_11d_level15_bridge_validation.json"
FIRST_MOMENTUM_ARTIFACT = ROOT / "results/adynkra_11d_first_momentum_couplings_all.json"
SECOND_MOMENTUM_ARTIFACT = ROOT / "results/adynkra_11d_second_momentum_recoupling.json"
SECOND_MOMENTUM_RANK_ARTIFACT = ROOT / "results/adynkra_11d_second_momentum_full_77_rank_p0.json"
EQ40_FIBER_ARTIFACT = ROOT / "results/adynkra_11d_eq40_independent_a3_fiber.json"
OUTPUT = ROOT / "results/adynkra_11d_direct_spinor_common_parent_inventory.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_character_oracle():
    spec = importlib.util.spec_from_file_location("adynkra_b5_character_oracle", CHARACTER_ORACLE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load exact B5 character oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_lower_level_fixtures() -> list[list[tuple[str, int]]]:
    source = PREPOTENTIAL_SOURCE.read_text()
    try:
        block = source.split("const LOWER_LEVEL_FIXTURES:", 1)[1].split("];", 1)[0]
    except IndexError as error:
        raise RuntimeError("LOWER_LEVEL_FIXTURES not found") from error
    fixtures = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', block)
    if len(fixtures) != 17:
        raise RuntimeError(f"expected 17 lower-level fixtures, found {len(fixtures)}")
    output: list[list[tuple[str, int]]] = []
    for fixture in fixtures:
        terms: list[tuple[str, int]] = []
        for token in fixture.split():
            if "*" in token:
                multiplicity_text, label = token.split("*", 1)
                multiplicity = int(multiplicity_text)
            else:
                label = token
                multiplicity = 1
            if len(label) != 5 or not label.isdigit() or multiplicity <= 0:
                raise RuntimeError(f"invalid fixture token {token!r}")
            terms.append((label, multiplicity))
        output.append(terms)
    return output


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def build_inventory() -> dict[str, Any]:
    b5 = load_character_oracle()
    fixtures = parse_lower_level_fixtures()
    derivative = read_json(DERIVATIVE_ARTIFACT)
    scalar = read_json(SCALAR_NEGATIVE_CONTROL)
    first = read_json(FIRST_MOMENTUM_ARTIFACT)
    second = read_json(SECOND_MOMENTUM_ARTIFACT)
    second_rank = read_json(SECOND_MOMENTUM_RANK_ARTIFACT)
    eq40 = read_json(EQ40_FIBER_ARTIFACT)

    if not derivative.get("passed") or derivative.get("derivative_matrix_rank") != 7:
        raise RuntimeError("direct-spinor derivative certificate is not green at rank 7")
    if derivative.get("derivative_matrix_nullity") != 5:
        raise RuntimeError("direct-spinor derivative nullity is not 5")
    if not scalar.get("passed"):
        raise RuntimeError("scalar negative-control certificate is not green")
    if "cannot cancel" not in scalar.get("boundary", ""):
        raise RuntimeError("scalar negative-control no-cancellation boundary is absent")
    if not first.get("passed") or first.get("embedded_maps_certified") != 44:
        raise RuntimeError("first-momentum 44-map certificate is not green")
    if not second.get("passed") or not second.get("representation_inventory_matches"):
        raise RuntimeError("second-momentum representation inventory is not green")
    if not second_rank.get("passed") or second_rank.get("rank_over_gaussian_extension") != 77:
        raise RuntimeError("second-momentum 77-column rank certificate is not green")
    if not eq40.get("passed") or eq40.get("exact_image_residual_rows", 0) <= 0:
        raise RuntimeError("Eq40 physical-fiber obstruction certificate is not green")

    conformal_graviton = b5.symmetric_character(b5.VECTOR_BASIS, 2)
    conformal_graviton[(0, 0, 0, 0, 0)] -= 1
    conformal_graviton = +conformal_graviton
    a3 = b5.exterior_character(b5.VECTOR_BASIS, 3)
    target_specs = [
        ("P_H", "Hhat", "10001", 16, b5.T),
        ("P_h", "conformal_graviton", "20000", 17, conformal_graviton),
        ("P_A", "A3", "00100", 17, a3),
        ("P_psi", "conformal_gravitino", "10001", 18, b5.T),
    ]
    expected_block_counts = {
        "P_H": [12, 44, 77, 100, 81, 41, 21, 9, 1],
        "P_h": [2, 14, 24, 31, 30, 17, 8, 4, 2],
        "P_A": [8, 30, 49, 64, 58, 32, 16, 9, 2],
        "P_psi": [8, 57, 109, 136, 133, 85, 41, 21, 9, 1],
    }
    candidates: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    ordinal = 0
    for block_name, target_name, target_label, total_weight, target_character in target_specs:
        orders: list[dict[str, Any]] = []
        block_start = ordinal
        for momentum_degree in range(total_weight // 2 + 1):
            exterior_degree = total_weight - 2 * momentum_degree
            source_level = min(exterior_degree, 32 - exterior_degree)
            source_terms = fixtures[source_level]
            symmetric_momentum = b5.symmetric_character(b5.VECTOR_BASIS, momentum_degree)
            intermediate_character = b5.convolve(symmetric_momentum, target_character)
            possible_intermediates = sorted(
                {
                    label
                    for source_label, _ in source_terms
                    for label in b5.tensor_spinor_labels(source_label)
                }
            )
            intermediate_channels = []
            for label in possible_intermediates:
                multiplicity = b5.irreducible_multiplicity(intermediate_character, label)
                if multiplicity:
                    intermediate_channels.append((label, multiplicity))

            pair_count = 0
            source_copy_incidences = 0
            order_start = ordinal
            for intermediate_label, intermediate_multiplicity in intermediate_channels:
                for source_label, source_multiplicity in source_terms:
                    if intermediate_label not in b5.tensor_spinor_labels(source_label):
                        continue
                    pair_count += 1
                    source_copy_incidences += source_multiplicity
                    for intermediate_copy in range(1, intermediate_multiplicity + 1):
                        for source_copy in range(1, source_multiplicity + 1):
                            candidates.append(
                                {
                                    "ordinal": ordinal,
                                    "block": block_name,
                                    "target_name": target_name,
                                    "target_dynkin_label": target_label,
                                    "total_operator_weight": total_weight,
                                    "momentum_degree": momentum_degree,
                                    "exterior_spinor_degree": exterior_degree,
                                    "source_fixture_level": source_level,
                                    "normal_form": (
                                        f"Sym^{momentum_degree}(V) tensor "
                                        f"Lambda^{exterior_degree}(S) tensor S_Psi -> "
                                        f"{target_name}({target_label})"
                                    ),
                                    "intermediate_dynkin_label": intermediate_label,
                                    "intermediate_copy": intermediate_copy,
                                    "source_dynkin_label": source_label,
                                    "source_fixture_copy": source_copy,
                                }
                            )
                            ordinal += 1
            direct_character = b5.convolve(
                b5.convolve(symmetric_momentum, b5.S), target_character
            )
            direct_multiplicity = sum(
                source_multiplicity
                * b5.irreducible_multiplicity(direct_character, source_label)
                for source_label, source_multiplicity in source_terms
            )
            coefficient_count = ordinal - order_start
            if direct_multiplicity != coefficient_count:
                raise RuntimeError(
                    f"Frobenius cross-check failed for {block_name} at momentum degree "
                    f"{momentum_degree}: {direct_multiplicity} != {coefficient_count}"
                )
            orders.append(
                {
                    "momentum_degree": momentum_degree,
                    "exterior_spinor_degree": exterior_degree,
                    "source_fixture_level": source_level,
                    "operator_symbol": f"p^{momentum_degree} D^{exterior_degree} Psi",
                    "intermediate_channels": [
                        {"dynkin_label": label, "multiplicity": multiplicity}
                        for label, multiplicity in intermediate_channels
                    ],
                    "intermediate_channel_copies": sum(
                        multiplicity for _, multiplicity in intermediate_channels
                    ),
                    "source_intermediate_pairs": pair_count,
                    "source_copy_incidences": source_copy_incidences,
                    "coefficient_count": coefficient_count,
                    "ordinal_range": [order_start, ordinal],
                    "direct_character_cross_check": direct_multiplicity,
                }
            )
        actual_counts = [entry["coefficient_count"] for entry in orders]
        if actual_counts != expected_block_counts[block_name]:
            raise RuntimeError(f"unexpected {block_name} coefficient counts {actual_counts}")
        blocks.append(
            {
                "block": block_name,
                "target_name": target_name,
                "target_dynkin_label": target_label,
                "total_operator_weight": total_weight,
                "orders": orders,
                "coefficient_counts_by_momentum_degree": actual_counts,
                "total_coefficients": sum(actual_counts),
                "leading_coefficients": actual_counts[0],
                "lower_symbol_coefficients": sum(actual_counts[1:]),
                "ordinal_range": [block_start, ordinal],
            }
        )

    if len(candidates) != 1386:
        raise RuntimeError(f"expected 1386 common-parent candidates, found {len(candidates)}")
    if derivative.get("source_basis") != [
        "10000#1", "20000#1", "00100#1", "00100#2", "00010#1", "00010#2",
        "00002#1", "10100#1", "10010#1", "10002#1", "10002#2", "10002#3",
    ]:
        raise RuntimeError("leading 12-column order drifted")

    witness = eq40["first_off_image_witness"]
    if witness["residual"] != {"real": [-1, 56], "imaginary": [0, 1]}:
        raise RuntimeError("the pinned -1/56 obstruction witness drifted")
    if eq40.get("candidate_potential_bidegree") != [1, 0]:
        raise RuntimeError("Eq40 candidate bidegree drifted")

    candidate_manifest_sha = canonical_sha256(candidates)
    block_summary_sha = canonical_sha256(blocks)
    hhat = blocks[0]
    hhat_candidates = [candidate for candidate in candidates if candidate["block"] == "P_H"]
    hhat_manifest_sha = canonical_sha256(hhat_candidates)
    total_missing_hhat = sum(hhat["coefficient_counts_by_momentum_degree"][3:])
    report = {
        "schema_version": "adynkra-11d-direct-spinor-common-parent-inventory-v2",
        "passed_representation_inventory": True,
        "obstruction_launch_ready": False,
        "engineering_convention": {
            "weight_D": 1,
            "weight_p": 2,
            "target_total_weights": {"P_H": 16, "P_h": 17, "P_A": 17, "P_psi": 18},
            "pbw_normal_forms": "p^q D^(w-2q) Psi for q=0..floor(w/2); p is commuting and D is exterior after torsion normal ordering",
            "hom_formula": "Hom_B5(Sym^q(V) tensor Lambda^(w-2q)(S) tensor S_Psi, target)",
            "completeness": "all local polynomial Lorentz-equivariant symbols at each declared target weight before source gauge quotient",
        },
        "inventory": {
            "blocks": blocks,
            "total_coefficients": len(candidates),
            "leading_coefficients": sum(block["leading_coefficients"] for block in blocks),
            "lower_symbol_coefficients": sum(block["lower_symbol_coefficients"] for block in blocks),
            "hhat_coefficients": hhat["total_coefficients"],
            "hhat_q0_q1_q2_negative_control_columns": sum(hhat["coefficient_counts_by_momentum_degree"][:3]),
            "hhat_new_q3_through_q8_coefficients": total_missing_hhat,
            "candidate_ordering": "target block P_H,P_h,P_A,P_psi; momentum degree; intermediate Dynkin label; source fixture order; intermediate copy; source copy",
            "candidate_manifest_sha256": candidate_manifest_sha,
            "hhat_candidate_manifest_sha256": hhat_manifest_sha,
            "block_summary_sha256": block_summary_sha,
            "candidates": candidates,
        },
        "leading_direct_spinor_gate": {
            "leading_columns": 12,
            "derivative_matrix_shape": [7, 12],
            "derivative_rank": 7,
            "derivative_nullity": 5,
            "kernel_basis": derivative["primitive_integer_kernel_basis"],
            "scalar_factorizing_coordinates": derivative["scalar_factorizing_coordinates"],
            "scalar_factorizing_line_is_in_kernel": derivative["scalar_factorizing_hook_image_is_zero"],
            "interpretation": "the scalar-factorizing line is one negative-control direction inside the five-dimensional direct-spinor leading kernel, not the common-parent ansatz",
        },
        "scalar_negative_control": {
            "artifact_sha256": sha256(SCALAR_NEGATIVE_CONTROL),
            "passed": scalar["passed"],
            "result": "the complete local p D13 V correction has rank 2 while the augmented rank is 3, so it cannot cancel the scalar bridge Eq2.7 residual",
            "use_in_this_inventory": "negative control only",
        },
        "closed_bounded_hhat_negative_controls": {
            "q0_plus_q1": {
                "columns": 56,
                "exact_rank": 56,
                "nullity": 0,
                "builder": "eleven_dimensional_level16_couplings::build_joint_compatibility_matrix",
                "durable_report_path": None,
                "provenance_note": "the exact builder and completed measurement exist, but no tracked report is present at results/adynkra_11d_joint_compatibility_matrix.json",
            },
            "q2": {
                "columns": 77,
                "exact_rank": 77,
                "nullity": 0,
                "artifact_sha256": sha256(SECOND_MOMENTUM_RANK_ARTIFACT),
                "matrix_sha256": second_rank["matrix_sha256"],
            },
            "interpretation": "these are already-closed bounded P_H negative controls. They do not constrain the 253 q>=3 P_H symbols or construct the coupled P_h, P_A, and P_psi blocks.",
        },
        "obstruction_functional": {
            "definition": "O = (1 - P_physical) D(d Psi_[3])",
            "sign_convention": "residual = reconstructed physical image - candidate",
            "domain": "Hhat PBW source stream after a candidate common-parent map P_H",
            "existing_hhat_source0_canary": {
                "source_ordinal": 0,
                "row_key": witness["monomial"] | {"target_coordinate": witness["target_coordinate"]},
                "candidate": witness["candidate"],
                "reconstructed_physical_image": witness["reconstructed_image"],
                "residual": witness["residual"],
            },
            "existing_eq40_candidate_rows": eq40["candidate_rows"],
            "existing_eq40_off_image_residual_rows": eq40["exact_image_residual_rows"],
            "existing_eq40_candidate_stream_sha256": eq40["candidate_stream_sha256"],
            "logical_scope": "the -1/56 value is an exact canary for O on Hhat source basis vector 0. It is not O composed with any of the 386 direct-Psi maps.",
        },
        "source_constraint_and_decisive_matrix": {
            "required_source_constraint": "construct the all-order source and common-parent supersymmetry matrix C on the 1386 coupled coefficients, then let K be a certified basis of ker(C)",
            "obstruction_matrix": "stack exact rows of O composed with P_H K",
            "no_go_gate": "rank(O P_H K) equals dim(K)",
            "survivor_gate": "a nonzero kernel vector survives all source, obstruction, and later gauge-descent rows",
            "raw_hhat_obstruction_columns": 386,
            "raw_common_parent_columns": 1386,
            "shortest_unrestricted_hhat_no_go_minor_shape": [386, 386],
            "shortest_unrestricted_common_parent_no_go_minor_shape": [1386, 1386],
            "shortest_constrained_no_go_minor_shape": "k by k, where k = dim ker(C)",
            "single_canary_shape": [1, 386],
            "single_canary_is_decisive": False,
        },
        "precise_blocker": {
            "missing": "canonical Cartesian/PBW emitters for the 253 q>=3 P_H symbols and the coupled P_h, P_A, P_psi blocks, followed by the all-order common-parent constraint matrix and P_H composition into the frozen physical-fiber left inverse",
            "missing_hhat_coefficients": total_missing_hhat,
            "missing_coupled_component_coefficients_before_relation_solves": sum(block["total_coefficients"] for block in blocks[1:]),
            "why_bounded_hhat_tests_are_not_enough": "the omitted P_H symbols occupy the same total engineering weight, while the other three blocks are required to test one common parent rather than an isolated Hhat map",
            "fail_closed_result": "retain the 56-column, 77-column, and scalar-factorizing computations only as closed negative controls; do not call them the direct-spinor common-parent test",
        },
        "gpu_handoff_contract": {
            "candidate_manifest_sha256": candidate_manifest_sha,
            "canonical_column_count": 1386,
            "hhat_obstruction_column_count": 386,
            "canonical_column_order": "use inventory.candidates exactly",
            "canonical_row_key": [
                "constraint_family", "source_exterior_mask", "source_momentum_exponents",
                "output_exterior_mask", "output_momentum_exponents", "target_coordinate",
            ],
            "families": ["source_constraint", "common_parent_supersymmetry", "physical_fiber_off_image", "relative_normalization"],
            "prime_rule": "three distinct admissible primes with denominator checks before reduction",
            "device_output": "deterministic sparse rows, per-column hashes, three-prime ranks, pivot row keys, and first nonzero residual",
            "cpu_replay": "reconstruct the retained actual-row square minor over Q(i), verify its determinant or RREF exactly, and replay the -1/56 canary",
            "mutation": "replace the physical left inverse by the pinned off-image mutation and require changed residual stream hash and row count",
            "publication": "immutable dependency manifest, checkpoint at candidate boundaries, report-last atomic rename",
        },
        "dependencies": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                CHARACTER_ORACLE, PREPOTENTIAL_SOURCE, DERIVATIVE_ARTIFACT,
                SCALAR_NEGATIVE_CONTROL, FIRST_MOMENTUM_ARTIFACT,
                SECOND_MOMENTUM_ARTIFACT, SECOND_MOMENTUM_RANK_ARTIFACT,
                EQ40_FIBER_ARTIFACT,
            ]
        },
        "boundary": "This proves the complete unconstrained B5 Hom counts and canonical 1386-column order for P_H, P_h, P_A, and P_psi at their declared weights. It does not construct the new Cartesian maps, solve the coupled common-parent relations, quotient source gauge, or compose a direct-spinor parent with the physical obstruction. The scalar bridge and bounded 56-column and 77-column Hhat spaces are already excluded and are retained only as negative controls.",
    }
    return report


def main() -> None:
    report = build_inventory()
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "total_coefficients": report["inventory"]["total_coefficients"],
        "block_totals": {
            block["block"]: block["total_coefficients"]
            for block in report["inventory"]["blocks"]
        },
        "candidate_manifest_sha256": report["inventory"]["candidate_manifest_sha256"],
        "obstruction_launch_ready": report["obstruction_launch_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
