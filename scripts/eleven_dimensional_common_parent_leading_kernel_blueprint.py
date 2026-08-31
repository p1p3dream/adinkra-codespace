#!/usr/bin/env python3
"""Audit the explicit leading P_A and P_psi common-parent generator inputs."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "results/adynkra_11d_direct_spinor_common_parent_inventory.json"
LEVEL15 = ROOT / "results/adynkra_11d_level15_bridge_validation.json"
LEVEL18 = ROOT / "results/adynkra_11d_level18_embedded_maps.json"
OUTPUT = ROOT / "results/adynkra_11d_common_parent_leading_kernel_blueprint.json"
SPINOR_WEIGHTS = list(itertools.product((1, -1), repeat=5))
SIMPLE_ROOTS = [
    (2, -2, 0, 0, 0),
    (0, 2, -2, 0, 0),
    (0, 0, 2, -2, 0),
    (0, 0, 0, 2, -2),
    (0, 0, 0, 0, 2),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def highest_weight(label: str) -> tuple[int, int, int, int, int]:
    digits = list(map(int, label))
    return tuple(2 * sum(digits[index:4]) + digits[4] for index in range(5))


def exterior_weight_counts(maximum_degree: int) -> list[Counter[tuple[int, ...]]]:
    counts = [Counter() for _ in range(maximum_degree + 1)]
    counts[0][(0, 0, 0, 0, 0)] = 1
    for weight in SPINOR_WEIGHTS:
        for degree in range(maximum_degree, 0, -1):
            for current, multiplicity in list(counts[degree - 1].items()):
                target = tuple(left + right for left, right in zip(current, weight))
                counts[degree][target] += multiplicity
    return counts


def system_shape(
    counts: list[Counter[tuple[int, ...]]],
    degree: int,
    label: str,
    expected_nullity: int,
) -> dict[str, Any]:
    highest = highest_weight(label)
    raising = [
        counts[degree][tuple(left + right for left, right in zip(highest, root))]
        for root in SIMPLE_ROOTS
    ]
    return {
        "dynkin_label": label,
        "exterior_degree": degree,
        "highest_weight_doubled_coordinates": list(highest),
        "source_weight_space_columns": counts[degree][highest],
        "raising_block_rows": raising,
        "total_raising_rows": sum(raising),
        "expected_kernel_dimension": expected_nullity,
        "expected_rank": counts[degree][highest] - expected_nullity,
    }


def candidate_rows(inventory: dict[str, Any], block: str) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in inventory["inventory"]["candidates"]
        if candidate["block"] == block and candidate["momentum_degree"] == 0
    ]


def existing_level17_fixture(label: str, copy: int, copies: int) -> Path:
    suffix = "" if copies == 1 else f"_{copy}"
    return ROOT / (
        "data/eleven_dimensional_spinor_bridge/"
        f"level17_{label}_highest_weight_kernel{suffix}.i16le"
    )


def ppsi_map_path(source_label: str, source_copy: int) -> Path:
    return ROOT / (
        "results/eleven_dimensional_level18_embedded/"
        f"embedded_10001_from_{source_label}_copy{source_copy}.json"
    )


def main() -> None:
    inventory = read_json(INVENTORY)
    level15 = read_json(LEVEL15)
    level18 = read_json(LEVEL18)
    if inventory.get("passed_representation_inventory") is not True:
        raise RuntimeError("common-parent representation inventory is not green")
    if level15.get("passed") is not True or level18["report"].get("passed") is not True:
        raise RuntimeError("a pinned source-map dependency is not green")
    if level18["report"]["exact_embedded_maps_by_target"].get("10001") != 8:
        raise RuntimeError("level18 target-10001 map count is not 8")

    counts = exterior_weight_counts(18)
    pa_multiplicities = {"00001": 2, "10001": 1, "01001": 2, "00101": 3}
    ppsi_multiplicities = {
        "00100": 2,
        "00010": 2,
        "10100": 1,
        "10010": 1,
        "10002": 2,
    }
    pa_shapes = [
        system_shape(counts, 17, label, multiplicity)
        for label, multiplicity in pa_multiplicities.items()
    ]
    ppsi_shapes = [
        system_shape(counts, 18, label, multiplicity)
        for label, multiplicity in ppsi_multiplicities.items()
    ]

    pa_candidates = candidate_rows(inventory, "P_A")
    ppsi_candidates = candidate_rows(inventory, "P_psi")
    if len(pa_candidates) != 8 or len(ppsi_candidates) != 8:
        raise RuntimeError("leading candidate count drifted")
    expected_pa = [
        ("00001", 1), ("00001", 2), ("10001", 1), ("01001", 1),
        ("01001", 2), ("00101", 1), ("00101", 2), ("00101", 3),
    ]
    expected_ppsi = [
        ("00100", 1), ("00100", 2), ("00010", 1), ("00010", 2),
        ("10100", 1), ("10010", 1), ("10002", 1), ("10002", 2),
    ]
    if [(row["source_dynkin_label"], row["source_fixture_copy"]) for row in pa_candidates] != expected_pa:
        raise RuntimeError("P_A candidate order drifted")
    if [(row["source_dynkin_label"], row["source_fixture_copy"]) for row in ppsi_candidates] != expected_ppsi:
        raise RuntimeError("P_psi candidate order drifted")

    pa_generators = []
    for local_ordinal, candidate in enumerate(pa_candidates):
        label = candidate["source_dynkin_label"]
        copy = candidate["source_fixture_copy"]
        multiplicity = pa_multiplicities[label]
        fixture = existing_level17_fixture(label, copy, multiplicity)
        if fixture.exists():
            source_status = "existing exact level17 fixture"
            source_path = str(fixture.relative_to(ROOT))
            source_sha = sha256(fixture)
        elif label == "00001":
            level15_path = ROOT / (
                "data/eleven_dimensional_bridge/"
                f"00001_highest_weight_kernel_{copy}.i16le"
            )
            if not level15_path.exists():
                raise RuntimeError(f"missing level15 Hodge input {level15_path}")
            source_status = "construct by exact longest-Weyl descent of the level15 fixture followed by exterior Hodge complement"
            source_path = str(level15_path.relative_to(ROOT))
            source_sha = sha256(level15_path)
        else:
            source_status = "missing exact level17 highest-weight kernel"
            source_path = None
            source_sha = None
        pa_generators.append(
            {
                "local_ordinal": local_ordinal,
                "global_manifest_ordinal": candidate["ordinal"],
                "source_dynkin_label": label,
                "source_copy": copy,
                "source_status": source_status,
                "source_path_or_hodge_input": source_path,
                "source_sha256": source_sha,
                "abstract_coupling": f"unique multiplicity-one {label} tensor 00001 -> 00100",
                "target": "lexicographic Cartesian Lambda3(V), dimension 165",
            }
        )

    ppsi_generators = []
    for local_ordinal, candidate in enumerate(ppsi_candidates):
        label = candidate["source_dynkin_label"]
        copy = candidate["source_fixture_copy"]
        path = ppsi_map_path(label, copy)
        embedded = read_json(path)
        if not embedded.get("passed") or embedded.get("certified_irrep_image_rank") != 320:
            raise RuntimeError(f"P_psi embedded map is not green: {path}")
        if embedded["certificate"].get("exact_raising_residual_terms_by_simple_root") != [0] * 5:
            raise RuntimeError(f"P_psi raising residual is nonzero: {path}")
        ppsi_generators.append(
            {
                "local_ordinal": local_ordinal,
                "global_manifest_ordinal": candidate["ordinal"],
                "source_dynkin_label": label,
                "source_copy": copy,
                "embedded_map_path": str(path.relative_to(ROOT)),
                "embedded_map_sha256": sha256(path),
                "coupled_map_sha256": embedded["coupled_map_sha256"],
                "source_fixture": embedded["source_fixture"],
                "source_fixture_sha256": embedded["source_fixture_sha256"],
                "certified_irrep_image_rank": embedded["certified_irrep_image_rank"],
            }
        )

    report = {
        "schema_version": "adynkra-11d-common-parent-leading-kernel-blueprint-v1",
        "passed_inventory_audit": True,
        "pa_component_construction_ready": False,
        "ppsi_abstract_maps_ready": True,
        "ppsi_cartesian_stream_ready": False,
        "canonical_order": {
            "P_A": pa_generators,
            "P_psi": ppsi_generators,
        },
        "source_highest_weight_systems": {
            "P_A_level17": pa_shapes,
            "P_psi_level18": ppsi_shapes,
        },
        "P_A_blueprint": {
            "coefficient_dimension": 8,
            "source_fixture_status": {
                "level17_existing_copies": 3,
                "level15_exact_hodge_inputs_for_00001": 2,
                "missing_level17_00101_copies": 3,
            },
            "shortest_source_completion": [
                "Expose the existing level15 descendant generator, descend each 00001 highest vector to the lowest weight, apply the signed exterior Hodge complement, and verify all level17 raising rows exactly.",
                "Adapt the deterministic sparse highest-weight kernel generator to exterior degree17 label00101, requiring rank161429 and nullity3, then verify every integer raising row.",
                "Add a generic multiplicity-one abstract coupling problem with target00100 and target highest weight [2,2,2,0,0] for source labels00001,10001,01001,00101.",
                "Apply each abstract coupling to all eight exterior fixtures and join its 165 target states to lexicographic Cartesian three-form masks.",
            ],
            "target_projector_route": "No spectral projector is needed. Lambda3(V) is already the irreducible 00100. Use the exact 3-slot antisymmetrizer and lexicographic 165-coordinate basis; a Casimir eigen-residual is an optional redundant gate.",
            "minimal_completeness_gates": [
                "source rank-nullity and exact integer raising residual for all eight source fixtures",
                "four abstract multiplicity-one coupling residuals are zero under all five Chevalley raising operators",
                "eight embedded source maps have pairwise bound source hashes and nonzero target highest coefficient",
                "Cartesian target join has exact rank165, all six antisymmetry identities, and zero residual for all55 Lorentz generators",
                "the eight full PBW coefficient streams have exact column rank8 on retained actual rows",
                "one source coefficient mutation and one omitted antisymmetry route are both rejected",
            ],
        },
        "P_psi_blueprint": {
            "coefficient_dimension": 8,
            "representation_maps_already_complete": True,
            "aggregate_level18_artifact_sha256": sha256(LEVEL18),
            "shortest_component_completion": [
                "Replay the eight existing embedded target10001 maps in canonical order.",
                "Generate the abstract target lowering basis once and solve its exact join to canonical_gamma_traceless_frame_basis().",
                "Emit source exterior mask, free Psi spinor, and canonical Hhat ordinal rows for each coefficient column.",
            ],
            "target_projector_route": "Use the exact P320 gamma-trace projector on the 352-dimensional vector-spinor, then the canonical 320-frame basis and its spatial-slot left inverse. This is stronger and cheaper than a new Casimir projector.",
            "minimal_completeness_gates": [
                "all eight pinned embedded maps pass, have rank320, and have zero exact raising residual",
                "P320 is idempotent, has rank320 and trace-kernel rank32, and every target column has zero gamma trace",
                "abstract-to-Cartesian join has rank320, exact two-sided reconstruction, and zero residual for all55 Lorentz generators",
                "the eight full PBW coefficient streams have exact column rank8 on retained actual rows",
                "collapsing two source copies and mutating the time metric or gamma-trace subtraction are rejected",
            ],
        },
        "publication_contract": {
            "row_key": [
                "target_block", "source_exterior_mask", "free_Psi_spinor",
                "target_coordinate",
            ],
            "column_order": "use canonical_order without sorting by filenames",
            "hashes": "bind source fixture, abstract coupling, target join, full stream, and dependency source hashes independently",
            "exactness": "three-prime construction is acceptable only with denominator checks and exact Q replay of the retained rank8 actual-row minor",
            "observability": "checkpoint at generator boundaries and publish the final report by atomic rename after all mutations pass",
        },
        "dependencies": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                INVENTORY,
                LEVEL15,
                LEVEL18,
                ROOT / "src/eleven_dimensional_level16_couplings.rs",
                ROOT / "src/eleven_dimensional_h_hat_jet.rs",
                ROOT / "src/eleven_dimensional_independent_a3_adapter.rs",
            ]
        },
        "boundary": "This audit closes the exact source-system shapes, canonical order, and reuse boundary for the sixteen leading P_A and P_psi generators. It does not construct the missing level17 00101 kernels, the P_A abstract couplings, either Cartesian stream, lower PBW symbols, or the coupled common-parent relations.",
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "pa_generators": len(pa_generators),
        "ppsi_generators": len(ppsi_generators),
        "missing_level17_00101_kernels": 3,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
