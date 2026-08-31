#!/usr/bin/env python3
"""Exact Fierz-channel seed inventory for the (2,1) D G4 Hom space."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "higher_hom": (
        ROOT / "results/adynkra_11d_higher_bidegree_hom_inventory.json",
        "0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f",
    ),
    "source_fierz_projectors": (
        ROOT / "results/adynkra_11d_clifford_projectors_validation.json",
        "0f5efdec8c36c7449d27a80be9eab0cd8c39db970b72d9910ae2e6fd385686b2",
    ),
    "target_casimir_projectors": (
        ROOT / "results/adynkra_11d_dg4_casimir_projectors.json",
        "a616e996fb8b002473743840051df5792dfeef6d5b43c7fe378d8a9d0e2cab6d",
    ),
}
PRIMES = [1_073_741_783, 1_073_741_723, 1_073_741_719]
SECTORS = ["00001", "00011", "00101", "01001", "10001"]
EXPECTED = {
    "scalar": {"00001": 1, "00011": 0, "00101": 0, "01001": 1, "10001": 1},
    "lambda3": {"00001": 3, "00011": 2, "00101": 5, "01001": 6, "10001": 6},
    "lambda4": {"00001": 3, "00011": 5, "00101": 6, "01001": 7, "10001": 6},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_character_oracle():
    path = ROOT / "scripts/eleven_dimensional_higher_bidegree_hom_oracle.py"
    spec = importlib.util.spec_from_file_location("higher_hom_oracle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    observed = {name: sha(path) for name, (path, _) in DEPENDENCIES.items()}
    expected_hashes = {name: digest for name, (_, digest) in DEPENDENCIES.items()}
    if observed != expected_hashes:
        raise SystemExit(f"dependency digest drift: expected={expected_hashes}, observed={observed}")

    h = load_character_oracle()
    trivial = h.Counter({(0, 0, 0, 0, 0): 1})
    characters = {"scalar": trivial, "lambda3": h.A3, "lambda4": h.G4}
    computed = {}
    source_dimensions = {}
    for channel, character in characters.items():
        domain = h.convolve(h.convolve(character, h.V), h.T)
        computed[channel] = h.target_multiplicities(domain, SECTORS)
        source_dimensions[channel] = sum(domain.values())
    assert computed == EXPECTED
    totals = {sector: sum(computed[channel][sector] for channel in computed) for sector in SECTORS}
    assert totals == {"00001": 7, "00011": 7, "00101": 11, "01001": 14, "10001": 13}
    assert sum(totals.values()) == 52
    assert source_dimensions == {"scalar": 3520, "lambda3": 580800, "lambda4": 1161600}
    assert sum(source_dimensions.values()) == 1_745_920

    seed_slots = []
    ordinal = 0
    for channel in ("scalar", "lambda3", "lambda4"):
        for sector in SECTORS:
            for copy in range(computed[channel][sector]):
                seed_slots.append(
                    {
                        "ordinal": ordinal,
                        "fierz_channel": channel,
                        "target_sector": sector,
                        "multiplicity_ordinal": copy,
                        "construction": "reserved canonical slot from the exact character multiplicity; Cartesian Lorentz-commutator seed is pending",
                        "status": "abstract_exact_seed_pending_cartesian_stream",
                    }
                )
                ordinal += 1
    assert len(seed_slots) == 52

    report = {
        "schema_version": "adynkra-11d-d21-manifestly-equivariant-seed-inventory-v1",
        "dependencies": observed,
        "source": "Lambda2(S) tensor V* tensor Hhat",
        "source_fierz_decomposition": {
            "scalar": {"form_degree": 0, "dimension": 1, "source_dimension_after_V_Hhat": source_dimensions["scalar"]},
            "lambda3": {"form_degree": 3, "dimension": 165, "source_dimension_after_V_Hhat": source_dimensions["lambda3"]},
            "lambda4": {"form_degree": 4, "dimension": 330, "source_dimension_after_V_Hhat": source_dimensions["lambda4"]},
            "total_dimension": sum(source_dimensions.values()),
        },
        "target": "S tensor Lambda4(V*)",
        "minimal_seed_counts_by_fierz_channel_and_target_sector": computed,
        "target_sector_rank_totals": totals,
        "required_cartesian_basis_dimension": len(seed_slots),
        "seed_slots": seed_slots,
        "all_vector_contractions": {
            "coverage": "exhaustive modulo Clifford, Hodge, exterior, symmetric-momentum, and gamma-trace identities",
            "raw_diagram_generators": [
                "Gamma_[q] with q=0..5; q=6..11 are represented by the 11D Hodge-dual member",
                "every perfect eta pairing among non-external vector legs",
                "zero or one epsilon_11 vertex, with more reduced by epsilon-epsilon identities",
                "every choice of four antisymmetrized external target legs",
                "the single formal momentum leg and the upper Hhat vector leg with the repository time metric",
            ],
            "canonicalization": [
                "numeric source-form and target-form masks",
                "sort metric edges lexicographically",
                "move epsilon indices into ascending order and retain permutation sign",
                "apply gamma Hodge duality to rank at most five",
                "annihilate repeated exterior legs and the gamma trace Gamma_a H^a",
                "merge identical sparse Cartesian streams before RREF",
            ],
            "completeness_reason": "the orthogonal invariant tensor algebra is generated by eta and epsilon, while the spinor endomorphism algebra is the Clifford basis; the exact commutator-nullspace dimension supplies the independent upper bound",
            "raw_syntactic_diagram_count": None,
            "raw_count_boundary": "do not publish a raw count before canonicalization because it depends on presentation and is not the Hom dimension",
        },
        "exact_rref_completeness_gates": {
            "source_fierz_projector_ranks": {"scalar": 1, "lambda3": 165, "lambda4": 330},
            "source_fierz_projector_sum_rank": 496,
            "source_fierz_projector_cross_products": 0,
            "lorentz_generators": 55,
            "target_projector_ranks": {"00001": 32, "00011": 5280, "00101": 3520, "01001": 1408, "10001": 320},
            "required_ranks_by_channel_and_sector": computed,
            "required_combined_ranks_by_sector": totals,
            "required_combined_rank": 52,
            "ordered_primes": PRIMES,
            "denominator_gate": "every Fierz, Hhat, and target-projector denominator must be coprime to all three primes before reduction",
            "stream_order": "Fierz channel scalar,lambda3,lambda4; target sector canonical order; multiplicity ordinal; canonical PBW row",
            "nullspace_seed_rule": "solve all 55 equations rho_target(X)F-F rho_source(X)=0 inside each projected channel-sector block and choose deterministic first-free-column RREF basis",
            "diagram_span_rule": "project and flatten every canonical contraction diagram; its three-prime RREF rank must equal the commutator-nullspace multiplicity in every channel-sector block",
            "exact_replay_rule": "replay the selected pivot minor over Q and require a nonzero determinant after denominator clearing",
            "full_stream_mutations": [
                "drop the Lambda4 Fierz channel",
                "omit all epsilon/Hodge diagrams",
                "remove the Lorentz time metric from one contraction",
                "replace numeric target masks with lexicographic masks",
                "use symmetric-pair rather than exterior-spinor Fierz signs",
            ],
        },
        "passed_exact_multiplicity_inventory": True,
        "passed_cartesian_generator_construction": False,
        "boundary": "This is an exact 52-dimensional abstract Hom inventory and Cartesian-construction completeness contract. Its 52 records reserve canonical coefficient slots, not constructed seed maps. It does not claim the Cartesian streams exist until the commutator-nullspace and contraction-diagram RREF gates pass. No 51-column factorization or one-dimensional complement is assumed.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
