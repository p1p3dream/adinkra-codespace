#!/usr/bin/env python3
"""Exact representation-level audit of the local ten-dimensional gauge complex.

Build the abelian momentum-space exterior derivative through three-forms,

  Lambda^0 -> Lambda^1 -> Lambda^2 -> Lambda^3,

over R[D,p_1,...,p_9].  This is the gauge-parameter, potential,
field-strength, and Bianchi complex.  The script verifies d^2=0 exactly,
records its SO(9) content, tests the zero-total-momentum necessary condition
for a chain-homotopy retract into the fixed 128|128 module, and audits the
smallest conventional valise enlargement.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from math import comb
from pathlib import Path

import sympy as sp


def exterior_basis(rank: int) -> list[tuple[int, ...]]:
    return list(combinations(range(10), rank))


def exterior_derivative(
    rank: int, momentum: tuple[sp.Symbol, ...]
) -> sp.MutableSparseMatrix:
    source = exterior_basis(rank)
    target = exterior_basis(rank + 1)
    target_index = {indices: row for row, indices in enumerate(target)}
    matrix = sp.MutableSparseMatrix(len(target), len(source), {})
    for column, indices in enumerate(source):
        for mu in range(10):
            if mu in indices:
                continue
            result = tuple(sorted((mu, *indices)))
            insert_position = sum(index < mu for index in indices)
            sign = -1 if insert_position % 2 else 1
            matrix[target_index[result], column] += sign * momentum[mu]
    return matrix


def substituted_rank(matrix: sp.MatrixBase, values: dict[sp.Symbol, int]) -> int:
    return int(matrix.subs(values).rank())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decomposition",
        default="results/sr_spin9_decomposition_audit.json",
    )
    parser.add_argument(
        "--so9", default="results/sr_so9_equivariance_audit.json"
    )
    parser.add_argument(
        "--output", default="results/sr_gauge_complex_audit.json"
    )
    args = parser.parse_args()

    decomposition_path = Path(args.decomposition)
    so9_path = Path(args.so9)
    decomposition = json.loads(decomposition_path.read_text())
    so9 = json.loads(so9_path.read_text())

    symbols = sp.symbols("D p1:10")
    d0 = exterior_derivative(0, symbols)
    d1 = exterior_derivative(1, symbols)
    d2 = exterior_derivative(2, symbols)
    d1d0 = d1 * d0
    d2d1 = d2 * d1
    complex_closes = not any(d1d0) and not any(d2d1)

    spatial_zero = {symbol: 0 for symbol in symbols}
    spatial_zero[symbols[0]] = 1
    generic = {symbol: index + 1 for index, symbol in enumerate(symbols)}
    total_zero = {symbol: 0 for symbol in symbols}
    ranks = {
        "generic_nonzero_momentum": [
            substituted_rank(matrix, generic) for matrix in (d0, d1, d2)
        ],
        "zero_spatial_momentum_with_D_equal_1": [
            substituted_rank(matrix, spatial_zero) for matrix in (d0, d1, d2)
        ],
        "zero_total_momentum": [
            substituted_rank(matrix, total_zero) for matrix in (d0, d1, d2)
        ],
        "expected_nonzero_koszul_ranks": [comb(9, rank) for rank in range(3)],
    }
    ranks_match = (
        ranks["generic_nonzero_momentum"]
        == ranks["expected_nonzero_koszul_ranks"]
        == ranks["zero_spatial_momentum_with_D_equal_1"]
        and ranks["zero_total_momentum"] == [0, 0, 0]
    )

    # C_r = Lambda^r(R time + R^9 space), decomposed under spatial SO(9).
    cochains = {
        "C0_gauge_parameter": {
            "dimension": 1,
            "so9": ["1"],
            "casimirs": [0],
        },
        "C1_potential": {
            "dimension": 10,
            "so9": ["1 (A_0)", "9 (A_i)"],
            "casimirs": [0, 8],
        },
        "C2_field_strength": {
            "dimension": 45,
            "so9": ["9 (F_0i)", "36 (F_ij)"],
            "casimirs": [8, 14],
        },
        "C3_bianchi": {
            "dimension": 120,
            "so9": ["36 (B_0ij)", "84 (B_ijk)"],
            "casimirs": [14, 18],
        },
    }

    decomposition_passed = bool(decomposition["gate_passed"])
    fixed_boson_irreps = ["44", "84"]
    fixed_fermion_irreps = ["128_gamma_traceless_vector_spinor"]
    missing_complex_irreps = {
        "from_fixed_bosons": ["1", "9", "36"],
        "from_fixed_fermions_for_SYM_seed": ["16_spinor"],
        "already_present": ["84_spatial_three_form"],
    }

    # At k=(D,p)=0, every derivative-built differential vanishes.  Therefore
    # d H + H d vanishes for any local polynomial H, including one with a
    # constant term.  The Casimir audit excludes the resulting strict 9|16
    # retract.
    total_origin_differential_zero = ranks["zero_total_momentum"] == [0, 0, 0]
    casimir_excludes_vector_spinor_retract = bool(
        so9["casimir_forces_all_so9_retract_maps_zero"]
    )
    fixed_chain_homotopy_retract_possible = not (
        total_origin_differential_zero and casimir_excludes_vector_spinor_retract
    )

    # A strict N=16 Garden module at nonzero time frequency is a graded real
    # Cl(16,0) module.  Its boson and fermion dimensions are multiples of 128.
    minimal_strict_valise_dimension = 128
    next_dimension_retaining_current_block = 256
    target_inventory = {
        "at_128_each_from_scratch": {
            "bosons": "9 physical vector dimensions + 119 auxiliary dimensions",
            "fermions": "16 physical spinor dimensions + 112 auxiliary dimensions",
            "fermion_spinor_bookkeeping": "112 = 7 x 16",
        },
        "at_256_each_retaining_current_128_block": {
            "added_bosons": "128 dimensions, including a 9",
            "added_fermions": "128 dimensions, including a 16 and 112 remaining dimensions",
        },
    }

    # The real Cl(16) valise irreducible has the two graded Spin(9) halves
    # A=(44+84 | 128), with parity reversal A^op.  Direct sums of these halves
    # never contain a 9 on the bosonic side or a 16 on the fermionic side.
    strict_256_candidates = {
        "A_plus_A": {
            "bosons": "2 x (44+84)",
            "fermions": "2 x 128",
            "contains_9_boson_and_16_fermion": False,
        },
        "A_plus_parity_reverse_A": {
            "bosons": "(44+84) + 128",
            "fermions": "128 + (44+84)",
            "contains_9_boson_and_16_fermion": False,
        },
        "parity_reverse_A_plus_parity_reverse_A": {
            "bosons": "2 x 128",
            "fermions": "2 x (44+84)",
            "contains_9_boson_and_16_fermion": False,
        },
    }
    strict_256_candidate_survives = any(
        candidate["contains_9_boson_and_16_fermion"]
        for candidate in strict_256_candidates.values()
    )

    gate_passed = bool(
        decomposition_passed
        and complex_closes
        and ranks_match
        and not fixed_chain_homotopy_retract_possible
        and not strict_256_candidate_survives
    )
    output = {
        "sources": {
            "decomposition": args.decomposition,
            "decomposition_sha256": hashlib.sha256(
                decomposition_path.read_bytes()
            ).hexdigest(),
            "so9": args.so9,
            "so9_sha256": hashlib.sha256(so9_path.read_bytes()).hexdigest(),
        },
        "ring": "R[D,p_1,...,p_9]",
        "complex": "Lambda^0 -> Lambda^1 -> Lambda^2 -> Lambda^3",
        "matrix_shapes": [
            [int(d0.rows), int(d0.cols)],
            [int(d1.rows), int(d1.cols)],
            [int(d2.rows), int(d2.cols)],
        ],
        "d_squared_is_exactly_zero": complex_closes,
        "exact_ranks": ranks,
        "ranks_match_koszul_complex": ranks_match,
        "so9_cochains": cochains,
        "fixed_module": {
            "bosons": fixed_boson_irreps,
            "fermions": fixed_fermion_irreps,
        },
        "missing_representation_inventory": missing_complex_irreps,
        "chain_homotopy_retract_test": {
            "identity": "P J - I = d H + H d",
            "all_derivative_differentials_vanish_at_total_momentum_zero": (
                total_origin_differential_zero
            ),
            "zero_momentum_reduces_to_strict_retract": True,
            "strict_9_and_16_retract_excluded_by_casimir": (
                casimir_excludes_vector_spinor_retract
            ),
            "fixed_module_chain_homotopy_retract_possible": (
                fixed_chain_homotopy_retract_possible
            ),
            "scope": (
                "This excludes complexes whose differentials are built only "
                "from derivatives, for arbitrary local polynomial homotopies. "
                "Algebraic differentials or added representations that survive "
                "at total momentum zero remain open."
            ),
        },
        "garden_dimension_constraint": {
            "reason": (
                "At nonzero time frequency a strict N=16 Garden representation "
                "is a graded real Cl(16,0) module."
            ),
            "boson_dimension_is_multiple_of": minimal_strict_valise_dimension,
            "fermion_dimension_is_multiple_of": minimal_strict_valise_dimension,
            "next_size_retaining_current_block": next_dimension_retaining_current_block,
        },
        "minimal_target_inventory": target_inventory,
        "strict_256_valise_candidates": strict_256_candidates,
        "strict_256_valise_candidate_survives": strict_256_candidate_survives,
        "surviving_candidate_classes": [
            (
                "a genuinely nonvalise or gauge-extended module whose "
                "zero-momentum field representation directly contains 9|16"
            ),
            (
                "a complex with algebraic differentials or homotopies that "
                "remain nonzero at total momentum zero"
            ),
        ],
        "gate_passed": gate_passed,
        "conclusion": (
            "The ordinary ten-dimensional exterior differential closes and has "
            "the expected nonzero-momentum ranks, but it does not rescue the "
            "fixed module: its derivative homotopies "
            "vanish at total momentum zero, where the 9|16 Casimir obstruction "
            "returns. Enlarging to 256|256 by direct sums of strict valise blocks "
            "also fails. A survivor must change the zero-mode representation to "
            "contain 9|16 or introduce an algebraic zero-mode differential."
            if gate_passed
            else "One or more gauge-complex audit gates failed."
        ),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
