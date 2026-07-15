#!/usr/bin/env python3
"""Exact verifier for the uniform-section Q-lambda linear system.

Reads the integer constraint certificate emitted by the minimal-valise Rust
analysis, computes ranks
over QQ with SymPy, and, when consistent, emits an exact rational boson
projection P_B with an entrywise verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp
from sympy import ZZ
from sympy.polys.matrices import DomainMatrix


def rational_token(value: sp.Expr) -> str:
    value = sp.cancel(value)
    if value.q == 1:
        return str(value.p)
    return f"{value.p}/{value.q}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default="results/sr_hole_minimal.json",
        help="minimal-valise JSON certificate",
    )
    parser.add_argument(
        "--output",
        default="results/sr_uniform_section_audit.json",
        help="exact rank/solution audit output",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text())
    audits = []
    for topology in report["topologies"]:
        witness = topology["auxiliary_projection"].get("witness")
        system = witness.get("uniform_section_system") if witness else None
        if not system:
            continue

        m = system["equations"]
        n = system["full_boson_variables"]
        outputs = system["output_components"]
        a_rows = [[0] * n for _ in range(m)]
        t_rows = [[0] * outputs for _ in range(m)]
        for row_index, row in enumerate(system["rows"]):
            for entry in row["input"]:
                a_rows[row_index][entry["index"]] = entry["coefficient"]
            t_rows[row_index] = row["target"]

        # DomainMatrix uses exact fraction-free/domain-aware elimination and is
        # dramatically faster here than generic symbolic Matrix.rank().
        a_domain = DomainMatrix.from_list(a_rows, ZZ)
        augmented_rows = [a_rows[i] + t_rows[i] for i in range(m)]
        rank_a = a_domain.rank()
        rank_augmented = DomainMatrix.from_list(augmented_rows, ZZ).rank()
        per_output_augmented_ranks = [
            DomainMatrix.from_list(
                [a_rows[i] + [t_rows[i][output]] for i in range(m)], ZZ
            ).rank()
            for output in range(outputs)
        ]
        consistent = rank_a == rank_augmented
        audit = {
            "catalog_index": topology["catalog_index"],
            "section": "J_F = (1/8) P_F^T",
            "right_inverse_verified_by_rust": system["right_inverse_verified"],
            "equations": m,
            "unknowns_per_output": n,
            "outputs": outputs,
            "rank_a_over_Q": rank_a,
            "rank_augmented_over_Q": rank_augmented,
            "rank_augmented_per_output_over_Q": per_output_augmented_ranks,
            "consistent": consistent,
        }

        if consistent:
            a = sp.Matrix(a_rows)
            target = sp.Matrix(t_rows)
            solution, parameters = a.gauss_jordan_solve(target)
            substitutions = {symbol: 0 for symbol in solution.free_symbols}
            solution = solution.xreplace(substitutions)
            residual = a * solution - target
            exact = all(value == 0 for value in residual)
            sparse_projection = []
            for full_boson in range(n):
                for physical_vector in range(outputs):
                    value = solution[full_boson, physical_vector]
                    if value != 0:
                        sparse_projection.append(
                            {
                                "full_boson": full_boson,
                                "physical_vector": physical_vector,
                                "coefficient": rational_token(value),
                            }
                        )
            audit.update(
                {
                    "exact_solution_verified": exact,
                    "free_parameters_set_to_zero": len(parameters),
                    "nonzero_projection_entries": len(sparse_projection),
                    "boson_projection": sparse_projection,
                }
            )
        audits.append(audit)

    output = {
        "source": args.input,
        "method": "exact rational rank and Gauss-Jordan elimination over QQ",
        "audits": audits,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
