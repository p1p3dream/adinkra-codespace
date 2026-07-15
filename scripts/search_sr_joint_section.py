#!/usr/bin/env python3
"""Alternating linear search for a joint fermion section and boson projection.

The exact one-sided witness fixes P_F and J_B. This script searches for J_F and
P_B satisfying

    P_F J_F = I_16,
    P_B J_B = I_9,
    P_B L_a J_F = G_a  for a=0..15.

Both right/left-inverse constraints are enforced by nullspace
parameterizations. Only the bilinear linkage equation is optimized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg


def dense_signed_perm(perm: list[int], sign: list[int]) -> np.ndarray:
    d = len(perm)
    out = np.zeros((d, d), dtype=np.float64)
    out[np.arange(d), np.asarray(perm)] = np.asarray(sign)
    return out


def residual_metrics(
    pb: np.ndarray, ls: list[np.ndarray], jf: np.ndarray, targets: list[np.ndarray]
) -> tuple[float, float]:
    blocks = [pb @ ls[a] @ jf - targets[a] for a in range(16)]
    flat = np.concatenate([block.ravel() for block in blocks])
    return float(np.max(np.abs(flat))), float(np.linalg.norm(flat))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/sr_hole_minimal.json")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--newton-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", default="results/sr_joint_section_search.json")
    parser.add_argument(
        "--candidate-output",
        default="results/sr_joint_section_candidate.npz",
        help="best numerical P_B/J_F candidate",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text())
    topology = next(
        t
        for t in report["topologies"]
        if t["auxiliary_projection"].get("witness")
    )
    witness = topology["auxiliary_projection"]["witness"]
    data = witness["joint_section_data"]

    ls = [
        dense_signed_perm(perm, sign)
        for perm, sign in zip(data["l_perm"], data["l_sign"])
    ]
    gammas = [
        dense_signed_perm(perm, sign)
        for perm, sign in zip(data["gamma_perm"], data["gamma_sign"])
    ]
    targets = [
        np.stack([gammas[i][charge, :] for i in range(9)], axis=0)
        for charge in range(16)
    ]

    pf = np.zeros((16, 128), dtype=np.float64)
    fibers: list[list[tuple[int, int]]] = [[] for _ in range(16)]
    for entry in witness["projection"]:
        beta = entry["physical_beta"]
        full_f = entry["full_fermion"]
        sign = entry["sign"]
        pf[beta, full_f] = sign
        fibers[beta].append((full_f, sign))
    assert all(len(fiber) == 8 for fiber in fibers)

    jb = np.zeros((128, 9), dtype=np.float64)
    for i, full_b in enumerate(witness["boson_by_vector"]):
        jb[full_b, i] = 1.0

    # Exact affine right inverse J_F = J0 + N_F Z.
    j0 = pf.T / 8.0
    nf = np.zeros((128, 112), dtype=np.float64)
    col = 0
    for fiber in fibers:
        f0, s0 = fiber[0]
        for fk, sk in fiber[1:]:
            nf[fk, col] = 1.0
            nf[f0, col] = -sk / s0
            col += 1
    assert col == 112 and np.max(np.abs(pf @ nf)) == 0.0
    assert np.max(np.abs(pf @ j0 - np.eye(16))) == 0.0

    # Exact affine left inverse P_B = P0 + Z_B E^T. E has support only away
    # from the nine selected physical boson coordinates.
    physical_bosons = witness["boson_by_vector"]
    p0 = jb.T.copy()
    complement = [i for i in range(128) if i not in set(physical_bosons)]
    eb = np.eye(128, dtype=np.float64)[:, complement]
    assert np.max(np.abs(eb.T @ jb)) == 0.0
    assert np.max(np.abs(p0 @ jb - np.eye(9))) == 0.0

    target_all = np.hstack(targets)
    rng = np.random.default_rng(args.seed)
    starts = []
    best = None
    for start in range(args.starts):
        if start == 0:
            jf = j0.copy()
        else:
            jf = j0 + nf @ (0.05 * rng.standard_normal((112, 16)))
        history = []
        pb = p0.copy()
        for iteration in range(args.iterations):
            # P_B step, preserving P_B J_B = I exactly.
            linkage_inputs = np.hstack([ls[a] @ jf for a in range(16)])
            x = eb.T @ linkage_inputs
            rhs = target_all - p0 @ linkage_inputs
            zbt, *_ = scipy.linalg.lstsq(x.T, rhs.T, lapack_driver="gelsy")
            pb = p0 + zbt.T @ eb.T

            # J_F step, preserving P_F J_F = I exactly.
            m = np.vstack([pb @ ls[a] for a in range(16)])
            rhs_j = np.vstack([targets[a] for a in range(16)]) - m @ j0
            zf, *_ = scipy.linalg.lstsq(m @ nf, rhs_j, lapack_driver="gelsy")
            jf = j0 + nf @ zf

            max_residual, frobenius = residual_metrics(pb, ls, jf, targets)
            history.append(frobenius)
            if frobenius < 1e-10:
                break
            if len(history) > 6 and abs(history[-1] - history[-6]) < 1e-12:
                break

        max_residual, frobenius = residual_metrics(pb, ls, jf, targets)
        inverse_f = float(np.max(np.abs(pf @ jf - np.eye(16))))
        inverse_b = float(np.max(np.abs(pb @ jb - np.eye(9))))
        start_result = {
            "start": start,
            "iterations": len(history),
            "max_linkage_residual": max_residual,
            "frobenius_linkage_residual": frobenius,
            "fermion_right_inverse_residual": inverse_f,
            "boson_left_inverse_residual": inverse_b,
        }
        starts.append(start_result)
        if best is None or frobenius < best[0]:
            best = (frobenius, start_result, pb.copy(), jf.copy())

    assert best is not None
    # Newton correction in the affine coordinates. The Jacobian is rectangular
    # (2304 equations, 2863 variables); full row rank means the linkage map is
    # locally onto at the candidate.
    pb = best[2]
    jf = best[3]
    zb = (pb - p0) @ eb
    zf, *_ = scipy.linalg.lstsq(nf, jf - j0, lapack_driver="gelsy")
    newton_history = []
    jacobian_rank = None
    for _ in range(args.newton_steps):
        residual_blocks = [pb @ ls[a] @ jf - targets[a] for a in range(16)]
        residual = np.concatenate([block.ravel() for block in residual_blocks])
        jacobian = np.zeros((16 * 9 * 16, 9 * 119 + 112 * 16), dtype=np.float64)
        for charge in range(16):
            left = eb.T @ ls[charge] @ jf
            right = pb @ ls[charge] @ nf
            for i in range(9):
                for beta in range(16):
                    row = charge * 9 * 16 + i * 16 + beta
                    jacobian[row, i * 119 : (i + 1) * 119] = left[:, beta]
                    base = 9 * 119
                    jacobian[row, base + beta : base + 112 * 16 : 16] = right[i, :]
        delta, _, jacobian_rank, _ = scipy.linalg.lstsq(
            jacobian, -residual, lapack_driver="gelsy"
        )
        zb += delta[: 9 * 119].reshape(9, 119)
        zf += delta[9 * 119 :].reshape(112, 16)
        pb = p0 + zb @ eb.T
        jf = j0 + nf @ zf
        newton_history.append(residual_metrics(pb, ls, jf, targets))

    max_residual, frobenius = residual_metrics(pb, ls, jf, targets)
    best[1].update(
        {
            "max_linkage_residual_after_newton": max_residual,
            "frobenius_linkage_residual_after_newton": frobenius,
            "newton_steps": args.newton_steps,
            "jacobian_rank": int(jacobian_rank) if jacobian_rank is not None else None,
            "jacobian_rows": 16 * 9 * 16,
            "jacobian_columns": 9 * 119 + 112 * 16,
            "newton_history": [
                {"max_residual": maximum, "frobenius_residual": frob}
                for maximum, frob in newton_history
            ],
        }
    )
    output = {
        "source": args.input,
        "catalog_index": topology["catalog_index"],
        "ansatz": "general real J_F and P_B with exact affine inverse constraints",
        "method": "deterministic alternating constrained least squares",
        "seed": args.seed,
        "starts": args.starts,
        "iteration_limit": args.iterations,
        "best": best[1],
        "numerical_candidate_found": max_residual < 1e-10,
        "exact_certificate_found": False,
        "candidate_file": args.candidate_output,
        "all_starts": starts,
    }
    np.savez_compressed(
        args.candidate_output,
        boson_projection=pb,
        fermion_section=jf,
        fermion_projection=pf,
        boson_injection=jb,
        boson_affine=zb,
        fermion_affine=zf,
        boson_complement=np.asarray(complement, dtype=np.int64),
    )
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
