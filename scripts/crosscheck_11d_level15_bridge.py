#!/usr/bin/env python3
"""Independent numerical cross-check of the Rust level-15 bridge systems.

This script deliberately rebuilds the sparse raising equations without using
the Rust implementation.  It is not the primary verifier.  Its floating-point
eigenvalues check the expected kernel dimensions and spectral separation.
"""

import argparse
import collections
import itertools
import json
import math
import os

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


ROOTS = [
    (2, -2, 0, 0, 0),
    (0, 2, -2, 0, 0),
    (0, 0, 2, -2, 0),
    (0, 0, 0, 2, -2),
    (0, 0, 0, 0, 2),
]
CASES = {
    "00001": ((1, 1, 1, 1, 1), 2),
    "10001": ((3, 1, 1, 1, 1), 1),
}


def half_groups(weights):
    groups = collections.defaultdict(list)
    for mask in range(1 << 16):
        total = tuple(
            sum(weight[axis] for bit, weight in enumerate(weights) if mask >> bit & 1)
            for axis in range(5)
        )
        groups[(mask.bit_count(), total)].append(mask)
    return groups


def weight_basis(target, left, right):
    basis = []
    for (left_degree, left_weight), left_masks in left.items():
        right_masks = right.get(
            (
                15 - left_degree,
                tuple(target[axis] - left_weight[axis] for axis in range(5)),
            ),
            (),
        )
        for left_mask in left_masks:
            basis.extend(left_mask | (right_mask << 16) for right_mask in right_masks)
    return sorted(basis)


def lower_index(weight, root, weight_index):
    lower = list(weight)
    if root < 4:
        if lower[root] != 1 or lower[root + 1] != -1:
            return None
        lower[root], lower[root + 1] = -1, 1
    else:
        if lower[4] != 1:
            return None
        lower[4] = -1
    return weight_index[tuple(lower)]


def build_matrix(label, left, right, weights, weight_index):
    highest_weight, expected_nullity = CASES[label]
    source = weight_basis(highest_weight, left, right)
    columns = {mask: index for index, mask in enumerate(source)}
    row_indices = []
    column_indices = []
    values = []
    row_offset = 0
    block_rows = []

    for root, simple_root in enumerate(ROOTS):
        output_weight = tuple(
            highest_weight[axis] + simple_root[axis] for axis in range(5)
        )
        output = weight_basis(output_weight, left, right)
        block_rows.append(len(output))
        for local_row, output_mask in enumerate(output):
            for upper_index, weight in enumerate(weights):
                if not (output_mask >> upper_index & 1):
                    continue
                lower = lower_index(weight, root, weight_index)
                if lower is None or output_mask >> lower & 1:
                    continue
                source_mask = (output_mask ^ (1 << upper_index)) | (1 << lower)
                low, high = sorted((lower, upper_index))
                interval = ((1 << high) - 1) ^ ((1 << (low + 1)) - 1)
                parity = (source_mask & interval).bit_count() & 1
                row_indices.append(row_offset + local_row)
                column_indices.append(columns[source_mask])
                values.append(-1.0 if parity else 1.0)
        row_offset += len(output)

    matrix = sp.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(row_offset, len(source)),
        dtype=np.float64,
    )
    return matrix, block_rows, expected_nullity


def smallest_eigenvalues(matrix, count, seed, tolerance, iterations):
    columns = matrix.shape[1]

    def multiply(vector):
        return matrix.T @ (matrix @ vector)

    operator = sla.LinearOperator(
        (columns, columns), matvec=multiply, matmat=multiply, dtype=np.float64
    )
    diagonal = np.asarray(matrix.power(2).sum(axis=0)).ravel()

    def precondition(vector):
        denominator = diagonal[:, None] if vector.ndim == 2 else diagonal
        return vector / np.maximum(denominator, 1.0)

    preconditioner = sla.LinearOperator(
        (columns, columns),
        matvec=precondition,
        matmat=precondition,
        dtype=np.float64,
    )
    initial = np.random.default_rng(seed).normal(size=(columns, count))
    eigenvalues, eigenvectors = sla.lobpcg(
        operator,
        initial,
        M=preconditioner,
        largest=False,
        tol=tolerance,
        maxiter=iterations,
    )
    order = np.argsort(eigenvalues)
    return [float(eigenvalues[index]) for index in order], eigenvectors[:, order]


def extract_integer_kernels(label, matrix, eigenvectors, output_directory):
    expected_nullity = CASES[label][1]
    if expected_nullity == 1:
        normalized = eigenvectors[:, :1] / np.max(np.abs(eigenvectors[:, 0]))
        scales = [1320]
    else:
        _, _, pivots = la.qr(
            eigenvectors[:, :expected_nullity].T,
            pivoting=True,
            mode="economic",
        )
        pivot_block = eigenvectors[pivots[:expected_nullity], :expected_nullity]
        normalized = eigenvectors[:, :expected_nullity] @ np.linalg.inv(pivot_block)
        scales = [7920, 1]

    os.makedirs(output_directory, exist_ok=True)
    artifacts = []
    for index, scale in enumerate(scales):
        approximation = normalized[:, index] * scale
        coefficients = np.rint(approximation).astype(np.int64)
        maximum_rounding_residual = float(np.max(np.abs(approximation - coefficients)))
        if maximum_rounding_residual > 1e-5:
            raise RuntimeError(
                f"integer reconstruction residual {maximum_rounding_residual} is too large"
            )
        coefficient_gcd = 0
        for coefficient in coefficients:
            coefficient_gcd = math.gcd(coefficient_gcd, abs(int(coefficient)))
        coefficients //= coefficient_gcd
        if coefficients.min() < -32768 or coefficients.max() > 32767:
            raise RuntimeError("integer kernel does not fit signed 16-bit storage")
        residual = matrix @ coefficients
        if np.count_nonzero(residual):
            raise RuntimeError("reconstructed integer vector has a nonzero raising residual")
        path = os.path.join(
            output_directory,
            f"{label}_highest_weight_kernel_{index + 1}.i16le"
            if expected_nullity > 1
            else f"{label}_highest_weight_kernel.i16le",
        )
        coefficients.astype("<i2").tofile(path)
        artifacts.append(
            {
                "path": path,
                "scale_before_primitive_reduction": scale,
                "coefficient_gcd_removed": coefficient_gcd,
                "maximum_rounding_residual": maximum_rounding_residual,
                "nonzero_coefficients": int(np.count_nonzero(coefficients)),
                "minimum_coefficient": int(coefficients.min()),
                "maximum_coefficient": int(coefficients.max()),
            }
        )
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["00001", "10001"], required=True)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--vectors-output")
    parser.add_argument("--integer-artifact-directory")
    args = parser.parse_args()

    weights = list(itertools.product([1, -1], repeat=5))
    weight_index = {weight: index for index, weight in enumerate(weights)}
    left = half_groups(weights[:16])
    right = half_groups(weights[16:])
    matrix, block_rows, expected_nullity = build_matrix(
        args.label, left, right, weights, weight_index
    )
    eigenvalues, eigenvectors = smallest_eigenvalues(
        matrix,
        expected_nullity + 3,
        args.seed,
        args.tolerance,
        args.iterations,
    )
    threshold = 1e-6
    report = {
        "schema_version": "adynkra.11d.level15-bridge-crosscheck.v1",
        "role": "independent floating-point cross-check of the Rust exact sparse system",
        "dynkin_label": args.label,
        "matrix_rows": matrix.shape[0],
        "matrix_columns": matrix.shape[1],
        "nonzero_entries": matrix.nnz,
        "raising_block_rows": block_rows,
        "expected_kernel_dimension": expected_nullity,
        "smallest_eigenvalues_of_transpose_product": eigenvalues,
        "numerical_zero_threshold": threshold,
        "observed_numerical_kernel_dimension": sum(
            value < threshold for value in eigenvalues
        ),
        "passed": sum(value < threshold for value in eigenvalues)
        == expected_nullity,
        "boundary": "The eigenvalue calculation is numerical. It checks nullity and separation but does not supply an exact kernel basis.",
    }
    if args.integer_artifact_directory:
        report["integer_kernel_candidates"] = extract_integer_kernels(
            args.label,
            matrix,
            eigenvectors,
            args.integer_artifact_directory,
        )
        report["boundary"] = (
            "The eigenvalue calculation and integer reconstruction are numerical. "
            "The Rust verifier independently checks every stored integer coefficient "
            "against every exact raising equation."
        )
    rendered = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if args.vectors_output:
        np.savez_compressed(
            args.vectors_output,
            eigenvalues=np.asarray(eigenvalues),
            eigenvectors=eigenvectors[:, :expected_nullity],
        )
    print(rendered)


if __name__ == "__main__":
    main()
