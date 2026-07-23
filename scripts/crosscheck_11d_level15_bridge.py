#!/usr/bin/env python3
"""Independent numerical cross-check of the Rust bridge source systems.

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
    (15, "00001"): ((1, 1, 1, 1, 1), 2),
    (15, "10001"): ((3, 1, 1, 1, 1), 1),
    (13, "00001"): ((1, 1, 1, 1, 1), 1),
    (13, "01001"): ((3, 3, 1, 1, 1), 2),
    (16, "10000"): ((2, 0, 0, 0, 0), 1),
    (16, "20000"): ((4, 0, 0, 0, 0), 1),
    (16, "00100"): ((2, 2, 2, 0, 0), 2),
    (16, "00010"): ((2, 2, 2, 2, 0), 2),
    (16, "00002"): ((2, 2, 2, 2, 2), 1),
    (16, "10100"): ((4, 2, 2, 0, 0), 1),
    (16, "10010"): ((4, 2, 2, 2, 0), 1),
    (16, "10002"): ((4, 2, 2, 2, 2), 3),
    (17, "10001"): ((3, 1, 1, 1, 1), 1),
    (17, "01001"): ((3, 3, 1, 1, 1), 2),
    (17, "20001"): ((5, 1, 1, 1, 1), 1),
    (17, "11001"): ((5, 3, 1, 1, 1), 3),
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


def weight_basis(degree, target, left, right):
    basis = []
    for (left_degree, left_weight), left_masks in left.items():
        if left_degree > degree or degree - left_degree > 16:
            continue
        right_masks = right.get(
            (
                degree - left_degree,
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


def build_matrix(degree, label, left, right, weights, weight_index):
    highest_weight, expected_nullity = CASES[(degree, label)]
    source = weight_basis(degree, highest_weight, left, right)
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
        output = weight_basis(degree, output_weight, left, right)
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


def discover_integer_scale(vector, maximum=32767, tolerance=2e-5):
    """Find the smallest scale that makes a normalized kernel integral."""
    active = np.flatnonzero(np.abs(vector) > 1e-7)
    if active.size > 2048:
        active = active[np.linspace(0, active.size - 1, 2048, dtype=np.int64)]
    sample = vector[active]
    best = None
    for start in range(1, maximum + 1, 1024):
        scales = np.arange(start, min(start + 1024, maximum + 1), dtype=np.float64)
        products = scales[:, None] * sample[None, :]
        residuals = np.max(np.abs(products - np.rint(products)), axis=1)
        for local in np.flatnonzero(residuals < tolerance):
            scale = int(scales[local])
            full = vector * scale
            maximum_residual = float(np.max(np.abs(full - np.rint(full))))
            if maximum_residual < tolerance:
                return scale
            if best is None or maximum_residual < best[1]:
                best = (scale, maximum_residual)
    detail = "none" if best is None else f"{best[0]} with residual {best[1]}"
    raise RuntimeError(f"no integral scale through {maximum}; best candidate {detail}")


def integer_candidate(vector, maximum=32767, tolerance=2e-5):
    normalized = vector / np.max(np.abs(vector))
    scale = discover_integer_scale(normalized, maximum, tolerance)
    approximation = normalized * scale
    coefficients = np.rint(approximation).astype(np.int64)
    if np.max(np.abs(approximation - coefficients)) > tolerance:
        raise RuntimeError("integer candidate exceeded the rounding tolerance")
    coefficient_gcd = 0
    for coefficient in coefficients:
        coefficient_gcd = math.gcd(coefficient_gcd, abs(int(coefficient)))
    return coefficients // coefficient_gcd


def extract_rank_three_clustered_basis(eigenvectors):
    """Recover the level-17 (11001) lattice from its numerical nullspace."""
    basis = eigenvectors[:, :3]
    _, _, pivots = la.qr(basis.T, pivoting=True, mode="economic")
    normalized = basis @ np.linalg.inv(basis[pivots[:3], :])

    first = None
    for column in normalized.T:
        try:
            first = integer_candidate(column)
            break
        except RuntimeError:
            continue
    if first is None:
        raise RuntimeError("no first integer direction in the rank-three kernel")

    first_coordinates = basis.T @ first
    complement_coordinates = la.null_space(first_coordinates.reshape(1, 3))
    complement = basis @ complement_coordinates
    angles = np.mod(np.arctan2(complement[:, 1], complement[:, 0]), np.pi)
    active = np.linalg.norm(complement, axis=1) > 1e-10
    rounded = np.round(angles[active], 6)
    values, counts = np.unique(rounded, return_counts=True)

    for cluster in np.argsort(counts)[::-1][:32]:
        selected = active & (np.abs(angles - values[cluster]) < 2e-6)
        rows = complement[selected].copy()
        reference = rows[0]
        rows[np.sum(rows * reference, axis=1) < 0] *= -1
        row = rows.mean(axis=0)
        for direction in (np.asarray([-row[1], row[0]]), row):
            try:
                second = integer_candidate(complement @ direction)
            except RuntimeError:
                continue
            coordinate_constraints = np.vstack(
                (first_coordinates, basis.T @ second)
            )
            remaining_coordinates = la.null_space(coordinate_constraints)
            if remaining_coordinates.shape != (3, 1):
                continue
            try:
                third = integer_candidate(
                    basis @ remaining_coordinates[:, 0]
                )
            except RuntimeError:
                continue
            integers = np.column_stack((first, second, third))
            if np.linalg.matrix_rank(basis.T @ integers) == 3:
                return integers
    raise RuntimeError("the rank-three clustered lattice recovery failed")


def extract_integer_kernels(degree, label, matrix, eigenvectors, output_directory):
    expected_nullity = CASES[(degree, label)][1]
    if (degree, label) == (17, "11001"):
        normalized = extract_rank_three_clustered_basis(eigenvectors).astype(np.float64)
        scales = [1] * expected_nullity
    elif expected_nullity == 1:
        normalized = eigenvectors[:, :1] / np.max(np.abs(eigenvectors[:, 0]))
        scales = [discover_integer_scale(normalized[:, 0])]
    elif (degree, label) == (13, "01001"):
        angles = np.mod(np.arctan2(eigenvectors[:, 1], eigenvectors[:, 0]), np.pi)
        active = np.linalg.norm(eigenvectors[:, :2], axis=1) > 1e-10
        rounded = np.round(angles[active], 6)
        values, counts = np.unique(rounded, return_counts=True)
        dominant = values[np.argmax(counts)]
        rows = eigenvectors[np.abs(angles - dominant) < 2e-6, :2].copy()
        reference = rows[0]
        rows[np.sum(rows * reference, axis=1) < 0] *= -1
        row = rows.mean(axis=0)
        sparse_direction = np.asarray([-row[1], row[0]])
        sparse = eigenvectors[:, :2] @ sparse_direction
        sparse /= np.max(np.abs(sparse))
        sparse_scale = discover_integer_scale(sparse)
        sparse_integer = np.rint(sparse * sparse_scale)

        coordinates = eigenvectors[:, :2].T @ sparse_integer
        complementary_direction = np.asarray([-coordinates[1], coordinates[0]])
        complementary = eigenvectors[:, :2] @ complementary_direction
        complementary /= np.max(np.abs(complementary))
        normalized = np.column_stack((sparse, complementary))
        scales = [sparse_scale, discover_integer_scale(complementary)]
    else:
        _, _, pivots = la.qr(
            eigenvectors[:, :expected_nullity].T,
            pivoting=True,
            mode="economic",
        )
        pivot_block = eigenvectors[pivots[:expected_nullity], :expected_nullity]
        normalized = eigenvectors[:, :expected_nullity] @ np.linalg.inv(pivot_block)
        scales = [discover_integer_scale(normalized[:, index]) for index in range(expected_nullity)]

    os.makedirs(output_directory, exist_ok=True)
    artifacts = []
    for index, scale in enumerate(scales):
        approximation = normalized[:, index] * scale
        coefficients = np.rint(approximation).astype(np.int64)
        maximum_rounding_residual = float(np.max(np.abs(approximation - coefficients)))
        if maximum_rounding_residual > 2e-5:
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
        prefix = f"level{degree}_" if degree != 15 else ""
        path = os.path.join(
            output_directory,
            f"{prefix}{label}_highest_weight_kernel_{index + 1}.i16le"
            if expected_nullity > 1
            else f"{prefix}{label}_highest_weight_kernel.i16le",
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
    parser.add_argument("--degree", type=int, choices=[13, 15, 16, 17], default=15)
    parser.add_argument(
        "--label",
        choices=sorted({label for _, label in CASES}),
        required=True,
    )
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
    if (args.degree, args.label) not in CASES:
        parser.error(f"label {args.label} is not configured at degree {args.degree}")
    matrix, block_rows, expected_nullity = build_matrix(
        args.degree, args.label, left, right, weights, weight_index
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
        "schema_version": "adynkra.11d.bridge-crosscheck.v2",
        "role": "independent floating-point cross-check of the Rust exact sparse system",
        "dynkin_label": args.label,
        "exterior_degree": args.degree,
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
    if args.vectors_output:
        np.savez_compressed(
            args.vectors_output,
            eigenvalues=np.asarray(eigenvalues),
            eigenvectors=eigenvectors[:, :expected_nullity],
        )
    if args.integer_artifact_directory:
        report["integer_kernel_candidates"] = extract_integer_kernels(
            args.degree,
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
    print(rendered)


if __name__ == "__main__":
    main()
