#!/usr/bin/env python3
"""Generate exact level-12 kernels needed by the second-momentum 11D bridge.

The candidate list is the representation-theoretic preimage of
``Sym^2(V) tensor (10001)`` under the minuscule spinor product at scalar
superfield level 12.  The calculation uses sparse row echelon reduction over
the Mersenne prime 2^31-1, rational reconstruction, and a full integer raising
residual check.  No modular result is accepted without the integer check.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import struct
import tempfile
import time
from fractions import Fraction
from pathlib import Path

PRIME = 2_147_483_647
DEGREE = 12
ROOTS = (
    (2, -2, 0, 0, 0),
    (0, 2, -2, 0, 0),
    (0, 0, 2, -2, 0),
    (0, 0, 0, 2, -2),
    (0, 0, 0, 0, 2),
)

# Multiplicities are the published level-12 scalar-superfield multiplicities.
SPECS = {
    "00000": 1,
    "00010": 1,
    "00100": 1,
    "01002": 4,
    "01100": 2,
    "02000": 2,
    "10002": 2,
    "11002": 5,
    "11010": 4,
    "11100": 3,
    "12000": 1,
    "20002": 2,
    "20010": 3,
    "20100": 2,
    "30002": 3,
    "30010": 2,
    "30100": 1,
    "31000": 1,
    "40000": 1,
}

WEIGHTS = tuple(
    tuple(1 if ((index >> (4 - axis)) & 1) == 0 else -1 for axis in range(5))
    for index in range(32)
)
WEIGHT_INDEX = {weight: index for index, weight in enumerate(WEIGHTS)}
RAISE = tuple(
    tuple(
        WEIGHT_INDEX.get(
            tuple(WEIGHTS[index][axis] + ROOTS[root][axis] for axis in range(5)),
            -1,
        )
        for index in range(32)
    )
    for root in range(5)
)


def highest(label: str) -> tuple[int, ...]:
    digits = tuple(map(int, label))
    return tuple(2 * sum(digits[index:4]) + digits[4] for index in range(5))


def half_groups(offset: int):
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    half_weights = WEIGHTS[offset : offset + 16]
    for mask in range(1 << 16):
        weight = [0] * 5
        remainder = mask
        while remainder:
            bit = (remainder & -remainder).bit_length() - 1
            remainder &= remainder - 1
            for axis in range(5):
                weight[axis] += half_weights[bit][axis]
        groups.setdefault((mask.bit_count(), tuple(weight)), []).append(mask)
    return groups


def weight_basis(degree: int, target, left, right):
    result: list[int] = []
    for left_degree in range(max(0, degree - 16), min(16, degree) + 1):
        right_degree = degree - left_degree
        for (candidate_degree, left_weight), left_masks in left.items():
            if candidate_degree != left_degree:
                continue
            needed = tuple(target[axis] - left_weight[axis] for axis in range(5))
            right_masks = right.get((right_degree, needed))
            if right_masks:
                result.extend(x | (y << 16) for x in left_masks for y in right_masks)
    result.sort()
    return result


def raising_rows(label: str, left, right):
    target = highest(label)
    source = weight_basis(DEGREE, target, left, right)
    blocks = []
    row_count = 0
    for root in range(5):
        output_weight = tuple(target[axis] + ROOTS[root][axis] for axis in range(5))
        basis = weight_basis(DEGREE, output_weight, left, right)
        blocks.append({mask: index + row_count for index, mask in enumerate(basis)})
        row_count += len(basis)
    rows = [{} for _ in range(row_count)]
    for column, mask in enumerate(source):
        for root in range(5):
            for lower in range(32):
                upper = RAISE[root][lower]
                if upper < 0 or not (mask >> lower & 1) or mask >> upper & 1:
                    continue
                output = (mask ^ (1 << lower)) | (1 << upper)
                low, high = sorted((lower, upper))
                interval = (
                    0
                    if high == low + 1
                    else ((1 << high) - 1) ^ ((1 << (low + 1)) - 1)
                )
                sign = -1 if (mask & interval).bit_count() % 2 else 1
                rows[blocks[root][output]][column] = sign
    return source, rows


def sparse_echelon(rows, prime: int):
    pivots = {}
    maximum_pivot_width = 0
    for source_row in sorted(rows, key=len):
        row = {column: value % prime for column, value in source_row.items()}
        while row:
            column = min(row)
            coefficient = row[column]
            pivot = pivots.get(column)
            if pivot is None:
                inverse = pow(coefficient, -1, prime)
                row = {
                    key: (value * inverse) % prime
                    for key, value in row.items()
                    if (value * inverse) % prime
                }
                pivots[column] = row
                maximum_pivot_width = max(maximum_pivot_width, len(row))
                break
            for key, value in pivot.items():
                reduced = (row.get(key, 0) - coefficient * value) % prime
                if reduced:
                    row[key] = reduced
                else:
                    row.pop(key, None)
    return pivots, maximum_pivot_width


def rational_reconstruct(residue: int, modulus: int):
    residue %= modulus
    bound = math.isqrt(modulus // 2)
    old_remainder, remainder = modulus, residue
    old_denominator, denominator = 0, 1
    while abs(remainder) > bound:
        quotient = old_remainder // remainder
        old_remainder, remainder = remainder, old_remainder - quotient * remainder
        old_denominator, denominator = denominator, old_denominator - quotient * denominator
    if denominator == 0:
        return None
    if denominator < 0:
        remainder, denominator = -remainder, -denominator
    divisor = math.gcd(abs(remainder), denominator)
    numerator = remainder // divisor
    denominator //= divisor
    if (
        abs(numerator) <= bound
        and denominator <= bound
        and (residue * denominator - numerator) % modulus == 0
    ):
        return Fraction(numerator, denominator)
    return None


def primitive_integer_nullspace(rows, columns: int, pivots, prime: int):
    free = [column for column in range(columns) if column not in pivots]
    result = []
    for free_column in free:
        modular = {free_column: 1}
        for column in sorted(pivots, reverse=True):
            row = pivots[column]
            value = -sum(
                coefficient * modular.get(other, 0)
                for other, coefficient in row.items()
                if other != column
            ) % prime
            if value:
                modular[column] = value
        rationals = []
        for column in range(columns):
            value = rational_reconstruct(modular.get(column, 0), prime)
            if value is None:
                raise RuntimeError(f"rational reconstruction failed at column {column}")
            rationals.append(value)
        denominator = 1
        for value in rationals:
            denominator = math.lcm(denominator, value.denominator)
        vector = [
            value.numerator * (denominator // value.denominator) for value in rationals
        ]
        divisor = 0
        for value in vector:
            divisor = math.gcd(divisor, abs(value))
        vector = [value // divisor for value in vector]
        first = next(value for value in vector if value)
        if first < 0:
            vector = [-value for value in vector]
        for row in rows:
            residual = sum(coefficient * vector[column] for column, coefficient in row.items())
            if residual:
                raise RuntimeError(f"nonzero exact raising residual {residual}")
        result.append(vector)
    return free, result


def write_label(label: str, copies: int, left, right, root: Path):
    started = time.time()
    source, rows = raising_rows(label, left, right)
    built = time.time()
    pivots, maximum_pivot_width = sparse_echelon(rows, PRIME)
    reduced = time.time()
    free, vectors = primitive_integer_nullspace(rows, len(source), pivots, PRIME)
    verified = time.time()
    if len(vectors) != copies:
        raise RuntimeError(f"{label}: expected nullity {copies}, found {len(vectors)}")
    maximum = max(abs(value) for vector in vectors for value in vector)
    width = 2 if maximum <= 32_767 else 4
    kernel_dir = root / "data/eleven_dimensional_spinor_bridge"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    staged_outputs = []
    for copy, vector in enumerate(vectors, 1):
        suffix = "" if copies == 1 else f"_{copy}"
        path = kernel_dir / (
            f"level12_{label}_highest_weight_kernel{suffix}.i{8 * width}le"
        )
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.staged")
        with temporary.open("wb") as stream:
            for value in vector:
                stream.write(struct.pack("<h" if width == 2 else "<i", value))
        staged_outputs.append((temporary, path))
        outputs.append(
            {
                "copy": copy,
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                "bytes": temporary.stat().st_size,
                "nonzero_coefficients": sum(value != 0 for value in vector),
                "maximum_absolute_coefficient": max(map(abs, vector)),
            }
        )
    system = {
        "dynkin_label": label,
        "exterior_degree": DEGREE,
        "source_columns": len(source),
        "raising_rows": len(rows),
        "nonzero_entries": sum(map(len, rows)),
        "prime": PRIME,
        "exact_modular_rank": len(pivots),
        "exact_nullity": len(vectors),
        "free_columns": free,
        "maximum_pivot_width": maximum_pivot_width,
        "coefficient_width_bytes": width,
        "outputs": outputs,
        "seconds": {
            "matrix": built - started,
            "echelon": reduced - built,
            "reconstruct_and_integer_verify": verified - reduced,
            "total": verified - started,
        },
        "passed": True,
    }
    return system, staged_outputs


def empty_artifact():
    return {
        "schema_version": "adynkra-11d-level12-second-momentum-kernel-generation-v1",
        "role": "exact level-12 source fixtures for p^2 D^12 second-momentum operators",
        "method": "deterministic sparse echelon over 2^31-1, rational reconstruction, and full integer residual verification",
        "systems": [],
    }


def verify_outputs(artifact, root: Path):
    if artifact.get("schema_version") != empty_artifact()["schema_version"]:
        raise RuntimeError("level-12 kernel checkpoint schema mismatch")
    systems = artifact.get("systems", [])
    labels = [system["dynkin_label"] for system in systems]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate systems in level-12 kernel checkpoint")
    resolved_root = root.resolve()
    for system in systems:
        verify_system_metadata(system)
        for output in system["outputs"]:
            path = (root / output["path"]).resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError as error:
                raise RuntimeError(f"kernel path escapes repository root: {path}") from error
            if not path.exists() or path.stat().st_size != output["bytes"]:
                raise RuntimeError(f"missing or truncated kernel {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != output["sha256"]:
                raise RuntimeError(f"kernel hash mismatch {path}")


def verify_system_metadata(system):
    label = system["dynkin_label"]
    if label not in SPECS:
        raise RuntimeError(f"unexpected level-12 kernel label {label}")
    if system["exterior_degree"] != DEGREE or not system["passed"]:
        raise RuntimeError(f"invalid kernel certificate metadata for {label}")
    if system["exact_modular_rank"] + system["exact_nullity"] != system["source_columns"]:
        raise RuntimeError(f"rank-nullity mismatch for {label}")
    if system["exact_nullity"] != SPECS[label]:
        raise RuntimeError(f"published multiplicity mismatch for {label}")
    if system["coefficient_width_bytes"] not in (2, 4):
        raise RuntimeError(f"unsupported coefficient width for {label}")
    if len(system["outputs"]) != SPECS[label]:
        raise RuntimeError(f"output-count mismatch for {label}")
    expected_copies = list(range(1, SPECS[label] + 1))
    if [output["copy"] for output in system["outputs"]] != expected_copies:
        raise RuntimeError(f"copy-order mismatch for {label}")
    if len(set(system["free_columns"])) != system["exact_nullity"] or any(
        column < 0 or column >= system["source_columns"]
        for column in system["free_columns"]
    ):
        raise RuntimeError(f"invalid free-column certificate for {label}")
    width = system["coefficient_width_bytes"]
    for output in system["outputs"]:
        suffix = "" if SPECS[label] == 1 else f"_{output['copy']}"
        expected_path = (
            "data/eleven_dimensional_spinor_bridge/"
            f"level12_{label}_highest_weight_kernel{suffix}.i{8 * width}le"
        )
        if output["path"] != expected_path:
            raise RuntimeError(f"unexpected kernel path for {label} copy {output['copy']}")
        if output["bytes"] != system["source_columns"] * width:
            raise RuntimeError(f"kernel byte-count mismatch for {label}")
        if len(output["sha256"]) != 64:
            raise RuntimeError(f"invalid kernel hash metadata for {label}")


def cleanup_staged_outputs(staged_outputs):
    for temporary, _ in staged_outputs:
        temporary.unlink(missing_ok=True)


def verify_staged_outputs(system, root: Path, staged_outputs):
    verify_system_metadata(system)
    resolved_root = root.resolve()
    by_final = {final.resolve(): temporary for temporary, final in staged_outputs}
    if len(by_final) != len(system["outputs"]):
        raise RuntimeError(f"incomplete staged output set for {system['dynkin_label']}")
    for output in system["outputs"]:
        final = (root / output["path"]).resolve()
        try:
            final.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(f"kernel path escapes repository root: {final}") from error
        temporary = by_final.get(final)
        if temporary is None or not temporary.exists():
            raise RuntimeError(f"missing staged kernel for {final}")
        if temporary.stat().st_size != output["bytes"]:
            raise RuntimeError(f"truncated staged kernel {temporary}")
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != output["sha256"]:
            raise RuntimeError(f"staged kernel hash mismatch {temporary}")


def publish_checkpoint(checkpoint: Path, root: Path, staged_systems):
    """Publish staged binaries and their checkpoint under one advisory lock.

    A conflicting same-label certificate is rejected before any final binary is
    replaced.  Filesystems do not offer one transaction spanning several
    binaries and the JSON checkpoint, so new binaries are published first and
    the checkpoint last.  A crash can therefore leave unreferenced binaries,
    but it cannot leave the checkpoint referring to missing new binaries.
    """
    lock_path = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    all_staged_outputs = [item for _, staged in staged_systems for item in staged]
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            artifact = (
                json.loads(checkpoint.read_text())
                if checkpoint.exists()
                else empty_artifact()
            )
            verify_outputs(artifact, root)
            by_label = {
                system["dynkin_label"]: system
                for system in artifact.get("systems", [])
            }
            pending_by_label = {}
            pending_final_paths = set()

            # Validate every candidate and detect conflicts before publishing a
            # single final binary.
            for system, staged_outputs in staged_systems:
                verify_staged_outputs(system, root, staged_outputs)
                label = system["dynkin_label"]
                if label in pending_by_label:
                    raise RuntimeError(f"duplicate staged kernel system for {label}")
                pending_by_label[label] = system
                existing = by_label.get(label)
                if existing is not None:
                    old_certificate = [
                        (output["copy"], output["path"], output["sha256"], output["bytes"])
                        for output in existing["outputs"]
                    ]
                    new_certificate = [
                        (output["copy"], output["path"], output["sha256"], output["bytes"])
                        for output in system["outputs"]
                    ]
                    if old_certificate != new_certificate:
                        raise RuntimeError(
                            f"conflicting exact kernels for {label}"
                        )
                for temporary, final in staged_outputs:
                    resolved_final = final.resolve()
                    if resolved_final in pending_final_paths:
                        raise RuntimeError(f"duplicate staged kernel path {final}")
                    pending_final_paths.add(resolved_final)
                    expected_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
                    if (
                        existing is None
                        and final.exists()
                        and hashlib.sha256(final.read_bytes()).hexdigest() != expected_hash
                    ):
                        raise RuntimeError(f"conflicting untracked kernel {final}")

            # Publish only after the complete conflict preflight succeeds.  If
            # an identical certificate already exists, preserve its metadata
            # and replace a final binary only when it needs repair.
            for system, staged_outputs in staged_systems:
                existing = by_label.get(system["dynkin_label"])
                for temporary, final in staged_outputs:
                    expected_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
                    final_matches = (
                        final.exists()
                        and hashlib.sha256(final.read_bytes()).hexdigest() == expected_hash
                    )
                    if not final_matches:
                        final.parent.mkdir(parents=True, exist_ok=True)
                        temporary.replace(final)
                if existing is None:
                    by_label[system["dynkin_label"]] = system

            artifact["systems"] = sorted(
                by_label.values(),
                key=lambda item: tuple(SPECS).index(item["dynkin_label"]),
            )
            artifact["completed_systems"] = len(artifact["systems"])
            artifact["completed_kernel_copies"] = sum(
                item["exact_nullity"] for item in artifact["systems"]
            )
            artifact["expected_systems"] = len(SPECS)
            artifact["expected_kernel_copies"] = sum(SPECS.values())
            artifact["inventory_complete"] = artifact["completed_systems"] == len(SPECS)
            artifact["passed"] = all(item["passed"] for item in artifact["systems"])
            verify_outputs(artifact, root)
            temporary = checkpoint.with_suffix(
                checkpoint.suffix + f".{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(artifact, indent=2) + "\n")
            temporary.replace(checkpoint)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return artifact
    finally:
        cleanup_staged_outputs(all_staged_outputs)


def publication_conflict_self_test():
    """Prove a rejected same-label publication preserves pinned state."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "results/checkpoint.json"
        checkpoint.parent.mkdir(parents=True)
        final = root / "data/eleven_dimensional_spinor_bridge/level12_00000_highest_weight_kernel.i16le"
        final.parent.mkdir(parents=True)

        def candidate(payload: bytes, tag: str):
            staged = final.with_suffix(final.suffix + f".{tag}.staged")
            staged.write_bytes(payload)
            system = {
                "dynkin_label": "00000",
                "exterior_degree": DEGREE,
                "source_columns": 1,
                "exact_modular_rank": 0,
                "exact_nullity": 1,
                "free_columns": [0],
                "coefficient_width_bytes": 2,
                "outputs": [
                    {
                        "copy": 1,
                        "path": str(final.relative_to(root)),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                ],
                "passed": True,
            }
            return system, [(staged, final)]

        publish_checkpoint(checkpoint, root, [candidate(b"\x01\x00", "first")])
        pinned_binary = final.read_bytes()
        pinned_checkpoint = checkpoint.read_bytes()
        try:
            publish_checkpoint(checkpoint, root, [candidate(b"\x02\x00", "second")])
        except RuntimeError as error:
            if "conflicting exact kernels" not in str(error):
                raise
        else:
            raise RuntimeError("publication conflict self-test did not reject conflict")
        if final.read_bytes() != pinned_binary:
            raise RuntimeError("publication conflict mutated the pinned binary")
        if checkpoint.read_bytes() != pinned_checkpoint:
            raise RuntimeError("publication conflict mutated the checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="*", choices=tuple(SPECS), default=tuple(SPECS))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test-publication", action="store_true")
    args = parser.parse_args()
    if args.self_test_publication:
        publication_conflict_self_test()
        print("publication conflict self-test passed", flush=True)
        return
    root = args.root.resolve()
    checkpoint = root / "results/adynkra_11d_level12_second_momentum_kernel_generation.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(checkpoint.read_text()) if checkpoint.exists() else empty_artifact()
    complete = {system["dynkin_label"] for system in existing.get("systems", [])}
    labels = [label for label in args.labels if args.force or label not in complete]
    left = half_groups(0)
    right = half_groups(16)
    for label in labels:
        system, staged_outputs = write_label(label, SPECS[label], left, right, root)
        artifact = publish_checkpoint(checkpoint, root, [(system, staged_outputs)])
        print(
            json.dumps(
                {
                    "dynkin_label": label,
                    "source_columns": system["source_columns"],
                    "exact_rank": system["exact_modular_rank"],
                    "exact_nullity": system["exact_nullity"],
                    "seconds": system["seconds"]["total"],
                    "completed_systems": artifact["completed_systems"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
