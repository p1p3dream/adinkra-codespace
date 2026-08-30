#!/usr/bin/env python3
"""Exact modular foundation for the 11D first-momentum F_X calculation.

The production runner stays over Gaussian rationals.  This independent tool
checks that completed operator checkpoints can make the round trip

    Q(i) -> F_p[i] for several p = 3 (mod 4) -> CRT -> Q(i)

without loss.  Its optional CUDA benchmark exercises the regular sparse
Gaussian contraction that can replace the final contraction stage.  It does
not claim to accelerate the irregular highest-weight state construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, Sequence


# These are deterministic Miller-Rabin certified below before use.
DEFAULT_PRIMES = (1_073_741_783, 1_073_741_723, 1_073_741_719)


def is_prime_32(n: int) -> bool:
    """Deterministic primality test for unsigned 32-bit integers."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # This base set is deterministic well beyond the 30-bit primes used here.
    for base in (2, 3, 5, 7, 11):
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def validate_prime(p: int) -> None:
    if p % 4 != 3:
        raise ValueError(f"prime {p} is not 3 modulo 4")
    if not is_prime_32(p):
        raise ValueError(f"modulus {p} is not prime")
    # Euler's criterion proves X^2 + 1 has no root over F_p.
    if pow(p - 1, (p - 1) // 2, p) != p - 1:
        raise AssertionError(f"-1 unexpectedly has a square root modulo {p}")


@dataclass(frozen=True)
class GaussianResidue:
    """An element a + b*i of F_p[i], where i^2 = -1."""

    real: int
    imag: int
    prime: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "real", self.real % self.prime)
        object.__setattr__(self, "imag", self.imag % self.prime)

    def _same_field(self, other: "GaussianResidue") -> None:
        if self.prime != other.prime:
            raise ValueError("cannot mix different residue fields")

    def __add__(self, other: "GaussianResidue") -> "GaussianResidue":
        self._same_field(other)
        return GaussianResidue(
            self.real + other.real, self.imag + other.imag, self.prime
        )

    def __sub__(self, other: "GaussianResidue") -> "GaussianResidue":
        self._same_field(other)
        return GaussianResidue(
            self.real - other.real, self.imag - other.imag, self.prime
        )

    def __mul__(self, other: "GaussianResidue") -> "GaussianResidue":
        self._same_field(other)
        return GaussianResidue(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
            self.prime,
        )

    def conjugate(self) -> "GaussianResidue":
        return GaussianResidue(self.real, -self.imag, self.prime)


def crt(residues: Sequence[int], moduli: Sequence[int]) -> tuple[int, int]:
    """Return the unique x in [0, M) and M for pairwise-coprime moduli."""
    if len(residues) != len(moduli) or not residues:
        raise ValueError("CRT requires equal nonempty residue and modulus lists")
    x = residues[0] % moduli[0]
    modulus = moduli[0]
    for residue, next_modulus in zip(residues[1:], moduli[1:]):
        if math.gcd(modulus, next_modulus) != 1:
            raise ValueError("CRT moduli are not pairwise coprime")
        correction = ((residue - x) * pow(modulus, -1, next_modulus)) % next_modulus
        x += modulus * correction
        modulus *= next_modulus
    return x, modulus


def centered_crt(residues: Sequence[int], moduli: Sequence[int]) -> tuple[int, int]:
    x, modulus = crt(residues, moduli)
    return (x - modulus if x > modulus // 2 else x), modulus


def rational_reconstruct(
    residue: int,
    modulus: int,
    numerator_bound: int,
    denominator_bound: int,
) -> Fraction:
    """Reconstruct a bounded rational using the extended Euclidean algorithm.

    Uniqueness is certified by 2*N*D < modulus.  The returned fraction is in
    lowest terms and is verified against the supplied modular residue.
    """
    if numerator_bound < 0 or denominator_bound < 1:
        raise ValueError("invalid rational reconstruction bounds")
    if 2 * numerator_bound * denominator_bound >= modulus:
        raise ValueError("modulus does not certify unique rational reconstruction")
    r0, r1 = modulus, residue % modulus
    s0, s1 = 0, 1
    while abs(r1) > numerator_bound:
        quotient = r0 // r1
        r0, r1 = r1, r0 - quotient * r1
        s0, s1 = s1, s0 - quotient * s1
    numerator, denominator = r1, s1
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    if denominator == 0:
        raise ValueError("rational reconstruction produced zero denominator")
    value = Fraction(numerator, denominator)
    if abs(value.numerator) > numerator_bound:
        raise ValueError("reconstructed numerator exceeds bound")
    if value.denominator > denominator_bound:
        raise ValueError("reconstructed denominator exceeds bound")
    if (value.denominator * residue - value.numerator) % modulus != 0:
        raise ValueError("rational reconstruction failed residue check")
    return value


@dataclass(frozen=True)
class GaussianRational:
    real: Fraction
    imag: Fraction


def checkpoint_values(document: dict) -> Iterator[GaussianRational]:
    for key in ("x2_rows", "x5_rows"):
        for row in document[key]:
            for value in row:
                yield GaussianRational(
                    Fraction(
                        int(value["real_numerator"]),
                        int(value["real_denominator"]),
                    ),
                    Fraction(
                        int(value["imaginary_numerator"]),
                        int(value["imaginary_denominator"]),
                    ),
                )


def common_denominator(values: Iterable[GaussianRational]) -> int:
    denominator = 1
    for value in values:
        denominator = math.lcm(
            denominator, value.real.denominator, value.imag.denominator
        )
    return denominator


def primes_for_bound(bound: int, denominator_bound: int) -> tuple[int, ...]:
    """Select enough primes for both integer and rational uniqueness bounds."""
    required = 2 * max(bound, bound * denominator_bound)
    modulus = 1
    selected: list[int] = []
    for prime in DEFAULT_PRIMES:
        validate_prime(prime)
        selected.append(prime)
        modulus *= prime
        if modulus > required:
            return tuple(selected)
    raise ValueError(f"configured primes do not exceed certified bound {required}")


def encode_rational(value: Fraction, prime: int) -> int:
    return value.numerator * pow(value.denominator, -1, prime) % prime


def verify_checkpoint(path: Path) -> dict:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not document.get("complete"):
        raise ValueError(f"checkpoint is not complete: {path}")
    values = list(checkpoint_values(document))
    denominator = common_denominator(values)
    numerator_bound = max(
        (abs(part.numerator) for value in values for part in (value.real, value.imag)),
        default=0,
    )
    denominator_bound = max(
        (part.denominator for value in values for part in (value.real, value.imag)),
        default=1,
    )
    cleared = [
        (
            value.real.numerator * (denominator // value.real.denominator),
            value.imag.numerator * (denominator // value.imag.denominator),
        )
        for value in values
    ]
    cleared_bound = max((abs(x) for pair in cleared for x in pair), default=0)
    # The stronger rational bound also implies the centered integer CRT bound.
    primes = primes_for_bound(max(cleared_bound, numerator_bound), denominator_bound)
    modulus = math.prod(primes)

    started = time.perf_counter()
    mismatches = 0
    reconstructed_digest = hashlib.sha256()
    for value, cleared_pair in zip(values, cleared):
        rebuilt_parts: list[Fraction] = []
        for original, cleared_integer in zip((value.real, value.imag), cleared_pair):
            integer_residues = [cleared_integer % prime for prime in primes]
            rebuilt_integer, rebuilt_modulus = centered_crt(integer_residues, primes)
            if rebuilt_modulus != modulus or abs(rebuilt_integer) > cleared_bound:
                raise AssertionError("centered CRT reconstruction violated bound")
            fixed_denominator_value = Fraction(rebuilt_integer, denominator)

            rational_residues = [encode_rational(original, prime) for prime in primes]
            rational_residue, rational_modulus = crt(rational_residues, primes)
            general_value = rational_reconstruct(
                rational_residue,
                rational_modulus,
                numerator_bound,
                denominator_bound,
            )
            if fixed_denominator_value != original or general_value != original:
                mismatches += 1
            rebuilt_parts.append(general_value)
        reconstructed_digest.update(
            f"{rebuilt_parts[0].numerator}/{rebuilt_parts[0].denominator},"
            f"{rebuilt_parts[1].numerator}/{rebuilt_parts[1].denominator}\n".encode()
        )
    elapsed = time.perf_counter() - started
    if mismatches:
        raise AssertionError(f"{mismatches} modular round-trip mismatches")

    return {
        "schema_version": "adynkra-11d-fx-modular-proof-v1",
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_schema_version": document.get("schema_version"),
        "gauge_form_degree": document.get("gauge_form_degree"),
        "operator_ordinal": document.get("operator_ordinal"),
        "gaussian_entries": len(values),
        "scalar_components": 2 * len(values),
        "common_denominator": denominator,
        "maximum_reduced_numerator": numerator_bound,
        "maximum_reduced_denominator": denominator_bound,
        "maximum_cleared_coefficient": cleared_bound,
        "primes": list(primes),
        "crt_modulus": str(modulus),
        "integer_uniqueness_condition": f"{modulus} > {2 * cleared_bound}",
        "rational_uniqueness_condition": (
            f"{modulus} > {2 * numerator_bound * denominator_bound}"
        ),
        "irreducible_gaussian_extension": all(prime % 4 == 3 for prime in primes),
        "mismatches": mismatches,
        "reconstructed_stream_sha256": reconstructed_digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "proof_boundary": (
            "This proves lossless modular encoding, CRT, and rational "
            "reconstruction for the completed checkpoint. It does not prove "
            "that a GPU generated the checkpoint's upstream sparse terms."
        ),
    }


def verify_checkpoint_set(paths: Sequence[Path]) -> dict:
    """Verify one fixed denominator and one CRT basis across many checkpoints."""
    if not paths:
        raise ValueError("checkpoint set is empty")
    documents = []
    values: list[GaussianRational] = []
    for path in paths:
        document = json.loads(path.read_text())
        if not document.get("complete"):
            raise ValueError(f"checkpoint is not complete: {path}")
        documents.append(document)
        values.extend(checkpoint_values(document))
    denominator = common_denominator(values)
    numerator_bound = max(
        abs(part.numerator) for value in values for part in (value.real, value.imag)
    )
    denominator_bound = max(
        part.denominator for value in values for part in (value.real, value.imag)
    )
    cleared_bound = max(
        abs(part.numerator * (denominator // part.denominator))
        for value in values
        for part in (value.real, value.imag)
    )
    primes = primes_for_bound(max(cleared_bound, numerator_bound), denominator_bound)
    modulus = math.prod(primes)

    started = time.perf_counter()
    mismatches = 0
    for value in values:
        for original in (value.real, value.imag):
            cleared_integer = original.numerator * (
                denominator // original.denominator
            )
            rebuilt_integer, rebuilt_modulus = centered_crt(
                [cleared_integer % prime for prime in primes], primes
            )
            rational_residue, rational_modulus = crt(
                [encode_rational(original, prime) for prime in primes], primes
            )
            rebuilt_rational = rational_reconstruct(
                rational_residue,
                rational_modulus,
                numerator_bound,
                denominator_bound,
            )
            if (
                rebuilt_modulus != modulus
                or Fraction(rebuilt_integer, denominator) != original
                or rebuilt_rational != original
            ):
                mismatches += 1
    if mismatches:
        raise AssertionError(f"{mismatches} modular checkpoint-set mismatches")
    return {
        "schema_version": "adynkra-11d-fx-modular-checkpoint-set-proof-v1",
        "checkpoints": len(paths),
        "gauge_form_degrees": sorted(
            {document["gauge_form_degree"] for document in documents}
        ),
        "gaussian_entries": len(values),
        "scalar_components": 2 * len(values),
        "global_common_denominator": denominator,
        "maximum_reduced_numerator": numerator_bound,
        "maximum_reduced_denominator": denominator_bound,
        "maximum_globally_cleared_coefficient": cleared_bound,
        "primes": list(primes),
        "crt_modulus": str(modulus),
        "integer_uniqueness_condition": f"{modulus} > {2 * cleared_bound}",
        "rational_uniqueness_condition": (
            f"{modulus} > {2 * numerator_bound * denominator_bound}"
        ),
        "mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - started,
        "proof_boundary": (
            "This proves one fixed denominator and one modular basis cover all "
            "completed checkpoints present in the supplied set at scan time."
        ),
    }


def cuda_sparse_benchmark(
    path: Path,
    outputs: int,
    terms_per_output: int,
    repeats: int,
) -> dict:
    """Benchmark a regular Gaussian sparse contraction over one certified prime."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the CUDA benchmark") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    document = json.loads(path.read_text())
    values = list(checkpoint_values(document))
    denominator = common_denominator(values)
    coefficients = [
        (
            value.real.numerator * (denominator // value.real.denominator),
            value.imag.numerator * (denominator // value.imag.denominator),
        )
        for value in values
    ]
    prime = DEFAULT_PRIMES[0]
    validate_prime(prime)
    input_count = len(coefficients)
    edge_count = outputs * terms_per_output

    # Deterministic regular COO schedule. Each output has exactly the same degree.
    schedule_started = time.perf_counter()
    output_index: list[int] = []
    source_index: list[int] = []
    weight_real: list[int] = []
    weight_imag: list[int] = []
    for output in range(outputs):
        for term in range(terms_per_output):
            output_index.append(output)
            source_index.append((output * 131 + term * 8191) % input_count)
            weight_real.append(((17 * term + 3 * output) % 15) - 7)
            weight_imag.append(((13 * term + 5 * output) % 15) - 7)
    schedule_build_seconds = time.perf_counter() - schedule_started

    started = time.perf_counter()
    cpu_real = [0] * outputs
    cpu_imag = [0] * outputs
    for edge, output in enumerate(output_index):
        source_real, source_imag = coefficients[source_index[edge]]
        wr = weight_real[edge]
        wi = weight_imag[edge]
        cpu_real[output] = (
            cpu_real[output] + source_real * wr - source_imag * wi
        ) % prime
        cpu_imag[output] = (
            cpu_imag[output] + source_real * wi + source_imag * wr
        ) % prime
    cpu_seconds = time.perf_counter() - started

    setup_started = time.perf_counter()
    torch.set_num_threads(1)
    source_cpu = torch.tensor(source_index, dtype=torch.int64)
    target_cpu = torch.tensor(output_index, dtype=torch.int64)
    wr_cpu = torch.tensor(weight_real, dtype=torch.int64) % prime
    wi_cpu = torch.tensor(weight_imag, dtype=torch.int64) % prime
    coeff_real_cpu = torch.tensor(
        [pair[0] % prime for pair in coefficients], dtype=torch.int64
    )
    coeff_imag_cpu = torch.tensor(
        [pair[1] % prime for pair in coefficients], dtype=torch.int64
    )
    cpu_tensor_setup_seconds = time.perf_counter() - setup_started

    def contract(source, target, wr, wi, coeff_real, coeff_imag):
        sr = coeff_real[source]
        si = coeff_imag[source]
        term_real = torch.remainder(sr * wr - si * wi, prime)
        term_imag = torch.remainder(sr * wi + si * wr, prime)
        out_real = torch.zeros(outputs, dtype=torch.int64, device=source.device)
        out_imag = torch.zeros(outputs, dtype=torch.int64, device=source.device)
        out_real.index_add_(0, target, term_real)
        out_imag.index_add_(0, target, term_imag)
        return torch.remainder(out_real, prime), torch.remainder(out_imag, prime)

    contract(
        source_cpu,
        target_cpu,
        wr_cpu,
        wi_cpu,
        coeff_real_cpu,
        coeff_imag_cpu,
    )
    started = time.perf_counter()
    for _ in range(repeats):
        torch_cpu_real, torch_cpu_imag = contract(
            source_cpu,
            target_cpu,
            wr_cpu,
            wi_cpu,
            coeff_real_cpu,
            coeff_imag_cpu,
        )
    torch_cpu_seconds = (time.perf_counter() - started) / repeats
    if torch_cpu_real.tolist() != cpu_real or torch_cpu_imag.tolist() != cpu_imag:
        raise AssertionError("optimized CPU tensor contraction disagrees with reference")

    device = torch.device("cuda")
    transfer_started = time.perf_counter()
    source = source_cpu.to(device)
    target = target_cpu.to(device)
    wr = wr_cpu.to(device)
    wi = wi_cpu.to(device)
    coeff_real = coeff_real_cpu.to(device)
    coeff_imag = coeff_imag_cpu.to(device)
    torch.cuda.synchronize()
    host_to_device_seconds = time.perf_counter() - transfer_started

    contract(source, target, wr, wi, coeff_real, coeff_imag)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        gpu_real, gpu_imag = contract(
            source, target, wr, wi, coeff_real, coeff_imag
        )
    torch.cuda.synchronize()
    gpu_seconds = (time.perf_counter() - started) / repeats
    download_started = time.perf_counter()
    rebuilt_real = gpu_real.cpu().tolist()
    rebuilt_imag = gpu_imag.cpu().tolist()
    device_to_host_seconds = time.perf_counter() - download_started
    mismatches = sum(
        left != right
        for left, right in zip(cpu_real, rebuilt_real)
    ) + sum(left != right for left, right in zip(cpu_imag, rebuilt_imag))
    if mismatches:
        raise AssertionError(f"CUDA sparse contraction has {mismatches} mismatches")

    return {
        "schema_version": "adynkra-11d-fx-modular-cuda-prototype-v1",
        "checkpoint": str(path),
        "prime": prime,
        "inputs": input_count,
        "outputs": outputs,
        "terms_per_output": terms_per_output,
        "edges": edge_count,
        "schedule_build_seconds": schedule_build_seconds,
        "cpu_reference_seconds": cpu_seconds,
        "cpu_tensor_setup_seconds": cpu_tensor_setup_seconds,
        "optimized_single_core_cpu_kernel_seconds": torch_cpu_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "cuda_kernel_seconds": gpu_seconds,
        "device_to_host_seconds": device_to_host_seconds,
        "cuda_kernel_speedup_over_optimized_single_core_cpu": (
            torch_cpu_seconds / gpu_seconds
        ),
        "cuda_kernel_speedup_over_python_reference": cpu_seconds / gpu_seconds,
        "mismatches": mismatches,
        "proof_boundary": (
            "The CUDA result is bit-exact for a deterministic regular sparse "
            "Gaussian contraction seeded by a real checkpoint. The schedule "
            "is synthetic because production primitive term streams are not "
            "yet materialized in a reusable format."
        ),
    }


def self_test() -> None:
    for prime in DEFAULT_PRIMES:
        validate_prime(prime)
    prime = DEFAULT_PRIMES[0]
    left = GaussianResidue(2, 3, prime)
    right = GaussianResidue(5, 7, prime)
    assert left * right == GaussianResidue(-11, 29, prime)
    assert left * left.conjugate() == GaussianResidue(13, 0, prime)
    integers = (-9_223_372_036, -1, 0, 1, 46_171_432_800)
    for integer in integers:
        residues = [integer % prime for prime in DEFAULT_PRIMES[:2]]
        rebuilt, _ = centered_crt(residues, DEFAULT_PRIMES[:2])
        assert rebuilt == integer
        mutated = list(residues)
        mutated[0] = (mutated[0] + 1) % DEFAULT_PRIMES[0]
        mutated_value, _ = centered_crt(mutated, DEFAULT_PRIMES[:2])
        assert mutated_value != integer
    modulus = math.prod(DEFAULT_PRIMES[:2])
    for value in (Fraction(-328_588, 15), Fraction(0), Fraction(119_552, 15)):
        residue, _ = crt(
            [encode_rational(value, prime) for prime in DEFAULT_PRIMES[:2]],
            DEFAULT_PRIMES[:2],
        )
        assert rational_reconstruct(residue, modulus, 500_000, 105) == value
    try:
        validate_prime(1_073_741_789)  # 1 modulo 4, so i is not adjoined this way.
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Gaussian-extension prime was accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-checkpoint")
    verify.add_argument("checkpoint", type=Path)
    verify.add_argument("--output", type=Path)

    verify_set = subparsers.add_parser("verify-directory")
    verify_set.add_argument("checkpoint_root", type=Path)
    verify_set.add_argument("--output", type=Path)

    benchmark = subparsers.add_parser("cuda-benchmark")
    benchmark.add_argument("checkpoint", type=Path)
    benchmark.add_argument("--outputs", type=int, default=4096)
    benchmark.add_argument("--terms-per-output", type=int, default=517)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--output", type=Path)

    subparsers.add_parser("self-test")
    arguments = parser.parse_args()
    if arguments.command == "self-test":
        self_test()
        print("all modular arithmetic self-tests passed")
        return
    if arguments.command == "verify-checkpoint":
        result = verify_checkpoint(arguments.checkpoint)
    elif arguments.command == "verify-directory":
        paths = sorted(arguments.checkpoint_root.glob("form-*/operator-*.json"))
        result = verify_checkpoint_set(paths)
    else:
        result = cuda_sparse_benchmark(
            arguments.checkpoint,
            arguments.outputs,
            arguments.terms_per_output,
            arguments.repeats,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
