#!/usr/bin/env python3
"""Build the fail-closed Phase 0 ledger for the 11D p3 three-prime run."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_PRIMES = [1073741783, 1073741723, 1073741719]
EXPECTED_MATRIX_SHA = [
    "23f21cec2040002338a6913e1f799da4ca3b6094e608affcf3b1034dc9f3b965",
    "a004dfe1323b31560470fe6ac3e0bd78f6d7f37cec466e840c2fa76d0f9be1d8",
    "949757fcadd88bf88cd70c198c95e96691e05eec154c75da04abe6d9a35dc493",
]
EXPECTED_ROLE_COUNTS = {
    "checkpoint": 44,
    "column_artifact": 231,
    "event_log": 44,
    "job_report": 132,
    "job_status": 44,
    "launch_provenance": 1,
    "lock_record": 132,
    "production_manifest": 1,
    "rank_certificate": 3,
    "rank_log": 3,
}
EXPECTED_PREFLIGHT = {
    "fixture_count": 24,
    "fixture_aggregate": "daa6f8187e4a57657a15a70c639d0bd05dfa103940e0c28f24f8cf8b516c371f",
    "frozen_path_count": 236,
    "freeze_aggregate": "5c49ba68986e288aa8ddfdeb41c5f12030d7819ea97ee5a88669a5627aa2645e",
    "map_file_count": 69,
    "repository_commit": "7ab0069",
    "isolated_commit": "856718e",
    "binary_sha256": "c1e22b834497b0d5c79ff31c6574e6c18b554d159a693282faa172650ac8ff41",
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def role(path: Path) -> str:
    name = path.name
    text = path.as_posix()
    if name.startswith("p3-all77-rank-") and name.endswith(".json"):
        return "rank_certificate"
    if name.startswith("p3-all77-rank-") and name.endswith(".log"):
        return "rank_log"
    if name == "p3-production-manifest.json":
        return "production_manifest"
    if name == "launch-provenance.txt":
        return "launch_provenance"
    if text.startswith("columns/"):
        return "column_artifact"
    if text.startswith("jobs/"):
        return {
            "job-report.json": "job_report",
            "checkpoint.json": "checkpoint",
            "events.jsonl": "event_log",
            "status.json": "job_status",
        }.get(name, "job_state")
    if text.startswith(".locks/"):
        return "lock_record"
    return "other"


def inventory(root: Path):
    entries = []
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode(),
    )
    for path in paths:
        relative = path.relative_to(root)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha(path),
                "role": role(relative),
            }
        )
    return entries


def parse_provenance(path: Path):
    values = {}
    source_hashes = []
    for line in path.read_text().splitlines():
        if "=" in line and line[:1].isalnum():
            key, value = line.split("=", 1)
            values[key] = value
        elif "  " in line:
            digest, source_path = line.split(None, 1)
            source_hashes.append({"sha256": digest, "path": source_path})
    values["source_hashes"] = source_hashes
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--denominator-certificate", type=Path, required=True)
    parser.add_argument("--normalized-rank-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-raw-inventory-sha256", required=True)
    parser.add_argument("--local-raw-inventory-sha256", required=True)
    args = parser.parse_args()

    failures = []
    production_entries = inventory(args.production_root)
    counts = dict(sorted(Counter(entry["role"] for entry in production_entries).items()))
    if counts != EXPECTED_ROLE_COUNTS:
        failures.append(f"role counts differ: {counts}")
    if len(production_entries) != 635:
        failures.append("production inventory is not 635 files")
    if args.source_raw_inventory_sha256 != args.local_raw_inventory_sha256:
        failures.append("source and local raw inventories differ")

    denominator = load_json(args.denominator_certificate)
    denominator_sha = file_sha(args.denominator_certificate)
    if not denominator.get("passed"):
        failures.append("denominator certificate did not pass")
    prime_audits = {entry["prime_slot"]: entry for entry in denominator["primes"]}
    if [prime_audits[index]["prime"] for index in range(3)] != EXPECTED_PRIMES:
        failures.append("denominator certificate primes differ")

    manifest_path = args.production_root / "p3-production-manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("physical_columns") != 77 or manifest.get("groups") != 44:
        failures.append("manifest dimensions differ")
    jobs = manifest.get("jobs", [])
    if len(jobs) != 132:
        failures.append("manifest does not contain 132 jobs")

    reports = []
    artifact_ordinals = {slot: set() for slot in range(3)}
    artifact_paths = set()
    for job in jobs:
        group = job["group_index"]
        slot = job["prime_index"]
        report_path = args.production_root / f"jobs/p3-g{group}-p{slot}/job-report.json"
        report = load_json(report_path)
        if not report.get("passed"):
            failures.append(f"failed report {report_path}")
        if report.get("prime") != EXPECTED_PRIMES[slot]:
            failures.append(f"wrong prime in {report_path}")
        if report.get("flat_plan_sha256") != prime_audits[slot]["modular_flat_plan_sha256"]:
            failures.append(f"flat plan mismatch in {report_path}")
        if report.get("manifest_sha256") != manifest.get("manifest_sha256"):
            failures.append(f"manifest hash mismatch in {report_path}")
        for artifact in report.get("artifacts", []):
            ordinal = artifact["global_ordinal"]
            relative = artifact["relative_path"]
            artifact_path = args.production_root / relative
            if file_sha(artifact_path) != artifact["artifact_sha256"]:
                failures.append(f"artifact hash mismatch: {relative}")
            if relative in artifact_paths:
                failures.append(f"duplicate artifact path: {relative}")
            artifact_paths.add(relative)
            artifact_ordinals[slot].add(ordinal)
        reports.append(
            {
                "path": report_path.relative_to(args.production_root).as_posix(),
                "sha256": file_sha(report_path),
                "prime_slot": slot,
                "group_index": group,
                "passed": report.get("passed"),
            }
        )
    for slot in range(3):
        if artifact_ordinals[slot] != set(range(77)):
            failures.append(f"prime slot {slot} does not cover ordinals 0..76")

    ranks = []
    for slot in range(3):
        path = args.production_root / f"p3-all77-rank-p{slot}.json"
        normalized_path = (
            args.normalized_rank_dir
            / f"adynkra_11d_p3_all77_rank_prime_slot_{slot}.json"
        )
        value = load_json(path)
        if (
            value.get("prime") != EXPECTED_PRIMES[slot]
            or value.get("rank_over_gaussian_extension") != 77
            or value.get("nullity_upper_bound") != 0
            or value.get("column_ordinals") != list(range(77))
            or value.get("matrix_sha256") != EXPECTED_MATRIX_SHA[slot]
        ):
            failures.append(f"rank certificate mismatch in prime slot {slot}")
        if not normalized_path.is_file() or file_sha(normalized_path) != file_sha(path):
            failures.append(f"normalized rank certificate mismatch in prime slot {slot}")
        ranks.append(
            {
                "prime_slot": slot,
                "historical_path": path.name,
                "normalized_path": normalized_path.as_posix(),
                "file_sha256": file_sha(path),
                "prime": value.get("prime"),
                "rank": value.get("rank_over_gaussian_extension"),
                "nullity_upper_bound": value.get("nullity_upper_bound"),
                "matrix_sha256": value.get("matrix_sha256"),
            }
        )

    preflight_entries = inventory(args.preflight_root)
    preflight = load_json(args.preflight_root / "preflight.json")
    if preflight != EXPECTED_PREFLIGHT:
        failures.append("preflight values differ from the pinned launch gate")
    provenance = parse_provenance(args.production_root / "launch-provenance.txt")
    for key in ("repository_commit", "isolated_commit", "binary_sha256"):
        if provenance.get(key) != preflight.get(key):
            failures.append(f"provenance/preflight mismatch for {key}")

    passed = not failures
    output = {
        "schema_version": "adynkra-11d-p3-three-prime-phase0-ledger-v1",
        "status": "authoritative" if passed else "invalid",
        "passed": passed,
        "phase0_complete": passed,
        "claim_boundary": (
            "durable denominator-admissible certificate for the bounded p3 D11 "
            "one-seed axis-retained diagnostic; not complete physical F, K, quotient, "
            "or an irreducibility theorem"
        ),
        "failures": failures,
        "source": {
            "host": "brandon@192.168.68.71",
            "production_root": "/home/brandon/adynkra-runs/p3-production-three-prime-fused-20260825T0902MDT",
            "preflight_root": "/home/brandon/adynkra-runs/p3-g0-20-launch-preflight-20260827T0812MDT",
        },
        "local_archive": str(args.production_root),
        "transfer_verification": {
            "source_raw_inventory_sha256": args.source_raw_inventory_sha256,
            "local_raw_inventory_sha256": args.local_raw_inventory_sha256,
            "path_size_sha256_sets_equal": args.source_raw_inventory_sha256
            == args.local_raw_inventory_sha256,
        },
        "production": {
            "file_count": len(production_entries),
            "total_bytes": sum(entry["bytes"] for entry in production_entries),
            "role_counts": counts,
            "manifest_sha256": file_sha(manifest_path),
            "entries": production_entries,
        },
        "preflight": {
            "values": preflight,
            "entries": preflight_entries,
        },
        "launch_provenance": provenance,
        "denominator_admissibility": {
            "path": str(args.denominator_certificate),
            "file_sha256": denominator_sha,
            "exact_coefficient_record_count": denominator["exact_coefficient_record_count"],
            "exact_coefficient_stream_sha256": denominator["exact_coefficient_stream_sha256"],
            "ordered_denominator_stream_sha256": denominator[
                "ordered_denominator_stream_sha256"
            ],
            "common_denominator_lcm": denominator["common_denominator_lcm"],
            "primes": denominator["primes"],
            "passed": denominator["passed"],
        },
        "rank_certificates": ranks,
        "job_reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("Phase 0 failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
