#!/usr/bin/env python3
"""Safely stage, run, inspect, and promote shared 11D F_X checkpoints.

The production checkpoint directory can be local or remote, but this program
must be invoked on the host that can see it as a normal filesystem path.  The
shared Rust test hard-codes its reference path, so each worker runs in an
isolated runtime directory containing an immutable snapshot of the complete
references under ``--root``.

No production checkpoint is modified by ``status`` or ``run``.  ``run`` first
copies validated complete production references byte-for-byte into a separate
candidate root, then asks the operator-major runner to compute only pairs that
are still absent or partial in production.  Partial production payloads are
never copied into, or resumed by, the candidate run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable


SCHEMA = "adynkra-11d-first-momentum-partial-fx-checkpoint-v4"
CURVATURE_SHA256 = "c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f"
TARGET_ORDINAL = 319
FORM_DEGREES = tuple(range(6))
OPERATOR_ORDINALS = tuple(range(56))
TEST_NAME = (
    "eleven_dimensional_physical_curvature::tests::"
    "write_first_momentum_fx_shared_operator_batch"
)


class RolloutError(RuntimeError):
    pass


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    raw: bytes
    complete: bool

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


@dataclass
class Inventory:
    root: Path
    complete: dict[tuple[int, int], Checkpoint]
    partial: dict[tuple[int, int], Checkpoint]
    missing: list[tuple[int, int]]

    def summary(self) -> dict[str, object]:
        by_form = {}
        for degree in FORM_DEGREES:
            by_form[str(degree)] = {
                "complete": sum(key[0] == degree for key in self.complete),
                "partial": sum(key[0] == degree for key in self.partial),
                "missing": sum(key[0] == degree for key in self.missing),
            }
        return {
            "root": str(self.root),
            "complete": len(self.complete),
            "partial": len(self.partial),
            "missing": len(self.missing),
            "by_form": by_form,
        }


def checkpoint_path(root: Path, degree: int, operator: int) -> Path:
    return root / f"form-{degree}" / f"operator-{operator:02}.json"


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, document: object) -> None:
    atomic_write(
        path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def load_checkpoint(path: Path, degree: int, operator: int) -> Checkpoint:
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RolloutError(f"invalid checkpoint JSON {path}: {error}") from error
    expected = {
        "schema_version": SCHEMA,
        "curvature_artifact_sha256": CURVATURE_SHA256,
        "gauge_form_degree": degree,
        "target_basis_ordinal": TARGET_ORDINAL,
        "operator_ordinal": operator,
        "parameter_components_selected": [0],
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise RolloutError(
                f"refusing stale/misrouted checkpoint {path}: "
                f"{field}={document.get(field)!r}, expected {value!r}"
            )
    if not isinstance(document.get("complete"), bool):
        raise RolloutError(f"checkpoint {path} has no boolean complete flag")
    for field in ("emitted_target_terms", "source_entries_unique", "source_entries_processed"):
        value = document.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RolloutError(f"checkpoint {path} has invalid {field}={value!r}")
    unique = document["source_entries_unique"]
    processed = document["source_entries_processed"]
    if processed > unique:
        raise RolloutError(f"checkpoint {path} processed count exceeds unique count")
    if document["complete"] and processed != unique:
        raise RolloutError(f"checkpoint {path} is complete but its counts disagree")
    if not isinstance(document.get("x2_rows"), list) or not isinstance(
        document.get("x5_rows"), list
    ):
        raise RolloutError(f"checkpoint {path} has no exact row payloads")
    return Checkpoint(path=path, raw=raw, complete=document["complete"])


def inventory(root: Path) -> Inventory:
    root = root.expanduser().resolve()
    complete: dict[tuple[int, int], Checkpoint] = {}
    partial: dict[tuple[int, int], Checkpoint] = {}
    expected_paths = {
        checkpoint_path(root, degree, operator): (degree, operator)
        for degree in FORM_DEGREES
        for operator in OPERATOR_ORDINALS
    }
    if root.exists():
        for path in root.glob("form-*/operator-*.json"):
            resolved = path.resolve()
            if resolved not in expected_paths:
                raise RolloutError(f"refusing unexpected checkpoint pathname: {path}")
    for path, key in expected_paths.items():
        if not path.exists():
            continue
        checkpoint = load_checkpoint(path, *key)
        (complete if checkpoint.complete else partial)[key] = checkpoint
    missing = [
        (degree, operator)
        for degree in FORM_DEGREES
        for operator in OPERATOR_ORDINALS
        if (degree, operator) not in complete and (degree, operator) not in partial
    ]
    return Inventory(root, complete, partial, missing)


def assert_separate_roots(production: Path, candidate: Path) -> None:
    production = production.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    if production == candidate or production in candidate.parents or candidate in production.parents:
        raise RolloutError("production and candidate roots must be separate, non-nested paths")


def stage_complete_references(production: Inventory, candidate_root: Path) -> int:
    """Copy complete references only. Existing candidates must match exactly."""
    candidate = inventory(candidate_root)
    copied = 0
    for key, reference in production.complete.items():
        existing = candidate.complete.get(key)
        if existing is not None:
            if existing.raw != reference.raw:
                raise RolloutError(
                    f"candidate/reference byte mismatch for form {key[0]} operator {key[1]}"
                )
            continue
        if key in candidate.partial:
            raise RolloutError(
                f"refusing partial candidate for form {key[0]} operator {key[1]}"
            )
        atomic_write(checkpoint_path(candidate_root, *key), reference.raw)
        copied += 1
    return copied


def compare_all_references(production: Inventory, candidate: Inventory) -> None:
    for key, reference in production.complete.items():
        built = candidate.complete.get(key)
        if built is None:
            raise RolloutError(
                f"candidate lacks available reference form {key[0]} operator {key[1]}"
            )
        if built.raw != reference.raw:
            raise RolloutError(
                f"candidate/reference byte mismatch for form {key[0]} operator {key[1]}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_operator_tokens(values: Iterable[str], default: float) -> dict[int, float]:
    tokens = {operator: default for operator in OPERATOR_ORDINALS}
    for value in values:
        try:
            raw_operator, raw_tokens = value.split(":", 1)
            operator = int(raw_operator)
            amount = float(raw_tokens)
        except ValueError as error:
            raise RolloutError(f"invalid --operator-token {value!r}; expected ORDINAL:GIB") from error
        if operator not in OPERATOR_ORDINALS or amount <= 0:
            raise RolloutError(f"invalid --operator-token {value!r}")
        tokens[operator] = amount
    return tokens


def prepare_runtime(repo_root: Path, production: Inventory, runtime: Path) -> None:
    runtime.mkdir(parents=True, exist_ok=False)
    (runtime / "data").symlink_to(repo_root / "data", target_is_directory=True)
    reference_root = runtime / "results" / "eleven_dimensional_first_momentum_fx_checkpoints"
    # The Rust test's reference path is fixed.  Give it an immutable snapshot
    # containing complete references only.  In particular, never expose a
    # partial production file that it could mistake for resumable candidate
    # state.  Hard links are safe because production publishers use rename;
    # fall back to a byte copy across filesystems.
    for key, checkpoint in production.complete.items():
        destination = checkpoint_path(reference_root, *key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(checkpoint.path, destination)
        except OSError:
            atomic_write(destination, checkpoint.raw)


def write_execution(path: Path, **fields: object) -> None:
    atomic_json(
        path,
        {
            "schema_version": "adynkra-11d-fx-shared-rollout-execution-v1",
            **fields,
        },
    )


def guarded_command(arguments: argparse.Namespace, binary: Path) -> list[str]:
    test_command = [str(binary), TEST_NAME, "--exact", "--ignored", "--nocapture"]
    if not arguments.systemd_memory_guard:
        return test_command
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        raise RolloutError("--systemd-memory-guard requires systemd-run")
    return [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"Slice={arguments.systemd_slice}",
        "-p",
        f"MemoryHigh={arguments.memory_high}",
        "-p",
        f"MemoryMax={arguments.memory_max}",
        "-p",
        f"MemorySwapMax={arguments.memory_swap_max}",
        *test_command,
    ]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def acquire_lock(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RolloutError(f"rollout lock already exists: {path}") from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def release_lock(path: Path) -> None:
    path.unlink(missing_ok=True)


def run_rollout(arguments: argparse.Namespace) -> dict[str, object]:
    production_root = arguments.root.expanduser().resolve()
    candidate_root = arguments.candidate_root.expanduser().resolve()
    repo_root = arguments.repo_root.expanduser().resolve()
    binary = arguments.test_binary.expanduser().resolve()
    assert_separate_roots(production_root, candidate_root)
    production = inventory(production_root)
    candidate = inventory(candidate_root)
    if candidate.partial:
        raise RolloutError("refusing to resume partial candidate checkpoints")
    jobs = {
        operator: [
            degree
            for degree in FORM_DEGREES
            if (degree, operator) not in production.complete
            and (degree, operator) not in candidate.complete
        ]
        for operator in OPERATOR_ORDINALS
    }
    jobs = {operator: degrees for operator, degrees in jobs.items() if degrees}
    tokens = parse_operator_tokens(arguments.operator_token, arguments.default_token_gib)
    if any(tokens[operator] > arguments.memory_budget_gib for operator in jobs):
        raise RolloutError("an operator token exceeds the total memory-token budget")
    plan = {
        "schema_version": "adynkra-11d-fx-shared-rollout-plan-v1",
        "production": production.summary(),
        "candidate": candidate.summary(),
        "operators": [
            {"operator": operator, "degrees": degrees, "memory_tokens_gib": tokens[operator]}
            for operator, degrees in sorted(jobs.items())
        ],
        "pairs_to_compute": sum(map(len, jobs.values())),
        "memory_budget_gib": arguments.memory_budget_gib,
        "max_workers": arguments.max_workers,
        "systemd_memory_guard": arguments.systemd_memory_guard,
    }
    if arguments.dry_run:
        return plan
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RolloutError(f"test binary is not executable: {binary}")
    listed = subprocess.run(
        [str(binary), "--list"], capture_output=True, text=True, check=False
    )
    if listed.returncode != 0 or TEST_NAME not in listed.stdout:
        raise RolloutError(f"test binary does not expose {TEST_NAME}")
    command_template = guarded_command(arguments, binary)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    control = candidate_root / ".rollout"
    run_root = control / "runs" / run_id
    runtime = run_root / "runtime"
    lock = control / "active.lock"
    binary_sha = sha256_file(binary)
    lock_document = {
        "schema_version": "adynkra-11d-fx-shared-rollout-lock-v1",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_utc": utc_now(),
        "run_id": run_id,
        "binary": str(binary),
        "binary_sha256": binary_sha,
    }
    acquire_lock(lock, lock_document)
    try:
        copied = stage_complete_references(production, candidate_root)
        prepare_runtime(repo_root, production, runtime)
        atomic_json(run_root / "plan.json", plan)
    except BaseException:
        release_lock(lock)
        raise
    stop_launching = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_launching
        stop_launching = True

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    pending = sorted(jobs.items(), key=lambda item: (-tokens[item[0]], item[0]))
    active: dict[int, dict[str, object]] = {}
    outcomes: list[dict[str, object]] = []
    reserved = 0.0
    try:
        while pending or active:
            launched = False
            while pending and not stop_launching and len(active) < arguments.max_workers:
                fitting_index = next(
                    (
                        index
                        for index, (queued_operator, _) in enumerate(pending)
                        if reserved + tokens[queued_operator] <= arguments.memory_budget_gib
                    ),
                    None,
                )
                if fitting_index is None:
                    break
                operator, planned_degrees = pending[fitting_index]
                need = tokens[operator]
                # Production workers may have completed pairs after planning.
                current_production = inventory(production_root)
                stage_complete_references(current_production, candidate_root)
                current_candidate = inventory(candidate_root)
                degrees = [
                    degree
                    for degree in planned_degrees
                    if (degree, operator) not in current_production.complete
                    and (degree, operator) not in current_candidate.complete
                ]
                pending.pop(fitting_index)
                if not degrees:
                    outcomes.append({"operator": operator, "degrees": [], "status": "skipped"})
                    continue
                operator_root = run_root / f"operator-{operator:02}"
                operator_root.mkdir(parents=True, exist_ok=False)
                stdout_path = operator_root / "stdout.log.running"
                stderr_path = operator_root / "stderr.log.running"
                stdout = stdout_path.open("wb")
                stderr = stderr_path.open("wb")
                environment = os.environ.copy()
                environment.update(
                    {
                        "ADINKRA_FX_OPERATOR": str(operator),
                        "ADINKRA_FX_SHARED_DEGREES": ",".join(map(str, degrees)),
                        "ADINKRA_FX_SHARED_CHECKPOINT_ROOT": str(candidate_root),
                    }
                )
                command = command_template
                process = subprocess.Popen(
                    command,
                    cwd=runtime,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                started = utc_now()
                write_execution(
                    operator_root / "execution.json",
                    status="running",
                    operator=operator,
                    degrees=degrees,
                    memory_tokens_gib=need,
                    pid=process.pid,
                    host=socket.gethostname(),
                    binary=str(binary),
                    binary_sha256=binary_sha,
                    command=command,
                    started_utc=started,
                )
                atomic_write(operator_root / "worker.pid", f"{process.pid}\n".encode())
                active[operator] = {
                    "process": process,
                    "stdout": stdout,
                    "stderr": stderr,
                    "root": operator_root,
                    "degrees": degrees,
                    "tokens": need,
                    "started": started,
                    "command": command,
                }
                reserved += need
                launched = True

            finished = []
            for operator, state in active.items():
                process = state["process"]
                assert isinstance(process, subprocess.Popen)
                returncode = process.poll()
                if returncode is None:
                    continue
                finished.append(operator)
                state["stdout"].close()
                state["stderr"].close()
                operator_root = state["root"]
                assert isinstance(operator_root, Path)
                os.replace(operator_root / "stdout.log.running", operator_root / "stdout.log")
                os.replace(operator_root / "stderr.log.running", operator_root / "stderr.log")
                degrees = state["degrees"]
                assert isinstance(degrees, list)
                status = "failed"
                error_message = None
                if returncode == 0:
                    try:
                        current_candidate = inventory(candidate_root)
                        for degree in degrees:
                            if (degree, operator) not in current_candidate.complete:
                                raise RolloutError(
                                    f"runner returned success without complete form {degree} "
                                    f"operator {operator}"
                                )
                        current_production = inventory(production_root)
                        stage_complete_references(current_production, candidate_root)
                        compare_all_references(current_production, inventory(candidate_root))
                        status = "complete"
                    except RolloutError as error:
                        error_message = str(error)
                write_execution(
                    operator_root / "execution.json",
                    status=status,
                    operator=operator,
                    degrees=degrees,
                    memory_tokens_gib=state["tokens"],
                    pid=process.pid,
                    host=socket.gethostname(),
                    binary=str(binary),
                    binary_sha256=binary_sha,
                    command=state["command"],
                    started_utc=state["started"],
                    finished_utc=utc_now(),
                    exit_status=returncode,
                    error=error_message,
                )
                os.replace(operator_root / "worker.pid", operator_root / "worker.pid.done")
                outcome = {
                    "operator": operator,
                    "degrees": degrees,
                    "status": status,
                    "exit_status": returncode,
                }
                if error_message:
                    outcome["error"] = error_message
                outcomes.append(outcome)
                reserved -= float(state["tokens"])
                if status != "complete":
                    stop_launching = True
            for operator in finished:
                del active[operator]
            if not launched and not finished and active:
                time.sleep(1.0)
            elif not active and pending and stop_launching:
                break
        final_production = inventory(production_root)
        stage_complete_references(final_production, candidate_root)
        final_candidate = inventory(candidate_root)
        compare_all_references(final_production, final_candidate)
        report = {
            "schema_version": "adynkra-11d-fx-shared-rollout-report-v1",
            "run_id": run_id,
            "production": final_production.summary(),
            "candidate": final_candidate.summary(),
            "references_staged": copied,
            "binary": str(binary),
            "binary_sha256": binary_sha,
            "outcomes": sorted(outcomes, key=lambda value: value["operator"]),
            "interrupted": stop_launching,
            "passed": not stop_launching
            and all(outcome["status"] in {"complete", "skipped"} for outcome in outcomes),
            "finished_utc": utc_now(),
        }
        atomic_json(run_root / "report.json", report)
        return report
    finally:
        # Never orphan a worker or release the global lock while a child still
        # owns candidate paths.  An interrupt stops new launches; it does not
        # cancel an exact job already in progress.
        for operator, state in list(active.items()):
            process = state["process"]
            assert isinstance(process, subprocess.Popen)
            returncode = process.wait()
            state["stdout"].close()
            state["stderr"].close()
            operator_root = state["root"]
            assert isinstance(operator_root, Path)
            running_stdout = operator_root / "stdout.log.running"
            running_stderr = operator_root / "stderr.log.running"
            if running_stdout.exists():
                os.replace(running_stdout, operator_root / "stdout.log")
            if running_stderr.exists():
                os.replace(running_stderr, operator_root / "stderr.log")
            running_pid = operator_root / "worker.pid"
            if running_pid.exists():
                os.replace(running_pid, operator_root / "worker.pid.done")
            write_execution(
                operator_root / "execution.json",
                status="finished-unreviewed-after-supervisor-error",
                operator=operator,
                degrees=state["degrees"],
                memory_tokens_gib=state["tokens"],
                pid=process.pid,
                host=socket.gethostname(),
                binary=str(binary),
                binary_sha256=binary_sha,
                command=state["command"],
                started_utc=state["started"],
                finished_utc=utc_now(),
                exit_status=returncode,
            )
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        release_lock(lock)


def promote(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.confirm != "PROMOTE_COMPLETE_V4":
        raise RolloutError("promotion requires --confirm PROMOTE_COMPLETE_V4")
    if arguments.production_idle_ack != "LIVE_PRODUCTION_WORKERS_CHECKED_IDLE":
        raise RolloutError(
            "promotion requires --production-idle-ack LIVE_PRODUCTION_WORKERS_CHECKED_IDLE"
        )
    production_root = arguments.root.expanduser().resolve()
    candidate_root = arguments.candidate_root.expanduser().resolve()
    assert_separate_roots(production_root, candidate_root)
    lock = candidate_root / ".rollout" / "active.lock"
    if lock.exists():
        raise RolloutError(f"candidate rollout still has an active lock: {lock}")
    production = inventory(production_root)
    candidate = inventory(candidate_root)
    if candidate.partial or candidate.missing:
        raise RolloutError("promotion requires all 336 candidate checkpoints complete")
    compare_all_references(production, candidate)
    if production.partial and not arguments.replace_partial:
        raise RolloutError(
            "production contains partial checkpoints; pass --replace-partial to archive and replace"
        )
    actions = []
    for key in sorted(candidate.complete):
        source = candidate.complete[key]
        destination = checkpoint_path(production_root, *key)
        if key in production.complete:
            actions.append({"form": key[0], "operator": key[1], "action": "verified-existing"})
        elif key in production.partial:
            actions.append({"form": key[0], "operator": key[1], "action": "replace-partial"})
        else:
            actions.append({"form": key[0], "operator": key[1], "action": "copy-missing"})
    if arguments.dry_run:
        return {
            "schema_version": "adynkra-11d-fx-shared-promotion-plan-v1",
            "production": production.summary(),
            "candidate": candidate.summary(),
            "actions": actions,
        }
    promotion_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    promotion_lock = production_root / ".rollout" / "promotion.lock"
    acquire_lock(
        promotion_lock,
        {
            "schema_version": "adynkra-11d-fx-shared-promotion-lock-v1",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_utc": utc_now(),
            "promotion_id": promotion_id,
        },
    )
    try:
        promotion_root = production_root / ".rollout" / "promotions" / promotion_id
        backup_root = promotion_root / "partial-backup"
        promotion_root.mkdir(parents=True, exist_ok=False)
        copied = replaced = verified = 0
        for key in sorted(candidate.complete):
            source = candidate.complete[key]
            destination = checkpoint_path(production_root, *key)
            # Re-read immediately before each action to reduce the live-writer race.
            if destination.exists():
                current = load_checkpoint(destination, *key)
                if current.complete:
                    if current.raw != source.raw:
                        raise RolloutError(f"production changed during promotion: {destination}")
                    verified += 1
                    continue
                if not arguments.replace_partial:
                    raise RolloutError(f"partial checkpoint appeared during promotion: {destination}")
                backup = checkpoint_path(backup_root, *key)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                replaced += 1
            else:
                copied += 1
            atomic_write(destination, source.raw)
            if destination.read_bytes() != source.raw:
                raise RolloutError(f"post-copy byte verification failed: {destination}")
        final = inventory(production_root)
        compare_all_references(final, candidate)
        if final.partial or final.missing:
            raise RolloutError("production is not complete after promotion")
        report = {
            "schema_version": "adynkra-11d-fx-shared-promotion-report-v1",
            "promotion_id": promotion_id,
            "candidate_root": str(candidate_root),
            "production_root": str(production_root),
            "verified_existing": verified,
            "copied_missing": copied,
            "replaced_partial": replaced,
            "candidate_sha256": {
                f"form-{degree}/operator-{operator:02}": checkpoint.sha256
                for (degree, operator), checkpoint in sorted(candidate.complete.items())
            },
            "passed": True,
            "finished_utc": utc_now(),
        }
        atomic_json(promotion_root / "report.json", report)
        return report
    finally:
        release_lock(promotion_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def roots(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--root", type=Path, required=True, help="production checkpoint root")
        subparser.add_argument("--candidate-root", type=Path, required=True)

    status = subparsers.add_parser("status", help="validate and inventory both roots")
    roots(status)

    run = subparsers.add_parser("run", help="stage references and run missing pairs")
    roots(run)
    run.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    run.add_argument("--test-binary", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--memory-budget-gib", type=float, default=40.0)
    run.add_argument("--default-token-gib", type=float, default=20.0)
    run.add_argument("--operator-token", action="append", default=[], metavar="ORDINAL:GIB")
    run.add_argument("--max-workers", type=int, default=2)
    run.add_argument("--systemd-memory-guard", action="store_true")
    run.add_argument("--systemd-slice", default="adinkra-fx.slice")
    run.add_argument("--memory-high", default="16G")
    run.add_argument("--memory-max", default="20G")
    run.add_argument("--memory-swap-max", default="512M")

    promotion = subparsers.add_parser("promote", help="explicitly promote a complete candidate")
    roots(promotion)
    promotion.add_argument("--dry-run", action="store_true")
    promotion.add_argument("--replace-partial", action="store_true")
    promotion.add_argument("--confirm")
    promotion.add_argument("--production-idle-ack")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "status":
            production_root = arguments.root.expanduser().resolve()
            candidate_root = arguments.candidate_root.expanduser().resolve()
            assert_separate_roots(production_root, candidate_root)
            production = inventory(production_root)
            candidate = inventory(candidate_root)
            shared_keys = sorted(production.complete.keys() & candidate.complete.keys())
            for key in shared_keys:
                if production.complete[key].raw != candidate.complete[key].raw:
                    raise RolloutError(
                        f"candidate/reference byte mismatch for form {key[0]} operator {key[1]}"
                    )
            absent_references = sorted(production.complete.keys() - candidate.complete.keys())
            active_lock_path = candidate_root / ".rollout" / "active.lock"
            active_lock = None
            if active_lock_path.exists():
                try:
                    active_lock = json.loads(active_lock_path.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise RolloutError(f"invalid active rollout lock {active_lock_path}: {error}")
            result = {
                "schema_version": "adynkra-11d-fx-shared-rollout-status-v1",
                "production": production.summary(),
                "candidate": candidate.summary(),
                "available_references_in_candidate": len(shared_keys),
                "production_references_absent_from_candidate": len(absent_references),
                "all_shared_references_byte_identical": True,
                "active_rollout": active_lock,
            }
        elif arguments.command == "run":
            if arguments.memory_budget_gib <= 0 or arguments.default_token_gib <= 0:
                raise RolloutError("memory tokens and budget must be positive")
            if arguments.max_workers <= 0:
                raise RolloutError("--max-workers must be positive")
            result = run_rollout(arguments)
        else:
            result = promote(arguments)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("passed", True) else 1
    except (OSError, RolloutError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
