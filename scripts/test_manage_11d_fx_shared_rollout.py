#!/usr/bin/env python3

import importlib.util
import json
import argparse
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("manage_11d_fx_shared_rollout.py")
SPEC = importlib.util.spec_from_file_location("fx_rollout", MODULE_PATH)
assert SPEC and SPEC.loader
rollout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rollout
SPEC.loader.exec_module(rollout)


def payload(degree: int, operator: int, *, complete: bool = True, schema=None) -> bytes:
    document = {
        "schema_version": schema or rollout.SCHEMA,
        "curvature_artifact_sha256": rollout.CURVATURE_SHA256,
        "gauge_form_degree": degree,
        "target_basis_ordinal": rollout.TARGET_ORDINAL,
        "operator_ordinal": operator,
        "parameter_components_selected": [0],
        "emitted_target_terms": 0,
        "source_entries_unique": 1,
        "source_entries_processed": 1 if complete else 0,
        "complete": complete,
        "x2_rows": [],
        "x5_rows": [],
    }
    return (json.dumps(document, indent=2) + "\n").encode()


def write(root: Path, degree: int, operator: int, **keywords) -> Path:
    path = rollout.checkpoint_path(root, degree, operator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload(degree, operator, **keywords))
    return path


class RolloutTests(unittest.TestCase):
    def test_inventory_refuses_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, 0, 0, schema="stale")
            with self.assertRaisesRegex(rollout.RolloutError, "refusing stale"):
                rollout.inventory(root)

    def test_staging_copies_complete_and_excludes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            production = base / "production"
            candidate = base / "candidate"
            complete = write(production, 0, 0)
            write(production, 1, 0, complete=False)
            copied = rollout.stage_complete_references(
                rollout.inventory(production), candidate
            )
            self.assertEqual(copied, 1)
            self.assertEqual(
                rollout.checkpoint_path(candidate, 0, 0).read_bytes(), complete.read_bytes()
            )
            self.assertFalse(rollout.checkpoint_path(candidate, 1, 0).exists())

    def test_reference_comparison_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            production = base / "production"
            candidate = base / "candidate"
            write(production, 0, 0)
            candidate_path = write(candidate, 0, 0)
            document = json.loads(candidate_path.read_text())
            candidate_path.write_text(json.dumps(document, separators=(",", ":")))
            with self.assertRaisesRegex(rollout.RolloutError, "byte mismatch"):
                rollout.compare_all_references(
                    rollout.inventory(production), rollout.inventory(candidate)
                )

    def test_runtime_snapshot_omits_partial_references(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            (repo / "data").mkdir(parents=True)
            production = base / "production"
            write(production, 0, 0)
            write(production, 1, 0, complete=False)
            runtime = base / "runtime"
            rollout.prepare_runtime(repo, rollout.inventory(production), runtime)
            reference = runtime / "results" / "eleven_dimensional_first_momentum_fx_checkpoints"
            self.assertTrue(rollout.checkpoint_path(reference, 0, 0).exists())
            self.assertFalse(rollout.checkpoint_path(reference, 1, 0).exists())

    def test_nested_roots_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            production = Path(directory) / "production"
            candidate = production / "candidate"
            with self.assertRaisesRegex(rollout.RolloutError, "non-nested"):
                rollout.assert_separate_roots(production, candidate)

    def test_run_stages_references_and_invokes_only_missing_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            (repo / "data").mkdir(parents=True)
            production = base / "production"
            candidate = base / "candidate"
            missing = (2, 17)
            for degree in rollout.FORM_DEGREES:
                for operator in rollout.OPERATOR_ORDINALS:
                    if (degree, operator) != missing:
                        write(production, degree, operator)
            runner = base / "fake-test-binary.py"
            runner.write_text(
                """#!/usr/bin/env python3
import json, os, pathlib, sys
TEST = 'eleven_dimensional_physical_curvature::tests::write_first_momentum_fx_shared_operator_batch'
if '--list' in sys.argv:
    print(TEST + ': test')
    raise SystemExit(0)
operator = int(os.environ['ADINKRA_FX_OPERATOR'])
root = pathlib.Path(os.environ['ADINKRA_FX_SHARED_CHECKPOINT_ROOT'])
for raw_degree in os.environ['ADINKRA_FX_SHARED_DEGREES'].split(','):
    degree = int(raw_degree)
    document = {
      'schema_version': 'adynkra-11d-first-momentum-partial-fx-checkpoint-v4',
      'curvature_artifact_sha256': 'c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f',
      'gauge_form_degree': degree, 'target_basis_ordinal': 319,
      'operator_ordinal': operator, 'parameter_components_selected': [0],
      'emitted_target_terms': 0, 'source_entries_unique': 1,
      'source_entries_processed': 1, 'complete': True,
      'x2_rows': [], 'x5_rows': []}
    path = root / f'form-{degree}' / f'operator-{operator:02}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + '\\n')
"""
            )
            runner.chmod(0o755)
            arguments = argparse.Namespace(
                root=production,
                candidate_root=candidate,
                repo_root=repo,
                test_binary=runner,
                operator_token=[],
                default_token_gib=1.0,
                memory_budget_gib=1.0,
                max_workers=1,
                dry_run=False,
                systemd_memory_guard=False,
                systemd_slice="adinkra-fx.slice",
                memory_high="16G",
                memory_max="20G",
                memory_swap_max="512M",
            )
            report = rollout.run_rollout(arguments)
            self.assertTrue(report["passed"])
            self.assertEqual(
                [outcome for outcome in report["outcomes"] if outcome["status"] == "complete"],
                [{"operator": 17, "degrees": [2], "status": "complete", "exit_status": 0}],
            )
            final = rollout.inventory(candidate)
            self.assertEqual(len(final.complete), 336)
            self.assertFalse(final.partial)
            self.assertFalse(final.missing)


if __name__ == "__main__":
    unittest.main()
