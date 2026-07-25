import json
import tempfile
import unittest
from pathlib import Path
import sys


SEMANTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEMANTIC))

import build_semantic
import validate_semantic


class FullSemanticTests(unittest.TestCase):
    def test_committed_artifacts_validate(self):
        result = validate_semantic.validate(SEMANTIC, check_determinism=False)
        self.assertEqual("pass", result["status"], result["errors"])
        self.assertEqual(166, result["counts"]["covered_papers"])
        self.assertEqual(166, result["counts"]["proposals"])

    def test_rebuild_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            build_semantic.build(
                build_semantic.DEFAULT_MANIFEST,
                build_semantic.DEFAULT_EXTRACTION,
                Path(left),
            )
            build_semantic.build(
                build_semantic.DEFAULT_MANIFEST,
                build_semantic.DEFAULT_EXTRACTION,
                Path(right),
            )
            for name in validate_semantic.ARTIFACTS:
                self.assertEqual((Path(left) / name).read_bytes(), (Path(right) / name).read_bytes())

    def test_every_proposal_is_pending_and_anchored(self):
        proposals = [
            json.loads(line)
            for line in (SEMANTIC / "proposals.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertTrue(proposals)
        self.assertTrue(all(row["review_status"] == "pending" for row in proposals))
        self.assertTrue(all(row["evidence"]["chunk_id"] for row in proposals))
        self.assertTrue(all(row["evidence"]["page_number"] >= 1 for row in proposals))


if __name__ == "__main__":
    unittest.main()
