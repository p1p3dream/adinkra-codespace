#!/usr/bin/env python3
"""Unit tests for deterministic citation parsing and resolution."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("extract_citations.py")
SPEC = importlib.util.spec_from_file_location("extract_citations", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VALIDATE_PATH = Path(__file__).with_name("validate.py")
VALIDATE_SPEC = importlib.util.spec_from_file_location("validate_citations", VALIDATE_PATH)
VALIDATE_MODULE = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC and VALIDATE_SPEC.loader
sys.modules[VALIDATE_SPEC.name] = VALIDATE_MODULE
VALIDATE_SPEC.loader.exec_module(VALIDATE_MODULE)


class CitationExtractionTests(unittest.TestCase):
    def test_identifier_normalization(self) -> None:
        identifiers = MODULE.extract_identifiers(
            "arXiv:2012.14015v2; hep-th/0108200, [hep-th: 0211034], and doi:10.1103/PhysRevD.14.3227."
        )
        self.assertEqual(identifiers["arxiv"], ["2012.14015", "hep-th/0108200", "hep-th/0211034"])
        self.assertEqual(identifiers["doi"], ["10.1103/physrevd.14.3227"])

    def test_reference_labels(self) -> None:
        self.assertEqual(MODULE.label_match("[12] S. J. Gates, ..."), ("12", "S. J. Gates, ..."))
        self.assertEqual(MODULE.label_match("7. M. Rocek, ..."), ("7", "M. Rocek, ..."))
        self.assertIsNone(MODULE.label_match("Gates discusses 7. examples"))

    def test_title_normalization(self) -> None:
        self.assertEqual(
            MODULE.normalize_title("Adinkras: 0-branes, Holoraumy & SUSY"),
            "adinkras 0 branes holoraumy susy",
        )

    def test_postgres_nul_is_replaced_and_detected(self) -> None:
        self.assertEqual(MODULE.normalize_space("alpha\x00 beta"), "alpha\ufffd beta")
        self.assertEqual(VALIDATE_MODULE.nul_paths({"excerpt": "alpha\x00beta"}), ["$.excerpt"])
        self.assertEqual(VALIDATE_MODULE.nul_paths({"excerpt": "alpha\ufffdbeta"}), [])

    def test_external_stub_threshold(self) -> None:
        paper = MODULE.Paper("inspire:1", "1", "x", "2020", "A", (), (), None, "")
        entry = MODULE.ReferenceEntry(paper, "1", 2, [2], ["A. Author, Phys. Rev. D 12 (2020) 1"])
        sufficient, signals = MODULE.sufficient_external_stub(entry, {"arxiv": [], "doi": [], "inspire": []})
        self.assertTrue(sufficient)
        self.assertIn("year", signals)
        self.assertIn("journal_or_publisher", signals)

    def test_loads_canonical_manifest_contract(self) -> None:
        payload = {
            "schema_version": "gates-full-corpus-manifest-v1",
            "papers": [{
                "paper_id": "2012.14015",
                "inspire_id": "1080779",
                "year": 2020,
                "title": "A paper",
                "authors": ["Gates, S.J.", "Author, A."],
                "identifiers": {
                    "canonical": "2012.14015",
                    "arxiv": ["2012.14015"],
                    "doi": ["10.1000/example"],
                    "inspire": ["1080779"],
                },
                "full_text": {"status": "metadata_only", "canonical_path": None, "sha256": None},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            papers = MODULE.load_manifest(path)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].paper_id, "2012.14015")
        self.assertEqual(papers[0].arxiv_ids, ("2012.14015",))
        self.assertEqual(papers[0].authors, "Gates, S.J.; Author, A.")


if __name__ == "__main__":
    unittest.main()
