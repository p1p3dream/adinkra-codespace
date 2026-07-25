import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


GRAPH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH_DIR))

import import_pilot  # noqa: E402


class PilotImportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.manifest = self.root / "manifest.json"
        self.extracted = self.root / "extracted.jsonl"

    def tearDown(self):
        self.tempdir.cleanup()

    def write_inputs(self, manifest, rows):
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.extracted.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_deduplicates_artifacts_by_overlapping_stable_identifier(self):
        self.write_inputs(
            [
                {
                    "title": "One Work",
                    "arxiv_ids": "2304.09830",
                    "authors": "Gates, S.J.; Example, A.",
                    "pdf_filename": "arxiv.pdf",
                    "sha256": "a" * 64,
                },
                {
                    "title": "One Work, Publisher Copy",
                    "arxiv_ids": "arXiv:2304.09830v2",
                    "dois": "https://doi.org/10.1000/EXAMPLE",
                    "pdf_filename": "publisher.pdf",
                    "sha256": "b" * 64,
                },
            ],
            [{"paper_id": "arxiv:2304.09830", "content": "Section text."}],
        )
        plan = import_pilot.build_plan(self.manifest, [self.extracted])
        self.assertEqual(len(plan.papers), 1)
        self.assertEqual(len(plan.artifacts), 2)
        paper = next(iter(plan.papers.values()))
        self.assertEqual(paper.stable_identifier, "arxiv:2304.09830")
        self.assertEqual(plan.identifiers[("arxiv", "2304.09830")], paper.paper_id)
        self.assertEqual(plan.identifiers[("doi", "10.1000/example")], paper.paper_id)

    def test_builds_required_nodes_edges_and_provenance(self):
        self.write_inputs(
            [
                {
                    "inspire_id": "100",
                    "title": "Source Paper",
                    "year": "2023",
                    "authors": "Gates, S.J.; Doe, J.",
                },
                {
                    "inspire_id": "200",
                    "title": "Cited Paper",
                    "year": "2020",
                    "authors": "Gates, S.J.",
                },
            ],
            [
                {
                    "paper_id": "inspire:100",
                    "chunk_id": "source:chunk:1",
                    "content": "A concept supports a stated result.",
                    "section": "Results",
                    "page": 4,
                    "concepts": [{"id": "garden", "name": "Garden algebra"}],
                    "claims": [{"id": "claim-1", "text": "A bounded claim"}],
                    "results": [{"id": "result-1", "text": "A reported result"}],
                    "series": [{"id": "series-1", "name": "Adynkra sequence"}],
                    "citations": [{"inspire_id": "200", "locator": "reference 5"}],
                    "relationships": [
                        {
                            "source": "garden",
                            "target": "claim-1",
                            "relationship": "SUPPORTS",
                            "basis": "automated_inference",
                            "excerpt": "A concept supports a stated result.",
                        }
                    ],
                }
            ],
        )
        plan = import_pilot.build_plan(self.manifest, [self.extracted])
        types = {node.node_type for node in plan.nodes.values()}
        self.assertTrue({"paper", "author", "concept", "claim", "result", "series"} <= types)
        relationships = {edge.relationship for edge in plan.edges.values()}
        self.assertTrue(
            {"AUTHORED_BY", "DISCUSSES", "MAKES_CLAIM", "REPORTS_RESULT", "PART_OF_SERIES", "CITES", "SUPPORTS"}
            <= relationships
        )
        inferred = [edge for edge in plan.edges.values() if edge.basis == "automated_inference"]
        self.assertTrue(inferred)
        self.assertTrue(all(edge.review_status == "pending" for edge in inferred))
        self.assertEqual(len(plan.nodes), len({ev.target_id for ev in plan.node_evidence.values()}))
        self.assertEqual(len(plan.edges), len({ev.target_id for ev in plan.edge_evidence.values()}))
        import_pilot.validate_plan(plan)

    def test_external_citation_becomes_provenanced_stub(self):
        self.write_inputs(
            [{"arxiv_ids": "2311.06842", "title": "Source", "authors": "Gates, S.J."}],
            [
                {
                    "paper_id": "arxiv:2311.06842",
                    "content": "References",
                    "citations": [
                        {
                            "doi": "10.9999/external",
                            "title": "External Work",
                            "locator": "reference 9",
                        }
                    ],
                }
            ],
        )
        plan = import_pilot.build_plan(self.manifest, [self.extracted])
        stubs = [paper for paper in plan.papers.values() if paper.is_stub]
        self.assertEqual(len(stubs), 1)
        self.assertEqual(stubs[0].stable_identifier, "doi:10.9999/external")
        cites = [edge for edge in plan.edges.values() if edge.relationship == "CITES"]
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0].review_status, "observed")
        self.assertTrue(any(ev.target_id == cites[0].edge_id for ev in plan.edge_evidence.values()))

    def test_automated_inference_cannot_be_observed(self):
        self.write_inputs(
            [{"arxiv_ids": "1.00001", "title": "Source", "authors": "A"}],
            [
                {
                    "paper_id": "arxiv:1.00001",
                    "content": "Text",
                    "concepts": [
                        {
                            "name": "Concept",
                            "basis": "automated_inference",
                            "review_status": "observed",
                        }
                    ],
                }
            ],
        )
        with self.assertRaisesRegex(import_pilot.InputError, "cannot be marked observed"):
            import_pilot.build_plan(self.manifest, [self.extracted])

    def test_default_cli_mode_never_opens_database(self):
        self.write_inputs(
            [{"arxiv_ids": "1.00001", "title": "Source", "authors": "A"}],
            [{"paper_id": "arxiv:1.00001", "content": "Text"}],
        )
        with patch.object(import_pilot, "apply_plan", side_effect=AssertionError("database used")):
            status = import_pilot.main(
                ["--manifest", str(self.manifest), "--extracted", str(self.extracted)]
            )
        self.assertEqual(status, 0)

    def test_preserves_flat_extractor_page_and_method_provenance(self):
        self.write_inputs(
            [{"arxiv_id": "2012.14015", "title": "Source", "authors": ["A"]}],
            [
                {
                    "paper_id": "2012.14015",
                    "chunk_id": "2012.14015:p0002:c000",
                    "chunk_index": 2,
                    "page_chunk_index": 0,
                    "page_number": 2,
                    "page_label": "1",
                    "page_line_start": 3,
                    "page_line_end": 20,
                    "bbox": [72.0, 90.0, 540.0, 710.0],
                    "section_heading": "Introduction",
                    "section_start_page": 2,
                    "section_heading_source": "pdf_outline",
                    "text": "Introduction\nSection text.",
                    "word_count": 3,
                    "token_count": 4,
                    "counting_provenance": {"token_method": "unicode_lexical_units"},
                    "extraction_provenance": {
                        "backend": "pymupdf",
                        "backend_version": "1.27.2.3",
                        "source_sha256": "f" * 64,
                    },
                }
            ],
        )
        plan = import_pilot.build_plan(self.manifest, [self.extracted])
        chunk = plan.chunks["2012.14015:p0002:c000"]
        self.assertEqual(chunk.page_start, 2)
        self.assertEqual(chunk.section_title, "Introduction")
        self.assertEqual(chunk.properties["bbox"], [72.0, 90.0, 540.0, 710.0])
        self.assertEqual(chunk.properties["page_line_start"], 3)
        self.assertEqual(chunk.properties["extraction_provenance"]["backend"], "pymupdf")

    def test_schema_does_not_mutate_shared_graph_tables(self):
        sql = (GRAPH_DIR / "schema.sql").read_text(encoding="utf-8").casefold()
        for destructive in ("drop table", "truncate", "delete from", "alter table"):
            self.assertNotIn(destructive, sql)
        self.assertNotIn("create table if not exists graphrag_", sql)
        self.assertIn("create table if not exists gates_pilot_", sql)


if __name__ == "__main__":
    unittest.main()
