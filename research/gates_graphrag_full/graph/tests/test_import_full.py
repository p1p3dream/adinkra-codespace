from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))
from import_full import (DEFAULT_CITATIONS, DEFAULT_MANIFEST, DEFAULT_SHARDS,
                         DEFAULT_UNRESOLVED, InputError, build_plan, upsert_sql,
                         validate_no_nul)


class FullImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(DEFAULT_MANIFEST,DEFAULT_SHARDS,DEFAULT_CITATIONS,DEFAULT_UNRESOLVED,[])

    def test_verified_manifest_contract(self):
        self.assertEqual(self.plan.corpus_id, "gates_graphrag_full")
        self.assertEqual(self.plan.counts()["corpus_papers"], 295)
        self.assertEqual(sum(p["full_text_status"] != "metadata_only" for p in self.plan.papers.values() if not p["is_external_stub"]), 166)

    def test_all_local_papers_and_shards_loaded(self):
        self.assertEqual(len(list(DEFAULT_SHARDS.glob("*.jsonl"))), 166)
        self.assertEqual(len({c["paper_id"] for c in self.plan.chunks.values()}), 166)
        self.assertGreater(len(self.plan.chunks), 10_000)

    def test_graph_references_are_closed(self):
        self.assertTrue(all(e["src_node_id"] in self.plan.nodes and e["dst_node_id"] in self.plan.nodes for e in self.plan.edges.values()))
        self.assertTrue(all(e["paper_id"] in self.plan.papers for e in self.plan.edge_evidence.values()))
        self.assertTrue(all(e["paper_id"] in self.plan.papers for e in self.plan.node_evidence.values()))

    def test_every_node_and_edge_has_evidence(self):
        self.assertLessEqual({e["edge_id"] for e in self.plan.edges.values()}, {e["edge_id"] for e in self.plan.edge_evidence.values()})
        self.assertLessEqual({n["node_id"] for n in self.plan.nodes.values()}, {e["node_id"] for e in self.plan.node_evidence.values()})

    def test_citation_review_controls(self):
        citations = [e for e in self.plan.edges.values() if e["relationship"] == "CITES"]
        self.assertEqual(len(citations), 3825)
        self.assertEqual(sum(e["review_status"] == "accepted" for e in citations), 995)
        self.assertEqual(sum(e["review_status"] == "pending" for e in citations), 2830)
        self.assertFalse(any(e["review_status"] == "observed" for e in citations))
        self.assertEqual(len({e["edge_id"] for e in citations}), len(citations))

    def test_pdf_artifact_reconciliation(self):
        canonical = [a for a in self.plan.artifacts.values() if a["is_canonical"]]
        alternate = [a for a in self.plan.artifacts.values() if not a["is_canonical"]]
        self.assertEqual(len(canonical), 166)
        self.assertEqual(len(alternate), 4)
        self.assertTrue(all(a["properties"].get("relationship") == "exact_byte_copy" for a in alternate))

    def test_plan_is_deterministic(self):
        rebuilt = build_plan(DEFAULT_MANIFEST,DEFAULT_SHARDS,DEFAULT_CITATIONS,DEFAULT_UNRESOLVED,[])
        self.assertEqual(self.plan.digest(), rebuilt.digest())
        self.assertEqual(self.plan.counts(), rebuilt.counts())

    def test_semantic_excerpt_mismatch_fails_closed(self):
        chunk = next(iter(self.plan.chunks.values()))
        paper = self.plan.papers[chunk["paper_id"]]
        proposal = {"proposal_id":"bad","source":{"type":"paper","key":paper["stable_identifier"],"name":paper["title"]},
            "relationship":"DISCUSSES","target":{"type":"concept","key":"concept:test","name":"test"},
            "evidence":{"paper_id":chunk["paper_id"],"chunk_id":chunk["chunk_id"],"page_number":chunk["page_start"],"excerpt":"not present 9d493f"},
            "basis":"explicit_text","review_status":"pending","confidence":0.9}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"bad.jsonl"; path.write_text(json.dumps(proposal)+"\n")
            with self.assertRaises(InputError):
                build_plan(DEFAULT_MANIFEST,DEFAULT_SHARDS,DEFAULT_CITATIONS,DEFAULT_UNRESOLVED,[path])

    def test_schema_is_isolated_and_corpus_scoped(self):
        sql="\n".join(p.read_text() for p in sorted((GRAPH/"migrations").glob("*.sql")))
        self.assertIn("gates_full_corpora",sql)
        self.assertIn("corpus_id",sql)
        self.assertNotIn("ALTER TABLE gates_pilot_",sql)
        self.assertNotIn("INSERT INTO gates_pilot_",sql)

    def test_edge_upsert_preserves_completed_review_from_pending(self):
        sql = upsert_sql(
            "gates_full_edges",
            ["edge_id", "src_node_id", "dst_node_id", "relationship", "review_status"],
            "edge_id",
        )
        expected = (
            "review_status=CASE WHEN gates_full_edges.review_status IN ('accepted','rejected') "
            "AND EXCLUDED.review_status='pending' THEN gates_full_edges.review_status "
            "ELSE EXCLUDED.review_status END"
        )
        self.assertIn(expected, sql)
        self.assertIn("ON CONFLICT (corpus_id,edge_id) DO UPDATE", sql)

        def expected_merge(stored, incoming):
            return stored if stored in {"accepted", "rejected"} and incoming == "pending" else incoming

        self.assertEqual(expected_merge("accepted", "pending"), "accepted")
        self.assertEqual(expected_merge("rejected", "pending"), "rejected")
        self.assertEqual(expected_merge("accepted", "rejected"), "rejected")
        self.assertEqual(expected_merge("rejected", "accepted"), "accepted")
        self.assertEqual(expected_merge("pending", "accepted"), "accepted")

    def test_recursive_nul_validation_reports_nested_property_path(self):
        value = {"properties": {"nested": ["safe", {"excerpt": "bad\x00text"}]}}
        with self.assertRaisesRegex(
            InputError,
            r"U\+0000 NUL character at plan\.nodes\['node:test'\]\.properties\.nested\[1\]\.excerpt",
        ):
            validate_no_nul(value, "plan.nodes['node:test']")

    def test_build_plan_rejects_nul_with_record_path(self):
        rows = DEFAULT_UNRESOLVED.read_text(encoding="utf-8").splitlines()
        first = json.loads(rows[0])
        first["excerpt"] += "\x00database-incompatible"
        rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unresolved-with-nul.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(InputError) as raised:
                build_plan(DEFAULT_MANIFEST, DEFAULT_SHARDS, DEFAULT_CITATIONS, path, [])
            message = str(raised.exception)
            self.assertRegex(
                message,
                r"U\+0000 NUL character at plan\.(?:node|edge)_evidence\['evidence:[^']+'\]\.excerpt",
            )
            self.assertIn(f"; source {path.resolve()}:1", message)


if __name__ == "__main__":
    unittest.main()
