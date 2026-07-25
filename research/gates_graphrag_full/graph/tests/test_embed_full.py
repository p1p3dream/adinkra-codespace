from __future__ import annotations

import sys
import unittest
from pathlib import Path


GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))
from embed_full import embed_missing


class FakeCursor:
    def __init__(self, data):
        self.data = data
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        table = "gates_full_chunks" if "gates_full_chunks" in sql else "gates_full_nodes"
        if sql.startswith("SELECT count"):
            self.result = [(len(self.data[table]),)]
            return
        limit = int(params[1])
        self.result = list(self.data[table].items())[:limit]

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


class FakeConnection:
    def __init__(self, chunks, nodes):
        self.data = {
            "gates_full_chunks": dict(chunks),
            "gates_full_nodes": dict(nodes),
        }
        self.commits = 0
        self.cursor_instance = FakeCursor(self.data)

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1


def fake_batch(cur, sql, params, page_size):
    table = "gates_full_chunks" if "gates_full_chunks" in sql else "gates_full_nodes"
    assert page_size == len(params)
    for _, _, object_id in params:
        cur.data[table].pop(object_id)


class EmbedBatchTests(unittest.TestCase):
    def test_one_embedding_call_per_database_batch(self):
        conn = FakeConnection(
            [(f"c{i}", f"chunk {i}") for i in range(5)],
            [(f"n{i}", f"node {i}") for i in range(3)],
        )
        calls = []

        def embed_many(texts, batch_size):
            calls.append((list(texts), batch_size))
            return [[float(i)] * 768 for i in range(len(texts))]

        counts = embed_missing(conn, "corpus", 2, None, embed_many, fake_batch)
        self.assertEqual(counts, {"chunks_embedded": 5, "nodes_embedded": 3})
        self.assertEqual(len(calls), 5)
        self.assertEqual(conn.commits, 5)
        self.assertTrue(all(batch_size == len(texts) <= 2 for texts, batch_size in calls))
        self.assertEqual(conn.data, {"gates_full_chunks": {}, "gates_full_nodes": {}})

    def test_global_limit_spans_tables_without_overshoot(self):
        conn = FakeConnection([("c0", "chunk")], [(f"n{i}", f"node {i}") for i in range(5)])
        calls = []

        def embed_many(texts, batch_size):
            calls.append(list(texts))
            return [[0.0] * 768 for _ in texts]

        counts = embed_missing(conn, "corpus", 2, 3, embed_many, fake_batch)
        self.assertEqual(counts, {"chunks_embedded": 1, "nodes_embedded": 2})
        self.assertEqual([len(call) for call in calls], [1, 2])
        self.assertEqual(conn.commits, 2)
        self.assertEqual(len(conn.data["gates_full_nodes"]), 3)

    def test_cardinality_error_prevents_update_and_commit(self):
        conn = FakeConnection([("c0", "chunk"), ("c1", "chunk")], [])
        with self.assertRaisesRegex(ValueError, "response count"):
            embed_missing(conn, "corpus", 2, None, lambda texts, batch_size: [[0.0] * 768], fake_batch)
        self.assertEqual(conn.commits, 0)
        self.assertEqual(len(conn.data["gates_full_chunks"]), 2)

    def test_dimension_error_prevents_update_and_commit(self):
        conn = FakeConnection([("c0", "chunk")], [])
        with self.assertRaisesRegex(ValueError, "expected 768"):
            embed_missing(conn, "corpus", 2, None, lambda texts, batch_size: [[0.0] * 767], fake_batch)
        self.assertEqual(conn.commits, 0)
        self.assertEqual(len(conn.data["gates_full_chunks"]), 1)

    def test_limit_must_be_positive(self):
        conn = FakeConnection([], [])
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            embed_missing(conn, "corpus", 2, 0, lambda texts, batch_size: [], fake_batch)


if __name__ == "__main__":
    unittest.main()
