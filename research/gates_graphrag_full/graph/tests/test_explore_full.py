from __future__ import annotations

import sys
import unittest
from pathlib import Path


GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))
from explore_full import (TraversedEdge, format_edge, identifier_lookup,
                          resolution_query, short_text, traversal_query)


class ExploreQueryTests(unittest.TestCase):
    def test_canonical_identifier_parsing(self):
        self.assertEqual(identifier_lookup("arxiv:2012.13308v2"), ("arxiv", "2012.13308"))
        self.assertEqual(identifier_lookup("doi:10.1000/ABC"), ("doi", "10.1000/abc"))
        self.assertEqual(identifier_lookup("inspire:1838225"), ("inspire", "1838225"))
        self.assertEqual(identifier_lookup("2012.13308"), (None, None))
        self.assertEqual(identifier_lookup("concept:adinkra"), (None, None))

    def test_resolution_query_supports_direct_and_identifier_paths(self):
        sql = resolution_query()
        self.assertIn("n.node_id=%s OR n.canonical_key=%s", sql)
        self.assertIn("FROM gates_full_identifiers i", sql)
        self.assertIn("JOIN gates_full_papers p", sql)
        self.assertIn("i.identifier_type=%s AND i.identifier_value=%s", sql)
        self.assertIn("n.properties->>'paper_id'=p.paper_id", sql)

    def test_traversal_query_returns_relationships_and_evidence(self):
        sql = traversal_query()
        self.assertIn("e.review_status IN ('observed','accepted')", sql)
        self.assertIn("FROM gates_full_edge_evidence ev", sql)
        self.assertIn("CASE WHEN e.src_node_id=r.from_node_id THEN 'outgoing' ELSE 'incoming' END", sql)
        self.assertIn("e.relationship,e.review_status", sql)
        self.assertIn("evidence.locator,evidence.excerpt", sql)


class ExploreFormatTests(unittest.TestCase):
    def edge(self, direction="outgoing", locator="physical PDF page 4", excerpt="supported text"):
        return TraversedEdge(
            1,"edge:1","node:a","paper","2012.13308","Paper A",
            "node:b","concept","concept:test","Test concept",
            "INTRODUCES","pending",direction,locator,excerpt,
        )

    def test_outgoing_edge_format(self):
        rendered = format_edge(self.edge())
        self.assertIn("Paper A [2012.13308] -[INTRODUCES | pending]-> Test concept [concept:test]", rendered)
        self.assertIn("evidence: physical PDF page 4", rendered)
        self.assertIn("excerpt: supported text", rendered)

    def test_incoming_edge_format_and_missing_evidence(self):
        rendered = format_edge(self.edge(direction="incoming", locator=None, excerpt=None))
        self.assertIn("<-[INTRODUCES | pending]-", rendered)
        self.assertIn("[physical locator unavailable]", rendered)
        self.assertIn("[excerpt unavailable]", rendered)

    def test_excerpt_is_whitespace_normalized_and_bounded(self):
        rendered = short_text("a\n  b " + "x" * 300, limit=20)
        self.assertEqual(len(rendered), 20)
        self.assertTrue(rendered.startswith("a b "))
        self.assertTrue(rendered.endswith("..."))


if __name__ == "__main__":
    unittest.main()
