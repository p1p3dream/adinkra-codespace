from __future__ import annotations

import sys
from pathlib import Path


ENRICHMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENRICHMENT_DIR))

from import_proposals import db_node_type, stable_id  # noqa: E402
from validate_proposals import validate_proposal  # noqa: E402


def proposal() -> dict:
    return {
        "proposal_id": "test-1",
        "source": {"type": "paper", "key": "arxiv:2304.09830", "name": "Paper"},
        "relationship": "INTRODUCES",
        "target": {"type": "concept", "key": "concept:hopping-operator", "name": "hopping operator"},
        "evidence": {
            "paper_id": "2304.09830",
            "chunk_id": "chunk-1",
            "page_number": 3,
            "section": "Introduction",
            "excerpt": "defines a new mapping operator",
        },
        "basis": "explicit_text",
        "review_status": "pending",
        "confidence": 0.9,
    }


def chunks() -> dict:
    return {
        "chunk-1": {
            "chunk_id": "chunk-1",
            "arxiv_id": "2304.09830",
            "page_number": 3,
            "text": "The paper\ndefines a new mapping operator for this construction.",
        }
    }


def test_valid_proposal_matches_normalized_chunk_text() -> None:
    assert validate_proposal(proposal(), chunks(), "fixture") == []


def test_excerpt_must_occur_in_evidence_chunk() -> None:
    item = proposal()
    item["evidence"]["excerpt"] = "unsupported text"
    errors = validate_proposal(item, chunks(), "fixture")
    assert any("excerpt does not occur" in error for error in errors)


def test_semantic_types_map_to_existing_graph_node_types() -> None:
    assert db_node_type("paper") == "paper"
    assert db_node_type("claim") == "claim"
    assert db_node_type("result") == "result"
    assert db_node_type("group") == "concept"
    assert db_node_type("method") == "concept"


def test_stable_ids_are_deterministic_and_order_sensitive() -> None:
    first = stable_id("edge", "a", "b")
    assert first == stable_id("edge", "a", "b")
    assert first != stable_id("edge", "b", "a")
