#!/usr/bin/env python3
"""Deterministically validate concept and method enrichment artifacts."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHUNKS = Path("/tmp/gates-graphrag-pilot/chunks-enriched.jsonl")
PROPOSALS = HERE / "proposals.jsonl"
ALIASES = HERE / "ENTITY_ALIASES.json"
ALLOWED = {"INTRODUCES", "DEFINES", "STUDIES", "USES", "DEPENDS_ON", "ASSUMES"}
EXPECTED_PAPERS = {
    "1911.00807", "2002.08502", "2006.03609", "2007.07390", "2012.13308",
    "2012.14015", "2304.09830", "2311.06842", "2407.09334",
}
KEY_RE = re.compile(r"^[a-z]+:[a-z0-9]+(?:-[a-z0-9]+)*$")


def norm(text: str) -> str:
    return " ".join(text.split())


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    chunks = {
        row["chunk_id"]: row
        for row in (json.loads(line) for line in CHUNKS.read_text().splitlines())
    }
    proposals = [json.loads(line) for line in PROPOSALS.read_text().splitlines() if line.strip()]
    aliases = json.loads(ALIASES.read_text())
    canonical = {row["key"]: row for row in aliases["entities"]}

    ids = []
    edge_keys = []
    papers = Counter()
    relationships = Counter()
    targets = set()

    for index, row in enumerate(proposals, 1):
        expected_id = f"concepts-methods-{index:03d}"
        if row.get("proposal_id") != expected_id:
            fail(f"proposal {index}: expected id {expected_id}")
        ids.append(row["proposal_id"])
        if row.get("relationship") not in ALLOWED:
            fail(f"{expected_id}: disallowed relationship {row.get('relationship')}")
        if row.get("basis") != "explicit_text":
            fail(f"{expected_id}: basis must be explicit_text")
        if row.get("review_status") != "pending":
            fail(f"{expected_id}: review_status must remain pending")
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            fail(f"{expected_id}: invalid confidence")

        source = row.get("source", {})
        evidence = row.get("evidence", {})
        target = row.get("target", {})
        paper_id = evidence.get("paper_id")
        if source.get("type") != "paper" or source.get("key") != f"arxiv:{paper_id}":
            fail(f"{expected_id}: source and evidence paper disagree")
        if not source.get("name"):
            fail(f"{expected_id}: missing source name")
        if paper_id not in EXPECTED_PAPERS:
            fail(f"{expected_id}: unexpected paper {paper_id}")

        chunk_id = evidence.get("chunk_id")
        chunk = chunks.get(chunk_id)
        if chunk is None:
            fail(f"{expected_id}: unknown chunk {chunk_id}")
        if chunk.get("paper_id") != paper_id:
            fail(f"{expected_id}: chunk belongs to another paper")
        if evidence.get("page_number") != chunk.get("page_number"):
            fail(f"{expected_id}: physical page mismatch")
        if evidence.get("section") != chunk.get("section_heading"):
            fail(f"{expected_id}: section mismatch")
        excerpt = evidence.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt.strip():
            fail(f"{expected_id}: missing excerpt")
        if norm(excerpt) not in norm(chunk["text"]):
            fail(f"{expected_id}: excerpt is not present after whitespace normalization")

        target_key = target.get("key", "")
        if target_key != f"{target.get('type')}:{target_key.split(':', 1)[-1]}":
            fail(f"{expected_id}: target type prefix mismatch")
        if not KEY_RE.fullmatch(target_key):
            fail(f"{expected_id}: unstable target key {target_key!r}")
        if not target.get("name"):
            fail(f"{expected_id}: missing target name")
        if target_key not in canonical:
            fail(f"{expected_id}: target absent from ENTITY_ALIASES.json")
        if canonical[target_key]["name"] != target["name"]:
            fail(f"{expected_id}: canonical target name mismatch")

        edge_key = (source["key"], row["relationship"], target_key)
        edge_keys.append(edge_key)
        papers[paper_id] += 1
        relationships[row["relationship"]] += 1
        targets.add(target_key)

    if len(ids) != len(set(ids)):
        fail("duplicate proposal ids")
    if len(edge_keys) != len(set(edge_keys)):
        duplicates = [edge for edge, count in Counter(edge_keys).items() if count > 1]
        fail(f"duplicate semantic edges: {duplicates}")
    if set(papers) != EXPECTED_PAPERS:
        fail(f"paper coverage mismatch: {sorted(set(papers) ^ EXPECTED_PAPERS)}")
    if set(canonical) != targets:
        fail("alias entity set differs from proposal target set")

    seen_aliases = {}
    for key, entity in canonical.items():
        if entity.get("type") != key.split(":", 1)[0]:
            fail(f"alias type mismatch for {key}")
        for alias in entity.get("aliases", []):
            folded = alias.casefold()
            previous = seen_aliases.get(folded)
            if previous and previous != key:
                fail(f"alias collision {alias!r}: {previous} and {key}")
            seen_aliases[folded] = key

    print(f"PASS: {len(proposals)} proposals")
    print(f"PASS: {len(targets)} canonical targets")
    print(f"PASS: all excerpts verified across {len(papers)} papers")
    print("relationships: " + ", ".join(f"{key}={relationships[key]}" for key in sorted(relationships)))
    print("papers: " + ", ".join(f"{key}={papers[key]}" for key in sorted(papers)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
