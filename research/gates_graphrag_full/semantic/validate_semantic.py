#!/usr/bin/env python3
"""Validate provenance, coverage and deterministic rebuild of semantic proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import build_semantic


HERE = Path(__file__).resolve().parent
ARTIFACTS = ("proposals.jsonl", "nodes.jsonl", "ENTITY_ALIASES.json", "COVERAGE.json")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path, check_determinism: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    manifest_payload, papers = build_semantic.load_manifest(build_semantic.DEFAULT_MANIFEST)
    downloaded = {
        key for key, paper in papers.items() if paper.get("full_text", {}).get("status") == "verified_local_pdf"
    }
    index = {
        row["paper_id"]: row
        for row in rows(build_semantic.DEFAULT_EXTRACTION / "extraction_index.jsonl")
        if row.get("status") == "success"
    }
    chunks: dict[str, dict[str, Any]] = {}
    for paper_id, row in index.items():
        for chunk in rows(build_semantic.DEFAULT_EXTRACTION / row["shard_path"]):
            chunks[chunk["chunk_id"]] = chunk

    proposals = rows(root / "proposals.jsonl")
    nodes = rows(root / "nodes.jsonl")
    aliases = json.loads((root / "ENTITY_ALIASES.json").read_text())
    coverage = json.loads((root / "COVERAGE.json").read_text())
    proposal_ids: set[str] = set()
    covered: Counter[str] = Counter()
    referenced_keys: set[str] = set()
    for proposal in proposals:
        proposal_id = proposal.get("proposal_id")
        if proposal_id in proposal_ids:
            errors.append(f"duplicate proposal_id: {proposal_id}")
        proposal_ids.add(proposal_id)
        if proposal.get("relationship") not in build_semantic.ALLOWED_RELATIONSHIPS:
            errors.append(f"disallowed relationship: {proposal_id}")
        if proposal.get("basis") != "explicit_text" or proposal.get("review_status") != "pending":
            errors.append(f"invalid basis or review status: {proposal_id}")
        if proposal.get("method") not in {
            "explicit_active_cue_v1",
            "explicit_passive_cue_v1",
            "explicit_special_cue_v1",
            "explicit_title_scope_v1",
        }:
            errors.append(f"invalid method: {proposal_id}")
        confidence = proposal.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"invalid confidence: {proposal_id}")
        evidence = proposal.get("evidence", {})
        paper_id = evidence.get("paper_id")
        covered[paper_id] += 1
        chunk = chunks.get(evidence.get("chunk_id"))
        if chunk is None:
            errors.append(f"unknown chunk: {proposal_id}")
        else:
            if chunk["paper_id"] != paper_id or chunk["page_number"] != evidence.get("page_number"):
                errors.append(f"chunk anchor mismatch: {proposal_id}")
            excerpt = build_semantic.normalize_space(evidence.get("excerpt", ""))
            if not excerpt or excerpt not in build_semantic.normalize_space(chunk["text"]):
                errors.append(f"evidence excerpt not in chunk: {proposal_id}")
        paper = papers.get(paper_id)
        source = proposal.get("source", {})
        target = proposal.get("target", {})
        if paper is None or source.get("name") != build_semantic.normalize_space(paper["title"]):
            errors.append(f"paper source mismatch: {proposal_id}")
        elif source.get("key") != build_semantic.paper_key(paper) or source.get("type") != "paper":
            errors.append(f"paper source key mismatch: {proposal_id}")
        if target.get("type") not in build_semantic.ALLOWED_TYPES:
            errors.append(f"invalid target type: {proposal_id}")
        if not str(target.get("key", "")).startswith(f"{target.get('type')}:"):
            errors.append(f"target key/type mismatch: {proposal_id}")
        if not build_semantic.normalize_space(target.get("name", "")):
            errors.append(f"empty target name: {proposal_id}")
        referenced_keys.update((source.get("key"), target.get("key")))

    node_keys = [node.get("key") for node in nodes]
    if len(node_keys) != len(set(node_keys)):
        errors.append("duplicate node keys")
    if set(node_keys) != referenced_keys:
        errors.append("node keys do not equal proposal entity keys")
    for node in nodes:
        if node.get("type") not in build_semantic.ALLOWED_TYPES:
            errors.append(f"invalid node type: {node.get('key')}")
        if not set(node.get("proposal_ids", [])).issubset(proposal_ids):
            errors.append(f"node has unknown proposal: {node.get('key')}")
    alias_keys = [row.get("key") for row in aliases.get("entities", [])]
    if len(alias_keys) != len(set(alias_keys)) or not set(alias_keys).issubset(set(node_keys)):
        errors.append("alias keys are duplicated or do not resolve to nodes")

    if set(covered) != downloaded or set(index) != downloaded:
        errors.append("downloaded, extracted and semantically covered paper sets differ")
    if any(count != 1 for count in covered.values()):
        errors.append("each full-text paper must have one conservative proposal")
    if coverage.get("covered_paper_count") != len(covered) or coverage.get("proposal_count") != len(proposals):
        errors.append("coverage counts do not match artifacts")
    if coverage.get("uncovered_paper_ids"):
        errors.append("coverage reports uncovered papers")

    deterministic = True
    deterministic_mismatches: list[str] = []
    if check_determinism:
        with tempfile.TemporaryDirectory(prefix="gates-full-semantic-") as tmp:
            tmp_root = Path(tmp)
            build_semantic.build(
                build_semantic.DEFAULT_MANIFEST,
                build_semantic.DEFAULT_EXTRACTION,
                tmp_root,
            )
            for name in ARTIFACTS:
                if (root / name).read_bytes() != (tmp_root / name).read_bytes():
                    deterministic = False
                    deterministic_mismatches.append(name)
                    errors.append(f"nondeterministic artifact: {name}")

    result = {
        "schema_version": "gates-full-semantic-validation-v1",
        "status": "pass" if not errors else "fail",
        "checks": {
            "allowed_vocabulary": not any("relationship" in error or "target type" in error for error in errors),
            "all_pending_review": all(row.get("review_status") == "pending" for row in proposals),
            "complete_full_text_coverage": set(covered) == downloaded == set(index),
            "one_proposal_per_paper": bool(proposals) and all(count == 1 for count in covered.values()),
            "evidence_is_chunk_anchored": not any("chunk" in error or "excerpt" in error for error in errors),
            "entity_references_resolve": set(node_keys) == referenced_keys,
            "deterministic_rebuild": deterministic,
        },
        "counts": {
            "metadata_records": len(papers),
            "full_text_papers": len(downloaded),
            "covered_papers": len(covered),
            "proposals": len(proposals),
            "nodes": len(nodes),
            "aliases": len(alias_keys),
            "chunks_checked": len(chunks),
        },
        "relationship_counts": dict(sorted(Counter(row["relationship"] for row in proposals).items())),
        "method_counts": dict(sorted(Counter(row["method"] for row in proposals).items())),
        "artifact_sha256": {name: sha256(root / name) for name in ARTIFACTS},
        "deterministic_mismatches": deterministic_mismatches,
        "errors": errors,
    }
    build_semantic.write_json(root / "VALIDATION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    result = validate(args.root, not args.skip_determinism)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
