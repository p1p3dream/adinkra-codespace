#!/usr/bin/env python3
"""Deterministically validate claims and results against the extracted chunks."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPOSALS = HERE / "proposals.jsonl"
ALIASES = HERE / "ENTITY_ALIASES.json"
CHUNKS = Path("/tmp/gates-graphrag-pilot/chunks-enriched.jsonl")
REPORT = HERE / "VALIDATION.json"
ALLOWED = {
    "MAKES_CLAIM", "REPORTS_RESULT", "DERIVES", "SUPPORTS", "CONTRADICTS",
    "QUALIFIES", "REQUIRES_ASSUMPTION",
}
REQUIRED = {
    "proposal_id", "source", "relationship", "target", "evidence",
    "basis", "review_status", "confidence",
}
ENTITY_REQUIRED = {"type", "key", "name"}
EVIDENCE_REQUIRED = {"paper_id", "chunk_id", "page_number", "section", "excerpt"}
ENTITY_TYPES = {"paper", "claim", "result", "scope"}
PILOT_PAPERS = {
    "1911.00807", "2002.08502", "2006.03609", "2007.07390", "2012.13308",
    "2012.14015", "2304.09830", "2311.06842", "2407.09334",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    errors: list[str] = []
    chunks = load_jsonl(CHUNKS) if CHUNKS.exists() else []
    if not chunks:
        errors.append(f"missing or empty chunk corpus: {CHUNKS}")
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    proposals = load_jsonl(PROPOSALS)
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))
    ids = []
    signatures = []

    for line_number, proposal in enumerate(proposals, 1):
        proposal_id = proposal.get("proposal_id")
        ids.append(proposal_id)
        missing = REQUIRED - set(proposal)
        if missing:
            errors.append(f"line {line_number}: missing fields {sorted(missing)}")
        relationship = proposal.get("relationship")
        if relationship not in ALLOWED:
            errors.append(f"{proposal_id}: invalid claims relationship {relationship!r}")
        if proposal.get("basis") != "explicit_text":
            errors.append(f"{proposal_id}: basis is not explicit_text")
        if proposal.get("review_status") != "pending":
            errors.append(f"{proposal_id}: review status is not pending")
        confidence = proposal.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{proposal_id}: invalid confidence")
        for side in ("source", "target"):
            endpoint = proposal.get(side, {})
            if ENTITY_REQUIRED - set(endpoint):
                errors.append(f"{proposal_id}: malformed {side}")
                continue
            kind = endpoint.get("type")
            key = endpoint.get("key", "")
            if kind not in ENTITY_TYPES:
                errors.append(f"{proposal_id}: unsupported {side} type {kind!r}")
            elif kind == "paper":
                if not key.startswith("arxiv:"):
                    errors.append(f"{proposal_id}: paper key must use arxiv prefix")
            elif not key.startswith(f"{kind}:"):
                errors.append(f"{proposal_id}: {side} key prefix does not match type")
        evidence = proposal.get("evidence", {})
        if EVIDENCE_REQUIRED - set(evidence):
            errors.append(f"{proposal_id}: malformed evidence")
        chunk = by_id.get(evidence.get("chunk_id"))
        if not chunk:
            errors.append(f"{proposal_id}: unknown chunk {evidence.get('chunk_id')!r}")
            continue
        if evidence.get("paper_id") != chunk.get("paper_id"):
            errors.append(f"{proposal_id}: evidence paper does not match chunk")
        if evidence.get("page_number") != chunk.get("page_number"):
            errors.append(f"{proposal_id}: physical page does not match chunk")
        if not isinstance(evidence.get("page_number"), int) or evidence.get("page_number", 0) < 1:
            errors.append(f"{proposal_id}: invalid physical page")
        if normalize(evidence.get("excerpt", "")) not in normalize(chunk.get("text", "")):
            errors.append(f"{proposal_id}: excerpt absent after whitespace normalization")
        signatures.append((
            proposal.get("source", {}).get("key"), relationship,
            proposal.get("target", {}).get("key"), evidence.get("chunk_id"),
        ))

    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate proposal IDs: {duplicate_ids}")
    duplicate_signatures = [signature for signature, count in Counter(signatures).items() if count > 1]
    if duplicate_signatures:
        errors.append(f"duplicate relationship signatures: {duplicate_signatures}")
    covered = {proposal.get("evidence", {}).get("paper_id") for proposal in proposals}
    if covered != PILOT_PAPERS:
        errors.append(f"paper coverage mismatch: missing={sorted(PILOT_PAPERS-covered)}, extra={sorted(covered-PILOT_PAPERS)}")

    alias_keys = []
    proposal_entity_keys = {
        endpoint["key"] for proposal in proposals for endpoint in (proposal["source"], proposal["target"])
    }
    for entry in aliases.get("entities", []):
        key = entry.get("key")
        alias_keys.append(key)
        if not entry.get("canonical_name") or not entry.get("aliases"):
            errors.append(f"malformed alias entry: {key}")
        if key not in proposal_entity_keys:
            errors.append(f"alias does not identify a proposed entity: {key}")
    duplicate_aliases = sorted(key for key, count in Counter(alias_keys).items() if count > 1)
    if duplicate_aliases:
        errors.append(f"duplicate alias keys: {duplicate_aliases}")

    report = {
        "schema_version": "gates-claims-results-validation-v1",
        "status": "pass" if not errors else "fail",
        "proposal_count": len(proposals),
        "paper_count": len(covered),
        "relationship_counts": dict(sorted(Counter(p["relationship"] for p in proposals).items())),
        "paper_counts": dict(sorted(Counter(p["evidence"]["paper_id"] for p in proposals).items())),
        "source_type_counts": dict(sorted(Counter(p["source"]["type"] for p in proposals).items())),
        "target_type_counts": dict(sorted(Counter(p["target"]["type"] for p in proposals).items())),
        "alias_entity_count": len(alias_keys),
        "proposals_sha256": hashlib.sha256(PROPOSALS.read_bytes()).hexdigest(),
        "entity_aliases_sha256": hashlib.sha256(ALIASES.read_bytes()).hexdigest(),
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
