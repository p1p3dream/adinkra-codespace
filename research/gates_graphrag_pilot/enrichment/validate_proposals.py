#!/usr/bin/env python3
"""Validate and merge evidence-backed semantic relationship proposals."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

RELATIONSHIPS = {
    "EXTENDS", "GENERALIZES", "SPECIALIZES", "USES_RESULT_FROM",
    "REUSES_METHOD_FROM", "COMPARES_WITH", "CORRECTS", "PRECEDES_IN_SERIES",
    "VERSION_OF", "INTRODUCES", "DEFINES", "STUDIES", "USES", "DEPENDS_ON",
    "ASSUMES", "CONSTRUCTS", "CLASSIFIES", "CATALOGS", "ENUMERATES",
    "COMPUTES", "MAPS_TO", "DECOMPOSES_INTO", "PARTITIONS_INTO", "REPRESENTS",
    "ENCODES", "REALIZES", "REDUCES_TO", "LIFTS_TO", "EQUIVALENT_TO",
    "ISOMORPHIC_TO", "GENERATED_BY", "EQUIVALENCE_CLASS_OF", "QUOTIENT_OF",
    "MAKES_CLAIM", "REPORTS_RESULT", "DERIVES", "SUPPORTS", "CONTRADICTS",
    "QUALIFIES", "REQUIRES_ASSUMPTION", "APPLIES_TO", "USES_GROUP",
    "USES_ALGEBRA", "DESCRIBES_REPRESENTATION", "DESCRIBES_MULTIPLET",
    "HAS_INPUT", "HAS_OUTPUT",
}
ENTITY_TYPES = {
    "paper", "concept", "method", "invariant", "algorithm", "mathematical_object",
    "group", "algebra", "representation", "multiplet", "atlas", "claim", "result",
    "equation", "scope", "construction", "dataset", "artifact", "computation",
    "problem", "theorem", "operation", "quantity", "parameter", "property",
    "assumption", "evidence",
}


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            raise ValueError(f"{path}:{line_number}: missing chunk_id")
        chunks[chunk_id] = row
    return chunks


def validate_proposal(item: dict[str, Any], chunks: dict[str, dict[str, Any]], origin: str) -> list[str]:
    errors = []
    proposal_id = str(item.get("proposal_id") or "")
    if not proposal_id:
        errors.append(f"{origin}: missing proposal_id")
    relation = str(item.get("relationship") or "")
    if relation not in RELATIONSHIPS:
        errors.append(f"{origin}:{proposal_id}: unsupported relationship {relation!r}")
    for endpoint in ("source", "target"):
        entity = item.get(endpoint)
        if not isinstance(entity, dict):
            errors.append(f"{origin}:{proposal_id}: missing {endpoint} entity")
            continue
        if entity.get("type") not in ENTITY_TYPES:
            errors.append(f"{origin}:{proposal_id}: unsupported {endpoint} type {entity.get('type')!r}")
        if not str(entity.get("key") or "").strip() or not str(entity.get("name") or "").strip():
            errors.append(f"{origin}:{proposal_id}: {endpoint} requires key and name")
        if entity.get("type") == "paper" and not str(entity.get("key")).startswith("arxiv:"):
            errors.append(f"{origin}:{proposal_id}: paper key must start with arxiv:")
    evidence = item.get("evidence")
    if not isinstance(evidence, dict):
        errors.append(f"{origin}:{proposal_id}: missing evidence")
        return errors
    chunk_id = str(evidence.get("chunk_id") or "")
    chunk = chunks.get(chunk_id)
    if not chunk:
        errors.append(f"{origin}:{proposal_id}: unknown chunk {chunk_id!r}")
        return errors
    excerpt = normalized_text(str(evidence.get("excerpt") or ""))
    if not excerpt:
        errors.append(f"{origin}:{proposal_id}: empty excerpt")
    elif excerpt not in normalized_text(str(chunk.get("text") or "")):
        errors.append(f"{origin}:{proposal_id}: excerpt does not occur in chunk {chunk_id}")
    if int(evidence.get("page_number") or 0) != int(chunk.get("page_number") or 0):
        errors.append(f"{origin}:{proposal_id}: page does not match chunk")
    paper_id = str(evidence.get("paper_id") or "").removeprefix("arxiv:")
    chunk_paper = str(chunk.get("arxiv_id") or chunk.get("paper_id") or "").removeprefix("arxiv:")
    if paper_id != chunk_paper:
        errors.append(f"{origin}:{proposal_id}: evidence paper does not match chunk")
    if item.get("basis") != "explicit_text":
        errors.append(f"{origin}:{proposal_id}: basis must be explicit_text")
    if item.get("review_status") != "pending":
        errors.append(f"{origin}:{proposal_id}: review_status must be pending")
    confidence = item.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append(f"{origin}:{proposal_id}: confidence must be between 0 and 1")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    chunks = load_chunks(args.chunks)
    proposals = []
    errors = []
    seen_ids = set()
    seen_signatures = set()
    duplicate_signatures = 0
    for path in args.input:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            origin = f"{path}:{line_number}"
            errors.extend(validate_proposal(item, chunks, origin))
            proposal_id = item.get("proposal_id")
            if proposal_id in seen_ids:
                errors.append(f"{origin}: duplicate proposal_id {proposal_id!r}")
            seen_ids.add(proposal_id)
            signature = (
                item.get("source", {}).get("type"), item.get("source", {}).get("key"),
                item.get("relationship"), item.get("target", {}).get("type"),
                item.get("target", {}).get("key"), item.get("evidence", {}).get("chunk_id"),
            )
            if signature in seen_signatures:
                duplicate_signatures += 1
                continue
            seen_signatures.add(signature)
            proposals.append(item)
    proposals.sort(key=lambda x: str(x.get("proposal_id")))
    report = {
        "valid": not errors,
        "proposal_count": len(proposals),
        "duplicate_signatures_removed": duplicate_signatures,
        "relationships": dict(sorted(Counter(p["relationship"] for p in proposals).items())),
        "entity_types": dict(sorted(Counter(
            e["type"] for p in proposals for e in (p["source"], p["target"])
        ).items())),
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
