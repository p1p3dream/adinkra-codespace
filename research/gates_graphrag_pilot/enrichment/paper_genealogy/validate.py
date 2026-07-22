#!/usr/bin/env python3
"""Validate paper-genealogy proposals against the Gates pilot chunk corpus."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROPOSALS = HERE / "proposals.jsonl"
CHUNKS = Path("/tmp/gates-graphrag-pilot/chunks-enriched.jsonl")
MANIFEST = ROOT / "research/gates_graphrag_pilot/metadata/manifest.json"
REPORT = HERE / "VALIDATION.json"
ALLOWED = {
    "EXTENDS", "GENERALIZES", "SPECIALIZES", "USES_RESULT_FROM",
    "REUSES_METHOD_FROM", "COMPARES_WITH", "CORRECTS",
    "PRECEDES_IN_SERIES", "VERSION_OF",
}
REQUIRED = {
    "proposal_id", "source", "relationship", "target", "evidence",
    "basis", "review_status", "confidence", "notes",
}
ENTITY_REQUIRED = {"type", "key", "name"}
EVIDENCE_REQUIRED = {
    "paper_id", "chunk_id", "page_number", "section", "excerpt",
}
PILOT_PAPERS = {
    "1911.00807", "2002.08502", "2006.03609", "2007.07390",
    "2012.13308", "2012.14015", "2304.09830", "2311.06842",
    "2407.09334",
}


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    errors: list[str] = []
    if not CHUNKS.exists():
        errors.append(f"missing chunk corpus: {CHUNKS}")
        chunks: list[dict] = []
    else:
        chunks = load_jsonl(CHUNKS)
    by_chunk = {chunk["chunk_id"]: chunk for chunk in chunks}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    titles = {paper["arxiv_id"]: paper["title"] for paper in manifest["papers"]}
    if set(titles) != PILOT_PAPERS:
        errors.append("pilot paper set differs from the canonical manifest")

    proposals = load_jsonl(PROPOSALS)
    proposal_ids: list[str] = []
    signatures: list[tuple[str, str, str]] = []
    for line_number, proposal in enumerate(proposals, 1):
        label = f"line {line_number}"
        missing = REQUIRED - set(proposal)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
        proposal_id = proposal.get("proposal_id")
        proposal_ids.append(proposal_id)
        relationship = proposal.get("relationship")
        if relationship not in ALLOWED:
            errors.append(f"{proposal_id}: invalid genealogy relationship {relationship!r}")
        if proposal.get("basis") != "explicit_text":
            errors.append(f"{proposal_id}: basis must be explicit_text")
        if proposal.get("review_status") != "pending":
            errors.append(f"{proposal_id}: review_status must be pending")
        confidence = proposal.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{proposal_id}: confidence must be between zero and one")

        endpoint_ids: dict[str, str] = {}
        for side in ("source", "target"):
            entity = proposal.get(side, {})
            if ENTITY_REQUIRED - set(entity):
                errors.append(f"{proposal_id}: malformed {side}")
                continue
            if entity.get("type") != "paper":
                errors.append(f"{proposal_id}: {side} must be a paper")
            key = str(entity.get("key", ""))
            if not key.startswith("arxiv:"):
                errors.append(f"{proposal_id}: {side} key must use arxiv prefix")
                continue
            arxiv_id = key.removeprefix("arxiv:")
            endpoint_ids[side] = arxiv_id
            if arxiv_id not in PILOT_PAPERS:
                errors.append(f"{proposal_id}: {side} is outside the pilot")
            elif entity.get("name") != titles[arxiv_id]:
                errors.append(f"{proposal_id}: {side} title differs from manifest")
        if endpoint_ids.get("source") == endpoint_ids.get("target"):
            errors.append(f"{proposal_id}: self-relationship is not allowed")
        if len(endpoint_ids) == 2:
            signatures.append((endpoint_ids["source"], relationship, endpoint_ids["target"]))

        evidence = proposal.get("evidence", {})
        if EVIDENCE_REQUIRED - set(evidence):
            errors.append(f"{proposal_id}: malformed evidence")
            continue
        chunk_id = evidence.get("chunk_id")
        chunk = by_chunk.get(chunk_id)
        if chunk is None:
            errors.append(f"{proposal_id}: missing chunk {chunk_id!r}")
            continue
        if evidence.get("paper_id") != chunk.get("paper_id"):
            errors.append(f"{proposal_id}: evidence paper does not match chunk")
        if evidence.get("page_number") != chunk.get("page_number"):
            errors.append(f"{proposal_id}: physical page does not match chunk")
        if evidence.get("section") != chunk.get("section_heading"):
            errors.append(f"{proposal_id}: section does not match chunk")
        excerpt = evidence.get("excerpt", "")
        if not excerpt:
            errors.append(f"{proposal_id}: excerpt is empty")
        elif normalize_ws(excerpt) not in normalize_ws(chunk.get("text", "")):
            errors.append(
                f"{proposal_id}: excerpt absent after whitespace normalization"
            )

    duplicate_ids = sorted(key for key, count in Counter(proposal_ids).items() if count > 1)
    duplicate_edges = sorted(key for key, count in Counter(signatures).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate proposal IDs: {duplicate_ids}")
    if duplicate_edges:
        errors.append(f"duplicate semantic edges: {duplicate_edges}")

    report = {
        "schema_version": "gates-paper-genealogy-validation-v1",
        "status": "pass" if not errors else "fail",
        "proposal_count": len(proposals),
        "relationship_counts": dict(sorted(Counter(
            proposal["relationship"] for proposal in proposals
        ).items())),
        "source_paper_counts": dict(sorted(Counter(
            proposal["source"]["key"].removeprefix("arxiv:")
            for proposal in proposals
        ).items())),
        "evidence_paper_counts": dict(sorted(Counter(
            proposal["evidence"]["paper_id"] for proposal in proposals
        ).items())),
        "reviewed_paper_count": len(PILOT_PAPERS),
        "reviewed_papers_without_proposals": sorted(
            PILOT_PAPERS - {
                proposal["source"]["key"].removeprefix("arxiv:")
                for proposal in proposals
            }
        ),
        "proposals_sha256": hashlib.sha256(PROPOSALS.read_bytes()).hexdigest(),
        "chunks_sha256": hashlib.sha256(CHUNKS.read_bytes()).hexdigest() if CHUNKS.exists() else None,
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
