#!/usr/bin/env python3
"""Deterministically validate construction proposals against extracted chunks."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROPOSALS = HERE / "proposals.jsonl"
ALIASES = HERE / "ENTITY_ALIASES.json"
CHUNKS = Path("/tmp/gates-graphrag-pilot/chunks-enriched.jsonl")
REPORT = HERE / "VALIDATION.json"
ALLOWED = {
    "CONSTRUCTS", "CLASSIFIES", "CATALOGS", "ENUMERATES", "COMPUTES",
    "MAPS_TO", "DECOMPOSES_INTO", "PARTITIONS_INTO", "REPRESENTS",
    "ENCODES", "REALIZES", "REDUCES_TO", "LIFTS_TO", "EQUIVALENT_TO",
    "ISOMORPHIC_TO", "GENERATED_BY", "EQUIVALENCE_CLASS_OF", "QUOTIENT_OF",
}
REQUIRED = {
    "proposal_id", "source", "relationship", "target", "evidence",
    "basis", "review_status", "confidence",
}
ENTITY_REQUIRED = {"type", "key", "name"}
EVIDENCE_REQUIRED = {"paper_id", "chunk_id", "page_number", "section", "excerpt"}
PILOT_PAPERS = {
    "1911.00807", "2002.08502", "2006.03609", "2007.07390", "2012.13308",
    "2012.14015", "2304.09830", "2311.06842", "2407.09334",
}

def normalize_ws(value):
    return re.sub(r"\s+", " ", value).strip()

def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def main():
    errors=[]
    if not CHUNKS.exists():
        errors.append(f"missing chunk corpus: {CHUNKS}")
        chunks=[]
    else:
        chunks=load_jsonl(CHUNKS)
    by_id={c["chunk_id"]:c for c in chunks}
    proposals=load_jsonl(PROPOSALS)
    aliases=json.loads(ALIASES.read_text(encoding="utf-8"))
    ids=[]
    for index,p in enumerate(proposals,1):
        label=f"line {index}"
        missing=REQUIRED-set(p)
        if missing: errors.append(f"{label}: missing fields {sorted(missing)}")
        pid=p.get("proposal_id")
        ids.append(pid)
        if p.get("relationship") not in ALLOWED:
            errors.append(f"{pid}: invalid construction relationship {p.get('relationship')}")
        if p.get("basis") != "explicit_text": errors.append(f"{pid}: basis is not explicit_text")
        if p.get("review_status") != "pending": errors.append(f"{pid}: review_status is not pending")
        confidence=p.get("confidence")
        if not isinstance(confidence,(int,float)) or not 0 <= confidence <= 1:
            errors.append(f"{pid}: invalid confidence")
        for side in ("source","target"):
            entity=p.get(side,{})
            if ENTITY_REQUIRED-set(entity): errors.append(f"{pid}: malformed {side}")
            key=entity.get("key","")
            if entity.get("type") == "paper":
                if not key.startswith("arxiv:"): errors.append(f"{pid}: paper key must use arxiv prefix")
            elif not key.startswith(entity.get("type","")+":"):
                errors.append(f"{pid}: {side} key prefix does not match type")
        evidence=p.get("evidence",{})
        if EVIDENCE_REQUIRED-set(evidence): errors.append(f"{pid}: malformed evidence")
        chunk=by_id.get(evidence.get("chunk_id"))
        if not chunk:
            errors.append(f"{pid}: missing chunk {evidence.get('chunk_id')}")
            continue
        if evidence.get("paper_id") != chunk.get("paper_id"):
            errors.append(f"{pid}: evidence paper does not match chunk")
        if evidence.get("page_number") != chunk.get("page_number"):
            errors.append(f"{pid}: physical page does not match chunk")
        if not isinstance(evidence.get("page_number"),int) or evidence["page_number"] < 1:
            errors.append(f"{pid}: invalid physical page")
        if normalize_ws(evidence.get("excerpt","")) not in normalize_ws(chunk.get("text","")):
            errors.append(f"{pid}: excerpt absent after whitespace normalization")
    duplicate_ids=sorted(k for k,v in Counter(ids).items() if v>1)
    if duplicate_ids: errors.append(f"duplicate proposal IDs: {duplicate_ids}")
    covered={p.get("evidence",{}).get("paper_id") for p in proposals}
    if covered != PILOT_PAPERS:
        errors.append(f"paper coverage mismatch: missing={sorted(PILOT_PAPERS-covered)}, extra={sorted(covered-PILOT_PAPERS)}")
    alias_keys=[]
    for entry in aliases.get("entities",[]):
        alias_keys.append(entry.get("key"))
        if not entry.get("canonical_name") or not entry.get("aliases"):
            errors.append(f"malformed alias entry: {entry.get('key')}")
    dup_alias=sorted(k for k,v in Counter(alias_keys).items() if v>1)
    if dup_alias: errors.append(f"duplicate alias keys: {dup_alias}")
    proposal_bytes=PROPOSALS.read_bytes()
    alias_bytes=ALIASES.read_bytes()
    report={
        "schema_version":"gates-construction-validation-v1",
        "status":"pass" if not errors else "fail",
        "proposal_count":len(proposals),
        "paper_count":len(covered),
        "relationship_counts":dict(sorted(Counter(p["relationship"] for p in proposals).items())),
        "paper_counts":dict(sorted(Counter(p["evidence"]["paper_id"] for p in proposals).items())),
        "alias_entity_count":len(alias_keys),
        "proposals_sha256":hashlib.sha256(proposal_bytes).hexdigest(),
        "entity_aliases_sha256":hashlib.sha256(alias_bytes).hexdigest(),
        "errors":errors,
    }
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
