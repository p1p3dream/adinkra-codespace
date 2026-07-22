#!/usr/bin/env python3
"""Build conservative semantic relationship proposals for the full Gates corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
FULL_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = FULL_ROOT / "metadata" / "manifest.json"
DEFAULT_EXTRACTION = FULL_ROOT / "extraction"
PILOT_ENRICHMENT = REPO_ROOT / "research" / "gates_graphrag_pilot" / "enrichment"

ALLOWED_RELATIONSHIPS = {
    token
    for token in re.findall(
        r"`([A-Z][A-Z_]+)`", (PILOT_ENRICHMENT / "RELATIONSHIPS.md").read_text()
    )
}
ALLOWED_TYPES = {
    token
    for token in re.findall(
        r"`([a-z][a-z_]+)`", (PILOT_ENRICHMENT / "RELATIONSHIPS.md").read_text()
    )
}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def slug(value: str, limit: int = 88) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    value = value[:limit].rstrip("-")
    return value or "unnamed"


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    papers = payload["papers"] if isinstance(payload, dict) else payload
    return payload, {row["paper_id"]: row for row in papers}


def split_passages(text: str) -> list[str]:
    """Split extracted prose while retaining verbatim normalized substrings."""
    text = normalize_space(text)
    if not text:
        return []
    parts = re.split(
        r"(?:(?<=[.!?])\s+|(?<=[.!?][\u201d\"'])\s+)(?=(?:[A-Z0-9\[\(\u201c\"]|We\b|This\b|The\b|In\b|Our\b))",
        text,
    )
    return [part.strip() for part in parts if len(part.strip()) >= 24]


def section_priority(chunk: dict[str, Any]) -> int:
    heading = normalize_space(chunk.get("section_heading") or "").lower()
    text_start = normalize_space(chunk.get("text") or "")[:900].lower()
    page = int(chunk["page_number"])
    if "abstract" in heading or re.search(r"\babstract\b", text_start):
        return 0
    if "intro" in heading or "prologue" in heading or "preface" in heading:
        return 1
    if any(word in heading for word in ("conclusion", "summary", "outlook")):
        return 2
    if page <= 3 and not any(word in heading for word in ("reference", "bibliograph")):
        return 3
    if page <= 6 and not any(word in heading for word in ("reference", "bibliograph")):
        return 4
    return 9


ACTIVE_RULES: list[dict[str, Any]] = [
    {
        "relationship": "DEFINES",
        "target_type": "concept",
        "confidence": 0.97,
        "verbs": r"defin(?:e|es)",
    },
    {
        "relationship": "INTRODUCES",
        "target_type": "concept",
        "confidence": 0.95,
        "verbs": r"introduc(?:e|es)|propos(?:e|es)|develop(?:s)?",
    },
    {
        "relationship": "CONSTRUCTS",
        "target_type": "construction",
        "confidence": 0.95,
        "verbs": r"construct(?:s)?|build(?:s)?|creat(?:e|es)",
    },
    {
        "relationship": "CLASSIFIES",
        "target_type": "dataset",
        "confidence": 0.96,
        "verbs": r"classif(?:y|ies)",
    },
    {
        "relationship": "CATALOGS",
        "target_type": "dataset",
        "confidence": 0.96,
        "verbs": r"catalog(?:s)?",
    },
    {
        "relationship": "ENUMERATES",
        "target_type": "dataset",
        "confidence": 0.96,
        "verbs": r"enumerat(?:e|es)",
    },
    {
        "relationship": "COMPUTES",
        "target_type": "computation",
        "confidence": 0.95,
        "verbs": r"comput(?:e|es)|calculat(?:e|es)",
    },
    {
        "relationship": "DERIVES",
        "target_type": "result",
        "confidence": 0.96,
        "verbs": r"deriv(?:e|es)|obtain(?:s)?",
    },
    {
        "relationship": "COMPARES_WITH",
        "target_type": "scope",
        "confidence": 0.94,
        "verbs": r"compar(?:e|es)",
    },
    {
        "relationship": "REPORTS_RESULT",
        "target_type": "result",
        "confidence": 0.96,
        "verbs": r"show(?:s)?|demonstrat(?:e|es)|find(?:s)?|establish(?:es)?|prov(?:e|es)",
    },
    {
        "relationship": "STUDIES",
        "target_type": "scope",
        "confidence": 0.92,
        "verbs": r"stud(?:y|ies)|investigat(?:e|es)|analy[sz](?:e|es)|examin(?:e|es)|explor(?:e|es)|discuss(?:es)?|review(?:s)?|consider(?:s)?|describ(?:e|es)|continu(?:e|es)",
    },
    {
        "relationship": "USES",
        "target_type": "method",
        "confidence": 0.88,
        "verbs": r"appl(?:y|ies)|us(?:e|es)",
    },
    {
        "relationship": "REPORTS_RESULT",
        "target_type": "result",
        "confidence": 0.90,
        "verbs": r"present(?:s)?|provid(?:e|es)|giv(?:e|es)",
    },
]

PASSIVE_RULES: list[dict[str, Any]] = [
    {
        "relationship": "DEFINES",
        "target_type": "concept",
        "confidence": 0.91,
        "participle": r"defined",
    },
    {
        "relationship": "INTRODUCES",
        "target_type": "concept",
        "confidence": 0.90,
        "participle": r"introduced|proposed|developed",
    },
    {
        "relationship": "CONSTRUCTS",
        "target_type": "construction",
        "confidence": 0.91,
        "participle": r"constructed|built|created",
    },
    {
        "relationship": "CLASSIFIES",
        "target_type": "dataset",
        "confidence": 0.92,
        "participle": r"classified|catalogued|enumerated",
    },
    {
        "relationship": "COMPUTES",
        "target_type": "computation",
        "confidence": 0.91,
        "participle": r"computed|calculated|undertaken",
    },
    {
        "relationship": "DERIVES",
        "target_type": "result",
        "confidence": 0.92,
        "participle": r"derived|obtained",
    },
    {
        "relationship": "STUDIES",
        "target_type": "scope",
        "confidence": 0.88,
        "participle": r"studied|investigated|analyzed|analysed|examined|explored|discussed|reviewed|considered",
    },
]


for rule in ACTIVE_RULES:
    rule["regex"] = re.compile(
        rf"\b(?P<actor>we|this (?:paper|work|article|letter)|the present (?:paper|work|article|letter))"
        rf"\s+(?:also\s+|have\s+|will\s+|can\s+|now\s+|here\s+|explicitly\s+|further\s+|first\s+|shall\s+)*"
        rf"(?P<verb>{rule['verbs']})\b",
        re.IGNORECASE,
    )

for rule in PASSIVE_RULES:
    rule["regex"] = re.compile(
        rf"\b(?P<aux>is|are|was|were|has been|have been)\s+(?:also\s+|explicitly\s+|first\s+)*"
        rf"(?P<verb>{rule['participle']})\b",
        re.IGNORECASE,
    )

SPECIAL_RULES = [
    (
        re.compile(r"\bresults? (?:are|is) (?:given|presented|reported)\b", re.IGNORECASE),
        "REPORTS_RESULT",
        "result",
        0.92,
    ),
    (
        re.compile(r"\bevidence is presented\b", re.IGNORECASE),
        "SUPPORTS",
        "claim",
        0.93,
    ),
    (
        re.compile(r"\bthere (?:is|are) shown to (?:exist|be)\b", re.IGNORECASE),
        "REPORTS_RESULT",
        "result",
        0.94,
    ),
    (
        re.compile(r"\bthe (?:main )?purpose of (?:this|these) (?:paper|work|article|lectures) is to\b", re.IGNORECASE),
        "STUDIES",
        "scope",
        0.89,
    ),
]


def compact_excerpt(passage: str, match_start: int, match_end: int, limit: int = 360) -> str:
    if len(passage) <= limit:
        return passage
    start = max(0, match_start - 140)
    end = min(len(passage), max(match_end + 260, start + limit))
    if end - start > limit:
        end = start + limit
    excerpt = passage[start:end].strip()
    if start:
        first_space = excerpt.find(" ")
        if 0 <= first_space < 30:
            excerpt = excerpt[first_space + 1 :]
    if end < len(passage):
        last_space = excerpt.rfind(" ")
        if last_space > len(excerpt) - 35:
            excerpt = excerpt[:last_space]
    return excerpt.strip(" ,;:")


def clean_target(value: str, relationship: str) -> str:
    value = normalize_space(value)
    value = re.sub(r"^\(([^)]{1,60})\)\s*", r"\1 ", value)
    value = re.sub(
        r"\barXiv:\S+(?:\s+\[[^\]]+\])?(?:\s+\d{1,2}\s+[A-Za-z]+\s+\d{4})?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:that|how|whether|the following|a|an)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:in (?:this|the present) (?:paper|work|article|letter),?\s*)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:in (?:this|the present) (?:paper|work|article|letter))\.?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\[[0-9,\-\s]+\]\s*$", "", value)
    value = value.strip(" .,:;\"'()")
    if relationship == "REPORTS_RESULT" and value and not value.lower().startswith("result"):
        value = f"result that {value}"
    if len(value) > 280:
        cut = value[:280]
        for delimiter in ("; ", ". ", " while ", " whereas "):
            pos = cut.rfind(delimiter)
            if pos >= 60:
                cut = cut[:pos]
                break
        value = cut.rsplit(" ", 1)[0].strip(" .,:;")
    return value


def direct_target(passage: str, match: re.Match[str], relationship: str) -> str:
    tail = passage[match.end() :]
    tail = re.split(r"(?<=[;])\s+|\s+(?:In Section|Section)\s+\d", tail, maxsplit=1)[0]
    return clean_target(tail, relationship)


def passive_target(passage: str, match: re.Match[str], relationship: str) -> str:
    head = passage[: match.start()]
    if re.search(r"\babstract\b", head, re.IGNORECASE):
        head = re.split(r"\babstract\b", head, flags=re.IGNORECASE)[-1]
    if re.search(r"\bpresented in this paper\b", head, re.IGNORECASE):
        head = re.split(r"\bpresented in this paper\b", head, flags=re.IGNORECASE)[-1]
    head = re.sub(r",\s+as (?:described|represented|formulated) [^,]+,\s*$", "", head, flags=re.IGNORECASE)
    reviewed_prefix = re.match(r"^.*?\b(?:after|following)\s+.+,\s*([^,]{12,100})$", head, re.IGNORECASE)
    if reviewed_prefix:
        head = reviewed_prefix.group(1)
    head = re.split(r"(?<=[.!?;:])\s+", head)[-1]
    head = re.sub(r"^(?:Abstract|ABSTRACT|Summary|SUMMARY)\s+", "", head)
    return clean_target(head, relationship)


def candidate_from_passage(
    chunk: dict[str, Any], passage: str, rank: int
) -> dict[str, Any] | None:
    for rule_index, rule in enumerate(ACTIVE_RULES):
        match = rule["regex"].search(passage)
        if not match:
            continue
        target = direct_target(passage, match, rule["relationship"])
        if 5 <= len(target) <= 280 and target.casefold() not in {"it", "this", "that", "these", "those"}:
            confidence = rule["confidence"]
            if section_priority(chunk) >= 4:
                confidence = round(confidence - 0.04, 2)
            return {
                "relationship": rule["relationship"],
                "target_type": rule["target_type"],
                "target_name": target,
                "confidence": confidence,
                "method": "explicit_active_cue_v1",
                "excerpt": compact_excerpt(passage, match.start(), match.end()),
                "chunk": chunk,
                "sort_key": (section_priority(chunk), 0, rank, rule_index),
                "notes": (
                    "The source passage uses the verb 'prove'; the controlled vocabulary records it as a reported result."
                    if re.match(r"prov", match.group("verb"), re.IGNORECASE)
                    else None
                ),
            }
    for rule_index, rule in enumerate(PASSIVE_RULES):
        match = rule["regex"].search(passage)
        if not match:
            continue
        tail = passage[match.end() :].lstrip()
        if rule["relationship"] == "STUDIES" and re.match(r"by\s+[A-Z]", tail):
            continue
        if re.match(r"\s*\[[0-9]", tail) and re.match(r"Recently\b", passage, re.IGNORECASE):
            continue
        target = passive_target(passage, match, rule["relationship"])
        if 5 <= len(target) <= 280 and target.casefold() not in {"it", "this", "that", "these", "those"}:
            confidence = rule["confidence"]
            if section_priority(chunk) >= 4:
                confidence = round(confidence - 0.04, 2)
            return {
                "relationship": rule["relationship"],
                "target_type": rule["target_type"],
                "target_name": target,
                "confidence": confidence,
                "method": "explicit_passive_cue_v1",
                "excerpt": compact_excerpt(passage, match.start(), match.end()),
                "chunk": chunk,
                "sort_key": (section_priority(chunk), 1, rank, rule_index),
                "notes": None,
            }
    for rule_index, (regex, relationship, target_type, confidence) in enumerate(SPECIAL_RULES):
        match = regex.search(passage)
        if not match:
            continue
        target = direct_target(passage, match, relationship)
        if len(target) < 5:
            target = clean_target(passage, relationship)
        if 5 <= len(target) <= 280:
            return {
                "relationship": relationship,
                "target_type": target_type,
                "target_name": target,
                "confidence": confidence,
                "method": "explicit_special_cue_v1",
                "excerpt": compact_excerpt(passage, match.start(), match.end()),
                "chunk": chunk,
                "sort_key": (section_priority(chunk), 0, rank, 70 + rule_index),
                "notes": None,
            }
    return None


def title_fallback(paper: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = min(chunks, key=lambda chunk: (chunk["page_number"], chunk["chunk_index"]))
    normalized_title = normalize_space(paper["title"])
    title_words = [
        re.escape(word)
        for word in re.findall(r"[A-Za-z0-9]+", normalize_space(paper["title"]))
        if len(word) >= 3
    ]
    selected = first
    excerpt = normalize_space(first["text"])[:320].strip()
    for chunk in sorted(chunks, key=lambda row: (row["page_number"], row["chunk_index"]))[:8]:
        text = normalize_space(chunk["text"])
        title_position = text.casefold().find(normalized_title.casefold())
        if title_position >= 0:
            selected = chunk
            excerpt = text[title_position : title_position + len(normalized_title)]
            break
        positions = [m.start() for word in title_words[:8] if (m := re.search(rf"\b{word}\b", text, re.I))]
        if len(positions) >= min(4, max(2, len(title_words) // 2)):
            selected = chunk
            start = max(0, min(positions) - 40)
            excerpt = text[start : start + 360].strip()
            break
    target = normalized_title
    return {
        "relationship": "STUDIES",
        "target_type": "scope",
        "target_name": target,
        "confidence": 0.72,
        "method": "explicit_title_scope_v1",
        "excerpt": excerpt,
        "chunk": selected,
        "sort_key": (8, 0, 0, 99),
        "notes": "No qualifying prose cue was selected. The paper title supplies the stated subject; manual review is required.",
    }


def select_candidates(paper: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rank = 0
    for chunk in sorted(chunks, key=lambda row: (section_priority(row), row["page_number"], row["chunk_index"])):
        if section_priority(chunk) >= 4:
            continue
        for passage in split_passages(chunk["text"]):
            rank += 1
            candidate = candidate_from_passage(chunk, passage, rank)
            if candidate:
                candidates.append(candidate)
    candidates.sort(key=lambda row: row["sort_key"])
    return candidates[:1] or [title_fallback(paper, chunks)]


def paper_key(paper: dict[str, Any]) -> str:
    arxiv_ids = paper.get("identifiers", {}).get("arxiv", []) or paper.get("arxiv_ids", [])
    return f"arxiv:{paper['paper_id']}" if arxiv_ids else paper["paper_id"]


def build_proposal(paper: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    relationship = candidate["relationship"]
    target_type = candidate["target_type"]
    target_name = candidate["target_name"]
    target_slug = slug(target_name)
    if len(slug(target_name, limit=400)) > 88:
        target_slug += "-" + hashlib.sha256(target_name.encode()).hexdigest()[:8]
    target_key = f"{target_type}:{target_slug}"
    identity = f"{paper['paper_id']}|{relationship}|{target_key}|{candidate['chunk']['chunk_id']}"
    proposal_id = "full-semantic-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    chunk = candidate["chunk"]
    proposal: dict[str, Any] = {
        "basis": "explicit_text",
        "confidence": candidate["confidence"],
        "evidence": {
            "chunk_id": chunk["chunk_id"],
            "excerpt": candidate["excerpt"],
            "page_number": chunk["page_number"],
            "paper_id": paper["paper_id"],
            "section": chunk.get("section_heading"),
        },
        "method": candidate["method"],
        "proposal_id": proposal_id,
        "relationship": relationship,
        "review_status": "pending",
        "source": {
            "key": paper_key(paper),
            "name": normalize_space(paper["title"]),
            "type": "paper",
        },
        "target": {
            "key": target_key,
            "name": target_name,
            "type": target_type,
        },
    }
    if candidate.get("notes"):
        proposal["notes"] = candidate["notes"]
    return proposal


def load_pilot_aliases() -> dict[str, dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    for path in sorted(PILOT_ENRICHMENT.glob("*/ENTITY_ALIASES.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("entities", []):
            key = row["key"]
            name = row.get("canonical_name") or row.get("name")
            entry = aliases.setdefault(key, {"key": key, "canonical_name": name, "aliases": set()})
            if not entry["canonical_name"]:
                entry["canonical_name"] = name
            entry["aliases"].update(row.get("aliases", []))
    return aliases


def build_nodes(proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        for role in ("source", "target"):
            entity = proposal[role]
            node = aggregated.setdefault(
                entity["key"],
                {
                    "key": entity["key"],
                    "name": entity["name"],
                    "type": entity["type"],
                    "paper_ids": set(),
                    "proposal_ids": set(),
                },
            )
            node["paper_ids"].add(proposal["evidence"]["paper_id"])
            node["proposal_ids"].add(proposal["proposal_id"])
    inherited = load_pilot_aliases()
    alias_rows: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    for key in sorted(aggregated):
        node = aggregated[key]
        alias_values: set[str] = set()
        if key in inherited:
            alias_values.update(inherited[key]["aliases"])
        if node["type"] == "paper":
            if key.startswith("arxiv:"):
                alias_values.add("arXiv:" + key.removeprefix("arxiv:"))
            elif key.startswith("inspire:"):
                alias_values.add("INSPIRE:" + key.removeprefix("inspire:"))
        nodes.append(
            {
                "aliases": sorted(alias_values, key=str.casefold),
                "key": key,
                "name": node["name"],
                "paper_ids": sorted(node["paper_ids"]),
                "proposal_ids": sorted(node["proposal_ids"]),
                "type": node["type"],
            }
        )
        if alias_values:
            alias_rows.append(
                {
                    "aliases": sorted(alias_values, key=str.casefold),
                    "canonical_name": node["name"],
                    "key": key,
                }
            )
    alias_payload = {
        "schema_version": "gates-full-semantic-aliases-v1",
        "source": "Pilot aliases are inherited only when the canonical key occurs in a full-corpus proposal.",
        "entities": alias_rows,
    }
    return nodes, alias_payload


def build(manifest_path: Path, extraction_root: Path, output_root: Path) -> dict[str, Any]:
    manifest, papers = load_manifest(manifest_path)
    index_rows = [
        json.loads(line)
        for line in (extraction_root / "extraction_index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    success_rows = [row for row in index_rows if row.get("status") == "success"]
    proposals: list[dict[str, Any]] = []
    paper_methods: dict[str, list[str]] = {}
    for index_row in sorted(success_rows, key=lambda row: row["paper_id"]):
        paper_id = index_row["paper_id"]
        paper = papers[paper_id]
        shard_path = extraction_root / index_row["shard_path"]
        chunks = [json.loads(line) for line in shard_path.read_text().splitlines() if line.strip()]
        candidates = select_candidates(paper, chunks)
        built = [build_proposal(paper, candidate) for candidate in candidates]
        proposals.extend(built)
        paper_methods[paper_id] = [proposal["method"] for proposal in built]

    proposals.sort(key=lambda row: (row["evidence"]["paper_id"], row["proposal_id"]))
    nodes, aliases = build_nodes(proposals)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "proposals.jsonl").write_text("".join(json_line(row) + "\n" for row in proposals))
    (output_root / "nodes.jsonl").write_text("".join(json_line(row) + "\n" for row in nodes))
    write_json(output_root / "ENTITY_ALIASES.json", aliases)

    covered = sorted({row["evidence"]["paper_id"] for row in proposals})
    fallback = sorted(
        paper_id for paper_id, methods in paper_methods.items() if "explicit_title_scope_v1" in methods
    )
    relation_counts = Counter(row["relationship"] for row in proposals)
    method_counts = Counter(row["method"] for row in proposals)
    type_counts = Counter(row["target"]["type"] for row in proposals)
    downloaded = sorted(
        row["paper_id"]
        for row in papers.values()
        if row.get("full_text", {}).get("status") == "verified_local_pdf"
    )
    coverage = {
        "schema_version": "gates-full-semantic-coverage-v1",
        "corpus_id": manifest.get("corpus_id") if isinstance(manifest, dict) else None,
        "downloaded_paper_count": len(downloaded),
        "extracted_paper_count": len(success_rows),
        "covered_paper_count": len(covered),
        "uncovered_paper_ids": sorted(set(downloaded) - set(covered)),
        "proposal_count": len(proposals),
        "node_count": len(nodes),
        "relationship_counts": dict(sorted(relation_counts.items())),
        "target_type_counts": dict(sorted(type_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "title_fallback_count": len(fallback),
        "title_fallback_paper_ids": fallback,
        "paper_proposal_counts": dict(sorted(Counter(row["evidence"]["paper_id"] for row in proposals).items())),
    }
    write_json(output_root / "COVERAGE.json", coverage)
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extraction-root", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--output-root", type=Path, default=HERE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(args.manifest, args.extraction_root, args.output_root), indent=2, sort_keys=True))
