#!/usr/bin/env python3
"""Build and optionally import the Gates literature GraphRAG pilot.

Dry-run is the default. Database writes require the explicit ``--apply`` flag.
The importer writes only to dedicated ``gates_pilot_*`` tables and never
deletes rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CORPUS_ID = "gates_literature_pilot"
NODE_TYPES = {"paper", "author", "concept", "claim", "result", "series"}
EDGE_BASES = {"bibliographic", "explicit_text", "automated_inference", "manual"}
REVIEW_STATUSES = {"observed", "pending", "accepted", "rejected"}


class InputError(ValueError):
    """Input violates the pilot import contract."""


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value[:80] or "unnamed"


def _id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
    return [value]


def _first(value: Any) -> str:
    values = _list(value)
    return str(values[0]).strip() if values else ""


def _normalize_doi(value: str) -> str:
    value = value.strip().casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value.removeprefix("doi:").strip()


def _normalize_arxiv(value: str) -> str:
    value = value.strip().casefold()
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value)
    value = value.removesuffix(".pdf")
    value = value.removeprefix("arxiv:")
    return re.sub(r"v\d+$", "", value).strip()


def _normalize_inspire(value: str) -> str:
    value = value.strip().casefold().rstrip("/")
    value = re.sub(r"^https?://inspirehep\.net/literature/", "", value)
    return value.removeprefix("inspire:").strip()


def normalize_identifier(kind: str, value: Any) -> tuple[str, str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    kind = kind.casefold().strip().replace("_id", "")
    if kind in {"inspire", "inspirehep"}:
        normalized = _normalize_inspire(text)
        kind = "inspire"
    elif kind in {"arxiv", "arxiv_ids"}:
        normalized = _normalize_arxiv(text)
        kind = "arxiv"
    elif kind in {"doi", "dois"}:
        normalized = _normalize_doi(text)
        kind = "doi"
    else:
        normalized = text.casefold()
    return (kind, normalized) if normalized else None


def identifiers_from_record(record: dict[str, Any]) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    field_map = {
        "inspire_id": "inspire",
        "inspire_ids": "inspire",
        "inspire_url": "inspire",
        "arxiv_id": "arxiv",
        "arxiv_ids": "arxiv",
        "doi": "doi",
        "dois": "doi",
    }
    for field_name, kind in field_map.items():
        for value in _list(record.get(field_name)):
            item = normalize_identifier(kind, value)
            if item:
                found.add(item)
    identifiers = record.get("identifiers", {})
    if isinstance(identifiers, dict):
        for kind, values in identifiers.items():
            for value in _list(values):
                item = normalize_identifier(kind, value)
                if item:
                    found.add(item)
    stable = record.get("stable_identifier") or record.get("stable_id")
    if stable:
        stable_text = str(stable)
        if ":" in stable_text:
            kind, value = stable_text.split(":", 1)
            item = normalize_identifier(kind, value)
            if item:
                found.add(item)
    return sorted(found)


def stable_identifier(record: dict[str, Any]) -> str:
    identifiers = identifiers_from_record(record)
    # The focused pilot manifest designates arXiv as its canonical paper key.
    # INSPIRE and DOI remain alternate identifiers used for deduplication.
    priority = {"arxiv": 0, "inspire": 1, "doi": 2}
    if identifiers:
        kind, value = sorted(identifiers, key=lambda item: (priority.get(item[0], 9), item))[0]
        return f"{kind}:{value}"
    title = str(record.get("title") or "").strip()
    if not title:
        raise InputError("paper record has neither a stable identifier nor a title")
    return f"title-sha256:{_sha256_text(' '.join(title.casefold().split()))}"


def _parse_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        names = []
        for author in value:
            if isinstance(author, dict):
                name = author.get("name") or author.get("full_name")
            else:
                name = author
            if name:
                names.append(str(name).strip())
        return names
    if not value:
        return []
    text = str(value).strip()
    # The collection manifest uses semicolons between authors. A comma often
    # separates family and initials, so it is not treated as a delimiter.
    return [part.strip() for part in text.split(";") if part.strip()]


@dataclass
class Paper:
    paper_id: str
    stable_identifier: str
    title: str
    publication_year: int | None
    abstract: str | None
    is_stub: bool
    properties: dict[str, Any]


@dataclass
class Artifact:
    artifact_id: str
    paper_id: str
    artifact_type: str
    local_path: str | None
    source_url: str | None
    sha256: str | None
    properties: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section_title: str | None
    section_path: str | None
    page_start: int | None
    page_end: int | None
    chunk_index: int
    content: str
    content_sha256: str
    properties: dict[str, Any]


@dataclass
class Node:
    node_id: str
    node_type: str
    canonical_key: str
    name: str
    description: str | None
    properties: dict[str, Any]


@dataclass
class Edge:
    edge_id: str
    src_node_id: str
    dst_node_id: str
    relationship: str
    description: str | None
    basis: str
    review_status: str
    confidence: float | None
    properties: dict[str, Any]


@dataclass
class Evidence:
    evidence_id: str
    target_id: str
    paper_id: str
    chunk_id: str | None
    source_kind: str
    locator: str | None
    excerpt: str | None
    extraction_method: str
    confidence: float | None
    properties: dict[str, Any]


@dataclass
class ImportPlan:
    corpus_id: str
    manifest_sha256: str
    extraction_sha256: str
    papers: dict[str, Paper] = field(default_factory=dict)
    identifiers: dict[tuple[str, str], str] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    node_evidence: dict[str, Evidence] = field(default_factory=dict)
    edge_evidence: dict[str, Evidence] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, Any]:
        return {
            "papers": len(self.papers),
            "identifiers": len(self.identifiers),
            "artifacts": len(self.artifacts),
            "chunks": len(self.chunks),
            "nodes": len(self.nodes),
            "nodes_by_type": dict(sorted(Counter(n.node_type for n in self.nodes.values()).items())),
            "edges": len(self.edges),
            "edges_by_relationship": dict(
                sorted(Counter(e.relationship for e in self.edges.values()).items())
            ),
            "edges_by_review_status": dict(
                sorted(Counter(e.review_status for e in self.edges.values()).items())
            ),
            "node_evidence": len(self.node_evidence),
            "edge_evidence": len(self.edge_evidence),
            "warnings": len(self.warnings),
        }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("papers") or data.get("records")
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise InputError("manifest must be a JSON list, {papers: [...]}, or CSV")
    return data


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise InputError(f"{path}:{line_number}: each JSONL record must be an object")
                row["_input_file"] = str(path)
                row["_input_line"] = line_number
                rows.append(row)
    return rows


def _merge_manifest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge artifacts that share any stable identifier using union-find."""
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    owners: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for identifier in identifiers_from_record(record):
            if identifier in owners:
                union(index, owners[identifier])
            else:
                owners[identifier] = index

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(root(index), []).append(record)

    merged: list[dict[str, Any]] = []
    for rows in groups.values():
        base = dict(rows[0])
        all_identifiers: dict[str, set[str]] = {}
        for row in rows:
            for kind, value in identifiers_from_record(row):
                all_identifiers.setdefault(kind, set()).add(value)
            for key, value in row.items():
                if not base.get(key) and value:
                    base[key] = value
        base["identifiers"] = {key: sorted(values) for key, values in all_identifiers.items()}
        base["_artifacts"] = rows
        merged.append(base)
    return merged


def _paper_reference(row: dict[str, Any]) -> str | None:
    return str(
        row.get("paper_id")
        or row.get("stable_identifier")
        or row.get("stable_id")
        or row.get("source_paper_id")
        or ""
    ).strip() or None


def _resolve_paper(plan: ImportPlan, reference: Any) -> str | None:
    if isinstance(reference, dict):
        candidates = identifiers_from_record(reference)
        title = str(reference.get("title") or "").strip()
    else:
        text = str(reference or "").strip()
        if not text:
            return None
        candidates = []
        if ":" in text:
            kind, value = text.split(":", 1)
            item = normalize_identifier(kind, value)
            if item:
                candidates.append(item)
        candidates += [item for kind in ("inspire", "doi", "arxiv") if (item := normalize_identifier(kind, text))]
        title = text
    for candidate in candidates:
        if candidate in plan.identifiers:
            return plan.identifiers[candidate]
    for paper in plan.papers.values():
        if title and paper.title.casefold() == title.casefold():
            return paper.paper_id
        if title in {paper.paper_id, paper.stable_identifier}:
            return paper.paper_id
    return None


def _manifest_evidence(plan: ImportPlan, target_id: str, paper_id: str, kind: str) -> Evidence:
    eid = _id("ev", "manifest", target_id, paper_id, kind)
    return Evidence(
        evidence_id=eid,
        target_id=target_id,
        paper_id=paper_id,
        chunk_id=None,
        source_kind="manifest",
        locator=None,
        excerpt=None,
        extraction_method="manifest",
        confidence=1.0,
        properties={},
    )


def _node(
    plan: ImportPlan,
    node_type: str,
    canonical_key: str,
    name: str,
    description: str | None = None,
    properties: dict[str, Any] | None = None,
) -> Node:
    node_type = node_type.casefold().strip()
    if node_type not in NODE_TYPES:
        raise InputError(f"unsupported node type: {node_type}")
    canonical_key = canonical_key.strip().casefold()
    node_id = _id("node", plan.corpus_id, node_type, canonical_key)
    candidate = Node(node_id, node_type, canonical_key, name.strip(), description, properties or {})
    current = plan.nodes.get(node_id)
    if not current or len(candidate.description or "") > len(current.description or ""):
        plan.nodes[node_id] = candidate
    return plan.nodes[node_id]


def _edge(
    plan: ImportPlan,
    src: Node,
    dst: Node,
    relationship: str,
    description: str | None,
    basis: str,
    review_status: str | None,
    confidence: float | None,
    properties: dict[str, Any] | None = None,
) -> Edge:
    relationship = re.sub(r"[^A-Z0-9]+", "_", relationship.upper()).strip("_")
    if not relationship:
        raise InputError("relationship is empty")
    basis = basis.casefold().strip()
    if basis not in EDGE_BASES:
        raise InputError(f"unsupported edge basis: {basis}")
    if review_status is None:
        review_status = "pending" if basis in {"automated_inference", "explicit_text"} else "observed"
    review_status = review_status.casefold().strip()
    if review_status not in REVIEW_STATUSES:
        raise InputError(f"unsupported review status: {review_status}")
    if basis == "automated_inference" and review_status == "observed":
        raise InputError("automated_inference edges cannot be marked observed")
    edge_id = _id("edge", plan.corpus_id, src.node_id, dst.node_id, relationship)
    edge = Edge(
        edge_id, src.node_id, dst.node_id, relationship, description, basis,
        review_status, confidence, properties or {},
    )
    current = plan.edges.get(edge_id)
    if not current or len(edge.description or "") > len(current.description or ""):
        plan.edges[edge_id] = edge
    return plan.edges[edge_id]


def _evidence_from_item(
    plan: ImportPlan,
    target_id: str,
    paper_id: str,
    item: dict[str, Any],
    chunk_id: str | None,
    default_method: str,
) -> Evidence:
    locator = item.get("locator") or item.get("section") or item.get("section_heading")
    page_number = item.get("page_number", item.get("page"))
    if page_number is not None:
        locator = locator or f"physical page {page_number}"
    excerpt = item.get("excerpt") or item.get("evidence_text")
    extraction_provenance = item.get("extraction_provenance", {})
    if not isinstance(extraction_provenance, dict):
        extraction_provenance = {}
    method = str(
        item.get("extraction_method")
        or extraction_provenance.get("strategy")
        or extraction_provenance.get("backend")
        or default_method
    )
    source_kind = str(item.get("source_kind") or ("chunk" if chunk_id else "extracted_record"))
    confidence = item.get("confidence")
    confidence = float(confidence) if confidence is not None else None
    eid = _id(
        "ev", plan.corpus_id, target_id, paper_id, chunk_id or "", source_kind,
        str(locator or ""), str(excerpt or ""), method,
    )
    return Evidence(
        eid, target_id, paper_id, chunk_id, source_kind,
        str(locator) if locator else None,
        str(excerpt) if excerpt else None,
        method, confidence,
        {
            "input_file": item.get("_input_file"),
            "input_line": item.get("_input_line"),
            "page_number": page_number,
            "page_label": item.get("page_label"),
            "page_line_start": item.get("page_line_start"),
            "page_line_end": item.get("page_line_end"),
            "bbox": item.get("bbox"),
            "section_heading": item.get("section_heading"),
            "section_heading_source": item.get("section_heading_source"),
            "extraction_provenance": extraction_provenance,
        },
    )


def _create_stub_paper(
    plan: ImportPlan,
    reference: dict[str, Any],
    source_paper_id: str,
    chunk_id: str | None,
) -> str:
    sid = stable_identifier(reference)
    paper_id = _id("paper", plan.corpus_id, sid)
    title = str(reference.get("title") or sid).strip()
    paper = Paper(paper_id, sid, title, None, None, True, {"citation_stub": True})
    plan.papers[paper_id] = paper
    for ident in identifiers_from_record(reference):
        plan.identifiers[ident] = paper_id
    paper_node = _node(plan, "paper", sid, title, properties={"paper_id": paper_id, "is_stub": True})
    evidence = _evidence_from_item(
        plan,
        paper_node.node_id,
        source_paper_id,
        reference,
        chunk_id,
        str(reference.get("extraction_method") or "reference_parser"),
    )
    plan.node_evidence[evidence.evidence_id] = evidence
    return paper_id


def build_plan(
    manifest_path: Path,
    extracted_paths: list[Path],
    corpus_id: str = DEFAULT_CORPUS_ID,
) -> ImportPlan:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", corpus_id):
        raise InputError("corpus_id must contain 3-64 lowercase letters, digits, underscores, or hyphens")
    records = _merge_manifest_records(load_manifest(manifest_path))
    plan = ImportPlan(
        corpus_id=corpus_id,
        manifest_sha256=_file_sha256(manifest_path),
        extraction_sha256=_sha256_text("\n".join(_file_sha256(path) for path in extracted_paths)),
    )

    for record in records:
        sid = stable_identifier(record)
        paper_id = _id("paper", corpus_id, sid)
        title = str(record.get("title") or "").strip()
        if not title:
            raise InputError(f"{sid}: missing title")
        year_text = str(
            record.get("year")
            or record.get("publication_year")
            or record.get("eprint_year")
            or ""
        ).strip()
        year = int(year_text) if year_text.isdigit() else None
        reserved = {
            "title", "year", "publication_year", "abstract", "authors", "identifiers",
            "_artifacts", "pdf_filename", "local_pdf_filename", "local_pdf_path",
            "pdf_source_url", "arxiv_pdf_url", "sha256",
        }
        properties = {key: value for key, value in record.items() if key not in reserved and value not in (None, "", [])}
        plan.papers[paper_id] = Paper(
            paper_id, sid, title, year,
            str(record.get("abstract") or "").strip() or None,
            False, properties,
        )
        for identifier in identifiers_from_record(record):
            owner = plan.identifiers.get(identifier)
            if owner and owner != paper_id:
                raise InputError(f"identifier collision after deduplication: {identifier}")
            plan.identifiers[identifier] = paper_id

        paper_node = _node(plan, "paper", sid, title, plan.papers[paper_id].abstract, {"paper_id": paper_id})
        evidence = _manifest_evidence(plan, paper_node.node_id, paper_id, "paper")
        plan.node_evidence[evidence.evidence_id] = evidence

        for author_name in _parse_authors(record.get("authors")):
            author = _node(plan, "author", " ".join(author_name.casefold().split()), author_name)
            author_ev = _manifest_evidence(plan, author.node_id, paper_id, "author")
            plan.node_evidence[author_ev.evidence_id] = author_ev
            edge = _edge(
                plan, paper_node, author, "AUTHORED_BY", None,
                "bibliographic", "observed", 1.0,
            )
            edge_ev = _manifest_evidence(plan, edge.edge_id, paper_id, "authorship")
            plan.edge_evidence[edge_ev.evidence_id] = edge_ev

        series_name = str(record.get("series") or "").strip()
        if series_name:
            series = _node(
                plan,
                "series",
                series_name,
                series_name.replace("_", " "),
                str(record.get("series_role") or "").strip() or None,
                {"source": "curated_manifest"},
            )
            series_ev = _manifest_evidence(plan, series.node_id, paper_id, "series")
            plan.node_evidence[series_ev.evidence_id] = series_ev
            confidence_label = str(record.get("selection_confidence") or "").casefold()
            review_status = (
                "pending"
                if "unconfirmed" in confidence_label or "inferred" in confidence_label
                else "accepted"
            )
            edge = _edge(
                plan,
                paper_node,
                series,
                "PART_OF_SERIES",
                str(record.get("series_role") or "").strip() or None,
                "manual",
                review_status,
                None,
                {"selection_confidence": record.get("selection_confidence")},
            )
            edge_ev = _manifest_evidence(plan, edge.edge_id, paper_id, "series_membership")
            plan.edge_evidence[edge_ev.evidence_id] = edge_ev

        for index, artifact_record in enumerate(record.get("_artifacts", [record])):
            local_path = str(
                artifact_record.get("local_pdf_path")
                or artifact_record.get("local_path")
                or artifact_record.get("local_pdf_filename")
                or artifact_record.get("pdf_filename")
                or ""
            ).strip() or None
            source_url = str(
                artifact_record.get("arxiv_pdf_url")
                or artifact_record.get("pdf_source_url")
                or artifact_record.get("source_url")
                or ""
            ).strip() or None
            sha256 = str(artifact_record.get("sha256") or "").strip().casefold() or None
            if not any((local_path, source_url, sha256)):
                continue
            artifact_id = _id("artifact", corpus_id, paper_id, sha256 or local_path or source_url or str(index))
            plan.artifacts[artifact_id] = Artifact(
                artifact_id, paper_id, "pdf", local_path, source_url, sha256,
                {"manifest_artifact_index": index},
            )

    extracted_rows = load_jsonl(extracted_paths)
    for row in extracted_rows:
        paper_id = _resolve_paper(plan, _paper_reference(row) or row)
        if not paper_id:
            raise InputError(
                f"{row.get('_input_file')}:{row.get('_input_line')}: cannot resolve source paper"
            )
        paper = plan.papers[paper_id]
        paper_node = _node(plan, "paper", paper.stable_identifier, paper.title, paper.abstract, {"paper_id": paper_id})

        chunk_items = row.get("chunks") if isinstance(row.get("chunks"), list) else []
        if row.get("content") or row.get("text"):
            chunk_items = [row]
        row_chunk_ids: list[str] = []
        for fallback_index, chunk_item in enumerate(chunk_items):
            content = str(chunk_item.get("content") or chunk_item.get("text") or "").strip()
            if not content:
                continue
            content_hash = _sha256_text(content)
            chunk_index = int(chunk_item.get("chunk_index", fallback_index))
            chunk_id = str(chunk_item.get("chunk_id") or _id("chunk", corpus_id, paper_id, content_hash))
            page_start = chunk_item.get(
                "page_start", chunk_item.get("page_number", chunk_item.get("page"))
            )
            page_end = chunk_item.get("page_end", page_start)
            chunk_reserved = {
                "content", "text", "chunk_id", "paper_id", "stable_id",
                "stable_identifier", "source_paper_id", "chunk_index",
                "page_start", "page_end", "page", "page_number",
                "section", "section_title", "section_heading", "section_path",
                "concepts", "claims", "results", "series", "citations",
                "relationships", "entities", "_input_file", "_input_line",
            }
            chunk_properties = {
                key: value
                for key, value in chunk_item.items()
                if key not in chunk_reserved and value not in (None, "", [])
            }
            chunk_properties.update(
                {"input_file": row.get("_input_file"), "input_line": row.get("_input_line")}
            )
            plan.chunks[chunk_id] = Chunk(
                chunk_id, paper_id,
                str(
                    chunk_item.get("section_heading")
                    or chunk_item.get("section_title")
                    or chunk_item.get("section")
                    or ""
                ).strip() or None,
                str(chunk_item.get("section_path") or "").strip() or None,
                int(page_start) if page_start is not None else None,
                int(page_end) if page_end is not None else None,
                chunk_index, content, content_hash, chunk_properties,
            )
            row_chunk_ids.append(chunk_id)
        default_chunk_id = str(row.get("chunk_id") or "") or (row_chunk_ids[0] if len(row_chunk_ids) == 1 else None)

        entity_lookup: dict[str, Node] = {
            paper.paper_id: paper_node,
            paper.stable_identifier: paper_node,
            paper.title.casefold(): paper_node,
        }
        entity_groups = [
            ("concept", row.get("concepts", [])),
            ("claim", row.get("claims", [])),
            ("result", row.get("results", [])),
            ("series", row.get("series", [])),
        ]
        generic_entities = row.get("entities", [])
        if isinstance(generic_entities, list):
            for item in generic_entities:
                if isinstance(item, dict):
                    entity_groups.append((str(item.get("type") or "concept"), [item]))
        for default_type, items in entity_groups:
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    item = {"name": item}
                if not isinstance(item, dict):
                    continue
                node_type = str(item.get("type") or default_type).casefold()
                name = str(item.get("name") or item.get("title") or item.get("text") or "").strip()
                if not name:
                    raise InputError(f"{paper.stable_identifier}: {node_type} has no name/text")
                key = str(item.get("canonical_key") or item.get("id") or _sha256_text(" ".join(name.casefold().split())))
                node = _node(
                    plan, node_type, key, name,
                    str(item.get("description") or item.get("text") or "").strip() or None,
                    {"source_paper_id": paper_id},
                )
                for alias in (str(item.get("id") or ""), name.casefold(), key):
                    if alias:
                        entity_lookup[alias] = node
                item_chunk_id = str(item.get("chunk_id") or "") or default_chunk_id
                evidence = _evidence_from_item(
                    plan, node.node_id, paper_id, {**row, **item}, item_chunk_id,
                    str(item.get("extraction_method") or "automated_extraction"),
                )
                plan.node_evidence[evidence.evidence_id] = evidence
                relation = {
                    "concept": "DISCUSSES",
                    "claim": "MAKES_CLAIM",
                    "result": "REPORTS_RESULT",
                    "series": "PART_OF_SERIES",
                }[node_type]
                basis = str(item.get("basis") or "automated_inference")
                edge = _edge(
                    plan, paper_node, node, relation,
                    str(item.get("relationship_description") or "").strip() or None,
                    basis, item.get("review_status"),
                    float(item["confidence"]) if item.get("confidence") is not None else None,
                )
                edge_evidence = _evidence_from_item(
                    plan, edge.edge_id, paper_id, {**row, **item}, item_chunk_id,
                    str(item.get("extraction_method") or "automated_extraction"),
                )
                plan.edge_evidence[edge_evidence.evidence_id] = edge_evidence

        citations = row.get("citations", [])
        if isinstance(citations, list):
            for citation in citations:
                if isinstance(citation, str):
                    citation = {"title": citation}
                if not isinstance(citation, dict):
                    continue
                target_paper_id = _resolve_paper(plan, citation)
                if not target_paper_id:
                    target_paper_id = _create_stub_paper(
                        plan,
                        {**row, **citation},
                        paper_id,
                        str(citation.get("chunk_id") or "") or default_chunk_id,
                    )
                target_paper = plan.papers[target_paper_id]
                target_node = _node(
                    plan, "paper", target_paper.stable_identifier, target_paper.title,
                    target_paper.abstract, {"paper_id": target_paper_id, "is_stub": target_paper.is_stub},
                )
                basis = str(citation.get("basis") or "bibliographic")
                edge = _edge(
                    plan, paper_node, target_node, "CITES",
                    str(citation.get("description") or "").strip() or None,
                    basis, citation.get("review_status"),
                    float(citation["confidence"]) if citation.get("confidence") is not None else 1.0,
                )
                evidence = _evidence_from_item(
                    plan, edge.edge_id, paper_id, {**row, **citation},
                    str(citation.get("chunk_id") or "") or default_chunk_id,
                    str(citation.get("extraction_method") or "reference_parser"),
                )
                plan.edge_evidence[evidence.evidence_id] = evidence

        relationships = row.get("relationships", [])
        if isinstance(relationships, list):
            for relation in relationships:
                if not isinstance(relation, dict):
                    continue
                source_key = str(relation.get("source") or relation.get("src") or "").strip()
                target_key = str(relation.get("target") or relation.get("dst") or "").strip()
                src = entity_lookup.get(source_key) or entity_lookup.get(source_key.casefold())
                dst = entity_lookup.get(target_key) or entity_lookup.get(target_key.casefold())
                if not src or not dst:
                    plan.warnings.append(
                        f"{paper.stable_identifier}: skipped unresolved relationship {source_key!r} -> {target_key!r}"
                    )
                    continue
                basis = str(relation.get("basis") or "automated_inference")
                edge = _edge(
                    plan, src, dst,
                    str(relation.get("relationship") or relation.get("type") or "RELATED_TO"),
                    str(relation.get("description") or "").strip() or None,
                    basis, relation.get("review_status"),
                    float(relation["confidence"]) if relation.get("confidence") is not None else None,
                )
                evidence = _evidence_from_item(
                    plan, edge.edge_id, paper_id, {**row, **relation},
                    str(relation.get("chunk_id") or "") or default_chunk_id,
                    str(relation.get("extraction_method") or "automated_extraction"),
                )
                plan.edge_evidence[evidence.evidence_id] = evidence

    validate_plan(plan)
    return plan


def validate_plan(plan: ImportPlan) -> None:
    errors: list[str] = []
    for paper in plan.papers.values():
        if not paper.title or not paper.stable_identifier:
            errors.append(f"paper {paper.paper_id} lacks title or stable identifier")
    for node in plan.nodes.values():
        if not any(ev.target_id == node.node_id for ev in plan.node_evidence.values()):
            errors.append(f"node {node.node_id} has no provenance")
    for edge in plan.edges.values():
        if edge.src_node_id not in plan.nodes or edge.dst_node_id not in plan.nodes:
            errors.append(f"edge {edge.edge_id} has a missing endpoint")
        if not any(ev.target_id == edge.edge_id for ev in plan.edge_evidence.values()):
            errors.append(f"edge {edge.edge_id} has no provenance")
        if edge.basis == "automated_inference" and edge.review_status == "observed":
            errors.append(f"automated edge {edge.edge_id} cannot be observed")
    for ev in list(plan.node_evidence.values()) + list(plan.edge_evidence.values()):
        if ev.paper_id not in plan.papers:
            errors.append(f"evidence {ev.evidence_id} references missing paper")
        if ev.chunk_id and ev.chunk_id not in plan.chunks:
            errors.append(f"evidence {ev.evidence_id} references missing chunk")
    if errors:
        raise InputError("validation failed:\n  - " + "\n  - ".join(errors))


def _execute_many(cur: Any, sql: str, rows: Iterable[tuple[Any, ...]]) -> None:
    for row in rows:
        cur.execute(sql, row)


def apply_plan(plan: ImportPlan, dsn: str) -> None:
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2 is required only for --apply") from exc
    if not dsn:
        raise RuntimeError("--apply requires --dsn or GATES_GRAPHRAG_DSN")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (plan.corpus_id,))
            cur.execute(
                """INSERT INTO gates_pilot_corpora
                   (corpus_id, description, manifest_sha256, extraction_sha256)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (corpus_id) DO UPDATE SET
                     manifest_sha256=EXCLUDED.manifest_sha256,
                     extraction_sha256=EXCLUDED.extraction_sha256,
                     updated_at=now()""",
                (plan.corpus_id, "Focused S. James Gates Jr. literature pilot", plan.manifest_sha256, plan.extraction_sha256),
            )
            _execute_many(cur, """INSERT INTO gates_pilot_papers
                (corpus_id,paper_id,stable_identifier,title,publication_year,abstract,is_stub,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,paper_id) DO UPDATE SET
                  stable_identifier=EXCLUDED.stable_identifier,title=EXCLUDED.title,
                  publication_year=EXCLUDED.publication_year,abstract=EXCLUDED.abstract,
                  is_stub=EXCLUDED.is_stub,properties=EXCLUDED.properties,updated_at=now()""", (
                    (plan.corpus_id,p.paper_id,p.stable_identifier,p.title,p.publication_year,p.abstract,p.is_stub,_json(p.properties))
                    for p in plan.papers.values()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_identifiers
                (corpus_id,identifier_type,identifier_value,paper_id) VALUES (%s,%s,%s,%s)
                ON CONFLICT (corpus_id,identifier_type,identifier_value) DO UPDATE SET paper_id=EXCLUDED.paper_id""", (
                    (plan.corpus_id,kind,value,paper_id) for (kind,value),paper_id in plan.identifiers.items()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_artifacts
                (corpus_id,artifact_id,paper_id,artifact_type,local_path,source_url,sha256,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,artifact_id) DO UPDATE SET
                  local_path=EXCLUDED.local_path,source_url=EXCLUDED.source_url,
                  sha256=EXCLUDED.sha256,properties=EXCLUDED.properties,updated_at=now()""", (
                    (plan.corpus_id,a.artifact_id,a.paper_id,a.artifact_type,a.local_path,a.source_url,a.sha256,_json(a.properties))
                    for a in plan.artifacts.values()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_chunks
                (corpus_id,chunk_id,paper_id,section_title,section_path,page_start,page_end,chunk_index,content,content_sha256,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,chunk_id) DO UPDATE SET
                  section_title=EXCLUDED.section_title,section_path=EXCLUDED.section_path,
                  page_start=EXCLUDED.page_start,page_end=EXCLUDED.page_end,
                  chunk_index=EXCLUDED.chunk_index,content=EXCLUDED.content,
                  content_sha256=EXCLUDED.content_sha256,properties=EXCLUDED.properties,updated_at=now()""", (
                    (plan.corpus_id,c.chunk_id,c.paper_id,c.section_title,c.section_path,c.page_start,c.page_end,c.chunk_index,c.content,c.content_sha256,_json(c.properties))
                    for c in plan.chunks.values()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_nodes
                (corpus_id,node_id,node_type,canonical_key,name,description,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,node_id) DO UPDATE SET
                  name=EXCLUDED.name,description=EXCLUDED.description,
                  properties=EXCLUDED.properties,updated_at=now()""", (
                    (plan.corpus_id,n.node_id,n.node_type,n.canonical_key,n.name,n.description,_json(n.properties))
                    for n in plan.nodes.values()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_edges
                (corpus_id,edge_id,src_node_id,dst_node_id,relationship,description,basis,review_status,confidence,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,edge_id) DO UPDATE SET
                  description=EXCLUDED.description,basis=EXCLUDED.basis,
                  review_status=CASE
                    WHEN gates_pilot_edges.review_status IN ('accepted','rejected')
                      AND EXCLUDED.review_status='pending'
                    THEN gates_pilot_edges.review_status
                    ELSE EXCLUDED.review_status
                  END,
                  confidence=EXCLUDED.confidence,
                  properties=EXCLUDED.properties,updated_at=now()""", (
                    (plan.corpus_id,e.edge_id,e.src_node_id,e.dst_node_id,e.relationship,e.description,e.basis,e.review_status,e.confidence,_json(e.properties))
                    for e in plan.edges.values()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_node_evidence
                (corpus_id,evidence_id,node_id,paper_id,chunk_id,source_kind,locator,excerpt,extraction_method,confidence,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,evidence_id) DO UPDATE SET
                  locator=EXCLUDED.locator,excerpt=EXCLUDED.excerpt,
                  confidence=EXCLUDED.confidence,properties=EXCLUDED.properties""", (
                    (plan.corpus_id,e.evidence_id,e.target_id,e.paper_id,e.chunk_id,e.source_kind,e.locator,e.excerpt,e.extraction_method,e.confidence,_json(e.properties))
                    for e in plan.node_evidence.values()
                ))
            _execute_many(cur, """INSERT INTO gates_pilot_edge_evidence
                (corpus_id,evidence_id,edge_id,paper_id,chunk_id,source_kind,locator,excerpt,extraction_method,confidence,properties)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (corpus_id,evidence_id) DO UPDATE SET
                  locator=EXCLUDED.locator,excerpt=EXCLUDED.excerpt,
                  confidence=EXCLUDED.confidence,properties=EXCLUDED.properties""", (
                    (plan.corpus_id,e.evidence_id,e.target_id,e.paper_id,e.chunk_id,e.source_kind,e.locator,e.excerpt,e.extraction_method,e.confidence,_json(e.properties))
                    for e in plan.edge_evidence.values()
                ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def report(plan: ImportPlan, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "corpus_id": plan.corpus_id,
        "manifest_sha256": plan.manifest_sha256,
        "extraction_sha256": plan.extraction_sha256,
        "counts": plan.counts(),
        "warnings": plan.warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, action="append", required=True)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print the import plan; default")
    mode.add_argument("--validate-only", action="store_true", help="validate input without database access")
    mode.add_argument("--apply", action="store_true", help="write to dedicated gates_pilot_* tables")
    parser.add_argument("--dsn", default=os.environ.get("GATES_GRAPHRAG_DSN"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.manifest, args.extracted, args.corpus_id)
        selected_mode = "apply" if args.apply else "validate-only" if args.validate_only else "dry-run"
        if args.apply:
            apply_plan(plan, args.dsn or "")
        output = report(plan, selected_mode)
        rendered = json.dumps(output, indent=2, sort_keys=True)
        print(rendered)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
        return 0
    except (InputError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
