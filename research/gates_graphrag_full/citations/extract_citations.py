#!/usr/bin/env python3
"""Extract conservative, page-anchored citations from the Gates PDF corpus.

Only bibliography entries are considered citations.  Resolution proceeds by
exact identifiers first and normalized title containment second.  The script
does not infer citations from prose mentions, co-occurrence, or similarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz


SCHEMA_VERSION = "gates-citation-v1"
EXTRACTION_METHOD = "pymupdf-reference-section-lines-v1"
POSTGRES_NUL_REPLACEMENT = "\ufffd"

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references|bibliography|"
    r"notes\s+and\s+references|references\s+and\s+notes)\s*$",
    re.IGNORECASE,
)
LABEL_PATTERNS = (
    re.compile(r"^\s*\[\s*(\d{1,4}[a-z]?)\s*\]\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*(\d{1,4}[a-z]?)[.)]\s+(.*)$", re.IGNORECASE),
)
ARXIV_NEW_RE = re.compile(
    r"(?<!\d)(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)",
    re.IGNORECASE,
)
ARXIV_OLD_RE = re.compile(
    r"(?<![\w-])((?:astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|"
    r"math-ph|nlin|nucl-ex|nucl-th|physics|quant-ph|cs|math)(?:/|:\s*)\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"(?<!\w)(10\.\d{4,9}/[^\s\]\[<>\"']+)", re.IGNORECASE)
INSPIRE_RE = re.compile(
    r"(?:inspirehep\.net/(?:literature|record)/|inspire\s*(?:id)?\s*[:#]?\s*)(\d{4,9})",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
JOURNAL_RE = re.compile(
    r"\b(?:Phys\.?\s*Rev|Nucl\.?\s*Phys|JHEP|JCAP|Class\.?\s*Quant|"
    r"Commun\.?\s*Math|Int\.?\s*J|Ann\.?\s*Phys|Mod\.?\s*Phys|"
    r"Adv\.?\s*Theor|Fortschr|Lett\.?\s*Math|SciPost|Springer|Elsevier|"
    r"Cambridge|World\s+Scientific|University\s+Press)\b",
    re.IGNORECASE,
)
AUTHOR_RE = re.compile(
    r"\b(?:[A-Z]\.){1,3}\s*[A-Z][A-Za-z'’-]+|\b[A-Z][a-z'’-]+,\s*(?:[A-Z]\.){1,3}"
)
QUOTED_TITLE_RE = re.compile(r"[\"“](.{12,240}?)[\"”]")


@dataclass(frozen=True)
class Paper:
    canonical_paper_id: str
    inspire_id: str
    title: str
    year: str
    authors: str
    arxiv_ids: tuple[str, ...]
    dois: tuple[str, ...]
    pdf_path: Path | None
    sha256: str

    @property
    def paper_id(self) -> str:
        return self.canonical_paper_id


@dataclass
class ReferenceEntry:
    source: Paper
    label: str
    start_page: int
    page_numbers: list[int] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    section_ordinal: int = 0

    @property
    def text(self) -> str:
        return normalize_space(" ".join(self.lines))

    @property
    def postgres_nul_replacements(self) -> int:
        return sum(line.count("\x00") for line in self.lines)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", POSTGRES_NUL_REPLACEMENT)).strip()


def collapse_space_preserving_nul(value: str) -> str:
    """Collapse layout whitespace while retaining NUL for replacement metrics."""
    return re.sub(r"\s+", " ", value).strip()


def normalize_arxiv(value: str) -> str:
    value = re.sub(r"^arxiv\s*:\s*", "", value.strip(), flags=re.IGNORECASE).lower()
    return re.sub(r"^([a-z-]+):\s*(\d{7})$", r"\1/\2", value)


def normalize_doi(value: str) -> str:
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", value.strip(), flags=re.I)
    value = value.rstrip(".,;:")
    while value.endswith(")") and value.count("(") < value.count(")"):
        value = value[:-1]
    return value.lower()


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = value.replace("ß", "ss")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_space(value)


def split_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(x.strip() for x in re.split(r"\s*[;,]\s*", str(value)) if x.strip())


def load_manifest(path: Path) -> list[Paper]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("papers")
        if not isinstance(records, list):
            raise ValueError("canonical source manifest must contain a papers list")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("source manifest must contain a papers list or a legacy JSON list")
    papers: list[Paper] = []
    for row in records:
        identifiers = row.get("identifiers") if isinstance(row.get("identifiers"), dict) else {}
        full_text = row.get("full_text") if isinstance(row.get("full_text"), dict) else {}
        pdf_path = None
        if full_text.get("status") == "verified_local_pdf" and full_text.get("canonical_path"):
            pdf_path = Path(str(full_text["canonical_path"])).expanduser().resolve()
        elif row.get("pdf_status") == "downloaded" and row.get("pdf_filename"):
            pdf_path = (path.parent / "pdfs" / str(row["pdf_filename"])).resolve()
        raw_authors = row.get("authors") or ""
        authors = "; ".join(str(value) for value in raw_authors) if isinstance(raw_authors, list) else str(raw_authors)
        inspire_id = str(row.get("inspire_id") or next(iter(identifiers.get("inspire") or []), "")).strip()
        canonical_paper_id = str(
            row.get("paper_id") or identifiers.get("canonical") or f"inspire:{inspire_id}"
        ).strip()
        arxiv_values = identifiers.get("arxiv") if identifiers else split_values(row.get("arxiv_ids"))
        doi_values = identifiers.get("doi") if identifiers else split_values(row.get("dois"))
        papers.append(
            Paper(
                canonical_paper_id=canonical_paper_id,
                inspire_id=inspire_id,
                title=str(row.get("title") or "").strip(),
                year=str(row.get("year") or "").strip(),
                authors=authors.strip(),
                arxiv_ids=tuple(dict.fromkeys(normalize_arxiv(str(x)) for x in arxiv_values or [])),
                dois=tuple(dict.fromkeys(normalize_doi(str(x)) for x in doi_values or [])),
                pdf_path=pdf_path,
                sha256=str(full_text.get("sha256") or row.get("sha256") or "").strip(),
            )
        )
    return papers


def extract_identifiers(text: str) -> dict[str, list[str]]:
    arxiv = {normalize_arxiv(x) for x in ARXIV_NEW_RE.findall(text)}
    arxiv.update(normalize_arxiv(x) for x in ARXIV_OLD_RE.findall(text))
    dois = {normalize_doi(x) for x in DOI_RE.findall(text)}
    inspires = set(INSPIRE_RE.findall(text))
    return {
        "arxiv": sorted(x for x in arxiv if x),
        "doi": sorted(x for x in dois if x),
        "inspire": sorted(inspires),
    }


def label_match(line: str) -> tuple[str, str] | None:
    for pattern in LABEL_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1).lower(), match.group(2).strip()
    return None


def page_lines(document: fitz.Document) -> list[list[str]]:
    """Return text lines in deterministic reading order.

    PyMuPDF's page-wide text sort can interleave two bibliography columns on
    the same output line.  Ordering text blocks by column before splitting
    lines preserves reference boundaries.
    """
    pages: list[list[str]] = []
    for page in document:
        blocks = [block for block in page.get_text("blocks") if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()]
        midpoint = page.rect.width / 2.0
        substantial = [
            block for block in blocks
            if len(collapse_space_preserving_nul(str(block[4]))) >= 40
            and (block[2] - block[0]) < page.rect.width * 0.58
        ]
        left = [block for block in substantial if block[2] < midpoint + 15 and block[0] < midpoint - 30]
        right = [block for block in substantial if block[0] > midpoint - 15]
        two_column = len(left) >= 2 and len(right) >= 2
        if two_column:
            column_top = min((block[1] for block in left + right), default=0.0)

            def block_key(block: tuple) -> tuple[float, float, float]:
                x0, y0, x1, _ = block[:4]
                crosses_midpoint = x0 < midpoint - 30 and x1 > midpoint + 30
                if crosses_midpoint and y0 <= column_top + 10:
                    column = -1.0
                elif x0 < midpoint:
                    column = 0.0
                elif x0 >= midpoint - 15:
                    column = 1.0
                else:
                    column = 2.0
                return column, y0, x0

            blocks.sort(key=block_key)
        else:
            blocks.sort(key=lambda block: (block[1], block[0]))
        lines = [
            collapse_space_preserving_nul(line)
            for block in blocks
            for line in str(block[4]).splitlines()
        ]
        pages.append([line for line in lines if line])
    return pages


def reference_heading_locations(pages: list[list[str]]) -> list[tuple[int, int]]:
    return [
        (page_index, line_index)
        for page_index, lines in enumerate(pages)
        for line_index, line in enumerate(lines)
        if REFERENCE_HEADING_RE.match(line)
    ]


def fallback_reference_location(pages: list[list[str]]) -> tuple[int, int] | None:
    """Find an unlabeled bibliography heading from a final-page [1] run.

    Some journal layouts omit the word ``References`` from extracted text.
    The fallback requires labels 1 through 6 in the final 35 percent of pages,
    which is much stronger than a single bracketed prose or equation number.
    """
    first_page = max(0, int(len(pages) * 0.65))
    for page_index in range(first_page, len(pages)):
        for line_index, line in enumerate(pages[page_index]):
            matched = label_match(line)
            if not matched or matched[0] != "1":
                continue
            observed: set[str] = set()
            for later_page in range(page_index, len(pages)):
                start = line_index if later_page == page_index else 0
                for later_line in pages[later_page][start:]:
                    later_match = label_match(later_line)
                    if later_match:
                        observed.add(later_match[0])
            if all(str(number) in observed for number in range(1, 7)):
                return page_index, line_index - 1
    return None


def candidate_section_entries(
    paper: Paper,
    pages: list[list[str]],
    heading_page: int,
    heading_line: int,
    section_ordinal: int,
    max_pages: int = 40,
) -> list[ReferenceEntry]:
    """Parse one numbered bibliography section conservatively.

    A heading is accepted only if at least two numbered entries follow it.  A
    repeated label after entries have started terminates the section, which
    avoids swallowing a later chapter in books.
    """
    entries: list[ReferenceEntry] = []
    current: ReferenceEntry | None = None
    seen_labels: set[str] = set()
    empty_pages = 0
    encountered_next_heading = False
    stop_page = min(len(pages), heading_page + max_pages + 1)
    for page_index in range(heading_page, stop_page):
        lines = pages[page_index]
        start = heading_line + 1 if page_index == heading_page else 0
        labels_on_page = 0
        for line in lines[start:]:
            if page_index != heading_page and REFERENCE_HEADING_RE.match(line):
                encountered_next_heading = True
                break
            matched = label_match(line)
            if matched:
                label, remainder = matched
                if label in seen_labels and len(entries) >= 2:
                    if current:
                        entries.append(current)
                    return entries
                if current:
                    entries.append(current)
                current = ReferenceEntry(
                    source=paper,
                    label=label,
                    start_page=page_index + 1,
                    page_numbers=[page_index + 1],
                    lines=[remainder] if remainder else [],
                    section_ordinal=section_ordinal,
                )
                seen_labels.add(label)
                labels_on_page += 1
            elif current:
                # Page headers and section headings are retained in the raw
                # evidence rather than removed by a heuristic.
                current.lines.append(line)
                if page_index + 1 not in current.page_numbers:
                    current.page_numbers.append(page_index + 1)
        if current and labels_on_page == 0:
            empty_pages += 1
        else:
            empty_pages = 0
        if current and empty_pages >= 2:
            break
        if encountered_next_heading:
            break
    if current:
        entries.append(current)
    return entries if len(entries) >= 2 else []


def extract_reference_entries(paper: Paper) -> tuple[list[ReferenceEntry], dict[str, int]]:
    if not paper.pdf_path or not paper.pdf_path.exists():
        return [], {"pages": 0, "reference_headings": 0, "accepted_sections": 0}
    with fitz.open(paper.pdf_path) as document:
        pages = page_lines(document)
    headings = reference_heading_locations(pages)
    explicit_heading_count = len(headings)
    used_fallback = False
    if not headings:
        fallback = fallback_reference_location(pages)
        if fallback:
            headings = [fallback]
            used_fallback = True
    all_entries: list[ReferenceEntry] = []
    accepted = 0
    fingerprints: set[tuple[str, str]] = set()
    for ordinal, (page_index, line_index) in enumerate(headings, 1):
        entries = candidate_section_entries(paper, pages, page_index, line_index, ordinal)
        if not entries:
            continue
        accepted += 1
        for entry in entries:
            fingerprint = (entry.label, normalize_title(entry.text))
            if fingerprint not in fingerprints:
                all_entries.append(entry)
                fingerprints.add(fingerprint)
    return all_entries, {
        "pages": len(pages),
        "reference_headings": explicit_heading_count,
        "reference_section_candidates": len(headings),
        "accepted_sections": accepted,
        "used_final_page_fallback": int(used_fallback),
        "postgres_nul_replacements": sum(entry.postgres_nul_replacements for entry in all_entries),
        "reference_entries_with_postgres_nul_replacements": sum(
            entry.postgres_nul_replacements > 0 for entry in all_entries
        ),
    }


def build_indexes(papers: Iterable[Paper]) -> dict[str, Any]:
    arxiv: dict[str, list[Paper]] = defaultdict(list)
    doi: dict[str, list[Paper]] = defaultdict(list)
    inspire: dict[str, list[Paper]] = defaultdict(list)
    titles: dict[str, list[Paper]] = defaultdict(list)
    for paper in papers:
        inspire[paper.inspire_id].append(paper)
        for value in paper.arxiv_ids:
            arxiv[value].append(paper)
        for value in paper.dois:
            doi[value].append(paper)
        title = normalize_title(paper.title)
        if len(title.split()) >= 4 and len(title) >= 20:
            titles[title].append(paper)
    return {"arxiv": arxiv, "doi": doi, "inspire": inspire, "title": titles}


def title_matches(text: str, title_index: dict[str, list[Paper]]) -> list[tuple[str, Paper]]:
    normalized = f" {normalize_title(text)} "
    matches: list[tuple[str, Paper]] = []
    for title, papers in title_index.items():
        if f" {title} " in normalized and len(papers) == 1:
            matches.append((title, papers[0]))
    # If titles nest, the longer title is the stronger evidence for that paper.
    matches.sort(key=lambda item: (-len(item[0]), item[1].paper_id))
    return matches


def evidence_excerpt(text: str, needle: str | None = None, limit: int = 700) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    start = 0
    if needle:
        position = text.casefold().find(needle.casefold())
        if position >= 0:
            start = max(0, position - 180)
    excerpt = text[start : start + limit]
    if start:
        excerpt = "..." + excerpt
    if start + limit < len(text):
        excerpt += "..."
    return excerpt


def citation_record(
    entry: ReferenceEntry,
    target: Paper,
    method: str,
    matched_value: str,
) -> dict[str, Any]:
    exact = method != "normalized_title_containment"
    key = f"{entry.source.paper_id}|{target.paper_id}|{entry.section_ordinal}|{entry.label}|{entry.start_page}|{method}|{matched_value}"
    return {
        "schema_version": SCHEMA_VERSION,
        "citation_id": "citation:" + hashlib.sha256(key.encode()).hexdigest()[:24],
        "source_paper_id": entry.source.paper_id,
        "target_paper_id": target.paper_id,
        "source_inspire_id": entry.source.inspire_id,
        "target_inspire_id": target.inspire_id,
        "reference_label": entry.label,
        "physical_page": entry.start_page,
        "physical_pages": entry.page_numbers,
        "section_ordinal": entry.section_ordinal,
        "excerpt": evidence_excerpt(entry.text, matched_value),
        "resolution_method": method,
        "matched_value": matched_value,
        "confidence": 1.0 if exact else 0.95,
        "review_status": "accepted_exact_identifier" if exact else "pending_title_review",
        "extraction_method": EXTRACTION_METHOD,
    }


def sufficient_external_stub(entry: ReferenceEntry, identifiers: dict[str, list[str]]) -> tuple[bool, list[str]]:
    signals: list[str] = []
    if any(identifiers.values()):
        signals.append("exact_identifier")
    if YEAR_RE.search(entry.text):
        signals.append("year")
    if JOURNAL_RE.search(entry.text):
        signals.append("journal_or_publisher")
    if AUTHOR_RE.search(entry.text):
        signals.append("author_pattern")
    if QUOTED_TITLE_RE.search(entry.text):
        signals.append("quoted_title")
    return ("exact_identifier" in signals or len(signals) >= 3), signals


def unresolved_record(
    entry: ReferenceEntry,
    identifiers: dict[str, list[str]],
    signals: list[str],
) -> dict[str, Any]:
    key = f"{entry.source.paper_id}|{entry.section_ordinal}|{entry.label}|{entry.start_page}|{entry.text}"
    quoted = QUOTED_TITLE_RE.search(entry.text)
    return {
        "schema_version": SCHEMA_VERSION,
        "stub_id": "external-citation:" + hashlib.sha256(key.encode()).hexdigest()[:24],
        "source_paper_id": entry.source.paper_id,
        "source_inspire_id": entry.source.inspire_id,
        "reference_label": entry.label,
        "physical_page": entry.start_page,
        "physical_pages": entry.page_numbers,
        "section_ordinal": entry.section_ordinal,
        "excerpt": evidence_excerpt(entry.text),
        "identifiers": identifiers,
        "years": sorted(set(YEAR_RE.findall(entry.text))),
        "title_candidate": normalize_space(quoted.group(1)) if quoted else None,
        "bibliographic_signals": signals,
        "review_status": "unresolved_external",
        "extraction_method": EXTRACTION_METHOD,
    }


def resolve_entries(entries: Iterable[ReferenceEntry], indexes: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    citations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    metrics: Counter = Counter()
    seen_citations: set[tuple[str, str, int, str, str]] = set()
    seen_stubs: set[str] = set()
    for entry in entries:
        metrics["reference_entries"] += 1
        identifiers = extract_identifiers(entry.text)
        targets: dict[str, tuple[Paper, str, str]] = {}
        for kind in ("arxiv", "doi", "inspire"):
            for value in identifiers[kind]:
                matches = indexes[kind].get(value, [])
                if len(matches) == 1:
                    paper = matches[0]
                    targets[paper.paper_id] = (paper, f"exact_{kind}_identifier", value)
        if not targets:
            for title, paper in title_matches(entry.text, indexes["title"]):
                targets[paper.paper_id] = (paper, "normalized_title_containment", title)
        if targets:
            for target_id, (target, method, value) in sorted(targets.items()):
                if target_id == entry.source.paper_id:
                    metrics["self_citations_suppressed"] += 1
                    continue
                dedupe = (entry.source.paper_id, target_id, entry.section_ordinal, entry.label, method)
                if dedupe in seen_citations:
                    continue
                citations.append(citation_record(entry, target, method, value))
                seen_citations.add(dedupe)
                metrics[f"resolved_{method}"] += 1
        else:
            sufficient, signals = sufficient_external_stub(entry, identifiers)
            if sufficient:
                stub = unresolved_record(entry, identifiers, signals)
                if stub["stub_id"] not in seen_stubs:
                    unresolved.append(stub)
                    seen_stubs.add(stub["stub_id"])
                    metrics["external_stubs"] += 1
            else:
                metrics["insufficient_unresolved_entries_omitted"] += 1
    return citations, unresolved, metrics


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def run(manifest: Path, output_dir: Path) -> dict[str, Any]:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    papers = load_manifest(manifest)
    local = [paper for paper in papers if paper.pdf_path]
    indexes = build_indexes(papers)
    all_entries: list[ReferenceEntry] = []
    paper_metrics: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for number, paper in enumerate(local, 1):
        try:
            entries, extraction = extract_reference_entries(paper)
            all_entries.extend(entries)
            paper_metrics.append(
                {
                    "paper_id": paper.paper_id,
                    "inspire_id": paper.inspire_id,
                    "pdf_filename": paper.pdf_path.name if paper.pdf_path else None,
                    "pages": extraction["pages"],
                    "reference_headings": extraction["reference_headings"],
                    "reference_section_candidates": extraction.get("reference_section_candidates", extraction["reference_headings"]),
                    "accepted_sections": extraction["accepted_sections"],
                    "used_final_page_fallback": extraction.get("used_final_page_fallback", 0),
                    "postgres_nul_replacements": extraction.get("postgres_nul_replacements", 0),
                    "reference_entries_with_postgres_nul_replacements": extraction.get(
                        "reference_entries_with_postgres_nul_replacements", 0
                    ),
                    "reference_entries": len(entries),
                    "status": "processed",
                }
            )
        except Exception as exc:  # retain per-paper failure without hiding it
            errors.append({"paper_id": paper.paper_id, "error": f"{type(exc).__name__}: {exc}"})
            paper_metrics.append(
                {
                    "paper_id": paper.paper_id,
                    "inspire_id": paper.inspire_id,
                    "pdf_filename": paper.pdf_path.name if paper.pdf_path else None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if number % 20 == 0:
            print(f"processed {number}/{len(local)} PDFs", flush=True)

    citations, unresolved, resolution_metrics = resolve_entries(all_entries, indexes)
    citations.sort(key=lambda row: (row["source_inspire_id"], row["physical_page"], row["reference_label"], row["target_inspire_id"]))
    unresolved.sort(key=lambda row: (row["source_inspire_id"], row["physical_page"], row["reference_label"]))
    write_jsonl(output_dir / "citations.jsonl", citations)
    write_jsonl(output_dir / "unresolved.jsonl", unresolved)

    aliases = {
        "schema_version": SCHEMA_VERSION,
        "canonical_key": "inspire_id",
        "arxiv_to_inspire": {key: sorted(p.inspire_id for p in values) for key, values in sorted(indexes["arxiv"].items())},
        "doi_to_inspire": {key: sorted(p.inspire_id for p in values) for key, values in sorted(indexes["doi"].items())},
        "normalized_title_to_inspire": {key: sorted(p.inspire_id for p in values) for key, values in sorted(indexes["title"].items())},
    }
    (output_dir / "aliases.json").write_text(json.dumps(aliases, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    distinct_edges = {(row["source_paper_id"], row["target_paper_id"]) for row in citations}
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_manifest_contract": "canonical_papers_object" if isinstance(manifest_payload, dict) else "legacy_records_list",
        "manifest_records": len(papers),
        "local_pdfs_expected": len(local),
        "local_pdfs_processed": sum(row["status"] == "processed" for row in paper_metrics),
        "local_pdf_errors": len(errors),
        "total_pdf_pages": sum(row.get("pages", 0) for row in paper_metrics),
        "papers_with_reference_headings": sum(row.get("reference_headings", 0) > 0 for row in paper_metrics),
        "papers_with_accepted_reference_sections": sum(row.get("accepted_sections", 0) > 0 for row in paper_metrics),
        "papers_with_resolved_internal_citations": len({row["source_paper_id"] for row in citations}),
        "reference_entries": len(all_entries),
        "postgres_nul_replacements": sum(entry.postgres_nul_replacements for entry in all_entries),
        "reference_entries_with_postgres_nul_replacements": sum(
            entry.postgres_nul_replacements > 0 for entry in all_entries
        ),
        "resolved_citation_occurrences": len(citations),
        "distinct_internal_citation_edges": len(distinct_edges),
        "external_unresolved_stubs": len(unresolved),
        "resolution_counts": dict(sorted(resolution_metrics.items())),
        "paper_metrics": paper_metrics,
        "errors": errors,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = run(args.manifest.expanduser().resolve(), args.output_dir.expanduser().resolve())
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"paper_metrics", "errors"}}, indent=2, sort_keys=True))
    return 1 if metrics["local_pdf_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
