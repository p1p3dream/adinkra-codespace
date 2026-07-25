#!/usr/bin/env python3
"""Deterministic, page-anchored PDF extraction for the Gates pilot corpus.

The extractor deliberately does not repair mathematical notation, dehyphenate
line endings, or infer missing section names.  Its output is suitable as an
evidence layer for later indexing, not as a replacement for the source PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import fitz


SCHEMA_VERSION = "gates-pdf-chunk-v1"
EXTRACTION_STRATEGY = "pymupdf-layout-lines-v1"
SECTION_STRATEGY = "pdf-outline-alignment-and-literal-typographic-heading-v1"
TOKEN_STRATEGY = "unicode-lexical-tokens-v1"

ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|\d+(?:\.\d+)*", re.UNICODE)
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|\d+(?:\.\d+)*|[^\w\s]", re.UNICODE)
NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+(?=[A-Z0-9‘'\"])", re.UNICODE
)
GENERIC_HEADING_RE = re.compile(
    r"^(?:abstract|contents|introduction|background|methodology|methods?|results?|"
    r"discussion|conclusions?|summary|acknowledg(?:e)?ments?|references|bibliography|"
    r"appendix(?:\s+[A-Z0-9]+)?(?:\s*[:.-].*)?)$",
    re.IGNORECASE,
)
REJECT_HEADING_RE = re.compile(
    r"^(?:figure|fig\.?|table|equation|eq\.?|pacs|keywords?)\s*\d*\s*[:.]?",
    re.IGNORECASE,
)


class ExtractionError(RuntimeError):
    """Raised when an input cannot be represented without ambiguity."""


@dataclass(frozen=True)
class PaperInput:
    paper_id: str
    pdf_path: Path
    arxiv_id: str | None = None
    inspire_id: str | None = None
    title: str | None = None


@dataclass
class Line:
    text: str
    page_line: int
    block_index: int
    bbox: tuple[float, float, float, float]
    max_font_size: float
    bold_ratio: float
    null_replacements: int = 0
    is_heading: bool = False
    heading_source: str | None = None


@dataclass
class ChunkBuffer:
    lines: list[Line] = field(default_factory=list)
    section_heading: str | None = None
    section_start_page: int | None = None
    section_heading_source: str | None = None

    @property
    def word_count(self) -> int:
        return sum(count_words(line.text) for line in self.lines)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_tokens(text: str) -> int:
    """Count lexical units, not model-specific tokenizer tokens."""
    return len(TOKEN_RE.findall(text))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def infer_arxiv_id(text: str) -> str | None:
    match = ARXIV_RE.search(text)
    return match.group(1) if match else None


def infer_paper_input(path: Path, paper_id: str | None = None) -> PaperInput:
    path = path.expanduser().resolve()
    arxiv_id = infer_arxiv_id(path.name)
    resolved_id = paper_id or arxiv_id or path.stem
    return PaperInput(paper_id=resolved_id, pdf_path=path, arxiv_id=arxiv_id)


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ExtractionError(f"{path}:{line_number} is not an object")
                    records.append(value)
        return records
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("papers", value.get("records"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ExtractionError(f"{path} must contain a list of paper objects")
        return value
    raise ExtractionError(f"unsupported manifest format: {path.suffix}")


def _resolve_manifest_pdf(manifest: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (manifest.parent / candidate).resolve()
    if direct.exists():
        return direct
    in_pdf_dir = (manifest.parent / "pdfs" / candidate).resolve()
    return in_pdf_dir if in_pdf_dir.exists() else direct


def load_manifest(path: Path) -> list[PaperInput]:
    path = path.expanduser().resolve()
    papers: list[PaperInput] = []
    for record in _manifest_records(path):
        raw_pdf = next(
            (
                normalize_identifier(record.get(key))
                for key in ("pdf_path", "local_pdf_path", "path", "file", "pdf_filename")
                if normalize_identifier(record.get(key))
            ),
            None,
        )
        if not raw_pdf:
            continue
        pdf_path = _resolve_manifest_pdf(path, raw_pdf)
        arxiv_id = normalize_identifier(record.get("arxiv_id") or record.get("arxiv_ids"))
        if arxiv_id and "," in arxiv_id:
            arxiv_id = arxiv_id.split(",", 1)[0].strip()
        arxiv_id = infer_arxiv_id(arxiv_id or "") or infer_arxiv_id(pdf_path.name)
        inspire_id = normalize_identifier(record.get("inspire_id"))
        paper_id = normalize_identifier(record.get("paper_id") or record.get("id"))
        paper_id = paper_id or arxiv_id or inspire_id or pdf_path.stem
        papers.append(
            PaperInput(
                paper_id=paper_id,
                pdf_path=pdf_path,
                arxiv_id=arxiv_id,
                inspire_id=inspire_id,
                title=normalize_identifier(record.get("title")),
            )
        )
    return papers


def _line_from_spans(spans: Sequence[dict[str, Any]], page_line: int, block_index: int) -> Line | None:
    visible = [span for span in spans if span.get("text")]
    raw_text = "".join(str(span["text"]) for span in visible)
    null_replacements = raw_text.count("\x00")
    text = raw_text.replace("\x00", "\ufffd").strip()
    if not text:
        return None
    weighted_chars = sum(max(1, len(str(span["text"]).strip())) for span in visible)
    bold_chars = sum(
        max(1, len(str(span["text"]).strip()))
        for span in visible
        if int(span.get("flags", 0)) & 16
    )
    x0 = min(float(span["bbox"][0]) for span in visible)
    y0 = min(float(span["bbox"][1]) for span in visible)
    x1 = max(float(span["bbox"][2]) for span in visible)
    y1 = max(float(span["bbox"][3]) for span in visible)
    return Line(
        text=text,
        page_line=page_line,
        block_index=block_index,
        bbox=(x0, y0, x1, y1),
        max_font_size=max(float(span.get("size", 0.0)) for span in visible),
        bold_ratio=bold_chars / weighted_chars,
        null_replacements=null_replacements,
    )


def page_lines(page: fitz.Page) -> list[Line]:
    lines: list[Line] = []
    page_line = 0
    for block_index, block in enumerate(page.get_text("dict", sort=True).get("blocks", [])):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            line = _line_from_spans(raw_line.get("spans", []), page_line, block_index)
            if line:
                lines.append(line)
                page_line += 1
    return lines


def body_font_size(all_pages: Sequence[Sequence[Line]]) -> float:
    weights: Counter[float] = Counter()
    for lines in all_pages:
        for line in lines:
            bucket = round(line.max_font_size * 2.0) / 2.0
            weights[bucket] += max(1, len(line.text))
    return weights.most_common(1)[0][0] if weights else 0.0


def _heading_shape(text: str) -> bool:
    words = WORD_RE.findall(text)
    if not words or len(words) > 24 or len(text) > 180:
        return False
    alpha_chars = sum(character.isalpha() for character in text)
    visible_chars = sum(not character.isspace() for character in text)
    if not visible_chars or alpha_chars / visible_chars < 0.45:
        return False
    if REJECT_HEADING_RE.match(text) or text.endswith((".", ",", ";")):
        return False
    return True


def _match_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", text)
    return "".join(character for character in text if character.isalnum())


def outline_heading_keys(document: fitz.Document) -> dict[int, set[str]]:
    """Return normalized PDF-outline titles keyed by zero-based target page."""
    result: dict[int, set[str]] = {}
    for entry in document.get_toc(simple=True):
        if len(entry) < 3:
            continue
        page_number = int(entry[2])
        key = _match_key(str(entry[1]))
        if page_number > 0 and key:
            result.setdefault(page_number - 1, set()).add(key)
    return result


def is_heading(
    line: Line,
    body_size: float,
    page_number: int,
    toc_page: bool,
    outline_keys: set[str],
) -> str | None:
    """Recognize only literal lines supported by typography and heading shape."""
    text = " ".join(line.text.split())
    if toc_page or not _heading_shape(text):
        return None
    line_key = _match_key(text)
    if line_key in outline_keys:
        return "pdf_outline"
    generic = bool(GENERIC_HEADING_RE.fullmatch(text))
    numbered = bool(NUMBERED_HEADING_RE.match(text))
    typographic = line.bold_ratio >= 0.60 or line.max_font_size >= body_size + 0.75
    if page_number == 1 and not generic:
        return None
    if generic or numbered:
        return "typography" if typographic else None
    # Unnumbered headings require stronger typography. This admits literal
    # article headings while rejecting ordinary numbered footnotes and prose.
    return "typography" if (
        line.max_font_size >= body_size + 0.75
        and line.bold_ratio >= 0.60
        and len(WORD_RE.findall(text)) >= 2
    ) else None


def detect_toc_pages(all_pages: Sequence[Sequence[Line]]) -> set[int]:
    """Mark a front-matter Contents page and contiguous dense continuations."""
    result: set[int] = set()
    active = False
    for page_index, lines in enumerate(all_pages):
        normalized = {" ".join(line.text.split()).casefold() for line in lines}
        bold_short = sum(
            line.bold_ratio >= 0.60 and count_words(line.text) <= 24 and len(line.text) <= 180
            for line in lines
        )
        if "contents" in normalized:
            active = True
            result.add(page_index)
            continue
        if active and bold_short >= 4:
            result.add(page_index)
            continue
        if active:
            break
    return result


def _join_chunk_lines(lines: Sequence[Line]) -> str:
    output: list[str] = []
    previous_block: int | None = None
    for line in lines:
        if output and previous_block != line.block_index:
            output.append("")
        output.append(line.text.rstrip())
        previous_block = line.block_index
    return "\n".join(output).strip()


def _bbox(lines: Sequence[Line]) -> list[float]:
    return [
        round(min(line.bbox[0] for line in lines), 3),
        round(min(line.bbox[1] for line in lines), 3),
        round(max(line.bbox[2] for line in lines), 3),
        round(max(line.bbox[3] for line in lines), 3),
    ]


def extract_paper(paper: PaperInput, target_words: int = 300) -> Iterator[dict[str, Any]]:
    if target_words < 50:
        raise ExtractionError("target_words must be at least 50")
    if not paper.pdf_path.is_file():
        raise ExtractionError(f"PDF not found: {paper.pdf_path}")
    digest = sha256_file(paper.pdf_path)
    try:
        document = fitz.open(paper.pdf_path)
    except Exception as error:
        raise ExtractionError(f"cannot open {paper.pdf_path}: {error}") from error
    if document.needs_pass:
        document.close()
        raise ExtractionError(f"encrypted PDF requires a password: {paper.pdf_path}")

    try:
        pages = [page_lines(page) for page in document]
        body_size = body_font_size(pages)
        toc_pages = detect_toc_pages(pages)
        outline_keys = outline_heading_keys(document)
        current_section: str | None = None
        section_start_page: int | None = None
        current_section_source: str | None = None
        chunk_index = 0

        for page_index, lines in enumerate(pages):
            page_number = page_index + 1
            for line in lines:
                line.heading_source = is_heading(
                    line,
                    body_size,
                    page_number,
                    page_index in toc_pages,
                    outline_keys.get(page_index, set()),
                )
                line.is_heading = line.heading_source is not None

            page_chunk_index = 0
            buffer = ChunkBuffer(
                section_heading=current_section,
                section_start_page=section_start_page,
                section_heading_source=current_section_source,
            )

            def emit() -> dict[str, Any] | None:
                nonlocal chunk_index, page_chunk_index, buffer
                if not buffer.lines:
                    return None
                text = _join_chunk_lines(buffer.lines)
                if not text:
                    buffer.lines.clear()
                    return None
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "chunk_id": f"{paper.paper_id}:p{page_number:04d}:c{page_chunk_index:03d}",
                    "paper_id": paper.paper_id,
                    "arxiv_id": paper.arxiv_id,
                    "inspire_id": paper.inspire_id,
                    "title": paper.title,
                    "chunk_index": chunk_index,
                    "page_chunk_index": page_chunk_index,
                    "page_number": page_number,
                    "page_label": document[page_index].get_label() or None,
                    "page_line_start": buffer.lines[0].page_line,
                    "page_line_end": buffer.lines[-1].page_line,
                    "bbox": _bbox(buffer.lines),
                    "section_heading": buffer.section_heading,
                    "section_start_page": buffer.section_start_page,
                    "section_heading_source": buffer.section_heading_source,
                    "text": text,
                    "word_count": count_words(text),
                    "token_count": count_tokens(text),
                    "counting_provenance": {
                        "word_count": "unicode-words-v1",
                        "token_count": TOKEN_STRATEGY,
                    },
                    "extraction_provenance": {
                        "backend": "PyMuPDF",
                        "backend_version": fitz.VersionBind,
                        "strategy": EXTRACTION_STRATEGY,
                        "section_strategy": SECTION_STRATEGY,
                        "source_pdf": paper.pdf_path.name,
                        "source_path": str(paper.pdf_path),
                        "source_sha256": digest,
                        "source_page_count": len(document),
                        "mathematical_text_policy": "preserve-extracted-lines-no-repair",
                        "null_character_policy": "replace-U+0000-with-U+FFFD-for-storage",
                        "null_replacement_count": sum(line.null_replacements for line in buffer.lines),
                    },
                }
                chunk_index += 1
                page_chunk_index += 1
                buffer = ChunkBuffer(
                    section_heading=current_section,
                    section_start_page=section_start_page,
                    section_heading_source=current_section_source,
                )
                return row

            for line in lines:
                if line.is_heading:
                    row = emit()
                    if row:
                        yield row
                    current_section = " ".join(line.text.split())
                    section_start_page = page_number
                    current_section_source = line.heading_source
                    buffer.section_heading = current_section
                    buffer.section_start_page = section_start_page
                    buffer.section_heading_source = current_section_source
                projected_words = buffer.word_count + count_words(line.text)
                if buffer.lines and projected_words > target_words:
                    row = emit()
                    if row:
                        yield row
                buffer.lines.append(line)
            row = emit()
            if row:
                yield row
    finally:
        document.close()


def collect_inputs(args: argparse.Namespace) -> list[PaperInput]:
    papers: list[PaperInput] = []
    for manifest in args.manifest:
        papers.extend(load_manifest(Path(manifest)))
    papers.extend(infer_paper_input(Path(pdf)) for pdf in args.pdfs)
    for specification in args.paper:
        if "=" not in specification:
            raise ExtractionError("--paper must have the form PAPER_ID=PDF")
        paper_id, raw_path = specification.split("=", 1)
        if not paper_id.strip() or not raw_path.strip():
            raise ExtractionError("--paper must have the form PAPER_ID=PDF")
        papers.append(infer_paper_input(Path(raw_path), paper_id.strip()))
    if args.paper_id:
        if len(papers) != 1:
            raise ExtractionError("--paper-id requires exactly one input PDF")
        original = papers[0]
        papers[0] = PaperInput(
            paper_id=args.paper_id,
            pdf_path=original.pdf_path,
            arxiv_id=original.arxiv_id,
            inspire_id=original.inspire_id,
            title=original.title,
        )
    if not papers:
        raise ExtractionError("provide a PDF, --paper, or --manifest")
    papers.sort(key=lambda paper: (paper.paper_id, str(paper.pdf_path)))
    duplicate_ids = [paper_id for paper_id, count in Counter(p.paper_id for p in papers).items() if count > 1]
    if duplicate_ids:
        raise ExtractionError(f"duplicate paper_id values: {', '.join(sorted(duplicate_ids))}")
    return papers


def write_jsonl(papers: Iterable[PaperInput], output: Path | None, target_words: int) -> tuple[int, int]:
    handle = output.open("w", encoding="utf-8", newline="\n") if output else sys.stdout
    paper_count = 0
    chunk_count = 0
    try:
        for paper in papers:
            paper_count += 1
            for row in extract_paper(paper, target_words=target_words):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
                chunk_count += 1
    finally:
        if output:
            handle.close()
    return paper_count, chunk_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", help="PDFs whose paper IDs can be inferred from filenames")
    parser.add_argument("--paper", action="append", default=[], metavar="ID=PDF", help="explicit paper ID and PDF")
    parser.add_argument("--paper-id", help="override the ID for a single input")
    parser.add_argument("--manifest", action="append", default=[], help="CSV, JSON, or JSONL paper manifest")
    parser.add_argument("--output", "-o", type=Path, help="output JSONL; stdout when omitted")
    parser.add_argument("--target-words", type=int, default=300, help="soft maximum words per chunk (default: 300)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        papers = collect_inputs(args)
        output = args.output.expanduser().resolve() if args.output else None
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
        paper_count, chunk_count = write_jsonl(papers, output, args.target_words)
    except (ExtractionError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"extracted {chunk_count} chunks from {paper_count} papers", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
