from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import fitz
import pytest

from research.gates_graphrag_pilot.extraction.extract_papers import (
    PaperInput,
    _line_from_spans,
    count_tokens,
    count_words,
    extract_paper,
    load_manifest,
    write_jsonl,
)


CORPUS = Path.home() / "Documents" / "S_James_Gates_Publications" / "pdfs"
PAPER_2012 = CORPUS / (
    "2020_2012.14015_A_note_on_exemplary_off_shell_constructions_of_4D_N_2_"
    "supersymmetry_representations.pdf"
)
ADYNKRA_2007 = CORPUS / (
    "2020_2007.07390_Component_decompositions_and_adynkra_libraries_for_"
    "supermultiplets_in_lower_dimensional_su.pdf"
)


def make_pdf(path: Path) -> None:
    document = fitz.open()
    title = document.new_page()
    title.insert_text((72, 100), "A Synthetic Source", fontsize=18, fontname="hebo")
    title.insert_text((72, 140), "Author Name", fontsize=11)
    page = document.new_page()
    page.insert_text((72, 72), "1 Introduction", fontsize=14, fontname="hebo")
    page.insert_text((72, 105), "The well-", fontsize=11)
    page.insert_text((72, 120), "formed source line remains split.", fontsize=11)
    page.insert_text((72, 150), "L_I R_J + L_J R_I = 2 delta_IJ", fontsize=11)
    page.insert_text((72, 190), "2 Result", fontsize=14, fontname="hebo")
    page.insert_text((72, 220), "No section label is supplied beyond this literal line.", fontsize=11)
    document.set_toc([[1, "1 Introduction", 2], [1, "2 Result", 2]])
    document.save(path)
    document.close()


def test_counting_is_explicit_and_deterministic() -> None:
    text = "L_I R_J + 2 delta-IJ"
    assert count_words(text) == 6
    assert count_tokens(text) == 7


def test_null_pdf_character_is_replaced_and_counted() -> None:
    line = _line_from_spans(
        [{"text": "A\x00B", "bbox": [0, 0, 1, 1], "size": 10, "flags": 0}],
        page_line=0,
        block_index=0,
    )
    assert line is not None
    assert line.text == "A\ufffdB"
    assert line.null_replacements == 1


def test_synthetic_pdf_preserves_lines_and_literal_headings(tmp_path: Path) -> None:
    pdf = tmp_path / "2401.01234_source.pdf"
    make_pdf(pdf)
    paper = PaperInput("2401.01234", pdf, arxiv_id="2401.01234")

    first = list(extract_paper(paper, target_words=50))
    second = list(extract_paper(paper, target_words=50))

    assert first == second
    assert {row["section_heading"] for row in first} >= {"1 Introduction", "2 Result"}
    assert all(row["section_heading"] != "A Synthetic Source" for row in first)
    combined = "\n".join(row["text"] for row in first)
    assert "well-\nformed" in combined
    assert "L_I R_J + L_J R_I = 2 delta_IJ" in combined
    assert all(row["page_number"] in {1, 2} for row in first)
    assert all(row["page_line_start"] <= row["page_line_end"] for row in first)
    assert all(row["extraction_provenance"]["source_sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest() for row in first)


def test_manifest_preserves_identifiers_and_metadata(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf = pdf_dir / "source.pdf"
    make_pdf(pdf)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "arxiv_ids", "inspire_id", "title", "pdf_filename"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "paper_id": "stable-paper-id",
                "arxiv_ids": "2401.01234",
                "inspire_id": "12345",
                "title": "Literal title",
                "pdf_filename": "source.pdf",
            }
        )

    papers = load_manifest(manifest)
    assert papers == [PaperInput("stable-paper-id", pdf.resolve(), "2401.01234", "12345", "Literal title")]
    output = tmp_path / "chunks.jsonl"
    paper_count, chunk_count = write_jsonl(papers, output, target_words=50)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert paper_count == 1
    assert chunk_count == len(rows) > 0
    assert {row["paper_id"] for row in rows} == {"stable-paper-id"}
    assert {row["inspire_id"] for row in rows} == {"12345"}


@pytest.mark.skipif(not PAPER_2012.exists(), reason="local Gates corpus is unavailable")
def test_real_2012_14015_section_and_page_quality() -> None:
    rows = list(extract_paper(PaperInput("2012.14015", PAPER_2012, "2012.14015")))
    headings = {row["section_heading"] for row in rows}
    assert "Introduction" in headings
    assert "4D, N = 1 SUSY and the Permutahedron" in headings
    assert "References" in headings
    assert len(rows) == 30
    assert {row["page_number"] for row in rows} == set(range(1, 16))
    assert sum(row["word_count"] for row in rows) > 5_000
    assert any("(1.1)" in row["text"] for row in rows)


@pytest.mark.skipif(not ADYNKRA_2007.exists(), reason="local Gates corpus is unavailable")
def test_real_adynkra_outline_recovers_subsections() -> None:
    rows = list(extract_paper(PaperInput("2007.07390", ADYNKRA_2007, "2007.07390")))
    headings = {row["section_heading"] for row in rows}
    assert "9D Minimal Scalar Superﬁeld Decomposition" in headings
    assert "9D Minimal Adinkra Diagram" in headings
    assert "Conclusion" in headings
    assert "References" in headings
    assert len(rows) == 128
    assert {row["page_number"] for row in rows} == set(range(1, 67))
    assert sum(row["word_count"] for row in rows) > 17_000
    assert all(row["section_heading"] != "Contents" for row in rows)
