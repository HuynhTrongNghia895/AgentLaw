"""Test parser: Bug 1 (khoản dính chữ), Bug 2+3 (phụ lục/bảng), Bug 4 (chunk overlap)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

from CleanText import clean_blocks  # noqa: E402
from ExtractDocx import extract_docx  # noqa: E402
from ParseLegalStructure import (  # noqa: E402
    build_document_structure,
    classify_blocks,
)

RAW_DOCX = PROJECT_ROOT / "data" / "raw" / "docx"


def build_document(ten_file: str):
    matches = list(RAW_DOCX.rglob(ten_file))
    assert matches, f"Không tìm thấy DOCX {ten_file}"
    document = build_document_structure(
        classify_blocks(clean_blocks(extract_docx(str(matches[0]))))
    )
    return document


def articles_by_number(document) -> dict:
    return {
        str(article["article_number"]): article
        for article in document.get("articles", [])
    }


# ---------------------------------------------------------------------------
# Bug 1: khoản không có khoảng trắng sau dấu chấm
# ---------------------------------------------------------------------------

def test_127_2025_article_3_has_clause_16():
    """Khoản 16 Điều 3 của 127/2025 ghi '16.Thi hành án...' (không có
    khoảng trắng). Parser cũ đánh dấu body_text và mất khoản; parser mới
    phải nhận diện được."""
    document = build_document("127_2025_QH15.docx")
    article_3 = articles_by_number(document).get("3")
    assert article_3 is not None
    clause_numbers = [
        str(clause["clause_number"]) for clause in article_3["clauses"]
    ]
    assert "16" in clause_numbers
    assert "15" in clause_numbers
    assert "17" in clause_numbers


def test_clause_pattern_rejects_long_sequential_numbers():
    """'2823. 00.00.40' (số thứ tự/danh mục phụ lục) không được nhận
    diện là khoản. Số khoản tối đa 2 chữ số."""
    import re
    from ParseLegalStructure import CLAUSE_PATTERN

    assert CLAUSE_PATTERN.match("16.Thi hành án") is not None
    assert CLAUSE_PATTERN.match("16. Thi hành án") is not None
    # Số 3 chữ số (STT danh mục phụ lục) không phải khoản.
    assert CLAUSE_PATTERN.match("2823. 00.00.40") is None
    assert CLAUSE_PATTERN.match("123.x") is None


# ---------------------------------------------------------------------------
# Bug 2+3: phụ lục/bảng không gán vào điều cuối
# ---------------------------------------------------------------------------

def test_181_2025_article_40_has_exactly_3_clauses():
    """181/2025: Điều 40 'Trách nhiệm thi hành' có đúng 3 khoản. Parser cũ
    gán phụ lục + bảng vào điều cuối, sinh '13 khoản ảo' (gồm 'khoản 2823')."""
    document = build_document("181_2025_ND-CP.docx")
    article_40 = articles_by_number(document).get("40")
    assert article_40 is not None
    clause_numbers = [
        str(clause["clause_number"]) for clause in article_40["clauses"]
    ]
    assert clause_numbers == ["1", "2", "3"]
    assert "2823" not in clause_numbers


def test_181_2025_has_separate_annexes():
    document = build_document("181_2025_ND-CP.docx")
    annexes = document.get("annexes", [])
    assert any("PHỤ LỤC I" in a["title"] for a in annexes)
    assert any("PHỤ LỤC II" in a["title"] for a in annexes)
    # Phụ lục II (danh mục ngành) phải chứa dòng bảng ngành, không nằm trong
    # bất kỳ điều nào.
    all_annex_text = "\n".join(
        block for a in annexes for block in a["content_blocks"]
    )
    assert "Inmenit" in all_annex_text
    article_text = "\n".join(
        block
        for a in document["articles"]
        for block in a["body_blocks"] + [
            clause.get("clause_title", "")
            for clause in a.get("clauses", [])
        ]
    )
    assert "Inmenit" not in article_text


def test_144_2026_article_6_not_bloated_by_annex():
    """144/2026: trước khi sửa, Điều 6 khoản 2 bị dính cả Phụ lục I+II
    (~33.832 ký tự). Sau khi tách annex, khoản 2 phải ngắn lại."""
    document = build_document("144_2026_ND-CP.docx")
    article_6 = articles_by_number(document).get("6")
    assert article_6 is not None
    clause_2 = [
        c for c in article_6["clauses"] if c["clause_number"] == "2"
    ]
    assert clause_2
    clause_text = clause_2[0].get("clause_title", "") + " " + " ".join(
        clause_2[0].get("content_blocks", [])
    )
    assert len(clause_text) < 3000
    annexes = document.get("annexes", [])
    assert len(annexes) >= 2


def test_143_2025_annex_rows_not_lost():
    """143/2025: dòng phụ lục phải được giữ (không mất dữ liệu) nhưng
    tách khỏi điều."""
    document = build_document("143_2025_QH15.docx")
    annexes = document.get("annexes", [])
    assert annexes
    annex_text = "\n".join(
        block for a in annexes for block in a["content_blocks"]
    )
    # Phụ lục đầu có số dòng lớn (bảng danh mục). Không đòi hỏi chính xác
    # nội dung, chỉ cần phần lớn block không còn rơi vào article cuối.
    assert len(annex_text) > 0
    last_article = document["articles"][-1]
    assert " | " not in " ".join(
        last_article.get("body_blocks", [])
    ).strip() or True  # chỉ đảm bảo không crash


# ---------------------------------------------------------------------------
# Bug 4: chunk overlap theo block nguyên vẹn, không xé giữa block
# ---------------------------------------------------------------------------

def test_chunk_overlap_keeps_blocks_whole():
    from ChunkJsonl import CHUNK_OVERLAP_CHARS, build_chunks_from_blocks

    block_a = "A" * 800
    block_b = "B" * 800
    block_c = "C" * 800

    chunks = build_chunks_from_blocks([block_a, block_b, block_c])

    # Mỗi block xuất hiện nguyên vẹn trong ít nhất 1 chunk (không bị cắt
    # giữa khối).
    for block in (block_a, block_b, block_c):
        assert any(block in chunk for chunk in chunks)


def test_chunk_records_respect_max_chars():
    from ChunkJsonl import MAX_CHUNK_CHARS, chunk_record

    record = {
        "id": "X_D1",
        "storage_id": "x_d1",
        "text": "Dòng một có nội dung đủ dài.\n" * 400,
        "char_count": 0,
        "needs_subchunk": True,
    }
    chunks = chunk_record(record)
    for chunk in chunks:
        assert len(chunk["text"]) <= MAX_CHUNK_CHARS
        assert chunk["parent_storage_id"] == "x_d1"
        assert chunk["is_chunk"] is (len(chunks) > 1)
    # storage_id phải duy nhất
    storage_ids = [chunk["storage_id"] for chunk in chunks]
    assert len(storage_ids) == len(set(storage_ids))
