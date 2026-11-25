"""Test FilterDocuments đọc đúng metadata v2 + ExportJsonl gắn amended_by (Bug 5)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

from CleanText import clean_blocks  # noqa: E402
from ExportJsonl import (  # noqa: E402
    document_to_records,
    load_relation_index,
    match_amended_by,
)
from ExtractDocx import extract_docx  # noqa: E402
from FilterDocuments import load_filter_rules  # noqa: E402
from ParseLegalStructure import (  # noqa: E402
    build_document_structure,
    classify_blocks,
)

RAW_DOCX = PROJECT_ROOT / "data" / "raw" / "docx"
REGISTRY = PROJECT_ROOT / "data" / "metadata" / "DocumentRegistry.csv"
RELATIONS = PROJECT_ROOT / "data" / "metadata" / "DocumentRelations.csv"


def load_rules():
    return load_filter_rules(REGISTRY)


def build_document(ten_file: str):
    matches = list(RAW_DOCX.rglob(ten_file))
    assert matches
    return build_document_structure(
        classify_blocks(clean_blocks(extract_docx(str(matches[0]))))
    )


# ---------------------------------------------------------------------------
# FilterDocuments đọc metadata tiếng Anh (v2)
# ---------------------------------------------------------------------------

def test_load_rules_reads_v2_registry():
    rules = load_rules()
    # metadata v2: 22 dòng; 1 dòng (09/2026) có 2 dataset -> tách 2 rule
    # -> 23 rule. (296/2026/NĐ-CP thêm 04/11, đã verified docid + Công báo.)
    assert len(rules) == 23
    so_hieu = [r["document_number"] for r in rules]
    assert "59/2020/QH14" in so_hieu
    assert "68/2026/NĐ-CP" in so_hieu
    assert "296/2026/NĐ-CP" in so_hieu


def test_rule_fields_use_v2_columns():
    rules = load_rules()
    by_doc = {}
    for r in rules:
        by_doc.setdefault(r["document_number"], []).append(r)
    rule_68 = by_doc["68/2026/NĐ-CP"][0]
    assert rule_68["source_filename"] == "68_2026_ND-CP.docx"
    assert rule_68["mode"] == "full"
    assert rule_68["dataset_target"] == "thue_tndn"
    # keep_header/footer không còn cột -> mặc định True (không làm thay đổi
    # hành vi: giữ header + footer).
    assert rule_68["keep_header"] is True
    assert rule_68["keep_footer"] is True


def test_multi_dataset_rule_splitted():
    rules = load_rules()
    datasets_09 = [
        r["dataset_target"] for r in rules if r["document_number"] == "09/2026/QH16"
    ]
    assert set(datasets_09) == {"thue_gtgt", "thue_tndn"}


def test_includes_clauses_validation():
    """include_clauses yêu cầu đúng 1 điều (L9) — kiểm tra filter_document
    ném lỗi khi truyền nhiều điều, không tự ép."""
    from FilterDocuments import filter_document, filter_articles

    document = build_document("59_2020_QH14.docx")
    rule = {
        "document_number": "59/2020/QH14",
        "dataset_target": "doanh_nghiep",
        "mode": "include_clauses",
        "included_articles": ["4", "5"],  # 2 điều -> phải lỗi
        "included_clauses": ["1"],
        "config_status": "ready",
        "scope_description": "",
        "keep_header": True,
        "keep_footer": True,
    }
    with pytest.raises(ValueError):
        filter_document(document, rule)


def test_custom_mode_raises():
    from FilterDocuments import filter_document

    document = build_document("59_2020_QH14.docx")
    rule = {
        "document_number": "59/2020/QH14",
        "dataset_target": "doanh_nghiep",
        "mode": "custom",
        "included_articles": [],
        "included_clauses": [],
        "config_status": "ready",
        "scope_description": "",
        "keep_header": True,
        "keep_footer": True,
    }
    with pytest.raises(ValueError):
        filter_document(document, rule)


# ---------------------------------------------------------------------------
# Bug 5: amended_by gắn từ DocumentRelations (phương án A)
# ---------------------------------------------------------------------------

def test_relation_index_keys_are_target_documents():
    index = load_relation_index(RELATIONS, REGISTRY)
    keys = set(index.keys())
    # Các văn bản bị tác động (to_so_hieu) phải có mặt.
    for expected in [
        "181/2025/NĐ-CP",
        "320/2025/NĐ-CP",
        "48/2024/QH15",
        "59/2020/QH14",
        "67/2025/QH15",
        "68/2026/NĐ-CP",
    ]:
        assert expected in keys


def test_relation_index_ignores_self_relations_and_guides():
    index = load_relation_index(RELATIONS, REGISTRY)
    # GUIDES không phải quan hệ sửa đổi -> không nằm trong index.
    for entries in index.values():
        for entry in entries:
            assert entry["relation_type"] in {
                "AMENDS",
                "SUPPLEMENTS",
                "REPEALS",
                "REPLACES",
            }
            assert entry["from_so_hieu"] != entry.get("to_self", "")


def test_amended_by_matches_clause_level():
    """48/2024 Điều 12 khoản 3 bị 149/2025 REPEALS (Điều 12 khoản 3).
    Record khoản 3 phải có amended_by chứa 149/2025; khoản 1 thì không."""
    index = load_relation_index(RELATIONS, REGISTRY)
    doc = "48/2024/QH15"
    entries = index.get(doc, [])
    assert any(
        e["from_so_hieu"] == "149/2025/QH15"
        and e["relation_type"] == "REPEALS"
        for e in entries
    )

    matched_k3 = match_amended_by(entries, article_number="12", clause_number="3")
    matched_k1 = match_amended_by(entries, article_number="12", clause_number="1")

    assert any(e["from_so_hieu"] == "149/2025/QH15" for e in matched_k3)
    # Khoản 1 không bị 149/2025 bãi bỏ -> không khớp REPEALS đó.
    assert not any(
        e["from_so_hieu"] == "149/2025/QH15" and e["relation_type"] == "REPEALS"
        for e in matched_k1
    )


def test_amended_by_multi_article_provision():
    """1 quan hệ REPEALS 149/2025/48/2024 có to_provision liệt kê 2 điều:
    'Điều 12 (khoản 3); Điều 15 (khoản 9 điểm c)'. Cả D12k3 VÀ D15k9
    phải khớp quan hệ đó (không chỉ điều đầu tiên)."""
    index = load_relation_index(RELATIONS, REGISTRY)
    entries = index.get("48/2024/QH15", [])

    matched_12 = match_amended_by(entries, article_number="12", clause_number="3")
    matched_15 = match_amended_by(entries, article_number="15", clause_number="9")

    assert any(e["from_so_hieu"] == "149/2025/QH15" for e in matched_12)
    assert any(e["from_so_hieu"] == "149/2025/QH15" for e in matched_15)

    # Khoản 15 khác (vd khoản 1) không bị bãi bỏ -> không khớp.
    matched_15_k1 = match_amended_by(entries, article_number="15", clause_number="1")
    assert not any(
        e["from_so_hieu"] == "149/2025/QH15" and e["relation_type"] == "REPEALS"
        for e in matched_15_k1
    )


def test_document_amended_by_attached_to_records():
    """Mọi record của văn bản bị sửa phải mang document_amended_by (level văn
    bản), bất kể khớp điều/khoản hay không."""
    index = load_relation_index(RELATIONS, REGISTRY)
    document = build_document("67_2025_QH15.docx")
    records = document_to_records(
        document,
        document_number="67/2025/QH15",
        dataset_target="thue_tndn",
        source_filename="67_2025_QH15.docx",
        relation_index=index,
    )
    assert records
    for record in records:
        doc_level = record.get("document_amended_by", [])
        amenders = {e["from_so_hieu"] for e in doc_level}
        # 67/2025 bị 6 văn bản sửa (116,127,133,141,143,09/2026).
        assert "127/2025/QH15" in amenders
        assert "09/2026/QH16" in amenders


def test_relation_index_missing_file_returns_empty():
    missing = PROJECT_ROOT / "data" / "metadata" / "KhongTonTai.csv"
    assert load_relation_index(missing, missing) == {}


def test_include_articles_filters_annex_by_annex_giu_lai():
    """Cột annex_giu_lai chỉ định giữ phụ lục nào với chế độ include_*:
    - 144/2026 (dieu=1..5, annex_giu_lai=1;2) -> giữ đúng phụ lục 1, 2.
    - 143/2025 (dieu=7;50;51, annex_giu_lai rỗng) -> bỏ hết 9 phụ lục."""
    from FilterDocuments import filter_document, load_filter_rules

    rules = load_rules()

    def doc_for(so_hieu: str):
        row = next(
            r for r in [x for x in rules if x["document_number"] == so_hieu]
        )
        matches = list(RAW_DOCX.rglob(row["source_filename"]))
        assert matches
        return build_document_structure(
            classify_blocks(
                clean_blocks(extract_docx(str(matches[0])))
            )
        ), row

    doc144, rule144 = doc_for("144/2026/NĐ-CP")
    out144 = filter_document(doc144, rule144)
    assert [a["number"] for a in out144["annexes"]] == [1, 2]

    doc143, rule143 = doc_for("143/2025/QH15")
    out143 = filter_document(doc143, rule143)
    assert out143["annexes"] == []


def test_full_mode_keeps_all_annexes():
    """Chế độ full giữ toàn bộ phụ lục (nội dung chính thức của văn bản)."""
    from FilterDocuments import filter_document, load_filter_rules

    rules = load_rules()
    rule = next(
        r for r in rules if r["document_number"] == "181/2025/NĐ-CP"
    )
    matches = list(RAW_DOCX.rglob(rule["source_filename"]))
    document = build_document_structure(
        classify_blocks(clean_blocks(extract_docx(str(matches[0]))))
    )
    out = filter_document(document, rule)
    assert len(out["annexes"]) >= 1


def test_no_relation_index_keeps_empty_amended_by():
    """Tương thích ngược: không truyền relation_index -> amended_by rỗng
    (không bịa dữ liệu)."""
    document = build_document("59_2020_QH14.docx")
    records = document_to_records(
        document,
        document_number="59/2020/QH14",
        dataset_target="doanh_nghiep",
        source_filename="59_2020_QH14.docx",
    )
    for record in records:
        assert record.get("amended_by") == []
        assert record.get("document_amended_by") == []
