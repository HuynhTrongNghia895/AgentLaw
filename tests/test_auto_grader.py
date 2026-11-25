"""Test mẫu offline cho `evaluation/auto_grader.py` (G7).

Không cần Qwen/Neo4j/mạng — chấm bằng rule trên data in-memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.auto_grader import (  # noqa: E402
    check_citations,
    check_fact_coverage,
    check_forbidden,
    check_unsupported_numbers,
    extract_numbers,
    grade_all,
    grade_case,
)


def _ev(nid: str, text: str) -> dict:
    return {"node_id": nid, "text": text, "citation": nid}


# ---------------------------------------------------------------------------
# 1) Citation
# ---------------------------------------------------------------------------

def test_citation_in_evidence_ok():
    ev = [
        _ev("168_2025_NĐ_CP_D3_K10", "Khoản 10 điều 3 quy định..."),
        _ev("108_2025_QH15_D4_K8", "Khoản 8 điều 4..."),
    ]
    res = check_citations(
        "Theo khoản 10 điều 3 NĐ 168/2025, người nộp thuế là...", ev
    )
    assert res["ok"]
    assert any("điều 3" in c["citation"] for c in res["cited"])


def test_citation_not_in_evidence_flagged():
    """Citation 'điều 99' không có trong evidence -> FLAG (bịa điều)."""
    ev = [_ev("168_2025_NĐ_CP_D3_K10", "Khoản 10 điều 3...")]
    res = check_citations("Căn cứ điều 99 khoản 2, mức phạt là...", ev)
    assert not res["ok"]
    assert any("điều 99" in m for m in res["missing"])


def test_citation_by_node_suffix_only():
    """Không có chữ 'điều X' trong text evidence, nhưng có NODE đúng hậu tố
    -> vẫn khớp (khớp node, không phụ thuộc text)."""
    ev = [_ev("108_2025_QH15_D4", "nội dung điều 4")]
    res = check_citations("quy định tại điều 4", ev)
    assert res["ok"]


def test_no_citations_ok():
    """Câu trả lời không trích điều/khoản nào -> không có gì để sai."""
    res = check_citations(
        "Chưa tìm thấy quy định phù hợp trong dữ liệu hiện có.",
        [_ev("A_D1", "x")],
    )
    assert res["ok"]
    assert res["cited"] == []


# ---------------------------------------------------------------------------
# 2) Forbidden claims
# ---------------------------------------------------------------------------

def test_forbidden_hit_fails_case():
    res = check_forbidden(
        "Mức phạt 400 triệu theo Luật Giao thông...",
        ["Luật Giao thông", "triệu"],
    )
    assert not res["ok"]
    assert "Luật Giao thông" in res["hits"]
    assert "triệu" in res["hits"]


def test_forbidden_absent_passes():
    res = check_forbidden(
        "Chưa tìm thấy trong dữ liệu (ngoài phạm vi doanh nghiệp/thuế).",
        ["Luật Giao thông", "triệu"],
    )
    assert res["ok"]
    assert res["hits"] == []


def test_forbidden_case_insensitive():
    res = check_forbidden("PHẠT theo LUẬT GIAO THÔNG", ["luật giao thông"])
    assert not res["ok"]


def test_empty_forbidden_list_ok():
    assert check_forbidden("bất kỳ", None)["ok"]
    assert check_forbidden("bất kỳ", [])["ok"]


# ---------------------------------------------------------------------------
# 3) Unsupported numbers (bịa số liệu)
# ---------------------------------------------------------------------------

def test_number_in_evidence_ok():
    ev = [_ev("A_D1", "mức phạt là 0,05% mỗi ngày và 90 ngày")]
    res = check_unsupported_numbers("phạt 0,05% trong 90 ngày", ev)
    assert res["ok"]
    assert res["unsupported"] == []


def test_number_not_in_evidence_flagged():
    ev = [_ev("A_D1", "kỳ tính thuế là tháng, quý, năm")]
    res = check_unsupported_numbers("nộp chậm phạt 0,05% mỗi ngày", ev)
    assert not res["ok"]
    assert res["unsupported"]


def test_no_numbers_is_fine():
    ev = [_ev("A_D1", "x")]
    res = check_unsupported_numbers("không có con số nào ở đây", ev)
    assert res["ok"]
    assert res["numbers"] == []


def test_extract_numbers_finds_money_percent_days():
    got = extract_numbers("phạt 200 triệu hoặc 0,03% mỗi ngày, tối đa 90 ngày")
    assert any("200 triệu" in n for n in got)
    assert any("0,03%" in n for n in got)
    assert any("90 ngày" in n for n in got)


# ---------------------------------------------------------------------------
# 4) Fact coverage (sàn)
# ---------------------------------------------------------------------------

def test_coverage_all_facts_covered():
    facts = [
        {"text": "người nộp thuế", "anchors": ["người nộp thuế"]},
        {"text": "kỳ tính thuế", "anchors": ["kỳ tính thuế", "thời kỳ thuế"]},
    ]
    res = check_fact_coverage(
        "Người nộp thuế là DN. Kỳ tính thuế là tháng hoặc năm.", facts
    )
    assert res["coverage"] == 1.0
    assert res["total"] == 2
    assert all(f["covered"] for f in res["facts"])


def test_coverage_strict_misses_synonyms():
    """Từ đồng nghĩa (không trong anchor) bị trượt -> SÀN (strict)."""
    facts = [
        {"text": "kỳ tính thuế", "anchors": ["kỳ tính thuế"]},
        {"text": "mức thuế GTGT", "anchors": ["mức thuế gtgt", "thuế gtgt"]},
    ]
    res = check_fact_coverage("Theo thời kỳ thuế... (bỏ ý thứ 2)", facts)
    assert res["coverage"] == 0.0
    assert res["total"] == 2


def test_coverage_no_facts_is_neutral():
    res = check_fact_coverage("câu trả lời bất kỳ", None)
    assert res["coverage"] == 1.0
    assert res["total"] == 0
    assert res["ok"]


def test_anchor_synonym_match_counts():
    """Anchor thứ 2 khớp (thay thế được) -> covered."""
    facts = [
        {"text": "kỳ tính thuế", "anchors": ["kỳ tính thuế", "thời kỳ thuế"]},
    ]
    res = check_fact_coverage("Thời kỳ thuế áp dụng...", facts)
    assert res["coverage"] == 1.0
    assert res["facts"][0]["anchors_matched"]


# ---------------------------------------------------------------------------
# 5) grade_case + grade_all
# ---------------------------------------------------------------------------

def _good_record() -> dict:
    return {
        "id": "G3-01",
        "answer": "Theo điều 3 khoản 10 NĐ 168/2025, người nộp thuế là... "
                  "kỳ tính thuế là tháng, quý hoặc năm.",
        "evidence": [
            _ev("168_2025_NĐ_CP_D3_K10", "Khoản 10 điều 3"),
            _ev("108_2025_QH15_D4_K8", "Khoản 8 điều 4, kỳ tính thuế"),
        ],
        "expected": {
            "evidence_node_ids": [
                "168_2025_NĐ_CP_D3_K10", "108_2025_QH15_D4_K8"
            ],
            "required_facts": [
                {"text": "người nộp thuế", "anchors": ["người nộp thuế"]},
                {"text": "kỳ tính thuế",
                 "anchors": ["kỳ tính thuế"]},
            ],
            "forbidden_claims": ["Luật Giao thông"],
            "min_fact_coverage": 0.5,
        },
    }


def _bad_record() -> dict:
    r = _good_record()
    r["id"] = "G3-02"
    # Bịa: dẫn 'điều 99' (không trong evidence) + bịa số + dính forbidden
    r["answer"] = (
        "Theo điều 99, mức phạt 400 triệu theo Luật Giao thông, "
        "người nộp thuế là..."
    )
    return r


def test_grade_case_pass():
    g = grade_case(_good_record())
    assert g["pass"]
    assert g["citations"]["ok"]
    assert g["forbidden"]["ok"]
    assert g["unsupported_numbers"]["ok"]
    assert g["fact_coverage"]["coverage"] == 1.0
    assert g["needs_human"] == []


def test_grade_case_fails_on_multiple_violations():
    g = grade_case(_bad_record())
    assert not g["pass"]
    assert not g["citations"]["ok"]            # điều 99 bịa
    assert not g["forbidden"]["ok"]           # dính "Luật Giao thông"
    assert not g["unsupported_numbers"]["ok"]  # "400 triệu" không có trong ev
    assert g["needs_human"]                    # "kỳ tính thuế" thiếu
    assert "kỳ tính thuế" in g["needs_human"]


def test_grade_case_fact_below_threshold():
    r = _good_record()
    r["expected"]["min_fact_coverage"] = 1.0
    r["answer"] = "Người nộp thuế là..."  # bỏ ý 2 (kỳ tính thuế)
    g = grade_case(r)
    assert not g["pass"]
    assert g["fact_coverage"]["coverage"] == 0.5


def test_grade_all_aggregates():
    s = grade_all([_good_record(), _bad_record()])
    assert s["n_cases"] == 2
    assert s["pass_rate"] == 0.5
    assert "G3-02" in s["failed_cases"]
    assert s["citation_ok"] == 0.5
    assert s["forbidden_ok"] == 0.5
    assert s["numbers_ok"] == 0.5
    assert s["needs_human"]["G3-02"]
