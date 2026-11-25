"""Test BuildPrompt sau X7 — lớp dựng prompt chuẩn của hệ thống.

Trước X7: `src/utils/BuildPrompt.py` import `SearchVectorDb` (module đã
xoá từ P1) → ImportError, và không ai import nó → **chết**.

Sau X7: module này là **call site duy nhất** dựng prompt cho Legal Agent:
nhận list `Evidence` → tự ghép context block + status + graphiti.

Verify:
  - import được, không lỗi (module đã xoá không còn được kéo theo)
  - prompt chứa citation + text evidence (KHÔNG merge/rewrite)
  - evidence rỗng → thông báo "không đủ căn cứ", KHÔNG raise
  - output GIỐNG prompt cũ của `agents.prompts.build_legal_prompt`
    (đổi call site không làm đổi chất lượng câu trả lời)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "utils",
    PROJECT_ROOT / "src" / "retrieval",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from graph.kg.materialize import Evidence  # noqa: E402

from agents.prompts import build_legal_prompt as legacy_build_legal_prompt  # noqa: E402
from utils.BuildPrompt import (  # noqa: E402
    NO_CONTEXT_MESSAGE,
    build_context,
    build_legal_prompt,
    build_reference,
    normalize_text,
)


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            node_id="76_2025_QH15_D1_K24",
            citation="Điều 1 Khoản 24 (76/2025/QH15)",
            text=(
                "Sửa đổi, bổ sung khoản 1 Điều 213 như sau: "
                "Chi nhánh không được hoạt động tại địa điểm kinh doanh..."
            ),
            kind="clause",
            node_type="clause",
        ),
        Evidence(
            node_id="59_2020_QH14_D213",
            citation="Điều 213 (59/2020/QH14)",
            text="Chi nhánh là đơn vị phụ thuộc của doanh nghiệp.",
            kind="article",
            node_type="article",
        ),
    ]


# ---------------------------------------------------------------------------
# Import được (module đã xoá không còn bị kéo theo)
# ---------------------------------------------------------------------------
def test_module_imports_without_dead_dependency():
    import BuildPrompt  # noqa: F401

    assert hasattr(BuildPrompt, "build_legal_prompt")


def test_no_import_of_deleted_search_vector_db():
    """`SearchVectorDb` đã bị xoá — module này không được còn import nó.

    Kiểm tra bằng **import thật** (không so chuỗi — docstring nhắc tên
    module đã xoá là hợp lệ).
    """
    import BuildPrompt

    assert not hasattr(BuildPrompt, "search_chunks")
    assert not hasattr(BuildPrompt, "DEFAULT_TOP_K")
    # Module đã xoá -> import nó phải fail, nên BuildPrompt đã không
    # phụ thuộc vào nó.
    with pytest.raises(ImportError):
        import SearchVectorDb  # noqa: F401


# ---------------------------------------------------------------------------
# Prompt chứa citation + text (không merge/rewrite văn bản pháp luật)
# ---------------------------------------------------------------------------
def test_prompt_contains_citation_and_evidence_text():
    prompt = build_legal_prompt("Câu hỏi?", _evidence())
    assert "Điều 1 Khoản 24 (76/2025/QH15)" in prompt
    assert "Điều 213 (59/2020/QH14)" in prompt
    assert "Sửa đổi, bổ sung khoản 1 Điều 213" in prompt
    assert "Chi nhánh là đơn vị phụ thuộc của doanh nghiệp." in prompt


def test_prompt_contains_question_and_sections():
    prompt = build_legal_prompt("Quy định mới liên quan đến điều 213 là gì?",
                                _evidence())
    assert "Quy định mới liên quan đến điều 213 là gì?" in prompt
    assert "NGỮ CẢNH PHÁP LÝ" in prompt
    assert "CÂU HỎI" in prompt


def test_prompt_includes_status_block_when_given():
    prompt = build_legal_prompt(
        "Câu hỏi?", _evidence(), status_block="Điều 213: hết hiệu lực 01/01/2026"
    )
    assert "TÌNH TRẠNG HIỆU LỰC" in prompt
    assert "Điều 213: hết hiệu lực 01/01/2026" in prompt


def test_prompt_includes_graphiti_block_when_given():
    prompt = build_legal_prompt(
        "Câu hỏi?", _evidence(),
        graphiti_block="76/2025/QH15 sửa Điều 213 của 59/2020/QH14",
    )
    assert "TRI THỨC MỚI" in prompt
    assert "76/2025/QH15 sửa Điều 213" in prompt


def test_empty_evidence_yields_notice_and_does_not_raise():
    """Không có căn cứ -> thông báo, KHÔNG raise (quyết định #12)."""
    prompt = build_legal_prompt("Câu hỏi ngoài phạm vi?", [])
    assert "Không tìm thấy đủ căn cứ pháp lý" in prompt


def test_none_evidence_yields_notice():
    prompt = build_legal_prompt("Câu hỏi?", None)
    assert "Không tìm thấy đủ căn cứ pháp lý" in prompt


# ---------------------------------------------------------------------------
# Đổi call site KHÔNG đổi chất lượng prompt
# ---------------------------------------------------------------------------
def test_output_matches_legacy_prompt_builder():
    """Prompt mới phải GIỐNG HỆT prompt cũ (chỉ đổi chỗ gọi)."""
    evs = _evidence()
    from graph.kg.materialize import format_evidence_block

    expected = legacy_build_legal_prompt(
        "Câu hỏi?",
        format_evidence_block(evs),
        "status",
        "graphiti",
    )
    actual = build_legal_prompt("Câu hỏi?", evs, "status", "graphiti")
    assert actual == expected


# ---------------------------------------------------------------------------
# Hàm phụ (tương thích cũ)
# ---------------------------------------------------------------------------
def test_normalize_text():
    assert normalize_text(None) == ""
    assert normalize_text("  x  ") == "x"
    assert normalize_text(5) == "5"


def test_build_reference_prefers_citation():
    assert build_reference({"citation": "Điều 1"}) == "Điều 1"


def test_build_reference_falls_back_to_numbers():
    ref = build_reference({
        "article_number": "213",
        "clause_number": "1",
        "document_number": "59/2020/QH14",
    })
    assert "Điều 213" in ref
    assert "Khoản 1" in ref
    assert "59/2020/QH14" in ref


def test_build_reference_unknown():
    assert build_reference({}) == "Không xác định căn cứ"


def test_build_context_skips_invalid_and_dedupes():
    context = build_context([
        {"storage_id": "a", "text": "Nội dung A",
         "metadata": {"citation": "Điều 1"}},
        {"storage_id": "a", "text": "Trùng"},          # trùng -> bỏ
        {"storage_id": "b", "text": ""},                # text rỗng -> bỏ
        {"text": "Không có id"},                        # thiếu id -> bỏ
    ])
    assert "Nội dung A" in context
    assert "Trùng" not in context
    assert "[NGUỒN 1]" in context


def test_build_context_empty_returns_message():
    assert build_context([]) == NO_CONTEXT_MESSAGE
