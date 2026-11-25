"""Test P7 X3 — QueryRefiner (skill hỏi lại + chuẩn hóa).

Tất cả test dùng LLM GIẢ (mock) -> không tốn token, không cần Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval.QueryRefiner import (  # noqa: E402
    KIND_CLEARED,
    KIND_GREETING,
    KIND_VAGUE,
    QueryRefiner,
    RefineResult,
    has_sufficient_anchor,
    is_greeting,
    parse_refiner_json,
)
from retrieval.QueryUnderstanding import understand  # noqa: E402


def test_parse_refiner_json_skips_think_block_and_invalid_brace_prefix():
    raw = '<think>draft {not json}</think> preface {broken} ```json\n{"intents": []}\n```'
    assert parse_refiner_json(raw) == {"intents": []}


def test_parse_refiner_json_does_not_return_think_content():
    raw = '<think>{"kind":"vague"}</think>{"kind":"cleared","question":"Q"}'
    assert parse_refiner_json(raw) == {"kind": "cleared", "question": "Q"}


class FakeLLM:
    """LLM giả trả về JSON cố định."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload


# ---------------------------------------------------------------------------
# Rule-based
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "q",
    ["Xin chào", "xin chào!", "Chào bạn", "hello", "Hi", "cảm ơn"],
)
def test_is_greeting_true(q):
    assert is_greeting(q) is True


@pytest.mark.parametrize(
    "q",
    [
        "Điều 49 Luật Doanh nghiệp 2020 bị sửa bởi văn bản nào?",
        "Khoản 3 Điều 12 Nghị định 168/2025 quy định gì?",
        "thuế GTGT hộ kinh doanh 2025 nộp thế nào",
    ],
)
def test_is_greeting_false(q):
    assert is_greeting(q) is False


def test_refiner_greeting_stops_pipeline():
    """Lời chào -> trả lời + hỏi lại, KHÔNG hỏi LLM."""
    llm = FakeLLM("{}")
    r = QueryRefiner(llm=llm)

    res = r.refine("Xin chào")

    assert res.kind == KIND_GREETING
    assert res.is_greeting is True
    assert res.needs_clarification is True
    assert res.answer  # có lời chào
    assert res.follow_up  # có câu hỏi lại
    assert llm.prompts == []  # KHÔNG gọi LLM


def test_refiner_cleared_when_anchor_sufficient():
    """Câu đã có số hiệu + điều -> cleared, KHÔNG gọi LLM."""
    llm = FakeLLM("{}")
    r = QueryRefiner(llm=llm)

    res = r.refine("Điều 49 Luật Doanh nghiệp 2020 bị sửa bởi văn bản nào?")

    assert res.kind == KIND_CLEARED
    assert res.needs_clarification is False
    assert res.confidence == 1.0
    assert llm.prompts == []  # KHÔNG gọi LLM


def test_refiner_vague_calls_llm_and_returns_choices():
    """Câu mơ hồ -> gọi LLM, trả câu hỏi lại + lựa chọn đóng."""
    llm = FakeLLM(
        '{"kind": "vague", "question": "Cách tính thuế GTGT doanh nghiệp 2025", '
        '"legal_group": "thue_gtgt", "intent": "tra_cuu", '
        '"missing": ["loại thuế"], '
        '"choices": ["Thuế GTGT", "Thuế TNDN", "Thời hạn nộp"], '
        '"clarifying_question": "Bạn muốn tra phần nào?", '
        '"confidence": 0.4}'
    )
    r = QueryRefiner(llm=llm)

    res = r.refine("thuế 2025 nộp thế nào")

    assert res.kind == KIND_VAGUE
    assert res.needs_clarification is True
    assert len(res.choices) == 3
    assert res.follow_up == "Bạn muốn tra phần nào?"
    assert len(llm.prompts) == 1


def test_refiner_vague_generates_followup_from_choices():
    """LLM trả choices nhưng không có clarifying_question -> tự dựng."""
    llm = FakeLLM(
        '{"kind": "vague", "question": "q", "choices": ["A", "B"], '
        '"confidence": 0.3}'
    )
    r = QueryRefiner(llm=llm)

    res = r.refine("mơ hồ")

    assert res.kind == KIND_VAGUE
    assert "1) A" in res.follow_up
    assert "2) B" in res.follow_up


def test_refiner_user_answer_rewrites_question():
    """User đã chọn -> câu được viết lại, không hỏi lại nữa."""
    llm = FakeLLM(
        '{"kind": "cleared", "question": "Cách tính thuế GTGT doanh nghiệp 2025", '
        '"legal_group": "thue_gtgt", "intent": "tra_cuu", "confidence": 0.9}'
    )
    r = QueryRefiner(llm=llm)

    res = r.refine("thuế 2025 nộp thế nào", user_answer="Thuế GTGT")

    assert res.kind == KIND_CLEARED
    assert res.question == "Cách tính thuế GTGT doanh nghiệp 2025"
    assert res.needs_clarification is False


def test_refiner_without_llm_degrades_to_cleared():
    """Không có LLM -> không chặn user, trả câu gốc."""
    r = QueryRefiner(llm=None)

    res = r.refine("thuế 2025 nộp thế nào")

    assert res.kind == KIND_CLEARED
    assert res.question == "thuế 2025 nộp thế nào"
    assert res.needs_clarification is False


def test_refiner_bad_llm_output_degrades():
    """LLM trả rác -> degrade, không crash."""
    llm = FakeLLM("không phải JSON")
    r = QueryRefiner(llm=llm)

    res = r.refine("thuế 2025 nộp thế nào")

    assert res.kind == KIND_CLEARED
    assert res.source == "rule"


def test_refiner_llm_exception_degrades():
    """LLM ném exception -> degrade, không crash."""

    class BoomLLM:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("ollama down")

    r = QueryRefiner(llm=BoomLLM())

    res = r.refine("thuế 2025 nộp thế nào")

    assert res.kind == KIND_CLEARED
    assert res.source == "rule"


# ---------------------------------------------------------------------------
# Parse JSON
# ---------------------------------------------------------------------------

def test_parse_json_plain():
    assert parse_refiner_json('{"a": 1}') == {"a": 1}


def test_parse_json_with_fence():
    raw = '```json\n{"kind": "vague", "choices": []}\n```'
    assert parse_refiner_json(raw)["kind"] == "vague"


def test_parse_json_with_surrounding_text():
    raw = 'Đây là kết quả:\n{"kind": "cleared"}\nHết.'
    assert parse_refiner_json(raw)["kind"] == "cleared"


def test_parse_json_invalid():
    assert parse_refiner_json("no json here") is None
    assert parse_refiner_json("") is None
    assert parse_refiner_json("{broken") is None


def test_parse_json_skips_invalid_brace_and_removes_think_block():
    raw = '<think>secret {"private": true}</think> prefix {broken then {"kind": "cleared"}'
    assert parse_refiner_json(raw) == {"kind": "cleared"}


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

def test_has_sufficient_anchor():
    assert has_sufficient_anchor(
        understand("Điều 49 Luật Doanh nghiệp 2020 bị sửa bởi ai?")
    ) is True
    # Không có số hiệu -> chưa đủ
    assert has_sufficient_anchor(understand("thuế GTGT nộp thế nào")) is False


# ---------------------------------------------------------------------------
# F4a: thiếu vb + có Điều/Khoản -> buộc HỎI + cấp menu fallback (không bịa)
# F4b: rã soát menu với registry (loại số hiệu vb LLM hallucinate)
# ---------------------------------------------------------------------------

DOCS = [
    "108/2025/QH15 (Quản lý thuế)",
    "48/2024/QH15 (Thuế GTGT)",
    "59/2020/QH14 (Doanh nghiệp)",
]


def test_f4a_missing_doc_with_article_forces_clarify():
    """S3: 'Điều 2 quy định gì về người nộp thuế?' — LLM nói cleared,
    nhưng CÓ Điều MÀ THIẾU vb -> down về vague + cấp menu vb có sẵn."""
    llm = FakeLLM('{"kind": "cleared"}')
    r = QueryRefiner(llm=llm, allowed_documents=DOCS)

    res = r.refine("Điều 2 quy định gì về người nộp thuế?")

    assert res.kind == KIND_VAGUE
    assert res.needs_clarification is True
    assert res.follow_up  # có câu hỏi lại
    assert res.choices  # có menu
    # Menu chỉ gồm vb CÓ SẴN (không bịa): mỗi lựa chọn chứa 1 số hiệu thật.
    assert all(any(d.split(" (")[0] in c for d in DOCS) for c in res.choices)


def test_f4a_no_article_no_doc_still_answers_when_group_clear():
    """Câu thiếu Điều, có ý rõ (topic đơn) -> không rơi vào F4a fallback;
    vẫn vague (chưa đủ), nhưng KHÔNG ép menu vb chỉ vì có Điều."""
    llm = FakeLLM('{"kind": "cleared"}')
    r = QueryRefiner(llm=llm, allowed_documents=DOCS)

    res = r.refine("thuế GTGT nộp theo kỳ nào?")

    # Có nhóm rõ (thue_gtgt) nhưng thiếu anchor vb + Điều -> vague;
    # KHÔNG có Điều/Khoản nên F4a KHÔNG cấp menu vb fallback.
    assert res.kind == KIND_VAGUE


def test_f4b_filters_fabricated_doc_from_choices():
    """S4: LLM bịa '48/2025/QH15' (registry chỉ có 48/2024) -> bị loại,
    chỉ giữ lựa chọn có số hiệu vb THẬT nằm trong danh sách vb có sẵn."""
    llm = FakeLLM(
        '{"kind": "vague", '
        '"choices": ["48/2025/QH15 – thuế", '
        '"48/2024/QH15 (Thuế GTGT)", '
        '"59/2020/QH14 (Doanh nghiệp)"]}'
    )
    r = QueryRefiner(llm=llm, allowed_documents=DOCS)

    res = r.refine("Điều 5 quy định gì?")

    assert "48/2025/QH15" not in " ".join(res.choices)
    assert any("48/2024/QH15" in c for c in res.choices)
    assert any("59/2020/QH14" in c for c in res.choices)
    assert len(res.choices) == 2  # đúng 2 vb thật, bịa đã bị loại


def test_f4b_no_docs_list_skips_validation():
    """Danh sách vb có sẵn rỗng -> không rã soát được, giữ nguyên menu
    (đường degrade an toàn, không crash, không cắt oan)."""
    llm = FakeLLM(
        '{"kind": "vague", "choices": ["ABC/9999/QH99", "XYZ/8888/NĐ-CP"]}'
    )
    r = QueryRefiner(llm=llm, allowed_documents=[])

    res = r.refine("Điều 5 quy định gì?")

    assert len(res.choices) == 2  # không có registry -> giữ nguyên
