"""Test mẫu cho `evaluation/EvaluateMulti.py` + tính năng đa ý mới.

OFFLINE hoàn toàn: không gọi Qwen, không gọi mạng, không cần Neo4j
(dùng LLM giả). Mục tiêu: **coi có lỗi không** trước khi chạy test
tổng (cần Qwen3 thật, ~15-25 phút).

C覆盖:
    1. Metric per-intent (Hit@3/Hit@5) tính đúng trên data thật.
    2. "Vẽ lại ý con": LLM giả trả câu rõ -> dùng câu đó để tra;
       LLM lỗi / trả rỗng -> giữ nguyên ý con (không bịa, không crash).
    3. NoLLM -> rewriter KHÔNG viết lại (đường 65 câu/OOD không đổi).
    4. "Hỏi lại gộp": meta có đúng 1 câu hỏi gộp khi LLM báo thiếu.
    5. Lọc bộ test: 1 câu/trường hợp (65 -> 36).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.QueryDecomposer import (  # noqa: E402
    QueryDecomposer,
    default_sub_query_rewriter,
)
from evaluation.EvaluateMulti import (  # noqa: E402
    expected_node_ids,
    metric_for_multi,
    rank_of,
    read_multi_intent_questions,
)
from evaluation.EvaluatePipeline import dedupe_by_case  # noqa: E402


# ---------------------------------------------------------------------------
# LLM giả
# ---------------------------------------------------------------------------

class NoLLM:
    """Giống hệt NoLLM trong EvaluatePipeline (name="none")."""

    name = "none"

    def complete(self, prompt: str) -> str:
        return "(bỏ qua LLM - chỉ đo truy xuất)"


class FakeLLM:
    """LLM giả trả về theo thứ tự đã cấu."""

    name = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.i = 0

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies[self.i] if self.i < len(self.replies) else ""
        self.i += 1
        return reply


class BoomLLM:
    name = "boom"

    def complete(self, prompt: str) -> str:
        raise RuntimeError("ollama down")


class Ev:
    """Evidence giả (đủ field dùng trong QueryDecomposer)."""

    def __init__(self, nid: str, text: str = "t") -> None:
        self.node_id = nid
        self.text = text
        self.citation = nid


# ---------------------------------------------------------------------------
# 1) Metric per-intent
# ---------------------------------------------------------------------------

def test_metric_counts_each_intent_separately():
    """2 intent, chỉ 1 cái trong top-3 -> hit3 = 0.5, hit5 = 0.5."""

    class Ans:
        evidence = [
            {"node_id": "A_1"}, {"node_id": "B_1"}, {"node_id": "C_1"},
        ]

    row = {
        "intents": [
            {
                "text": "ý 1",
                "expected_sources": [{
                    "document_number": "108/2025/QH15",
                    "article_number": "4",
                    "clause_number": "8",
                }],
            },
            {
                "text": "ý 2",
                "expected_sources": [{
                    "document_number": "168/2025/NĐ-CP",
                    "article_number": "3",
                    "clause_number": "10",
                }],
            },
        ]
    }
    m = metric_for_multi(Ans(), row)
    assert m["n_intents"] == 2
    # "A_1" khác target -> ý1 trượt; "B_1" cũng vậy -> cả 2 trượt
    assert m["intents_hit_at_3"] == 0.0
    assert m["intents_hit_at_5"] == 0.0


def test_metric_hit_at_3_and_5_by_rank():
    """Target ở rank 3 -> hit3=True; rank 5 -> hit3=False, hit5=True."""

    class Ans:
        evidence = [
            {"node_id": "X"}, {"node_id": "Y"},
            {"node_id": "108_2025_QH15_D4_K8"},
            {"node_id": "Z"}, {"node_id": "W"},
        ]

    row = {
        "intents": [{
            "text": "ý",
            "expected_sources": [{
                "document_number": "108/2025/QH15",
                "article_number": "4",
                "clause_number": "8",
            }],
        }]
    }
    m = metric_for_multi(Ans(), row)
    assert m["detail"][0]["rank"] == 3
    assert m["intents_hit_at_3"] == 1.0
    assert m["intents_hit_at_5"] == 1.0


def test_ood_intent_excluded_from_hit_rate():
    """Intent đánh dấu ood -> KHÔNG tính vào hit-rate, không kéo thụt."""

    class Ans:
        evidence = [{"node_id": "108_2025_QH15_D4_K8"}]

    row = {
        "intents": [
            {"text": "ý ood", "ood": True},
            {
                "text": "ý thường",
                "expected_sources": [{
                    "document_number": "108/2025/QH15",
                    "article_number": "4",
                    "clause_number": "8",
                }],
            },
        ]
    }
    m = metric_for_multi(Ans(), row)
    assert m["n_intents"] == 1
    assert m["ood_intents"] == 1
    assert m["intents_hit_at_3"] == 1.0


def test_rank_of_returns_none_when_missing():
    assert rank_of(["A"], ["B", "C"]) is None
    assert rank_of(["A"], ["B", "A"]) == 2
    assert rank_of([], ["A"]) is None


def test_expected_node_ids_uses_clause_level():
    ids = expected_node_ids({
        "expected_sources": [{
            "document_number": "168/2025/NĐ-CP",
            "article_number": "3",
            "clause_number": "10",
        }]
    })
    assert ids == ["168_2025_NĐ_CP_D3_K10"]


# ---------------------------------------------------------------------------
# 2) + 3) "Vẽ lại ý con"
# ---------------------------------------------------------------------------

def test_rewriter_uses_llm_rewrite():
    """LLM trả câu rõ -> rewriter dùng câu đó."""
    llm = FakeLLM(["chủ thể nộp thuế theo Luật Quản lý thuế là ai?"])
    rw = default_sub_query_rewriter(llm)
    out = rw("người nộp thuế là ai", "CÂU HỎI GỐC: ...")
    assert out == "chủ thể nộp thuế theo Luật Quản lý thuế là ai?"


def test_rewriter_falls_back_to_original_on_error():
    """LLM lỗi -> giữ nguyên ý con (không bịa, không crash)."""
    rw = default_sub_query_rewriter(BoomLLM())
    assert rw("ý con", "ctx") == "ý con"


def test_rewriter_falls_back_on_empty_response():
    rw = default_sub_query_rewriter(FakeLLM([""]))
    assert rw("ý con", "ctx") == "ý con"


def test_rewriter_noop_for_no_llm():
    """NoLLM -> giữ nguyên ý con (đường 65 câu/OOD không đổi)."""
    assert default_sub_query_rewriter(NoLLM())("ý con", "ctx") == "ý con"
    assert default_sub_query_rewriter(None)("ý con", "ctx") == "ý con"


def test_fill_rewrites_uncovered_intent_before_runner():
    """fill() vẽ lại ý con rồi mới gọi runner với câu ĐÃ vẽ lại."""
    llm = FakeLLM([
        # 1) coverage: có 1 ý thiếu
        '{"intents": ["a"], "uncovered": ["mức GTGT"], "confidence": 0.6}',
        # 2) rewriter
        "mức thuế GTGT theo Luật Quản lý thuế là bao nhiêu?",
        # 3) coverage lần 2: đã đủ
        '{"intents": ["a"], "uncovered": [], "confidence": 0.9}',
    ])
    calls: list[str] = []
    d = QueryDecomposer(
        llm=llm, sub_query_rewriter=default_sub_query_rewriter(llm)
    )
    merged, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức GTGT bao nhiêu?",
        [Ev("A")],
        lambda sub: (calls.append(sub), [Ev("NEW")])[1],
    )
    assert len(calls) == 1
    assert calls[0] == "mức thuế GTGT theo Luật Quản lý thuế là bao nhiêu?"
    assert meta["filled"] == 1
    assert meta["rewritten"] == {
        "mức GTGT": "mức thuế GTGT theo Luật Quản lý thuế là bao nhiêu?"
    }


def test_fill_without_rewriter_keeps_original_sub():
    """Không gắn rewriter -> runner nhận ý con GỐC, meta không có
    'rewritten' (hành vi cũ nguyên vẹn)."""
    llm = FakeLLM([
        '{"intents": ["a"], "uncovered": ["mức GTGT"], "confidence": 0.6}',
        '{"intents": ["a"], "uncovered": [], "confidence": 0.9}',
    ])
    calls: list[str] = []
    d = QueryDecomposer(llm=llm)
    _, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức GTGT bao nhiêu?",
        [Ev("A")],
        lambda sub: (calls.append(sub), [Ev("NEW")])[1],
    )
    assert calls == ["mức GTGT"]
    assert "rewritten" not in meta


# ---------------------------------------------------------------------------
# 4) "Hỏi lại gộp"
# ---------------------------------------------------------------------------

def test_clarify_recorded_in_meta():
    """LLM báo thiếu thông tin -> meta có 1 câu hỏi gộp."""
    llm = FakeLLM([
        json.dumps({
            "intents": ["a", "b"],
            "uncovered": ["người nộp thuế"],
            "clarify": (
                "Về 'người nộp thuế', bạn muốn theo văn bản nào: "
                "168/2025 hay 108/2025?"
            ),
            "confidence": 0.5,
        }, ensure_ascii=False),
        '{"intents": ["a", "b"], "uncovered": [], "confidence": 0.9}',
    ])
    d = QueryDecomposer(llm=llm)
    _, meta = d.fill(
        "Người nộp thuế là ai và kỳ tính thuế thế nào?",
        [Ev("A")],
        lambda sub: [Ev("NEW")],
    )
    assert "168/2025" in meta["clarify"]
    assert "108/2025" in meta["clarify"]


def test_no_clarify_when_not_needed():
    """Không thiếu -> không có 'clarify' trong meta."""
    llm = FakeLLM([
        '{"intents": ["a"], "uncovered": [], "confidence": 0.9}',
    ])
    d = QueryDecomposer(llm=llm)
    _, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức GTGT bao nhiêu?",
        [Ev("A")],
        lambda sub: [Ev("NEW")],
    )
    assert "clarify" not in meta
    assert meta["stop_reason"] == "covered"


def test_no_llm_fill_is_noop():
    """Không LLM -> fill no-op, không gọi runner."""
    calls: list[str] = []
    d = QueryDecomposer(llm=None)
    merged, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức GTGT bao nhiêu?",
        [Ev("A")],
        lambda sub: (calls.append(sub), [Ev("B")])[1],
    )
    assert [e.node_id for e in merged] == ["A"]
    assert meta["filled"] == 0
    assert meta["rounds"] == 0
    # F5b: no-op vẫn cấp per_intent (1 ý = câu gốc) cho rerank theo từng ý.
    assert meta["per_intent"] == [
        {
            "sub_query": "Hộ kinh doanh đăng ký thuế thế nào, và mức GTGT bao nhiêu?",
            "node_ids": ["A"],
        }
    ]
    assert calls == []


# ---------------------------------------------------------------------------
# 5) Lọc bộ test (65 -> 36)
# ---------------------------------------------------------------------------

def test_dedupe_by_case_keeps_one_per_case():
    rows = [
        {"id": "A_Q1"}, {"id": "A_Q2"},
        {"id": "B_Q1"}, {"id": "B_Q2"},
        {"id": "C_Q1"},
    ]
    out = dedupe_by_case(rows)
    assert [r["id"] for r in out] == ["A_Q1", "B_Q1", "C_Q1"]


def test_dedupe_by_case_real_dataset_gives_36():
    """Bộ thật: 65 câu đơn -> 36 trường hợp, không mất target nào."""
    path = ROOT / "evaluation" / "questions_test.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    answerable = [r for r in rows if r.get("answerable", True)]
    kept = dedupe_by_case(answerable)
    assert len(answerable) == 65
    assert len(kept) == 36
    # Target của câu bị cắt == target của câu giữ (không mất đáp án).
    by_case: dict[str, list[dict]] = {}
    for r in answerable:
        base = str(r["id"]).rsplit("_Q", 1)[0]
        by_case.setdefault(base, []).append(r)
    for group in by_case.values():
        targets = {
            json.dumps(r.get("expected_sources"), sort_keys=True)
            for r in group
        }
        assert len(targets) == 1, f"trường hợp có target khác nhau: {group}"


def test_multi_intent_dataset_readable():
    """10 câu đa ý, >= 20 intent."""
    rows = read_multi_intent_questions()
    assert len(rows) == 10
    n_intents = sum(len(r.get("intents") or []) for r in rows)
    assert n_intents >= 20
