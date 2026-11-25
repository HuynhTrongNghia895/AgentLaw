"""P-B (Checkpoint B) — early-stop per-intent + intent_verdict + B2 signals.

Tương ứng §13.6: T1 early-stop trigger, T2 intent OOD (qua gate F5b),
T3 intent hợp lệ nhưng retrieval không tiến triển (INSUFFICIENT, không OOD),
T4 nhiều intent độc lập (stopped không chặn intent khác), T7 no-retry đảo
ngược khi có thông tin mới. T5 regression = test có sẵn (test_query_decomposer
đường no-op/covered/max_rounds) + baseline stub (chạy user).

Nguyên tắc B1: early-stop KHÔNG tự kết luận OOD — decomposer chỉ đánh dấu
`pending`; LegalRetrieval._intent_verdict (gate F5b / KG) mới ra verdict.
Tất cả test dùng LLM giả + runner closure -> không tốn token.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.QueryDecomposer import (  # noqa: E402
    VERDICT_COVERED,
    VERDICT_FAILED,
    VERDICT_INSUFFICIENT,
    VERDICT_OOD,
    QueryDecomposer,
)

Q_MULTI = (
    "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?"
)


class Ev:
    def __init__(self, nid: str) -> None:
        self.node_id = nid
        self.text = "t"
        self.citation = nid


class FixedCoverage:
    """LLM coverage deterministic: trả fixed intents/uncovered."""

    def __init__(self, intents: list[str], uncovered: list[str]) -> None:
        self.intents = intents
        self.uncovered = uncovered

    def complete(self, prompt: str) -> str:
        return json.dumps(
            {"intents": self.intents, "uncovered": self.uncovered,
             "confidence": 0.5}
        )


# ---------------------------------------------------------------------------
# T1: early-stop trigger (intent kẹt: 2 vòng uncovered + không node mới)
# ---------------------------------------------------------------------------

def test_t1_early_stop_stuck_intent():
    calls: list[str] = []

    def runner(sub: str):
        calls.append(sub)
        return []  # "b" không bao giờ kéo node mới -> no-progress

    d = QueryDecomposer(llm=FixedCoverage(["a", "b"], ["b"]))
    _, meta = d.fill(Q_MULTI, [Ev("SEED")], runner, evidence_cap=30)

    # "b" stopped ở vòng 2 (2 vòng no-progress liên tiếp), pending ood_gate
    # (intent "b" không có required node -> ứng viên OOD, chưa kết luận).
    assert "b" in meta.get("stopped", {}), "intent kẹt phải early-stop"
    st = meta["stopped"]["b"]
    assert st["pending"] == "ood_gate"
    assert st["round"] == 2
    # Tiết kiệm: b fill vòng 1 + 2, vòng 3 bị skip -> runner("b") = 2 lần
    # (trước P-B: 3 lần). calls_saved ghi lại.
    assert calls.count("b") == 2, "intent stopped phải bị skip ở vòng sau"
    assert st["calls_saved"] == 1, "phải đếm đúng số LLM call tiết kiệm"
    # per_intent: "b" = stopped/EMPTY; "a" (lành, covered-by-llm) không bị ảnh
    pi = {p["sub_query"]: p for p in meta["per_intent"]}
    assert pi["b"]["status"] == "stopped"
    assert pi["b"]["execution"] == "EMPTY"
    assert pi["b"]["pending"] == "ood_gate"
    assert pi["a"]["status"] != "stopped"


# ---------------------------------------------------------------------------
# T7: có thông tin mới (node mới cho intent) -> streak reset, KHÔNG stop
# ---------------------------------------------------------------------------

def test_t7_new_info_resets_streak():
    b_state = {"n": 0}

    def runner(sub: str):
        if sub == "b":
            b_state["n"] += 1
            # Vòng 1: rỗng (no-progress). Vòng 2: node MỚI -> reset streak.
            return [] if b_state["n"] == 1 else [Ev("B_NEW")]
        return []

    d = QueryDecomposer(llm=FixedCoverage(["a", "b"], ["b"]), max_rounds=2)
    _, meta = d.fill(Q_MULTI, [Ev("SEED")], runner, evidence_cap=30)
    assert "stopped" not in meta, "node mới ở vòng 2 phải reset streak (T7)"
    # B_NEW được merge vào pool (provenance của b).
    pi = {p["sub_query"]: p for p in meta["per_intent"]}
    assert "B_NEW" in pi["b"]["node_ids"]


# ---------------------------------------------------------------------------
# T3: intent HỢP LỆ (có required node thật) nhưng retrieval không tiến triển
#     -> pending "insufficient" (KHÔNG phải ood_gate), verdict = INSUFFICIENT.
# ---------------------------------------------------------------------------

INTENT_D4 = "Điều 4 của 108/2025/QH15 quy định gì về kỳ tính thuế?"


def test_t3_valid_intent_no_progress_is_insufficient():
    def runner(sub: str):
        return []  # không kéo được node D4 (nó có thật trong KG)

    q = (
        "Điều 4 của 108/2025/QH15 quy định gì về kỳ tính thuế, và "
        "mức thuế GTGT bao nhiêu?"
    )
    d = QueryDecomposer(
        llm=FixedCoverage([INTENT_D4], [INTENT_D4]), max_rounds=3
    )
    _, meta = d.fill(q, [Ev("SEED")], runner, evidence_cap=30)
    assert INTENT_D4 in meta.get("stopped", {}), "no-progress phải stop"
    st = meta["stopped"][INTENT_D4]
    assert st["pending"] == "insufficient", (
        "intent có required node (provision có thật) phải chờ "
        "insufficient, KHÔNG phải ood_gate (T3: không gọi OOD oan)"
    )


# ---------------------------------------------------------------------------
# T4: intent stopped KHÔNG chặn gap-fill của intent độc lập khác
# ---------------------------------------------------------------------------

def test_t4_stopped_intent_does_not_block_others():
    state = {"a": 0, "b": 0}

    def runner(sub: str):
        if sub == "a":
            state["a"] += 1
            return [Ev(f"A_NEW{state['a']}")]  # tiến triển mỗi vòng
        if sub == "b":
            state["b"] += 1
            return []  # kẹt
        return []

    d = QueryDecomposer(
        llm=FixedCoverage(["a", "b"], ["a", "b"]), max_rounds=3
    )
    merged, meta = d.fill(Q_MULTI, [Ev("SEED")], runner, evidence_cap=30)
    # b stop ở vòng 2; a (tiến triển) fill ĐÚNG 3 vòng, không bị chặn.
    assert state["b"] == 2, "b: fill 2 vòng rồi stop (skip vòng 3)"
    assert state["a"] == 3, "intent độc lập vẫn gap-fill bình thường"
    assert "b" in meta.get("stopped", {})
    assert "a" not in meta.get("stopped", {})
    # Evidence của a vẫn được merge (pool không bị intent b kéo thụt).
    assert any(e.node_id.startswith("A_NEW") for e in merged)


# ---------------------------------------------------------------------------
# Verdict table (B1 + điều chỉnh #3): LegalRetrieval._intent_verdict
# - COVERED bền (lỗi bước phụ không xóa)
# - OOD CHỈ qua gate F5b (intent_ood), không dùng rerank score riêng
# - FAILED = lỗi THỰC, không đánh đồng "không tìm thấy" (EMPTY)
# ---------------------------------------------------------------------------

def test_intent_verdict_table():
    from agents.LegalRetrieval import LegalRetrievalAgent

    V = LegalRetrievalAgent._intent_verdict
    # (1) COVERED: evidence đủ trong pool -> COVERED (bền).
    assert V({"status": "covered", "execution": "OK"}, [], False) == VERDICT_COVERED
    # (2) Lỗi THỰC khi fill -> FAILED (không phải "không tìm thấy").
    assert V({"status": "open", "execution": "FAILED", "pending": "failed"},
             [], False) == VERDICT_FAILED
    # (3a) stopped + pending ood_gate + gate F5b xác nhận -> OOD.
    assert V({"status": "stopped", "execution": "EMPTY",
              "pending": "ood_gate"}, [], True) == VERDICT_OOD
    # (3b) stopped + pending ood_gate + gate KHÔNG xác nhận -> INSUFFICIENT
    #     (B1: không tự kết luận OOD khi thiếu căn cứ).
    assert V({"status": "stopped", "execution": "EMPTY",
              "pending": "ood_gate"}, [], False) == VERDICT_INSUFFICIENT
    # (3c) stopped + pending insufficient + legal anchor -> INSUFFICIENT,
    #     kể cả khi gate truyền cờ OOD nhầm/không nhất quán.
    assert V({"status": "stopped", "execution": "EMPTY",
              "pending": "insufficient"}, ["108_2025_QH15_D4"], True) == VERDICT_INSUFFICIENT
    # (3d) Pre-planned retrieval rỗng ban đầu gắn pending=insufficient;
    #     khi gate xác nhận OOD cho intent không có anchor, verdict phải là OOD.
    assert V({"status": "stopped", "execution": "EMPTY",
              "pending": "insufficient"}, [], True) == VERDICT_OOD
    # (4) open (chưa stopped, chưa covered) -> INSUFFICIENT.
    assert V({"status": "open", "execution": "OK"}, [], False) == VERDICT_INSUFFICIENT
    # Điều chỉnh #3: intent CÓ anchor (provision được nêu đích danh) thì
    # gate F5b không bao giờ xác nhận OOD -> INSUFFICIENT, không OOD.
    assert V({"status": "stopped", "execution": "EMPTY",
              "pending": "ood_gate"}, ["108_2025_QH15_D4"], False) == VERDICT_INSUFFICIENT


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
