"""Test P7 — QueryDecomposer (gap-filling cho câu nhiều ý).

Không "cắt N ý cố định": N do LLM đọc câu + evidence mà đếm; trần
duy nhất là NGÂN SÁCH (vòng / evidence / thời gian).
Tất cả test dùng LLM giả -> không tốn token.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.QueryDecomposer import (  # noqa: E402
    MAX_EVIDENCE,
    QueryDecomposer,
    coverage_prompt,
    is_multi_intent,
)
from retrieval.QueryUnderstanding import (  # noqa: E402
    extract_anchors, understand, parse_temporal_constraint,
)
from retrieval.AnchorVerifier import classify_query  # noqa: E402


class Ev:
    def __init__(self, nid: str, text: str = "t") -> None:
        self.node_id = nid
        self.text = text
        self.citation = nid


class NoLLM:
    def complete(self, prompt: str) -> str:
        return ""  # parse lỗi -> dừng an toàn


# ---------------------------------------------------------------------------
# Lọc rẻ (không gọi LLM)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "q",
    [
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu, thời hạn nộp khi nào?",
        "Đăng ký thuế Hộ kinh doanh thế nào và mức thuế GTGT là bao nhiêu?",
        "Điều 12 và điều 15 Nghị định 168/2025 quy định gì?",
    ],
)
def test_is_multi_intent_true(q):
    assert is_multi_intent(q) is True


@pytest.mark.parametrize(
    "q",
    [
        "Hộ kinh doanh nộp thuế GTGT theo kỳ nào?",
        "Quy định mới liên quan đến điều 213 là gì?",
        "khoản 3 điều 12 nghị định 168/2025 quy định gì",
    ],
)
def test_is_multi_intent_false(q):
    assert is_multi_intent(q) is False


def test_should_check_requires_llm():
    d = QueryDecomposer(llm=None)
    assert d.should_check("Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?") is False


def test_date_and_document_number_anchors_do_not_overlap_and_g7_routes_exact():
    query = "Trước ngày 23/07/2026, Điều 4 của Nghị định 168/2025/NĐ-CP quy định gì?"
    anchors = extract_anchors(query)
    assert anchors.dates == ["23/7/2026"]
    assert anchors.doc_numbers == ["168/2025/NĐ-CP"]
    assert "07/2026" not in anchors.doc_numbers
    short_number = understand("Điều 2 của 108/2025 quy định gì?")
    assert "108/2025/QH15" in short_number.anchors.doc_numbers
    understood = understand(query)

    class Source:
        def registry(self):
            return {"168/2025/NĐ-CP": {}}

        def node(self, node_id):
            return {"text": ""} if node_id == "168_2025_NĐ_CP_D4" else None

        def children(self, _node_id):
            return []

    decision = classify_query(query, understood, source=Source())
    assert decision.route == "exact"


def test_two_document_gate_calls_planner_but_llm_decides_intent_count():
    class PlannerLLM:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        def complete(self, _prompt):
            self.calls += 1
            return self.response

    compound = (
        "Người nộp thuế theo Điều 2 của 108/2025/QH15 là ai? Và tổ chức, "
        "cá nhân trong nước có thuộc đối tượng áp dụng của 168/2025/NĐ-CP không?"
    )
    two_intent_llm = PlannerLLM(
        '{"intents":[{"sub_query":"Người nộp thuế theo Điều 2 của 108/2025/QH15 là ai?"},'
        '{"sub_query":"Đối tượng áp dụng của 168/2025/NĐ-CP là ai?"}]}'
    )
    plan = QueryDecomposer(llm=two_intent_llm).plan(compound)
    assert two_intent_llm.calls == 1
    assert len(plan["intents"]) == 2
    fallback = QueryDecomposer(llm=None).plan(compound)
    assert fallback["status"] == "single_intent"
    assert len(fallback["intents"]) == 1
    assert fallback["intents"][0]["sub_query"] == compound

    relation_question = "Nghị định 296/2026/NĐ-CP sửa Điều 4 của Nghị định 168/2025/NĐ-CP thế nào?"
    one_intent_llm = PlannerLLM(
        '{"intents":[{"sub_query":"Nghị định 296/2026/NĐ-CP sửa Điều 4 '
        'của Nghị định 168/2025/NĐ-CP thế nào?"}]}'
    )
    one_plan = QueryDecomposer(llm=one_intent_llm).plan(relation_question)
    assert one_intent_llm.calls == 1
    assert len(one_plan["intents"]) == 1


def test_planner_distinguishes_llm_parse_and_invalid_plan_failures():
    question = "Ai nộp thuế theo Luật 108/2025/QH15, và điều kiện đăng ký doanh nghiệp là gì?"

    class Raising:
        def complete(self, _prompt):
            raise TimeoutError("local model timeout")

    class Reply:
        def __init__(self, value): self.value = value
        def complete(self, _prompt): return self.value

    assert QueryDecomposer(llm=Raising()).plan(question)["status"] == "llm_error"
    parsed_error = QueryDecomposer(llm=Reply('<think>{oops}</think> not json')).plan(question)
    assert parsed_error["status"] == "parse_error"
    assert "oops" not in parsed_error["planner_diagnostic"]["raw_output"]
    invalid = QueryDecomposer(llm=Reply('{"intents": []}')).plan(question)
    assert invalid["status"] == "invalid_plan"


def test_planner_rule_fallback_recovers_multi_document_question_on_llm_error():
    class Raising:
        def complete(self, _prompt):
            raise TimeoutError("local model timeout")

    question = (
        "Đối tượng áp dụng của khoản 1 Điều 2 Nghị định 168/2025/NĐ-CP là ai, "
        "và ai là người nộp thuế theo khoản 1 Điều 2 Luật 108/2025/QH15?"
    )
    plan = QueryDecomposer(llm=Raising()).plan(question)
    assert plan["status"] == "planned"
    assert len(plan["intents"]) == 2
    assert "168/2025/NĐ-CP" in plan["intents"][0]["sub_query"]
    assert "108/2025/QH15" in plan["intents"][1]["sub_query"]


def test_planner_rule_fallback_marks_weather_intent_out_of_domain():
    class Raising:
        def complete(self, _prompt):
            raise TimeoutError("local model timeout")

    question = (
        "Thời tiết Hà Nội ngày mai thế nào, "
        "và ai là người nộp thuế theo khoản 1 Điều 2 Luật 108/2025/QH15?"
    )
    plan = QueryDecomposer(llm=Raising()).plan(question)
    assert plan["status"] == "planned"
    assert len(plan["intents"]) == 2
    assert plan["intents"][0]["is_out_of_domain"] is True
    assert plan["intents"][0]["needs_clarification"] is False


def test_planner_rule_fallback_splits_two_date_comparison_on_invalid_output():
    class InvalidReply:
        def complete(self, _prompt):
            return '{"intents":[{"source_clause":"only one"}]}'

    question = (
        "So sánh Điều 4 Nghị định 168/2025/NĐ-CP trước ngày 23/07/2026 "
        "và từ ngày 23/07/2026"
    )
    plan = QueryDecomposer(llm=InvalidReply()).plan(question)
    assert plan["status"] == "planned"
    assert len(plan["intents"]) == 2
    assert [item["temporal_operator"] for item in plan["intents"]] == ["BEFORE", "FROM"]
    assert [item["normalized_target_date"] for item in plan["intents"]] == [
        "2026-07-22", "2026-07-23",
    ]


def test_planner_rule_fallback_splits_semicolon_condition_and_procedure():
    class InvalidReply:
        def complete(self, _prompt):
            return '{"intents":[]}'

    question = (
        "Để được cấp Giấy chứng nhận đăng ký doanh nghiệp cần đáp ứng điều kiện gì; "
        "nếu hồ sơ chưa hợp lệ thì Cơ quan đăng ký kinh doanh xử lý thế nào?"
    )
    plan = QueryDecomposer(llm=InvalidReply()).plan(question)
    assert plan["status"] == "planned"
    assert len(plan["intents"]) == 2
    assert "điều kiện" in plan["intents"][0]["sub_query"].lower()
    assert "hồ sơ chưa hợp lệ" in plan["intents"][1]["sub_query"].lower()


def test_clear_time_comparison_rule_preserves_document_anchor():
    class MustNotCallLLM:
        name = "must-not-call"

        def complete(self, _prompt):
            raise AssertionError("clear time comparison must not call LLM")

    question = (
        "So sánh Điều 4 Nghị định 168/2025/NĐ-CP trước ngày 23/07/2026 "
        "và từ ngày 23/07/2026"
    )
    plan = QueryDecomposer(llm=MustNotCallLLM()).plan(question)
    assert plan["status"] == "planned"
    assert len(plan["intents"]) == 2
    assert all("168/2025/NĐ-CP" in item["sub_query"] for item in plan["intents"])
    assert all("Điều 4" in item["sub_query"] for item in plan["intents"])


def test_planner_removes_cross_intent_document_anchor_and_asks_missing_tax_facts():
    question = (
        "Ai là người nộp thuế theo khoản 1 Điều 2 Luật 108/2025/QH15, "
        "và khoản đóng góp từ thiện của tôi có được giảm thuế không?"
    )
    payload = '''{"intents":[
      {"source_clause":"Ai là người nộp thuế theo khoản 1 Điều 2 Luật 108/2025/QH15",
       "sub_query":"Ai là người nộp thuế theo khoản 1 Điều 2 Luật 108/2025/QH15"},
      {"source_clause":"khoản đóng góp từ thiện của tôi có được giảm thuế không",
       "sub_query":"Khoản đóng góp từ thiện của tôi có được giảm thuế không theo Luật 108/2025/QH15"}
    ]}'''
    plan = QueryDecomposer(llm=type("L", (), {"complete": lambda self, _p: payload})()).plan(question)
    first, second = plan["intents"]
    assert first["source_clause"] in question
    assert "108/2025/QH15" not in second["sub_query"]
    assert second["needs_clarification"] is True
    assert all(term in second["clarification"] for term in ("sắc thuế", "đối tượng nộp thuế", "loại khoản đóng góp"))


@pytest.mark.parametrize(("phrase", "operator", "target"), [
    ("trước ngày 23/07/2026", "BEFORE", "2026-07-22"),
    ("sau ngày 22/07/2026", "AFTER", "2026-07-23"),
    ("từ ngày 23/07/2026", "FROM", "2026-07-23"),
    ("kể từ ngày 23/07/2026", "FROM", "2026-07-23"),
    ("đến hết ngày 22/07/2026", "UNTIL", "2026-07-22"),
    ("đến trước ngày 23/07/2026", "UNTIL", "2026-07-22"),
    ("tại ngày 23/07/2026", "AT", "2026-07-23"),
])
def test_temporal_constraint_normalizes_operator_and_target(phrase, operator, target):
    parsed = parse_temporal_constraint(f"Điều 4 của Nghị định 168/2025/NĐ-CP {phrase}")
    assert parsed["operator"] == operator
    assert parsed["normalized_target_date"] == target
    assert parsed["raw_date"]


def test_simple_two_tax_phrases_do_not_trigger_planner():
    class MustNotCallLLM:
        def complete(self, _prompt):
            raise AssertionError("single-question gate unexpectedly called LLM")

    question = "Thuế TNDN và thuế suất là gì?"
    planner = QueryDecomposer(llm=MustNotCallLLM())
    assert planner.plan(question)["status"] == "single_intent"


def test_pre_retrieval_plan_keeps_per_intent_clarification_independent():
    class PlannerLLM:
        def set_purpose(self, _purpose):
            pass

        def complete(self, _prompt):
            return (
                '{"intents": ['
                '{"sub_query":"Điều 4 của 168/2025/NĐ-CP quy định gì?",'
                '"needs_clarification":false,"clarification":""},'
                '{"sub_query":"Điều 22 của văn bản chưa nêu",'
                '"needs_clarification":true,"clarification":"Điều 22 thuộc văn bản nào?"}'
                ']}'
            )

    d = QueryDecomposer(llm=PlannerLLM())
    plan = d.plan(
        "Điều 4 của 168/2025/NĐ-CP quy định gì và Điều 22 của văn bản nào?"
    )
    assert plan["status"] == "planned"
    assert plan["intents"][0]["needs_clarification"] is False
    assert plan["intents"][1]["needs_clarification"] is True
    assert plan["intents"][1]["clarification"] == "Điều 22 thuộc văn bản nào?"


# ---------------------------------------------------------------------------
# Gap-fill
# ---------------------------------------------------------------------------

def test_single_intent_noop():
    """Câu đơn -> no-op, không gọi LLM, không gọi runner."""
    calls: list[str] = []

    def runner(sub: str):
        calls.append(sub)
        return [Ev("NEW")]

    d = QueryDecomposer(llm=NoLLM())
    merged, meta = d.fill(
        "Hộ kinh doanh nộp thuế GTGT theo kỳ nào?", [Ev("A")], runner
    )
    assert [e.node_id for e in merged] == ["A"]
    assert meta["filled"] == 0
    assert meta["rounds"] == 0
    # F5b: câu đơn -> per_intent có đúng 1 intent = câu gốc.
    assert meta["per_intent"] == [
        {"sub_query": "Hộ kinh doanh nộp thuế GTGT theo kỳ nào?", "node_ids": ["A"]}
    ]
    assert calls == []


class CoverThenFull:
    """Lần 1 nói thiếu ý 'GTGT', lần 2 nói đủ."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"intents": ["a", "b"], "uncovered": ["mức GTGT"], "confidence": 0.6}'
        return '{"intents": ["a", "b"], "uncovered": [], "confidence": 0.9}'


def test_fill_uncovered_intent():
    calls: list[str] = []

    def runner(sub: str):
        calls.append(sub)
        return [Ev("GTGT_48_2024_D5_K1")]

    d = QueryDecomposer(llm=CoverThenFull())
    merged, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu, thời hạn nộp khi nào?",
        [Ev("QLT_108_2025_D5_K2")],
        runner,
    )
    assert [e.node_id for e in merged] == ["QLT_108_2025_D5_K2", "GTGT_48_2024_D5_K1"]
    assert meta["filled"] == 1
    assert meta["stop_reason"] == "covered"
    assert "mức GTGT" in calls


def test_no_dup_evidence_when_runner_returns_same():
    def runner(sub: str):
        return [Ev("QLT_108_2025_D5_K2")]  # trùng evidence cũ

    d = QueryDecomposer(llm=CoverThenFull())
    merged, _ = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?",
        [Ev("QLT_108_2025_D5_K2")],
        runner,
    )
    ids = [e.node_id for e in merged]
    assert ids.count("QLT_108_2025_D5_K2") == 1


def test_llm_exception_stops_safely():
    class Boom:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("ollama down")

    d = QueryDecomposer(llm=Boom())
    merged, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?",
        [Ev("A")],
        lambda sub: [Ev("B")],
    )
    assert [e.node_id for e in merged] == ["A"]
    assert meta["stop_reason"] == "coverage_unparsed"


def test_evidence_cap_respected():
    class AlwaysUncovered:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            return '{"intents": ["a"], "uncovered": ["x"], "confidence": 0.5}'

    def runner(sub: str):
        return [Ev(f"NEW{len(_seen)}"), ]

    _seen: list[str] = []

    def runner2(sub: str):
        _seen.append(sub)
        return [Ev(f"N{len(_seen)}")]

    d = QueryDecomposer(llm=AlwaysUncovered(), max_rounds=3)
    # Trần nhỏ = 3 để test nhanh.
    merged, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?",
        [Ev("A")],
        runner2,
        evidence_cap=3,
    )
    assert len(merged) <= 3
    assert meta["stop_reason"] in ("budget_evidence", "max_rounds")
    assert len(merged) == 3


def test_budget_max_rounds():
    class AlwaysUncovered:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            return '{"uncovered": ["x"]}'

    seen: list[str] = []

    def runner(sub: str):
        seen.append(sub)
        return [Ev(f"N{len(seen)}")]

    d = QueryDecomposer(llm=AlwaysUncovered(), max_rounds=2)
    _, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?",
        [Ev("A")],
        runner,
        evidence_cap=100,
    )
    assert meta["rounds"] == 2
    assert meta["stop_reason"] == "max_rounds"


def test_max_evidence_constant_sane():
    assert 10 <= MAX_EVIDENCE <= 40


def test_clarified_intent_rewrite_keeps_original_and_rejects_invented_anchors():
    class RewriteLLM:
        name = "fixture"
        def __init__(self, response): self.response = response
        def complete(self, _prompt): return self.response

    source = "Tra cứu Điều 2 của Luật 108/2025/QH15"
    good = QueryDecomposer(llm=RewriteLLM(
        "Theo Luật 108/2025/QH15, Điều 2 quy định gì về người nộp thuế?"
    ))
    query, audit = good.rewrite_clarified_intent(
        original_question="Câu hỏi nhiều ý ban đầu",
        original_sub_query=source,
        clarification="Bạn hỏi nhóm nào?",
        user_answer="Người nộp thuế là ai",
    )
    assert query == "Theo Luật 108/2025/QH15, Điều 2 quy định gì về người nộp thuế?"
    assert audit["status"] == "ANCHORS_VALIDATED"
    assert audit["original_sub_query"] == source
    assert audit["rewritten_query"] == query

    hallucinating = QueryDecomposer(llm=RewriteLLM(
        "Theo Luật 999/2030/QH16, Điều 99 quy định gì?"
    ))
    rejected_query, rejected = hallucinating.rewrite_clarified_intent(
        original_question="Câu hỏi nhiều ý ban đầu",
        original_sub_query=source,
        clarification="Bạn hỏi nhóm nào?",
        user_answer="Người nộp thuế là ai",
    )
    assert rejected_query == ""
    assert rejected["status"] == "REJECTED"
    assert rejected["unsupported_anchors"]
    assert rejected["missing_anchors"]


def test_clarification_answers_require_numbering_when_multiple_intents():
    from agents.QueryDecomposer import (
        format_intent_clarifications,
        parse_intent_clarification_answers,
    )

    ids = ["intent-a", "intent-b"]
    assert format_intent_clarifications([
        {"needs_clarification": True, "clarification": "A cần văn bản nào?"},
        {"needs_clarification": True, "clarification": "B cần thời điểm nào?"},
    ]) == "1) A cần văn bản nào?\n2) B cần thời điểm nào?"
    assert parse_intent_clarification_answers(ids, "Luật 59/2020") == {}
    assert parse_intent_clarification_answers(
        ids, "1) Luật 59/2020\n2) ngày 01/01/2026"
    ) == {"intent-a": "Luật 59/2020", "intent-b": "ngày 01/01/2026"}


def test_coverage_prompt_contains_question_and_evidence():
    p = coverage_prompt("đăng ký thuế và mức GTGT?", ["A | text"])
    assert "đăng ký thuế và mức GTGT?" in p
    assert "A | text" in p
    assert "uncovered" in p


# ---------------------------------------------------------------------------
# F1: coverage theo NODE (không theo chữ) + F5b: per_intent
# ---------------------------------------------------------------------------

class LLMClaimsCovered:
    """Bug S6 thật: LLM nhìn chữ 'Điều 4' trong provision khác ->
    khai 'covered' (uncovered rỗng) dù node D4 KHÔNG trong pool."""

    def complete(self, prompt: str) -> str:
        return (
            '{"intents": ['
            '"Điều 2 của 108/2025/QH15 quy định gì?", '
            '"Điều 4 của 108/2025/QH15 quy định gì về kỳ tính thuế?"'
            "], \"uncovered\": [], \"confidence\": 0.9}"
        )


def test_f1_node_coverage_forces_missing_intent():
    """D4 không trong pool -> dù LLM nói 'covered', rewriter/runner
    vẫn chạy cho ý Điều 4 (chấm theo node, không theo chữ)."""
    calls: list[str] = []

    def runner(sub: str):
        calls.append(sub)
        return [Ev("108_2025_QH15_D4_K8")]  # D4 (node con) vào pool

    # Pool ban đầu: có D2, D9_K5 (text D9_K5 chứa chữ 'khoản 4 Điều này'
    # — đúng bẫy chữ của S6), KHÔNG có D4.
    pool = [Ev("108_2025_QH15_D2"), Ev("108_2025_QH15_D9_K5")]
    d = QueryDecomposer(llm=LLMClaimsCovered())
    merged, meta = d.fill(
        "Điều 2 và Điều 4 của Luật Quản lý thuế 108/2025/QH15 quy định gì?",
        pool,
        runner,
    )
    ids = [e.node_id for e in merged]
    # D4_K8 (node con của D4) đã được kéo về -> pool đủ cả 2 ý.
    assert "108_2025_QH15_D4_K8" in ids
    assert meta["stop_reason"] == "covered"
    assert meta["filled"] == 1
    # Runner được gọi cho ý ĐIỀU 4 (ý thiếu), không cho ý Điều 2 (đã có).
    assert any("Điều 4" in c or "điều 4" in c.lower() for c in calls)
    assert not any("Điều 2" in c for c in calls)


def test_f1_per_intent_structure():
    """F5b: meta['per_intent'] cho mỗi ý = {sub_query, node_ids},
    node_ids có seed pool + evidence ý đó tự sinh (provenance)."""

    def runner(sub: str):
        return [Ev("108_2025_QH15_D4_K8")]

    pool = [Ev("108_2025_QH15_D2")]
    d = QueryDecomposer(llm=LLMClaimsCovered())
    _, meta = d.fill(
        "Điều 2 và Điều 4 của Luật Quản lý thuế 108/2025/QH15 quy định gì?",
        pool,
        runner,
    )
    per = meta["per_intent"]
    assert len(per) == 2
    by_text = {p["sub_query"][:6]: p for p in per}
    # Ý Điều 4: seed (D2) + tự sinh (D4_K8).
    i4 = [p for p in per if "Điều 4" in p["sub_query"]][0]
    assert "108_2025_QH15_D4_K8" in i4["node_ids"]
    # Ý Điều 2: chỉ seed (không tự sinh gì mới).
    i2 = [p for p in per if "Điều 2" in p["sub_query"]][0]
    assert "108_2025_QH15_D4_K8" not in i2["node_ids"]
    assert "108_2025_QH15_D2" in i2["node_ids"]


class NoAnchorIntent:
    """Ý không có vb + Điều -> fallback: tin LLM về độ phủ (đường cũ)."""

    def complete(self, prompt: str) -> str:
        return (
            '{"intents": ["mức thuế GTGT của hộ kinh doanh"], '
            '"uncovered": ["mức thuế GTGT của hộ kinh doanh"], '
            '"confidence": 0.5}'
        )


def test_f1_no_anchor_falls_back_to_llm():
    def runner(sub: str):
        return [Ev("48_2024_QH15_D5_K1")]

    d = QueryDecomposer(llm=NoAnchorIntent())
    merged, meta = d.fill(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?",
        [Ev("108_2025_QH15_D5_K2")],
        runner,
    )
    # LLM nói thiếu -> runner chạy (đường cũ, không thay đổi).
    assert [e.node_id for e in merged] == [
        "108_2025_QH15_D5_K2",
        "48_2024_QH15_D5_K1",
    ]
    assert meta["filled"] == 1


def test_f1_both_intents_covered_no_runner():
    """Cả 2 node điều đều trong pool -> không chạy runner (no-op phủ)."""
    calls: list[str] = []

    def runner(sub: str):
        calls.append(sub)
        return [Ev("X")]

    pool = [Ev("108_2025_QH15_D2"), Ev("108_2025_QH15_D4_K8")]
    d = QueryDecomposer(llm=LLMClaimsCovered())
    _, meta = d.fill(
        "Điều 2 và Điều 4 của Luật Quản lý thuế 108/2025/QH15 quy định gì?",
        pool,
        runner,
    )
    assert calls == []  # không cần lấp gì
    assert meta["stop_reason"] == "covered"


def test_planner_normalizes_out_of_domain_intent_without_clarification():
    import json
    question = (
        "Th\u1eddi ti\u1ebft H\u00e0 N\u1ed9i ng\u00e0y mai th\u1ebf n\u00e0o, v\u00e0 ai l\u00e0 ng\u01b0\u1eddi n\u1ed9p thu\u1ebf "
        "theo kho\u1ea3n 1 \u0110i\u1ec1u 2 Lu\u1eadt 108/2025/QH15?"
    )
    payload = json.dumps({"intents": [
        {"source_clause": "Th\u1eddi ti\u1ebft H\u00e0 N\u1ed9i ng\u00e0y mai th\u1ebf n\u00e0o",
         "sub_query": "Th\u1eddi ti\u1ebft H\u00e0 N\u1ed9i ng\u00e0y mai th\u1ebf n\u00e0o",
         "needs_clarification": True,
         "clarification": "Ngo\u00e0i ph\u1ea1m vi ph\u00e1p lu\u1eadt, kh\u00f4ng c\u1ea7n l\u00e0m r\u00f5"},
        {"source_clause": "ai l\u00e0 ng\u01b0\u1eddi n\u1ed9p thu\u1ebf theo kho\u1ea3n 1 \u0110i\u1ec1u 2 Lu\u1eadt 108/2025/QH15",
         "sub_query": "ai l\u00e0 ng\u01b0\u1eddi n\u1ed9p thu\u1ebf theo kho\u1ea3n 1 \u0110i\u1ec1u 2 Lu\u1eadt 108/2025/QH15"}
    ]}, ensure_ascii=False)
    planner = QueryDecomposer(llm=type("L", (), {"complete": lambda self, _prompt: payload})())
    out_of_domain, legal = planner.plan(question)["intents"]
    assert out_of_domain["is_out_of_domain"] is True
    assert out_of_domain["needs_clarification"] is False
    assert out_of_domain["clarification"] == ""
    assert legal["is_out_of_domain"] is False


def test_planner_keeps_two_temporal_constraints_separate_and_current_is_preserved():
    import json
    question = (
        "So s\u00e1nh \u0110i\u1ec1u 4 Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP tr\u01b0\u1edbc ng\u00e0y 23/07/2026 "
        "v\u00e0 t\u1eeb ng\u00e0y 23/07/2026"
    )
    clauses = [
        "\u0110i\u1ec1u 4 Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP tr\u01b0\u1edbc ng\u00e0y 23/07/2026",
        "\u0110i\u1ec1u 4 Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP t\u1eeb ng\u00e0y 23/07/2026",
    ]
    payload = json.dumps({"intents": [
        {"source_clause": c, "sub_query": c} for c in clauses
    ]}, ensure_ascii=False)
    plan = QueryDecomposer(llm=type("L", (), {"complete": lambda self, _prompt: payload})()).plan(question)
    assert [i["temporal_constraint"]["operator"] for i in plan["intents"]] == ["BEFORE", "FROM"]
    assert [i["normalized_target_date"] for i in plan["intents"]] == ["2026-07-22", "2026-07-23"]
    assert all(i["temporal_expression"] for i in plan["intents"])
    current = parse_temporal_constraint("\u0110i\u1ec1u 4 Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP c\u00f2n \u00e1p d\u1ee5ng hi\u1ec7n nay kh\u00f4ng?")
    assert current["operator"] == "CURRENT"
    assert current["normalized_target_date"] is None
    assert current["temporal_expression"] == "hi\u1ec7n nay"


def test_planner_splits_two_date_comparison_even_if_llm_returns_one_intent():
    import json
    question = (
        "So s\u00e1nh \u0110i\u1ec1u 4 Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP tr\u01b0\u1edbc ng\u00e0y 23/07/2026 "
        "v\u00e0 t\u1eeb ng\u00e0y 23/07/2026"
    )
    payload = json.dumps({"intents": [{"source_clause": question, "sub_query": question}]}, ensure_ascii=False)
    plan = QueryDecomposer(llm=type("L", (), {"complete": lambda self, _prompt: payload})()).plan(question)
    assert len(plan["intents"]) == 2
    assert [item["temporal_operator"] for item in plan["intents"]] == ["BEFORE", "FROM"]
    assert [item["normalized_target_date"] for item in plan["intents"]] == ["2026-07-22", "2026-07-23"]
