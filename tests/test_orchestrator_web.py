"""Test Orchestrator — nhánh web + hợp nhất xung đột (P7).

Dùng STUB cho Web agent và LLM -> không cần Tavily/Gemini thật, không
đụng network. Verify đúng hành vi đã chốt:
  - gọi web khi fast-track không đủ (quyết định #12)
  - không gọi web khi fast-track đủ
  - nhánh web mới hơn -> ưu tiên web + ghi chú "DB nội bộ cần cập nhật"
  - web unavailable -> vẫn trả được câu trả lời nội bộ (không chết)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "utils",
    PROJECT_ROOT / "src" / "retrieval",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.LegalRetrieval import LegalAnswer  # noqa: E402
from agents.Orchestrator import Orchestrator  # noqa: E402
from agents.WebSearchAgent import WebResult, WebSource  # noqa: E402


class StubLLM:
    """LLM giả — trả về đúng chuỗi đã cấu hình."""

    def __init__(self, reply: str = "Câu trả lời từ LLM") -> None:
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.reply


class StubLegal:
    """Legal Agent giả — trả về cố định."""

    def __init__(self, answer: LegalAnswer) -> None:
        self._answer = answer
        self.llm = StubLLM()

    def answer(self, question: str, understanding: object = None, **kwargs) -> LegalAnswer:
        self.last_call = (question, kwargs)
        return self._answer


class StubWeb:
    """Web Agent giả — trả về danh sách nguồn cố định."""

    def __init__(self, result: WebResult) -> None:
        self._result = result
        self.llm = StubLLM()
        self.calls: list[str] = []

    def run(self, question: str) -> WebResult:
        self.calls.append(question)
        return self._result

    def search(self, question: str) -> WebResult:
        return self._result


def _legal_answer(route: str = "vector", evidence: bool = True) -> LegalAnswer:
    return LegalAnswer(
        answer="Câu trả lời nội bộ",
        route=route,
        evidence=(
            [{
                "node_id": "168_2025_NĐ_CP_D100_K1",
                "citation": "Điều 100 Khoản 1 (168/2025/NĐ-CP)",
                "text": "Nội dung Điều 100 Khoản 1 của Nghị định 168/2025/NĐ-CP.",
                "kind": "clause",
                "node_type": "clause",
                "metadata": {},
            }]
            if evidence else []
        ),
    )


def _web_result(so_hieu: str = "", date: str = "", available: bool = True):
    sources = []
    if so_hieu:
        sources.append(WebSource(
            title=f"Văn bản {so_hieu}",
            url=f"https://vbpl.vn/{so_hieu}",
            content=f"Văn bản {so_hieu} ban hành ngày {date}. Nội dung web.",
            date=date,
            authority_tier="official",
        ))
    return WebResult(sources=sources, summary="Tóm tắt web", available=available)


def _orchestrator(legal, web, resolver_llm=None):
    from agents.ConflictResolver import ConflictResolver

    orch = Orchestrator(legal=legal, web=web, resolver=ConflictResolver(llm=resolver_llm))
    return orch


# ---------------------------------------------------------------------------
# Quy tắc gọi web (quyết định #12)
# ---------------------------------------------------------------------------
def test_web_called_when_no_evidence():
    legal = StubLegal(_legal_answer(evidence=False))
    web = StubWeb(_web_result())
    orch = _orchestrator(legal, web)
    orch.run("câu hỏi")
    assert len(web.calls) == 1


def test_web_not_called_on_fast_track():
    legal = StubLegal(_legal_answer(route="fast_track"))
    web = StubWeb(_web_result())
    orch = _orchestrator(legal, web)
    orch.run("câu hỏi")
    assert web.calls == []


def test_web_unavailable_still_answers_internal():
    """Web không khả dụng -> KHÔNG chết, vẫn trả câu trả lời nội bộ."""
    legal = StubLegal(_legal_answer(evidence=False))
    web = StubWeb(WebResult(available=False, reason="không có API key"))
    orch = _orchestrator(legal, web)
    out = orch.run("câu hỏi")
    assert out["answer"] == "Câu trả lời nội bộ"
    assert out["web_available"] is False


def test_execution_error_stops_before_web_fallback():
    from types import SimpleNamespace
    from agents.Orchestrator import OrchestratorState
    state = OrchestratorState(
        understanding=SimpleNamespace(intent="tra_cuu"),
        route="error",
        internal_answer=LegalAnswer(
            answer="Planner unavailable", metadata={"execution": "ERROR"}
        ),
    )
    assert Orchestrator._needs_web(state) is False


def test_orchestrator_forwards_planner_state_to_fixed_intent_without_refining():
    class Refiner:
        called = False
        def refine(self, *_a, **_k):
            self.called = True
            raise AssertionError("targeted planner clarification must bypass general refiner")

    state = {"original_question": "ý A và ý B", "pending_intent_id": "intent-2",
             "intents": [{"intent_id": "intent-2", "sub_query": "ý B",
                          "needs_clarification": True, "clarification": "Làm rõ B?"}]}
    legal = StubLegal(_legal_answer())
    refiner = Refiner()
    orch = Orchestrator(legal=legal, use_web=False, refiner=refiner)
    class Graph:
        def invoke(self, state): return orch._run_internal(state)
    orch._lg_graph = Graph()
    orch.run("câu trả lời của người dùng", user_answer="văn bản B",
             planner_state=state)
    question, kwargs = legal.last_call
    assert question == "ý A và ý B"
    assert kwargs["planner_state"]["pending_intent_id"] == "intent-2"
    assert kwargs["user_answer"] == "văn bản B"
    assert refiner.called is False


def test_new_multi_intent_query_bypasses_global_refiner(monkeypatch):
    import agents.Orchestrator as orchestrator_module
    monkeypatch.setattr(orchestrator_module, "is_multi_intent", lambda _q: True)

    class Refiner:
        def refine(self, *_a, **_k):
            raise AssertionError("multi-intent question must go to Planner first")

    legal = StubLegal(_legal_answer())
    orch = Orchestrator(legal=legal, use_web=False, refiner=Refiner())
    class Graph:
        def invoke(self, state): return orch._run_internal(state)
    orch._lg_graph = Graph()
    orch.run("câu hỏi gồm hai ý độc lập đủ dài để Planner phân tách")
    assert legal.last_call[0] == "câu hỏi gồm hai ý độc lập đủ dài để Planner phân tách"


def test_refiner_original_and_rewritten_query_are_returned_for_audit():
    from types import SimpleNamespace
    class Refiner:
        def refine(self, question, **_kwargs):
            return SimpleNamespace(
                question="Điều 2 Luật 108/2025/QH15 quy định gì?",
                follow_up="", kind="cleared", source="llm",
                needs_clarification=False, choices=[],
            )

    legal = StubLegal(_legal_answer())
    orch = Orchestrator(legal=legal, use_web=False, refiner=Refiner())
    class Graph:
        def invoke(self, state): return orch._run_internal(state)
    orch._lg_graph = Graph()
    original = "Điều 2 của Luật 108/2025/QH15"
    out = orch.run(original)
    assert out["query_rewrite"]["original_question"] == original
    assert out["query_rewrite"]["rewritten_question"] == "Điều 2 Luật 108/2025/QH15 quy định gì?"
    assert out["query_rewrite"]["status"] == "ANCHORS_VALIDATED"


def test_refiner_rewrite_with_invented_anchor_stops_before_retrieval():
    from types import SimpleNamespace
    class Refiner:
        def refine(self, question, **_kwargs):
            return SimpleNamespace(
                question="Điều 99 của Luật 999/2030/QH16 quy định gì?",
                follow_up="", kind="cleared", source="llm",
                needs_clarification=False, choices=[],
            )

    legal = StubLegal(_legal_answer())
    orch = Orchestrator(legal=legal, use_web=False, refiner=Refiner())
    original = "Tra cứu quy định về người nộp thuế"
    out = orch.run(original)
    assert out["route"] == "refine"
    assert out["evidence"] == []
    assert out["query_rewrite"]["status"] == "REJECTED"


def test_orchestrator_discards_invalid_planner_state_without_answer():
    class Refiner:
        def refine(self, *_a, **_k): return None

    stale = {"original_question": "câu hỏi cũ",
             "pending_intent_id": "intent-missing",
             "intents": [{"intent_id": "intent-1", "sub_query": "ý cũ",
                          "needs_clarification": True}]}
    legal = StubLegal(_legal_answer())
    orch = Orchestrator(legal=legal, use_web=False, refiner=Refiner())
    class Graph:
        def invoke(self, state): return orch._run_internal(state)
    orch._lg_graph = Graph()
    orch.run("câu hỏi mới", planner_state=stale, user_answer="")
    question, kwargs = legal.last_call
    assert question == "câu hỏi mới"
    assert "planner_state" not in kwargs
    assert "user_answer" not in kwargs


def test_planner_runs_before_per_intent_refiner_and_plan_is_forwarded():
    from types import SimpleNamespace

    events = []
    class PlannedLegal(StubLegal):
        def plan_question(self, question):
            events.append("planner")
            return {"planned": True, "status": "planned", "intents": [
                {"intent_id": "i1", "sub_query": "A"},
                {"intent_id": "i2", "sub_query": "B"},
            ]}
        def answer(self, question, understanding=None, **kwargs):
            events.append("retrieval")
            self.last_call = (question, kwargs)
            return self._answer

    class Refiner:
        def refine(self, question, **_kwargs):
            events.append(f"refine:{question}")
            return SimpleNamespace(question=question, follow_up="", kind="cleared",
                                   source="fixture", needs_clarification=False, choices=[])

    legal = PlannedLegal(_legal_answer())
    orch = Orchestrator(legal=legal, use_web=False, refiner=Refiner())
    class Graph:
        def invoke(self, state): return orch._run_internal(state)
    orch._lg_graph = Graph()
    orch.run("A và B")
    assert events == ["planner", "refine:A", "refine:B", "retrieval"]
    assert legal.last_call[1]["planner_plan"]["intents"][1]["sub_query"] == "B"


def test_web_gate_runs_only_for_empty_intent_and_never_for_error():
    from agents.Orchestrator import OrchestratorState
    legal = StubLegal(LegalAnswer(
        answer="partial", route="vector", evidence=[], metadata={
            "per_intent": [
                {"intent_id": "i1", "sub_query": "question A", "execution": "SUCCESS",
                 "pending": None, "node_ids": ["n1"]},
                {"intent_id": "i2", "sub_query": "question B", "execution": "EMPTY",
                 "pending": "insufficient", "node_ids": []},
            ]
        }
    ))
    web = StubWeb(_web_result())
    orch = _orchestrator(legal, web)
    state = OrchestratorState(question="A và B", internal_answer=legal._answer,
                              route="vector")
    assert Orchestrator._needs_web(state) is True
    orch._run_web(state)
    assert web.calls == ["question B"]
    assert state.web_results[0][0] == "i2"

    legal._answer.metadata["per_intent"][1]["execution"] = "ERROR"
    legal._answer.metadata["per_intent"][1]["pending"] = "error"
    assert Orchestrator._needs_web(state) is False

    legal._answer.metadata["per_intent"][1].update({
        "sub_query": "Văn bản B bổ sung khoản 7 vào văn bản A?",
        "execution": "SUCCESS", "pending": None,
    })
    assert Orchestrator._needs_web(state) is True


def test_conflict_resolver_receives_evidence_and_web_sources_scoped_by_intent():
    from agents.Orchestrator import OrchestratorState

    class ScopedResolver:
        def __init__(self):
            self.payloads = None
        def resolve_per_intent(self, payloads):
            self.payloads = payloads
            return {
                "answer": "Ý 1 đã đối chiếu.\nÝ 2 đã đối chiếu.",
                "conflict_report": {"resolution_status": "per_intent", "intents": []},
                "llm_error": False,
            }

    ids = ["node_a", "node_b"]
    evidence = [
        {"node_id": ids[0], "citation": "Luật A Điều 1", "text": "Nội dung A",
         "metadata": {"so_hieu": "1/2024/QH15", "dieu": "1"}},
        {"node_id": ids[1], "citation": "Luật B Điều 2", "text": "Nội dung B",
         "metadata": {"so_hieu": "2/2025/QH15", "dieu": "2"}},
    ]
    answer = LegalAnswer(
        answer="Câu trả lời nội bộ gộp.", route="vector", evidence=evidence,
        primary_evidence=evidence, metadata={"per_intent": [
            {"intent_id": "i1", "sub_query": "Ý A", "node_ids": [ids[0]],
             "execution": "SUCCESS", "evidence_status": "covered"},
            {"intent_id": "i2", "sub_query": "Ý B", "node_ids": [ids[1]],
             "execution": "EMPTY", "pending": "insufficient", "evidence_status": "empty"},
        ]},
    )
    resolver = ScopedResolver()
    orch = Orchestrator(legal=StubLegal(answer), web=StubWeb(_web_result()), resolver=resolver)
    orch._registry_dates = lambda: {}
    state = OrchestratorState(
        question="Ý A và Ý B", internal_answer=answer, route="vector",
        web_results=[("i2", "Ý B", _web_result("2/2025/QH15", "2025-01-01"))],
    )
    orch._synthesize(state)

    assert [p["intent_id"] for p in resolver.payloads] == ["i1", "i2"]
    assert "Nội dung A" in resolver.payloads[0]["internal_block"]
    assert "Nội dung B" not in resolver.payloads[0]["internal_block"]
    assert resolver.payloads[0]["web_sources"] == []
    assert "Nội dung B" in resolver.payloads[1]["internal_block"]
    assert "Nội dung A" not in resolver.payloads[1]["internal_block"]
    assert all(s.intent_id == "i2" for s in resolver.payloads[1]["web_sources"])
    assert state.final_answer.startswith("Ý 1")


def test_official_web_extraction_proof_reaches_per_intent_resolver():
    from agents.ConflictResolver import ConflictResolver
    from agents.Orchestrator import OrchestratorState

    llm = StubLLM("Đoạn trích chính thức xác định phạm vi sửa đổi.")
    proof = {
        "verified": True, "relation_type": "AMENDS",
        "from_document": "2/2025/NĐ-CP", "to_document": "1/2024/NĐ-CP",
        "effective_date": "2025-01-01", "relationship_checked": True,
        "affected_provisions": [{
            "document_number": "1/2024/NĐ-CP", "article": "4", "clause": "7",
            "scope": "clause_added", "quote": "Bổ sung khoản 7 vào sau khoản 6 Điều 4.",
        }],
        "source": {
            "official": True, "url": "https://vbpl.vn/doc/2-2025",
            "citation": "Điều 2",
            "excerpt": "Điều 2: Bổ sung khoản 7 vào sau khoản 6 Điều 4 của Nghị định 1/2024/NĐ-CP.",
            "effective_date_excerpt": "Nghị định này có hiệu lực từ ngày 1 tháng 1 năm 2025.",
            "source_sha256": "b" * 64,
        },
    }
    web_result = _web_result("2/2025/NĐ-CP", "2025-01-01")
    web_result.sources[0].legal_extraction = {"relations": [{
        "status": "verified", "verified_relationship": proof,
    }]}
    answer = LegalAnswer(
        answer="Câu trả lời nội bộ.", route="vector",
        evidence=[{"node_id": "n1", "citation": "A Điều 4", "text": "Evidence nội bộ",
                   "metadata": {"so_hieu": "1/2024/NĐ-CP", "dieu": "4"}}],
        primary_evidence=[{"node_id": "n1", "citation": "A Điều 4", "text": "Evidence nội bộ",
                           "metadata": {"so_hieu": "1/2024/NĐ-CP", "dieu": "4"}}],
        metadata={"per_intent": [{
            "intent_id": "i1", "sub_query": "Nghị định 2/2025 bổ sung khoản nào của Nghị định 1/2024?",
            "node_ids": ["n1"], "execution": "SUCCESS", "evidence_status": "covered",
        }]},
    )
    web = StubWeb(web_result)
    orch = Orchestrator(legal=StubLegal(answer), web=web,
                        resolver=ConflictResolver(llm=llm))
    orch._registry_dates = lambda: {}
    state = OrchestratorState(question=answer.metadata["per_intent"][0]["sub_query"],
                              internal_answer=answer, route="vector")
    orch._run_web(state)
    orch._synthesize(state)

    assert llm.calls
    assert "Đoạn trích chính thức" in state.final_answer
    assert state.conflict_report["resolution_status"] == "per_intent"


# ---------------------------------------------------------------------------
# Hợp nhất + xung đột
# ---------------------------------------------------------------------------
def test_web_newer_does_not_resolve_status_by_issue_date():
    legal = StubLegal(_legal_answer())
    # 296/2026/NĐ-CP (23/07/2026) là văn bản mới nhất trong DB nội bộ.
    # Web tìm được 310/2026/NĐ-CP (15/08/2026) -> mới hơn.
    web = StubWeb(_web_result("310/2026/NĐ-CP", "2026-08-15"))
    orch = _orchestrator(legal, web, resolver_llm=StubLLM("Câu trả lời hợp nhất"))
    out = orch.run("Điều 100 Nghị định 168/2025 còn hiệu lực không?")
    report = out["conflict_report"]
    assert report is not None
    assert report["preferred_origin"] is None
    assert report["resolution_status"] == "unresolved"
    assert "Chưa đủ căn cứ" in out["answer"]


def test_web_older_does_not_resolve_status_by_issue_date():
    """Web trả về văn bản CŨ hơn DB -> không xung đột, ưu tiên nội bộ.

    168/2025/NĐ-CP ban hành 30/06/2025; web trả 59/2020/QH14 (17/06/2020)
    -> cũ hơn -> giữ nội bộ.
    """
    legal = StubLegal(_legal_answer())
    web = StubWeb(_web_result("59/2020/QH14", "2020-06-17"))
    orch = _orchestrator(legal, web, resolver_llm=StubLLM("Câu trả lời hợp nhất"))
    out = orch.run("Điều 100 Nghị định 168/2025 còn hiệu lực không?")
    report = out["conflict_report"]
    assert report["preferred_origin"] is None
    assert report["resolution_status"] == "unresolved"
    assert "Chưa đủ căn cứ" in out["answer"]


def test_status_query_does_not_use_llm_to_pick_a_source():
    legal = StubLegal(_legal_answer())
    web = StubWeb(_web_result("310/2026/NĐ-CP", "2026-08-15"))
    llm = StubLLM("Câu trả lời hợp nhất")
    orch = _orchestrator(legal, web, resolver_llm=llm)
    out = orch.run("Điều 100 Nghị định 168/2025 còn hiệu lực không?")
    assert "Chưa đủ căn cứ" in out["answer"]
    assert llm.calls == []
    assert "conflict: resolve + synthesize" in out["trace"]
    assert "web: search (domain-restricted)" in out["trace"]


def test_web_is_called_even_when_legal_status_route_is_fast_track():
    legal = StubLegal(_legal_answer(route="fast_track"))
    web = StubWeb(_web_result("168/2025/NĐ-CP", "2025-06-25"))
    orch = _orchestrator(legal, web)
    out = orch.run("Điều 100 của 168/2025/NĐ-CP còn hiệu lực không?")
    assert web.calls
    assert out["web_sources"][0]["authority_tier"] == "official"


def test_web_skipped_when_internal_evidence_sufficient():
    """Câu tra cứu thường + evidence đủ -> KHÔNG gọi web (quyết định #12)."""
    legal = StubLegal(_legal_answer())
    web = StubWeb(_web_result("310/2026/NĐ-CP", "2026-08-15"))
    orch = _orchestrator(legal, web, resolver_llm=StubLLM("x"))
    out = orch.run("Nội dung Điều 100 Nghị định 168/2025 là gì?")
    assert web.calls == []
    assert out["answer"] == "Câu trả lời nội bộ"


def test_no_web_means_no_second_llm_call():
    """Không có web -> trả thẳng câu trả lời Legal, KHÔNG gọi LLM lần 2."""
    legal = StubLegal(_legal_answer())
    web = StubWeb(WebResult(available=False, reason="x"))
    llm = StubLLM()
    from agents.ConflictResolver import ConflictResolver

    orch = Orchestrator(
        legal=legal, web=web, resolver=ConflictResolver(llm=llm)
    )
    out = orch.run("câu hỏi")
    assert out["answer"] == "Câu trả lời nội bộ"
    assert llm.calls == []


def test_no_web_status_query_abstains_instead_of_repeating_internal_claim():
    legal = StubLegal(_legal_answer(route="fast_track"))
    web = StubWeb(WebResult(available=False, reason="no key"))
    orch = _orchestrator(legal, web)
    out = orch.run("Điều 100 Nghị định 168/2025 còn hiệu lực không?")
    assert "Chưa xác minh được" in out["answer"]


def test_coordinator_receives_separate_primary_and_legal_context_with_web_provenance():
    from agents.ConflictResolver import ConflictResolver
    from agents.Orchestrator import OrchestratorState

    llm = StubLLM("Tóm tắt có căn cứ.")
    legal = LegalAnswer(
        answer="Nội dung nội bộ.",
        evidence=[
            {"node_id": "n1", "citation": "Điều 4 (168/2025/NĐ-CP)",
             "text": "Primary clause", "metadata": {"so_hieu": "168/2025/NĐ-CP", "dieu": "4"},
             "evidence_class": "primary"},
            {"citation": "AMENDS: X -> Điều 4", "text": "Graph relation",
             "metadata": {}, "evidence_class": "legal_context"},
        ],
        primary_evidence=[
            {"node_id": "n1", "citation": "Điều 4 (168/2025/NĐ-CP)",
             "text": "Primary clause", "metadata": {"so_hieu": "168/2025/NĐ-CP", "dieu": "4"}}
        ],
        legal_context=[
            {"citation": "AMENDS: X -> Điều 4", "text": "Graph relation", "metadata": {}}
        ],
    )
    web = StubWeb(_web_result("168/2025/NĐ-CP", "2025-06-25"))
    orch = Orchestrator(legal=StubLegal(legal), web=web, resolver=ConflictResolver(llm=llm))
    state = OrchestratorState(
        question="Tóm tắt nguồn nội bộ và web",
        internal_answer=legal,
        web_result=web._result,
    )
    orch._synthesize(state)
    prompt = llm.calls[0]
    assert "PRIMARY EVIDENCE" in prompt
    assert "LEGAL CONTEXT" in prompt
    assert "URL: https://vbpl.vn/168/2025/NĐ-CP" in prompt
    assert "Cấp nguồn: chính thức" in prompt


def test_empty_question_returns_follow_up():
    orch = Orchestrator(legal=StubLegal(_legal_answer()), web=StubWeb(_web_result()))
    out = orch.run("   ")
    assert out["answer"] == ""
    assert out["follow_up"]


def test_current_status_without_web_preserves_internal_provision_text():
    legal = StubLegal(_legal_answer())
    orch = _orchestrator(legal, web=None)
    orch.use_web = False
    result = orch.run("Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP c\u00f2n hi\u1ec7u l\u1ef1c hi\u1ec7n nay kh\u00f4ng?")
    assert result["answer"].startswith("C\u00e2u tr\u1ea3 l\u1eddi n\u1ed9i b\u1ed9")
    assert "T2" in result["answer"]


def test_web_targets_skip_out_of_domain_intent():
    from types import SimpleNamespace
    from agents.Orchestrator import OrchestratorState
    state = OrchestratorState(
        understanding=SimpleNamespace(intent="tra_cuu"),
        internal_answer=LegalAnswer(answer="", metadata={"per_intent": [
            {"intent_id": "ood", "sub_query": "weather in Hanoi",
             "is_out_of_domain": True, "execution": "SUCCESS"},
            {"intent_id": "legal",
             "sub_query": "Ngh\u1ecb \u0111\u1ecbnh 296/2026/N\u0110-CP b\u1ed5 sung kho\u1ea3n 7 v\u00e0o \u0110i\u1ec1u 4",
             "execution": "SUCCESS"},
        ]}),
    )
    assert Orchestrator._web_targets(state) == [
        ("legal", state.internal_answer.metadata["per_intent"][1]["sub_query"])
    ]


def test_multi_intent_plan_does_not_call_refiner_even_if_query_looks_single():
    class Refiner:
        called = False
        def refine(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("Refiner must not overwrite planned sub-questions")
    legal = StubLegal(_legal_answer())
    legal.plan_question = lambda _q: {"planned": True, "status": "planned", "intents": [
        {"intent_id": "i1", "sub_query": "sub-question one", "needs_clarification": False},
        {"intent_id": "i2", "sub_query": "sub-question two", "needs_clarification": False},
    ]}
    refiner = Refiner()
    orch = Orchestrator(legal=legal, use_web=False, refiner=refiner)
    class Graph:
        def invoke(self, state): return orch._run_internal(state)
    orch._lg_graph = Graph()
    orch.run("original query")
    assert refiner.called is False
