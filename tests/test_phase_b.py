"""Focused, token-free Phase B regression checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "src" / "graph"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import api_server
from agents.QueryDecomposer import QueryDecomposer


def test_abstain_mapping_does_not_promote_generic_evidence():
    result = api_server.map_orchestrator_output({
        "answer": "Chưa đủ căn cứ",
        "evidence": [{"node_id": "fallback", "citation": "không được dùng", "text": "x"}],
        "primary_evidence": [],
        "metadata": {"abstain": True, "retrieval_status": "INSUFFICIENT"},
    }, "q", "")
    assert result["evidences"] == []
    assert result["citations"] == []


def test_unanchored_clarified_rewrite_is_not_rejected_for_missing_anchor():
    class FakeLLM:
        name = "fake"
        def set_purpose(self, _purpose): pass
        def complete(self, _prompt): return "quy định về thời hạn nộp thuế"

    decomposer = QueryDecomposer(llm=FakeLLM())
    rewritten, audit = decomposer.rewrite_clarified_intent(
        original_question="Hỏi về thuế",
        original_sub_query="quy định về thuế",
        clarification="thời hạn",
        user_answer="nộp thuế",
    )
    assert rewritten
    assert audit["status"] == "ANCHORS_VALIDATED"


def test_anchored_clarified_rewrite_rejects_lost_anchor():
    class FakeLLM:
        name = "fake"
        def set_purpose(self, _purpose): pass
        def complete(self, _prompt): return "quy định chung về thuế"

    decomposer = QueryDecomposer(llm=FakeLLM())
    rewritten, audit = decomposer.rewrite_clarified_intent(
        original_question="Điều 5 của 48/2024/QH15",
        original_sub_query="Điều 5 của 48/2024/QH15",
        clarification="phạm vi",
        user_answer="thuế",
    )
    assert rewritten == ""
    assert audit["status"] == "REJECTED"


def test_planner_preserves_missing_scope_metadata_without_llm():
    class FakeLLM:
        name = "fake"
        def set_purpose(self, _purpose): pass
        def complete(self, _prompt):
            return '{"intents":[{"sub_query":"ưu đãi thuế","ambiguity_type":"scope","missing_scope":["tax_type"],"needs_clarification":true,"clarification":"Nêu sắc thuế"}]}'

    plan = QueryDecomposer(llm=FakeLLM()).plan("Hỏi ưu đãi thuế ngoài ra thời hạn nộp thuế")
    intent = plan["intents"][0]
    assert intent["ambiguity_type"] == "scope"
    assert intent["missing_scope"] == ["tax_type"]
    assert intent["needs_clarification"] is True
