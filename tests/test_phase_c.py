"""Focused Phase C regression checks; no network or real LLM calls."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "src" / "graph"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import api_server
from agents.LegalRetrieval import LegalAnswer, LegalRetrievalAgent
from agents.Orchestrator import Orchestrator, OrchestratorState


def test_time_travel_without_evidence_still_uses_explicit_activation(monkeypatch):
    class Web:
        def __init__(self): self.calls = 0
        def run(self, _q): self.calls += 1; return SimpleNamespace(available=True, sources=[], summary="")
    web = Web()
    state = OrchestratorState(question="quy định tại ngày 01/01/2025", route="vector", understanding=SimpleNamespace(intent="tra_cuu"), internal_answer=LegalAnswer(answer="x", evidence=[]))
    orch = Orchestrator(legal=SimpleNamespace(llm=None), web=web, use_web=True)
    monkeypatch.setattr(Orchestrator, "_web_targets", staticmethod(lambda _state: []))
    assert orch._needs_web(state) is False
    assert web.calls == 0


def test_api_keeps_context_only_citation_out_of_primary_group():
    out = api_server.map_orchestrator_output({
        "answer": "ok",
        "primary_evidence": [],
        "legal_context": [{"node_id": "ctx", "citation": "Điều 1", "text": "context"}],
    }, "q", "")
    assert out["evidences"][0]["type"] == "context"
    assert out["citations"][0]["sourceLabel"] == "Kho nội bộ"


def test_synthesis_error_does_not_expose_provider_payload():
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent._pin_ids = lambda *_a: []
    agent.kg = SimpleNamespace(node=lambda _nid: None)
    agent._status_block = lambda *_a: ""
    agent.top_k = 5
    agent.rerank_fn = None
    agent._rerank_fn_cache = None
    agent._needs_relation = lambda *_a: False
    agent._grounded_fallback_text = lambda *_a: "Căn cứ đã truy xuất."
    agent._tag_evidence_group = lambda ev, *_a: list(ev)
    agent._claim_evidence_review = lambda *_a: []
    agent._unsupported_document_citations = lambda *_a: []
    agent._unsupported_provision_citations = lambda *_a: []
    agent.llm = SimpleNamespace(set_purpose=lambda _p: None, complete=lambda _p: (_ for _ in ()).throw(RuntimeError("RAW_PROVIDER_SECRET stack trace")))
    evidence = SimpleNamespace(node_id="p", citation="Điều 1", text="text", to_dict=lambda: {"node_id": "p", "citation": "Điều 1", "text": "text"})
    result = agent._synthesize("q", SimpleNamespace(legal_group=None, intent="tra_cuu", anchors=SimpleNamespace(dates=[])), SimpleNamespace(route="vector", confidence=0.5, five_route="", datasets=[]), [evidence])
    assert "RAW_PROVIDER_SECRET" not in result.answer
    assert "RAW_PROVIDER_SECRET" not in repr(result.metadata)
