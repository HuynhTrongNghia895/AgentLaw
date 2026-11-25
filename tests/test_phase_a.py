"""Focused regression coverage for Phase A evidence/API contracts."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "graph"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import api_server  # noqa: E402
from api_server import ChatRequest, map_orchestrator_output  # noqa: E402
from graph.kg.loader import KgNode  # noqa: E402
from graph.kg.materialize import materialize  # noqa: E402


def _node(node_id: str, kind: str, text: str, **kwargs) -> KgNode:
    return KgNode(
        node_id=node_id,
        so_hieu="168/2025/NĐ-CP",
        kind=kind,
        dieu="100",
        text=text,
        **kwargs,
    )


def test_materialize_article_recursively_includes_clause_and_point_text():
    article = _node("article", "article", "", title="")
    clause = _node("clause", "clause", "", khoan="1", parent_id="article")
    point = _node(
        "point", "point", "Nguyên văn điểm a.", khoan="1", diem="a",
        parent_id="clause",
    )
    evidence = materialize(
        article, children={"article": [clause], "clause": [point]},
        include_status=False,
    )
    assert "Khoản 1." in evidence.text
    assert "a) Nguyên văn điểm a." in evidence.text


def test_api_marks_excerpt_unavailable_when_evidence_text_is_empty():
    message = map_orchestrator_output(
        {"answer": "ok", "primary_evidence": [{
            "node_id": "empty", "citation": "Điều 100 (168/2025/NĐ-CP)",
            "text": "", "metadata": {"dieu": "100", "so_hieu": "168/2025/NĐ-CP"},
        }]},
        "question", "",
    )
    evidence = message["evidences"][0]
    assert evidence["excerpt"] is None
    assert evidence["excerptStatus"] == "unavailable"


def test_api_normalizes_and_deduplicates_relations():
    relation = {
        "name": "AMENDS", "fact": "Adds a provision",
        "source_node": {"citation": "296/2026/NĐ-CP Điều 1"},
        "effective_date": "2026-07-23", "text": "Added text",
    }
    message = map_orchestrator_output(
        {"answer": "ok", "primary_evidence": [{
            "node_id": "n1", "citation": "168/2025/NĐ-CP Điều 100",
            "text": "Evidence", "metadata": {"relations": [relation, dict(relation)]},
        }]},
        "question", "",
    )
    relations = message["evidences"][0]["relations"]
    assert len(relations) == 1
    assert relations[0] == {
        "title": "Adds a provision",
        "citation": "296/2026/NĐ-CP Điều 1",
        "excerpt": "Added text",
        "relationType": "AMENDS",
        "effectiveDate": "2026-07-23",
    }


def test_synthesis_error_metadata_and_answer_do_not_expose_exception():
    from agents.LegalRetrieval import LegalRetrievalAgent
    from retrieval.QueryUnderstanding import understand

    secret = "provider-secret-raw-debug"
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.top_k = 1
    agent.rerank_fn = None
    agent._rerank_fn_cache = None
    agent.graphiti = None
    agent._graph_relation_keys = set()
    agent._final_rerank = lambda _q, ev, facts, **_kw: (ev, facts, 0.0)
    agent._pin_ids = lambda _u: []
    agent._needs_relation = lambda *_args: False
    agent._status_block = lambda *_args: ""
    agent._verified_pins = lambda *_args: []
    agent._grounded_fallback_text = lambda *_args: "Grounded fallback answer"
    agent._tag_evidence_group = lambda ev, *_args: list(ev)
    agent._claim_evidence_review = lambda *_args: []

    class RaisingLLM:
        def set_purpose(self, _purpose):
            pass

        def complete(self, _prompt):
            raise RuntimeError(secret)

    agent.llm = RaisingLLM()
    result = agent._synthesize(
        "question", understand("question"),
        SimpleNamespace(route="vector", confidence=0.5, five_route="", datasets=[]),
        [],
    )
    assert result.metadata.get("llm_error") is True
    assert "llm_error_detail" not in result.metadata
    assert secret not in result.answer
    assert secret not in repr(result.to_dict())


def test_chat_endpoint_error_response_is_generic(monkeypatch):
    from fastapi import HTTPException

    class ExplodingOrchestrator:
        def run(self, **_kwargs):
            raise RuntimeError("provider-secret-raw-debug")

    monkeypatch.setattr(api_server, "get_orchestrator", lambda: ExplodingOrchestrator())
    try:
        asyncio.run(api_server.chat_endpoint(ChatRequest(message="question")))
    except HTTPException as exc:
        assert "provider-secret-raw-debug" not in str(exc.detail)
        assert exc.status_code == 500
    else:
        raise AssertionError("Expected HTTPException")


def test_materialize_without_node_or_descendant_text_does_not_invent_excerpt():
    article = _node("empty-article", "article", "", title="")
    clause = _node("empty-clause", "clause", "", khoan="1", parent_id=article.node_id)
    evidence = materialize(
        article, children={article.node_id: [clause]}, include_status=False
    )
    assert evidence.text.strip() == "Điều 100"
    assert " quy định " not in evidence.text
    assert " nguyên văn " not in evidence.text


def test_relation_dedupe_preserves_all_canonical_fields(monkeypatch):
    monkeypatch.setattr(api_server.asyncio, "get_event_loop", lambda: SimpleNamespace(time=lambda: 0))
    relation = {
        "relation_type": "GUIDES", "fact": "Guides clause",
        "citation": "108/2025/QH15 Điều 2", "excerpt": "Guidance text",
        "effective_date": "2026-01-01",
    }
    message = map_orchestrator_output(
        {"answer": "ok", "legal_context": [{
            "node_id": "ctx", "citation": "108/2025/QH15 Điều 2",
            "text": "Context", "metadata": {"relations": [relation, dict(relation)]},
        }]},
        "question", "",
    )
    relations = message["evidences"][0]["relations"]
    assert len(relations) == 1
    assert relations[0] == {
        "title": "Guides clause",
        "citation": "108/2025/QH15 Điều 2",
        "excerpt": "Guidance text",
        "relationType": "GUIDES",
        "effectiveDate": "2026-01-01",
    }


def test_api_relation_string_is_normalized_to_nonempty_title(monkeypatch):
    monkeypatch.setattr(api_server.asyncio, "get_event_loop", lambda: SimpleNamespace(time=lambda: 0))
    message = map_orchestrator_output(
        {"answer": "ok", "primary_evidence": [{
            "node_id": "n", "citation": "Source", "text": "Text",
            "relations": ["- AMENDS: Nghị định 296/2026/NĐ-CP Điều 1"],
        }]},
        "question", "",
    )
    relation = message["evidences"][0]["relations"][0]
    assert relation["title"] == "Nghị định 296/2026/NĐ-CP Điều 1"
    assert relation["relationType"] == "AMENDS"
    assert relation["citation"] == "Nghị định 296/2026/NĐ-CP Điều 1"
    assert relation["excerpt"] is None
    assert relation["effectiveDate"] is None


import pytest  # noqa: E402


def test_clean_relation_string_extracts_type_and_title(monkeypatch):
    monkeypatch.setattr(api_server.asyncio, "get_event_loop", lambda: SimpleNamespace(time=lambda: 0))
    # String relation formats are supported while backend callers migrate.
    message = map_orchestrator_output(
        {"answer": "ok", "primary_evidence": [{
            "node_id": "n", "citation": "Source", "text": "Text",
            "relations": ["- AMENDS: Nghị định 296/2026/NĐ-CP Điều 1"],
        }]},
        "question", "",
    )
    relation = message["evidences"][0]["relations"][0]
    assert relation["relationType"] == "AMENDS"
    assert relation["title"] == "Nghị định 296/2026/NĐ-CP Điều 1"
    assert relation["citation"]
