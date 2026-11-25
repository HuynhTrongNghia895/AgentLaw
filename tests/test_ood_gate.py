"""Test P7 X4 — OOD gate (chặn câu ngoài phạm vi DB).

Ngưỡng OOD chốt bằng số đo (max OOD = 0.027, min IN = 0.258 -> ngưỡng
0.10). Test dùng LLM giả + reranker giả -> không tốn token, không cần
Ollama / Neo4j.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.LegalRetrieval import (  # noqa: E402
    OOD_REJECT_THRESHOLD,
    LegalRetrievalAgent,
)


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "trả lời"


class FakeEvidence:
    """Evidence giả (chỉ cần .node_id và .text)."""

    def __init__(self, node_id: str, text: str = "nội dung quy định") -> None:
        self.node_id = node_id
        self.text = text
        self.citation = node_id

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "citation": self.citation,
                "text": self.text, "metadata": {}}


def make_agent(rerank_score: float, n_evidence: int = 2):
    """Agent với reranker giả trả về điểm cố định."""

    def fake_rerank(query: str, pool: list[dict]) -> list[dict]:
        out = []
        for i, c in enumerate(pool):
            out.append({**c, "score": rerank_score, "rerank": True,
                        "_orig_index": i})
        out.sort(key=lambda x: -x["score"])
        return out

    class FakeStore:
        def search(self, *a, **kw):
            # Trả 1 hit -> có evidence -> luồng đi tới _synthesize
            # (nơi OOD gate nằm).
            return [{
                "text": "nội dung provision",
                "metadata": {
                    "document_number": "59/2020/QH14",
                    "article_number": "49",
                },
                "score": 0.5,
            }]

    class FakeKG:
        def evidence_for_many(self, node_ids):
            return [FakeEvidence(n) for n in node_ids]

        def lookup(self, node_ids):
            return []

        def node(self, nid):
            return None

        def status_of(self, so, dieu):
            return {"status": "unknown", "relations": []}

    return LegalRetrievalAgent(
        store=FakeStore(),
        kg=FakeKG(),
        llm=FakeLLM(),
        rerank_fn=fake_rerank,
    )


def test_ood_threshold_constant():
    assert 0.0 < OOD_REJECT_THRESHOLD < 0.5


def test_low_score_is_blocked_as_ood():
    """Điểm rerank thấp + không anchor + CÓ legal_group -> chặn OOD.

    Chọn câu có keyword nhóm (thue_gtgt) để VƯỢT LỠ nhánh follow-up
    sớm (không legal_group), đi tới _synthesize nơi OOD gate nằm.
    """
    a = make_agent(rerank_score=0.02)
    res = a.answer("thuế GTGT của hộ kinh doanh tính như thế nào")
    assert res.metadata.get("ood") is True
    assert res.evidence == []
    assert "Chưa tìm thấy" in res.answer or res.answer == ""


def test_high_score_is_not_blocked():
    """Điểm rerank cao -> KHÔNG chặn."""
    a = make_agent(rerank_score=0.9)
    res = a.answer("mức thuế GTGT hàng hóa dịch vụ 2025")
    assert res.metadata.get("ood") is not True


def test_anchor_bypasses_ood_gate():
    """Câu có anchor -> KHÔNG chặn OOD dù điểm thấp (đã có dữ liệu)."""
    a = make_agent(rerank_score=0.01)
    res = a.answer("Điều 49 Luật Doanh nghiệp 2020 quy định gì?")
    assert res.metadata.get("ood") is not True
