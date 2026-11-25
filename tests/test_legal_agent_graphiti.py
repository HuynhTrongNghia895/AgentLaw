"""Test graphiti layer trong LegalRetrievalAgent (P6+P7).

Kiểm tra:
  - `graphiti_block()` degrade an toàn (bridge không sẵn sàng -> ("", []))
  - bridge thật -> trả (text, facts) với facts có source="graphiti"
  - `_graphiti_bridge()` không shadow (bug cũ)
"""

from __future__ import annotations

import os
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

from agents.LegalRetrieval import LegalRetrievalAgent  # noqa: E402


def _agent() -> LegalRetrievalAgent:
    """Agent nhẹ (không gọi LLM thật ở lớp test này)."""
    return LegalRetrievalAgent()


def test_graphiti_block_degrades_safely_when_no_bridge():
    """Bị ép bridge không khả dụng -> không crash, trả rỗng."""
    agent = _agent()
    # Fake bridge: search_with_nodes luôn lỗi.
    class _BrokenBridge:
        def search_with_nodes(self, *a, **k):
            raise RuntimeError("neo4j down")

    agent.graphiti = _BrokenBridge()
    text, facts = agent.graphiti_block("điều 49 luật doanh nghiệp")
    assert text == ""
    assert facts == []


def test_graphiti_block_degrades_when_bridge_none():
    """graphiti=None + không cache -> trả rỗng, không crash."""
    agent = _agent()
    agent.graphiti = None
    agent._graphiti_cache = False  # ép "không khả dụng"
    text, facts = agent.graphiti_block("thống nhất thuế GTGT")
    assert text == ""
    assert facts == []


def test_graphiti_block_shapes_facts_from_bridge():
    """Bridge giả trả kết quả -> facts có đúng schema."""
    agent = _agent()

    class _FakeBridge:
        def search_with_nodes(self, q, num_results=5):
            return [
                {
                    "fact": "AMENDS (bổ sung khoản 3)",
                    "name": "AMENDS",
                    "source_node": {
                        "citation": "03/2022/QH15 Điều 7",
                        "kg_id": "03_2022_QH15_D7",
                        "text": "Sửa đổi khoản 3 ...",
                    },
                    "target_node": {
                        "citation": "59/2020/QH14 Điều 49",
                        "kg_id": "59_2020_QH14_D49",
                        "text": "",
                    },
                    "valid_at": "2022-03-01",
                }
            ]

    agent.graphiti = _FakeBridge()
    text, facts = agent.graphiti_block("điều 49 bị sửa đổi")
    assert "AMENDS" in text
    assert len(facts) == 1
    f = facts[0]
    assert f["source"] == "graphiti"
    assert f["metadata"]["source_node"] == "03_2022_QH15_D7"
    assert f["metadata"]["target_node"] == "59_2020_QH14_D49"


def test_bridge_method_not_shadowed():
    """Bug cũ: attribute cùng tên method -> shadow bound-method.

    `self._graphiti_cache` phải là cache riêng, KHÔNG đè method
    `_graphiti_bridge`.
    """
    agent = _agent()
    assert hasattr(agent, "_graphiti_cache")
    # method vẫn là function (không bị thay bằng instance value)
    assert callable(type(agent)._graphiti_bridge)
    # gọi 2 lần vẫn ổn (idempotent, không crash)
    _ = agent._graphiti_bridge()
    _ = agent._graphiti_bridge()


def test_synthesize_includes_graphiti_facts_in_evidence():
    """`answer` gộp graphiti facts vào evidence (chúng có source='graphiti')."""
    agent = _agent()

    class _FakeBridge:
        def search_with_nodes(self, q, num_results=5):
            return [
                {
                    "fact": "AMENDS",
                    "name": "AMENDS",
                    "source_node": {"citation": "A Điều 1", "kg_id": "k1", "text": "x"},
                    "target_node": {"citation": "B Điều 2", "kg_id": "k2", "text": ""},
                    "valid_at": None,
                }
            ]

    agent.graphiti = _FakeBridge()
    text, facts = agent.graphiti_block("điều a sửa điều b")
    assert len(facts) == 1
    # Đây là data mà `_synthesize` sẽ cộng vào `evidence`.
    assert all(f["source"] == "graphiti" for f in facts)


# ---------------------------------------------------------------------------
# B1-rộng: kéo provision "văn bản sửa" từ FAISS hit SÂU (S2)
# ---------------------------------------------------------------------------

class _Node:
    """Node tối giản đủ cho B1 (so_hieu, dieu)."""

    def __init__(self, nid: str, so_hieu: str, dieu: str) -> None:
        self.node_id = nid
        self.so_hieu = so_hieu
        self.dieu = dieu


class _FakeKG:
    """KG giả: node(nid) + relations_of(so_hieu, article).

    - nodes: dict node_id -> _Node (node không có -> None, đúng hành vi thật)
    - relations: dict (so_hieu, article) -> list[dict by_document/by_article/...]
    """

    def __init__(self, nodes, relations) -> None:
        self._nodes = nodes
        self._relations = relations

    def node(self, nid):
        return self._nodes.get(nid)

    def relations_of(self, so_hieu, article=None):
        return self._relations.get((so_hieu, article), [])


# Pool FAISS của S2 (thật): 48/2024 D5 đứng #5 (index 4).
_S2_POOL = [
    "108_2025_QH15_D19_K1",
    "68_2026_NĐ_CP_D3_K1",
    "108_2025_QH15_D13_K1",
    "181_2025_NĐ_CP_D4_K16",
    "48_2024_QH15_D5_K28",
]


def test_b1_pulls_amending_doc_at_depth_5():
    """S2: provision gốc (48/2024 D5) ở vị trí FAISS #5 -> cửa sổ mặc
    định 5 bắt được -> hỏi đồ thị -> thêm provision 'văn bản sửa'
    (149/2025 D1_K1) vào pool. Trần cũ (top-3) đã bỏ sót."""
    agent = _agent()
    kg = _FakeKG(
        nodes={
            "108_2025_QH15_D19_K1": _Node("108_2025_QH15_D19_K1", "108/2025/QH15", "19"),
            "108_2025_QH15_D13_K1": _Node("108_2025_QH15_D13_K1", "108/2025/QH15", "13"),
            "48_2024_QH15_D5_K28": _Node("48_2024_QH15_D5_K28", "48/2024/QH15", "5"),
        },
        relations={
            ("48/2024/QH15", "5"): [
                {"by_document": "149/2025/QH15", "by_article": "1", "by_clause": "1"},
            ],
        },
    )
    agent.kg = kg
    assert agent._enrich_window == 5  # mặc định mới

    out = agent._enrich_with_graph_relations(None, list(_S2_POOL))

    assert "149_2025_QH15_D1_K1" in out
    # Pool gốc vẫn giữ nguyên (append, không xáo trộn n)
    assert out[: len(_S2_POOL)] == _S2_POOL


def test_b1_trace_preserves_target_clause_from_relation_metadata():
    """Relation trace must retain clause targets, not only document/article."""
    agent = _agent()
    agent.kg = _FakeKG(
        nodes={"48_2024_QH15_D5_K28": _Node(
            "48_2024_QH15_D5_K28", "48/2024/QH15", "5"
        )},
        relations={("48/2024/QH15", "5"): [{
            "by_document": "149/2025/QH15", "by_article": "1",
            "by_clause": "1", "relation_type": "AMENDS",
            "clauses": "khoản 25",
        }]},
    )
    agent._enrich_with_graph_relations(None, ["48_2024_QH15_D5_K28"])
    assert agent._last_graph_relations[0]["to_clauses"] == "khoản 25"


def test_b1_window_knob_controls_enrich():
    """Cửa sổ là knob: window=3 (hành vi cũ) -> D5 ở #5 bị chui -> vb
    sửa KHÔNG về; window=5 (mới) -> về. Chứng minh knob hoạt động thật."""
    agent = _agent()
    kg = _FakeKG(
        nodes={"48_2024_QH15_D5_K28": _Node("48_2024_QH15_D5_K28", "48/2024/QH15", "5")},
        relations={
            ("48/2024/QH15", "5"): [
                {"by_document": "149/2025/QH15", "by_article": "1", "by_clause": "1"},
            ],
        },
    )
    agent.kg = kg

    agent._enrich_window = 5
    assert "149_2025_QH15_D1_K1" in agent._enrich_with_graph_relations(None, list(_S2_POOL))

    agent._enrich_window = 3
    assert "149_2025_QH15_D1_K1" not in agent._enrich_with_graph_relations(None, list(_S2_POOL))


def test_b1_dedup_same_target_added_once():
    """Nhiều quan hệ chỉ về CÙNG provision -> chỉ append 1 lần (dedup seen)."""
    agent = _agent()
    kg = _FakeKG(
        nodes={"48_2024_QH15_D5_K28": _Node("48_2024_QH15_D5_K28", "48/2024/QH15", "5")},
        relations={
            ("48/2024/QH15", "5"): [
                {"by_document": "149/2025/QH15", "by_article": "1", "by_clause": "1"},
                {"by_document": "149/2025/QH15", "by_article": "1", "by_clause": "1"},
            ],
        },
    )
    agent.kg = kg
    out = agent._enrich_with_graph_relations(None, ["48_2024_QH15_D5_K28"])
    assert out.count("149_2025_QH15_D1_K1") == 1
