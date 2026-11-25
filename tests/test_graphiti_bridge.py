"""Test GraphitiBridge (P6).

Kiểm tra phần logic thuần (không cần Ollama/Neo4j):
  - embedder BGE-M3 trả embedding PHẲNG đúng contract của graphiti
  - NullLLMClient / NullCrossEncoder đúng chữ ký abstract
  - ollama_available() không crash khi Ollama tắt
  - trạng thái/status báo cáo đúng

Các test cần Neo4j được đánh dấu `integration` -> skip nếu thiếu.
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

from agents.GraphitiBridge import (  # noqa: E402
    EMBEDDING_DIM,
    GROUP_ID,
    BgeM3Embedder,
    GraphitiBridge,
    NullCrossEncoder,
    NullLLMClient,
    ollama_available,
)


# ---------------------------------------------------------------------------
# Null clients: phải implement đúng abstract method
# ---------------------------------------------------------------------------
def test_null_llm_client_is_llmclient_subclass():
    from graphiti_core.llm_client.client import LLMClient

    assert issubclass(NullLLMClient, LLMClient)
    # Không được còn abstract method chưa implement.
    assert not NullLLMClient.__abstractmethods__


def test_null_cross_encoder_is_subclass():
    from graphiti_core.cross_encoder import CrossEncoderClient

    assert issubclass(NullCrossEncoder, CrossEncoderClient)
    assert not NullCrossEncoder.__abstractmethods__


@pytest.mark.asyncio
async def test_null_llm_client_raises_on_use():
    """Gọi LLM khi không có -> phải raise rõ ràng, không im lặng."""
    client = NullLLMClient()
    with pytest.raises(RuntimeError, match="không dùng LLM"):
        await client._generate_response([])


@pytest.mark.asyncio
async def test_null_cross_encoder_keeps_order():
    ce = NullCrossEncoder()
    out = await ce.rank("q", ["a", "b", "c"])
    assert [p for p, _ in out] == ["a", "b", "c"]
    assert all(score == 0.0 for _, score in out)


# ---------------------------------------------------------------------------
# Embedder: contract phẳng
# ---------------------------------------------------------------------------
def test_embedder_is_embedderclient_subclass():
    from graphiti_core.embedder import EmbedderClient

    assert issubclass(BgeM3Embedder, EmbedderClient)
    assert not BgeM3Embedder.__abstractmethods__


@pytest.mark.asyncio
async def test_embedder_create_returns_flat_vector():
    """`create` phải trả list[float] PHẲNG (name_embedding là mảng 1D)."""
    emb = BgeM3Embedder()
    vec = await emb.create(["Điều 100 Nghị định 168/2025"])
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert len(vec) == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_embedder_create_batch_returns_nested():
    emb = BgeM3Embedder()
    out = await emb.create_batch(["câu một", "câu hai"])
    assert len(out) == 2
    assert all(len(v) == EMBEDDING_DIM for v in out)


@pytest.mark.asyncio
async def test_embedder_create_empty_returns_empty():
    emb = BgeM3Embedder()
    assert await emb.create([]) == []


# ---------------------------------------------------------------------------
# Bridge: status / degrade
# ---------------------------------------------------------------------------
def test_ollama_available_does_not_crash():
    """Phải trả bool, không raise khi Ollama tắt."""
    assert isinstance(ollama_available(), bool)


def test_bridge_status_shape():
    bridge = GraphitiBridge(use_llm=False)
    status = bridge.status()
    for key in (
        "graphiti_available",
        "llm_backend",
        "embedder",
        "group_id",
        "neo4j_database",
        "note",
    ):
        assert key in status
    assert status["group_id"] == GROUP_ID
    assert "BGE-M3" in status["embedder"]


def test_bridge_group_id_isolates_from_legal_kg():
    """group_id phải KHÁC rỗng -> tách khỏi KG pháp lý chính."""
    assert GROUP_ID
    assert GROUP_ID != "neo4j"


def test_ingest_without_graphiti_returns_error_not_crash():
    """Neo4j down -> report lỗi, KHÔNG raise."""
    bridge = GraphitiBridge(
        uri="bolt://localhost:1",   # cổng không có gì
        user="x", password="y",
        use_llm=False,
    )
    report = bridge.ingest_deterministic(
        [{"uuid": "u1", "name": "n", "labels": ["LegalArticle"]}],
        [],
    )
    assert report.nodes_written == 0
    assert report.errors


# ---------------------------------------------------------------------------
# Tích hợp Neo4j (bỏ qua nếu không có DB)
# ---------------------------------------------------------------------------
def _neo4j_available() -> bool:
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase

        load_dotenv(PROJECT_ROOT / ".env")
        drv = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        drv.verify_connectivity()
        drv.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _neo4j_available(), reason="cần Neo4j chạy"
)
def test_ingest_deterministic_writes_node_and_edge():
    """Ingest 2 node + 1 edge -> node/edge có mặt trong Neo4j."""
    from neo4j import GraphDatabase

    gid = "agentlaw-p6-test"
    bridge = GraphitiBridge(group_id=gid, use_llm=False)
    bridge.clear_group()

    nodes = [
        {"uuid": "p6_t1", "name": "Test Article A", "labels": ["LegalArticle"],
         "summary": "a", "attributes": {"so_hieu": "1/2020/QH"}},
        {"uuid": "p6_t2", "name": "Test Article B", "labels": ["LegalArticle"],
         "summary": "b", "attributes": {"so_hieu": "2/2020/QH"}},
    ]
    edges = [
        {"uuid": "p6_e1", "name": "AMENDS", "fact": "AMENDS",
         "source_uuid": "p6_t1", "target_uuid": "p6_t2", "attributes": {}},
    ]
    report = bridge.ingest_deterministic(nodes, edges)
    assert report.nodes_written == 2, report.errors
    assert report.edges_written == 1, report.errors

    drv = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    with drv.session(database="neo4j") as s:
        n = s.run(
            "MATCH (n:Entity {uuid:'p6_t1'}) RETURN n.uuid AS u, n.group_id AS g"
        ).single()
        assert n is not None
        assert n["g"] == gid
        e = s.run(
            "MATCH ()-[r:RELATES_TO {uuid:'p6_e1'}]->() RETURN count(r) AS c"
        ).single()
        assert e["c"] == 1
    drv.close()
    bridge.clear_group()
