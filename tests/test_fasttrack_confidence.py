"""Test confidence fast-track dùng COSINE THẬT (dense_score), không
phải RRF fusion score (score).

Bug đã gặp: `VectorStore.search()` trả `score` = RRF fusion score (~0.03,
hằng số theo rank). `QueryRouter._default_confidence` đọc `score` -> so
với ngưỡng 0.7 luôn False -> fast-track KHÔNG BAO GIỜ chạy (dead code),
và `fasttrack_sweep.py` mọi câu ra cùng điểm 0.0328 -> không chọn
được ngưỡng nào.

Test ở đây dùng stub store (không cần FAISS/model) -> nhanh và ổn định.
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

from retrieval.QueryRouter import (  # noqa: E402
    FAST_TRACK_THRESHOLD,
    QueryRouter,
)


class StubStore:
    """Store giả — trả kết quả với cả `score` (RRF) và `dense_score`."""

    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def search(self, query, dataset=None, top_k=5, rerank_top_n=0):
        self.calls.append({
            "query": query, "dataset": dataset,
            "top_k": top_k, "rerank_top_n": rerank_top_n,
        })
        return self.hits


def _hit(dense: float, rrf: float = 0.0328) -> dict:
    return {
        "text": "nội dung",
        "metadata": {"document_number": "168/2025/NĐ-CP"},
        "score": rrf,
        "dense_score": dense,
    }


# ---------------------------------------------------------------------------
# _default_confidence
# ---------------------------------------------------------------------------
def test_confidence_uses_dense_score_not_rrf():
    """Confidence phải = cosine thật, KHÔNG phải RRF score."""
    store = StubStore([_hit(dense=0.79)])
    router = QueryRouter(store=store)
    conf = router._default_confidence("câu hỏi", None)
    assert conf == 0.79
    assert conf != 0.0328


def test_confidence_zero_when_no_hits():
    router = QueryRouter(store=StubStore([]))
    assert router._default_confidence("câu hỏi", None) == 0.0


def test_confidence_zero_when_no_store():
    router = QueryRouter(store=None)
    assert router._default_confidence("câu hỏi", None) == 0.0


def test_confidence_zero_on_search_error():
    class BrokenStore:
        def search(self, *a, **kw):
            raise RuntimeError("lỗi index")

    router = QueryRouter(store=BrokenStore())
    assert router._default_confidence("câu hỏi", None) == 0.0


# ---------------------------------------------------------------------------
# Ngưỡng đã chốt bằng sweep
# ---------------------------------------------------------------------------
def test_threshold_matches_sweep_result():
    """0.70 = ngưỡng chọn từ fasttrack_sweep (sai 0%, FT rate cao nhất).

    0.80 trở lên -> fast-track không bao giờ chạy.
    """
    assert FAST_TRACK_THRESHOLD == 0.70


def test_fast_track_activates_above_threshold():
    """Có anchor + confidence cao -> route fast_track."""
    store = StubStore([_hit(dense=0.85)])
    router = QueryRouter(
        store=store,
        graph_lookup=lambda ids: [{"node_id": i, "text": "x"} for i in ids],
    )
    decision = router.route("Nội dung Điều 100 Nghị định 168/2025/NĐ-CP")
    assert decision.route == "fast_track"
    assert decision.confidence == 0.85
    assert decision.target_node_ids


def test_falls_back_to_vector_below_threshold():
    """Có anchor nhưng confidence thấp -> đi nhánh đầy đủ (an toàn)."""
    store = StubStore([_hit(dense=0.45)])
    router = QueryRouter(
        store=store,
        graph_lookup=lambda ids: [{"node_id": i, "text": "x"} for i in ids],
    )
    decision = router.route("Nội dung Điều 100 Nghị định 168/2025/NĐ-CP")
    assert decision.route == "vector"
    assert decision.confidence == 0.45
    # Vẫn giữ anchor để bổ sung evidence ở nhánh đầy đủ.
    assert decision.target_node_ids


def test_no_graph_lookup_disables_fast_track():
    """Không có graph_lookup -> fast-track tắt (không trả lời khi
    không tra được KG)."""
    store = StubStore([_hit(dense=0.99)])
    router = QueryRouter(store=store, graph_lookup=None)
    decision = router.route("Nội dung Điều 100 Nghị định 168/2025/NĐ-CP")
    assert decision.route != "fast_track"
