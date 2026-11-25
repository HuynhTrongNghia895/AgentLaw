"""Test retrieval (bước 8).

Fast tests KHÔNG tải model BGE thật: mock embedder + FAISS index giả
(faiss chỉ cần khi build thật - mock bằng object Python).
Slow test (bỏ qua khi chạy thường): build thật 1 dataset + query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "utils"),
    str(PROJECT_ROOT / "src" / "retrieval"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

import vector_store  # noqa: E402
from Reranker import abstain, rerank  # noqa: Reranker, E402
from vector_store import (  # noqa: E402
    rrf_fuse,
    build_sparse_array,
    sparse_dot,
    VectorStore,
    DatasetIndex,
)


# ---------------------------------------------------------------------------
# Fake objects (không tải model thật)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Ghép vào monkeypatch embedder.embed_query/embed_texts (không gọi BGE)."""

    def __init__(self):
        pass

    def embed_query(self, q: str):
        dense = np.zeros(vector_store.embedder.dimension(),
                         dtype=np.float32)
        dense[0] = 1.0
        sparse = {1: 1.0, 2: 0.5}
        return dense, sparse

    def embed_texts(self, texts, model=None):
        n = len(texts)
        dense = np.zeros((n, vector_store.embedder.dimension()),
                          dtype=np.float32)
        dense[:, 0] = 1.0
        sparse = [{i: float(i + 1) for i in range(3)}
                  for i in range(n)]
        return dense, sparse


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def test_rrf_fuse_rank_order():
    """Row xuất hiện trong cả 2 list (top) phải đứng trên row chỉ 1 list."""
    a = [(0, 0.9), (1, 0.8)]  # dense
    b = [(1, 0.7), (2, 0.6)]  # sparse
    fused = rrf_fuse([a, b], limit=5)
    rows = [f[0] for f in fused]
    assert rows[0] == 1  # row 1 top-1 cả 2 list -> RRF cao nhất
    assert len(rows) == 3


def test_rrf_deterministic():
    """Cùng input -> cùng output (không phụ thuộc dict order)."""
    a = [(0, 0.9), (1, 0.8), (2, 0.7)]
    b = [(2, 0.9), (0, 0.8)]
    r1 = rrf_fuse([a, b], limit=5)
    r2 = rrf_fuse([a, b], limit=5)
    assert r1 == r2


def test_rrf_rank_per_signal():
    """Tín hiệu nào có row đó thì rank đúng; không có -> None (trống rỗng)."""
    a = [(0, 0.9)]
    b = [(0, 0.8), (5, 0.1)]
    fused = rrf_fuse([a, b], limit=5)
    row0 = [f for f in fused if f[0] == 0][0]
    assert row0[2] == {"s0": 1, "s1": 1}
    row5 = [f for f in fused if f[0] == 5][0]
    assert row5[2] == {"s1": 2}  # row 5 không có trong list s0 (dense)


# ---------------------------------------------------------------------------
# Sparse (inverted index)
# ---------------------------------------------------------------------------


def test_build_sparse_array_structure():
    sparse = [
        {10: 0.5, 20: 0.5},
        {20: 1.0},
        {30: 1.0},
    ]
    arr = build_sparse_array(sparse)
    assert list(arr["terms"]) == [10, 20, 30]
    assert list(arr["starts"]) == [0, 1, 3, 4]
    # Row 0: (10, 0.5), (20, 0.5) -> postings: 10->[0], 20->[0, 1], 30->[2]
    assert list(arr["rows"]) == [0, 0, 1, 2]
    assert np.allclose(arr["row_l2"], [
        float(np.sqrt(0.25 + 0.25)), 1.0, 1.0
    ])


def test_sparse_dot_scoring_and_tiebreak():
    sparse = [
        {10: 0.5, 20: 0.5},
        {20: 1.0},
        {30: 1.0},
    ]
    arr = build_sparse_array(sparse)
    # Query {20: 1.0}: row 0 dot = 0.5, row 1 dot = 1.0
    res = sparse_dot({20: 1.0}, arr)
    assert res[0][0] == 1  # row 1 cao nhất
    assert res[1][0] == 0


def test_sparse_dot_no_match():
    arr = build_sparse_array([{1: 0.5}])
    assert sparse_dot({99: 1.0}, arr) == []


def test_sparse_dot_empty_index():
    arr = build_sparse_array([])
    assert sparse_dot({1: 1.0}, arr) == []


# ---------------------------------------------------------------------------
# Reranker (mock model - không tải BGE-Reranker)
# ---------------------------------------------------------------------------


class FakeRerankerModel:
    def compute_score(self, sentence_pairs, **kw):
        # Giả lập: câu "ngắn" (doc text) score cao -> reverse
        return [1.0 / (i + 1) for i in range(len(sentence_pairs))]


def test_rerank_ordering_and_scores():
    candidates = [
        {"storage_id": "a", "text": "đạo luật doanh nghiệp"},
        {"storage_id": "b", "text": "thuế giá trị gia tăng"},
        {"storage_id": "c", "text": "thuế xuất nhập khẩu"},
    ]
    out = rerank("thuế", candidates, top_n=2, model=FakeRerankerModel())
    assert len(out) == 2
    # Fake model trả 1/(i+1): doc 0 cao nhất
    assert [o["storage_id"] for o in out] == ["a", "b"]
    assert out[0]["score"] >= out[1]["score"]
    # metadata giữ nguyên
    assert out[0]["text"] == "đạo luật doanh nghiệp"


def test_rerank_empty():
    assert rerank("q", [], top_n=5, model=FakeRerankerModel()) == []


def test_abstain_high_score_no_abstain():
    results = [
        {"score": 0.9, "rerank": True},
        {"score": 0.3, "rerank": True},
    ]
    assert abstain(results, threshold=0.5) is False


def test_abstain_low_score_abstains():
    results = [
        {"score": 0.2, "rerank": True},
        {"score": 0.1, "rerank": True},
    ]
    assert abstain(results, threshold=0.5) is True


def test_abstain_no_rerank_results():
    # Không có kết quả rerank -> abstain (an toàn: không bịa)
    assert abstain([], threshold=0.5) is True
    assert abstain([{"score": 0.9}], threshold=0.5) is True


# ---------------------------------------------------------------------------
# VectorStore (fake index - không cần faiss thật)
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, n_rows=2, dataset="testds"):
    """Tạo VectorStore với data JSONL + index FAISS fake (không load faiss)."""
    processed = tmp_path / "processed"
    idx_dir = tmp_path / "vector_db"
    (processed / dataset).mkdir(parents=True)
    records = [
        {
            "storage_id": f"{dataset}_d{i}_c1",
            "document_number": "1/2025/QH15",
            "article_number": str(i),
            "clause_number": "1",
            "text": f"nội dung điều {i}",
            "citation": f"Điều {i}",
            "dataset": dataset,
        }
        for i in range(n_rows)
    ]
    with (processed / dataset / "1_2025_QH15.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    store = VectorStore(
        processed_root=processed,
        index_root=idx_dir,
    )
    return store, idx_dir, dataset


def test_hash_aware_skip(tmp_path, monkeypatch):
    """Build lần 1 (mock embed) -> build. Lần 2 (hash không đổi) -> SKIP."""
    store, idx_dir, ds = _make_store(tmp_path, n_rows=2)
    # Chỉ mock embed_texts (không tải BGE); faiss thật xử lý vector giả.
    # dimension() thật (1024) - vector fake cũng 1024-d, khớp.
    fake = FakeEmbedder()
    monkeypatch.setattr(
        vector_store.embedder, "embed_texts",
        lambda texts, model=None: fake.embed_texts(texts)
    )

    report = store.build_dataset(ds)
    assert report["skipped"] is False
    assert report["n_rows"] == 2
    # Chạy lại -> SKIP (hash không đổi, không re-embed)
    report2 = store.build_dataset(ds)
    assert report2["skipped"] is True


# ---------------------------------------------------------------------------
# Smoke test thật (opt-in, bỏ qua khi chạy thường)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_real_index_thue_tndn():
    """Build index thue_tndn thật (328 vec) + 1 query + rerank top 10.
    Chạy: pytest tests/test_retrieval.py -m slow -x
    Đòi model BGE đã cache (tải ~4.5GB) + faiss."""
    store = VectorStore(
        processed_root=PROJECT_ROOT / "data" / "processed",
        index_root=PROJECT_ROOT / "data" / "vector_db",
    )
    report = store.build_dataset("thue_tndn")
    idx = store.load_dataset("thue_tndn")
    assert idx.n_rows >= 100  # thue_tndn có ~328 records

    results = store.search(
        "ưu đãi thuế TNDN doanh nghiệp công nghệ cao",
        dataset="thue_tndn",
        top_k=5,
        rerank_top_n=10,
    )
    assert len(results) == 5
    # storage_id hợp lệ (đúng format dataset_doc)
    for r in results:
        assert r["storage_id"].startswith("thue_tndn_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# build_embed_text (P1, 04/11): ghép Chương/Mục + text đầy đủ
# ---------------------------------------------------------------------------


def test_build_embed_text_includes_structure():
    """Chương/Mục được ghép lên đầu text để tăng khả năng khớp ngữ nghĩa."""
    from vector_store import build_embed_text

    text = build_embed_text({
        "text": "Điều 1. Phạm vi điều chỉnh\nNghị định này quy định...",
        "chapter": {"number": "I", "title": "QUY ĐỊNH CHUNG"},
        "section": {"number": "1", "title": "Phạm vi điều chỉnh"},
    })
    assert "I. QUY ĐỊNH CHUNG" in text
    assert "1. Phạm vi điều chỉnh" in text
    assert "Nghị định này quy định..." in text


def test_build_embed_text_without_structure():
    """Văn bản không có Chương/Mục -> text vẫn dùng được."""
    from vector_store import build_embed_text

    text = build_embed_text({"text": "Điều 5. Nội dung"})
    assert text == "Điều 5. Nội dung"


def test_build_embed_text_empty_when_no_body():
    """Điều có khoản có text='' -> không embed (nội dung nằm ở khoản con)."""
    from vector_store import build_embed_text

    assert build_embed_text({"text": ""}) == ""
