from __future__ import annotations

import threading
from typing import Any

import numpy as np

try:
    # Import trong package (src.retrieval.*)
    from . import embedder
except ImportError:
    # Import top-level (src/retrieval thêm vào sys.path) - cho script/tests
    import embedder

# Ngưỡng abstention cho điểm rerank (cross-encoder BGE-Reranker-M3 trả
# điểm sigmoid 0..1, không cần normalize thêm). Default 0.5 - tune 1 lần
# trên bộ OOD (bước 6) nếu cần; giữ ngưỡng "tròn", không overfit test.
ABSTAIN_THRESHOLD = 0.5
RERANK_BATCH_SIZE = 16


_model_cache: dict[str, Any] = {}
_lock = threading.Lock()


def _load_reranker(model_name: str) -> Any:
    """Load FlagReranker (BGE-Reranker-M3, zero-shot) một lần, cache."""
    with _lock:
        if model_name in _model_cache:
            return _model_cache[model_name]
        try:
            from FlagEmbedding import FlagReranker
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Thiếu 'flagembedding'. Cài: "
                ".venv/Scripts/python -m pip install flagembedding"
            ) from exc
        model = FlagReranker(
            model_name,
            use_fp16=embedder.USE_FP16,
        )
        _model_cache[model_name] = model
        return model


def get_reranker(
    model_name: str = embedder.reranker_model_id(),
) -> Any:
    """Trả reranker đã load (load 1 lần, dùng chung giữa các câu hỏi)."""
    return _load_reranker(model_name)


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
    model: Any | None = None,
    model_name: str = embedder.reranker_model_id(),
) -> list[dict[str, Any]]:
    """Xếp lại `candidates` (list dict có key `text`) theo độ liên quan
    của `query`, trả về top_n.

    - Thêm field `score` = điểm cross-encoder (0..1) lên mỗi kết quả.
    - Giữ nguyên các field khác của kết quả.
    - Deterministic: cùng input -> cùng thứ tự (tie-break theo score rồi
      vị trí gốc).
    """
    if not candidates:
        return []

    m = model if model is not None else _load_reranker(model_name)
    texts = [c.get("text", "") for c in candidates]

    # FlagEmbedding (version mới): compute_score(sentence_pairs),
    # sentence_pairs = [(query, doc), ...]. Trả ndarray điểm (đã sigmoid).
    sentence_pairs = [(query, text) for text in texts]
    raw = m.compute_score(
        sentence_pairs,
        batch_size=RERANK_BATCH_SIZE,
        normalize=True,
    )
    scores = (
        list(np.asarray(raw).reshape(-1))
        if isinstance(raw, np.ndarray)
        else (list(raw) if isinstance(raw, (list, tuple)) else [float(raw)])
    )

    decorated = [
        {
            **c,
            "score": float(score),
            "rerank": True,
            "_orig_index": i,
        }
        for i, (c, score) in enumerate(zip(candidates, scores))
    ]
    # Sort: điểm giảm dần; tie-break theo vị trí gốc (deterministic).
    decorated.sort(key=lambda x: (-x["score"], x["_orig_index"]))
    for item in decorated:
        item.pop("_orig_index")
    return decorated[:top_n]


def rerank_for_search(
    query: str,
    candidates: list[dict[str, Any]],
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Adapter đúng ký hiệu `rerank_fn` mà VectorStore.search mong đợi:
    `rerank_fn(query, pool) -> list` (không truyền top_n). Trả về pool đã
    xếp hạng (vector_store tự cắt top_k)."""
    return rerank(
        query,
        candidates,
        top_n=len(candidates),
        model=model,
    )


def abstain(
    results: list[dict[str, Any]],
    threshold: float = ABSTAIN_THRESHOLD,
) -> bool:
    """Trả True nếu câu hỏi thiếu căn cứ (không trả lời được).

    Chỉ xét các kết quả đã rerank (có key `rerank`). Max điểm rerank
    < threshold -> abstain. Không có kết quả đã rerank -> abstain
    (an toàn: không bịa khi chưa xếp hạng)."""
    reranked = [r for r in results if r.get("rerank")]
    if not reranked:
        return True
    top = max(r.get("score", 0.0) for r in reranked)
    return top < threshold


__all__ = [
    "rerank",
    "rerank_for_search",
    "abstain",
    "get_reranker",
    "ABSTAIN_THRESHOLD",
]
