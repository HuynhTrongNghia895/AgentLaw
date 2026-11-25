from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

# --- Cấu hình model ---------------------------------------------------------
# BGE-M3: multi-vector (dense 1024-d + sparse token weights + long-doc colbert),
# chúng ta chỉ dùng dense + sparse (hybrid nội tại, theo quyết định user).
MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024

# BGE-Reranker-M3 (cross-encoder) - dùng trong Reranker.py.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

ENCODER_BATCH_SIZE = 64
# Đã bỏ chunking (04/11): 1 record = 1 điều/khoản, text ĐẦY ĐỦ
# (điều + khoản + điểm) nên dài hơn nhiều so với chunk cũ.
#
# ĐO THẬT trên toàn bộ 2.393 record (tokenizer BGE-M3, 04/11):
#   median 94 token · p95 406 · p99 951 · max 6.026
#   >1024 token: 18 record (0.75%)   >2048: 8 (0.33%)   >4096: 4 (0.17%)
#
# Chọn 2048: che 99.67% record, chỉ 8 record bị cắt (đều là văn bản sửa
# đổi dài). Build đo được: 1024 -> 377s · 2048 -> 426s · 4096 -> 572s
# (dataset thue_gtgt, CPU fp32). Chênh lệch 1024 vs 4096 chỉ ~50%
# (phần lớn thời gian nằm ở padding theo batch, không ở record dài).
#
# LƯU Ý: BGE-M3 attention O(seq^2) + KHÔNG tự sort theo độ dài -> nếu
# 1 batch 64 chứa 1 text 16.859 ký tự (~6.000 token) thì CẢ 64 text
# đều bị pad lên 4096+ token => ~35s/text. Fix bằng LENGTH_BUCKETS.
MAX_LENGTH = 2048

# Chia batch nhiều TẦNG theo độ dài (mỗi tầng 1 batch size); trong tầng
# sắp xếp tăng dần -> mỗi batch đồng đều độ dài -> padding tối thiểu.
#
# Đo thật (i5-12450H, CPU, fp32, 04/11):
#   text ≤ 500 ký tự  -> 0.13–0.66 s/text
#   text 16.859 ký tự -> 35.2 s/text (batch 1 = batch 4)
# Chi phí 1 batch ≈ batch_size × seq_max².
LENGTH_BUCKETS: tuple[tuple[int, int], ...] = (
    (600, 64),
    (1500, 32),
    (3000, 16),
    (6000, 8),
    (10**9, 2),
)

# CPU-only (đã verify: không có torch/cuda trong repo). FP16 trên CPU không
# có lợi (chậm + trôi số) -> chạy fp32.
USE_FP16 = False

_MODEL_CACHE: dict[str, Any] = {}
_LOAD_LOCK = threading.Lock()


def _load_model(
    model_name: str = MODEL_NAME,
    use_fp16: bool = USE_FP16,
) -> Any:
    """Load BGEM3FlagModel một lần, cache theo (name, fp16).

    Version FlagEmbedding mới đổi class: `BGEM3FlagModel` (encode() trả
    dict {"dense_vecs","lexical_weights","colbert_vecs"}). Lazy import để
    module import được khi chưa cài (mock trong test); báo rõ nếu thiếu.
    """
    key = f"{model_name}|fp16={use_fp16}"
    with _LOAD_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Thiếu 'flagembedding'. Cài: "
                ".venv/Scripts/python -m pip install flagembedding"
            ) from exc

        # fp16=False: chạy CPU. normalize_embeddings=True (default) ->
        # dense đã L2-normalize (FAISS FlatIP = cosine).
        model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            normalize_embeddings=True,
        )
        _MODEL_CACHE[key] = model
        return model


def _as_int_weight_sparse(sparse: Any) -> dict[int, float]:
    """Chuẩn hóa dict sparse của BGE-M3 (token_id -> weight) về
    {int: float}. Chấm dứt các key rỗng / weight <= 0.

    Chứa token có weight dương; token-id là int (khóa inverted index).
    """
    if not sparse:
        return {}
    cleaned: dict[int, float] = {}
    for term, weight in sparse.items():
        weight = float(weight)
        if weight > 0.0:
            cleaned[int(term)] = weight
    return cleaned


def _dense_from_result(
    result: Any, index: int | None = None
) -> np.ndarray:
    """Trích dense vector từ kết quả dict của BGEM3FlagModel.encode().

    `dense_vecs`: 2-D [n, dim] khi input là list; 1-D [dim] khi là 1
    string. index=None -> cả n (cho corpus); index=i -> vec số i.
    Đã L2-normalize (normalize_embeddings=True); normalize thêm cho an
    toàn (idempotent)."""
    dense = np.asarray(result["dense_vecs"], dtype=np.float32)
    if index is not None:
        dense = dense[index]
    dense = dense.reshape(-1) if dense.ndim == 1 else dense
    if dense.ndim == 2:
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        dense = dense / norm
    else:
        norm = float(np.linalg.norm(dense))
        if norm != 0:
            dense = dense / norm
    return dense


def _sparse_from_result(
    result: Any, index: int | None = None
) -> Any:
    """Trích sparse (lexical_weights) từ kết quả encode().

    `lexical_weights`: list[n] dict khi input list; dict khi 1 string.
    index=None -> cả n; index=i -> dict số i. Clean về {int: float}.
    """
    lex = result["lexical_weights"]
    if index is not None:
        lex = lex[index]
    return _as_int_weight_sparse(lex)


def embed_texts(
    texts: Sequence[str],
    model: Any | None = None,
) -> tuple[np.ndarray, list[dict[int, float]]]:
    """Embed danh sách văn BẢN (chunk/document) -> (dense, sparse).

    dense: float32 [n, DENSE_DIM], đã L2-normalize.
    sparse: list[n] của dict {token_id:int, weight:float} (weight>0).

    `model`: đối tượng BGE-M3 (thêm để test mock). Mặc định dùng cache.
    Mock đơn giản: object có `.encode(list) -> {"dense_vecs": np 2-D,
    "lexical_weights": list[dict]}` (xem tests/test_retrieval.py).

    Chia batch theo độ dài (xem LONG_TEXT_CHARS ở trên) để tránh padding
    cả batch lên 4096 token khi có 1 text dài. Thứ tự kết quả GIỮ NGUYÊN
    theo input (sắp xếp nội bộ rồi khôi phục).
    """
    m = model if model is not None else _load_model()
    items = list(texts)
    if not items:
        return np.zeros((0, DENSE_DIM), dtype=np.float32), []

    dense_parts: dict[int, Any] = {}
    sparse_parts: dict[int, dict[int, float]] = {}

    def _run(idxs: list[int], batch_size: int) -> None:
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start:start + batch_size]
            res = m.encode(
                [items[i] for i in chunk],
                batch_size=batch_size,
                max_length=MAX_LENGTH,
                return_dense=True,
                return_sparse=True,
            )
            d = _dense_from_result(res)
            for j, i in enumerate(chunk):
                dense_parts[i] = d[j]
                sparse_parts[i] = _sparse_from_result(res, j)

    # Chia tầng theo độ dài; trong tầng sắp xếp tăng dần -> mỗi batch
    # đều đồng đều độ dài -> padding tối thiểu.
    for max_chars, batch_size in LENGTH_BUCKETS:
        idxs = [i for i, t in enumerate(items) if len(t) <= max_chars]
        if not idxs:
            continue
        idxs.sort(key=lambda i: len(items[i]))
        # Chỉ giữ các text CHƯA được xử lý (bucket trước đã lấy các
        # text ngắn hơn).
        idxs = [i for i in idxs if i not in dense_parts]
        _run(idxs, batch_size)

    dense = np.stack([dense_parts[i] for i in range(len(items))])
    sparse = [sparse_parts[i] for i in range(len(items))]
    return dense, sparse


def embed_query(
    query: str,
    model: Any | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """Embed 1 QUERY -> (dense, sparse).

    BGE-M3 không cần prefix [Query] (khác BGE v1.5) -> encode nguyên câu.

    dense: float32 [DENSE_DIM], đã L2-normalize.
    sparse: dict {token_id:int, weight:float}.
    """
    m = model if model is not None else _load_model()

    res = m.encode(
        [query],
        batch_size=ENCODER_BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
    )

    dense = _dense_from_result(res, 0).reshape(DENSE_DIM)
    sparse = _sparse_from_result(res, 0)
    return dense, sparse


def dimension() -> int:
    """Chiều dense vector BGE-M3 (trước mắt = 1024; ghi vào meta index)."""
    return DENSE_DIM


def model_id() -> str:
    """Nhận diện model, ghi vào meta.json để đối chiếu khi đổi model."""
    return MODEL_NAME


def reranker_model_id() -> str:
    return RERANKER_MODEL_NAME
