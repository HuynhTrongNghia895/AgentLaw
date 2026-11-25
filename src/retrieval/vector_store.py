from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - faiss được cài ở bước 1
    faiss = None  # type: ignore[assignment]

try:
    # Import trong package (src.retrieval.*)
    from . import embedder
except ImportError:
    # Import top-level (src/retrieval thêm vào sys.path) - cho script/tests
    import embedder

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

RRF_K = 60  # Reciprocal Rank Fusion, k=60 (chuẩn, stable)
DEFAULT_TOP_K = 5
DEFAULT_RERANK_TOP_N = 10
CANDIDATE_LIMIT = 60  # tối đa candidate mỗi tín hiệu (dense/sparse)

# Metadata giữ trong rows.json. Eval match theo
# document_number/article_number/clause_number; Bug 5 cần amended_by.
METADATA_FIELDS = (
    "storage_id",
    "citation",
    "dataset",
    "document_number",
    "document_type",
    "document_title",
    "record_type",
    "article_number",
    "clause_number",
    "issued_date",
    "effective_date",
    "amended_by",
    "document_amended_by",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _structure_label(raw: Any) -> str:
    """'chapter' trong record = {'number': 'I', 'title': 'QUY ĐỊNH CHUNG'}.

    Trả về chuỗi để ghép vào text embed ("Chương I. QUY ĐỊNH CHUNG").
    """
    if isinstance(raw, dict):
        number = str(raw.get("number") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not number and not title:
            return ""
        if number and title:
            return f"{number}. {title}"
        return number or title
    return str(raw or "").strip()


def build_embed_text(record: dict[str, Any]) -> str:
    """Text dùng để EMBED (để TÌM node liên quan).

    = Chương/Mục (nếu có) + text đầy đủ của record.

    Vì sao gộp Chương/Mục: "Chương VIII. HỘ KINH DOANH VÀ ĐĂNG KÝ HỘ KINH
    DOANH" là từ khóa phân biệt mạnh nhất giữa các nhóm nội dung, mà các
    điều/khoản đều chỉ có tiêu đề chung chung. Không gộp thì query "về hộ
    kinh doanh" khớp mọi điều như nhau.
    """
    parts: list[str] = []
    for key in ("part", "chapter", "section"):
        label = _structure_label(record.get(key))
        if label:
            parts.append(label)
    body = str(record.get("text") or "").strip()
    if body:
        parts.append(body)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sparse inverted index (term-major CSR, lưu npz)
#
# BGE-M3 sparse = {token_id: weight}. Struct:
#   terms:   int64  [n_terms]      (đã sort)
#   starts:  int64  [n_terms + 1]
#   rows:    int32  [n_postings]   (row = index trong rows.json)
#   weights: float32 [n_postings]
# Điểm sparse giữa query q và doc d = sum_t q(t)*d(t) / (||q||*||d||),
# tính trong O(số query term có trong index).
# ---------------------------------------------------------------------------


def build_sparse_array(
    sparse: Sequence[dict[int, float]],
) -> dict[str, np.ndarray]:
    term_postings: dict[int, list[tuple[int, float]]] = {}
    for row, mapping in enumerate(sparse):
        for term, weight in mapping.items():
            term_postings.setdefault(int(term), []).append(
                (row, float(weight))
            )

    terms = np.array(sorted(term_postings), dtype=np.int64)
    starts: list[int] = [0]
    flat_rows: list[int] = []
    flat_weights: list[float] = []
    for term in terms:
        for row, weight in term_postings[term]:
            flat_rows.append(row)
            flat_weights.append(weight)
        starts.append(len(flat_rows))

    row_l2 = np.zeros((len(sparse),), dtype=np.float32)
    for row, mapping in enumerate(sparse):
        if mapping:
            row_l2[row] = float(
                np.sqrt(sum(w * w for w in mapping.values()))
            )

    return {
        "terms": terms,
        "starts": np.array(starts, dtype=np.int64),
        "rows": np.array(
            flat_rows, dtype=np.int32
        ) if flat_rows else np.zeros(0, dtype=np.int32),
        "weights": np.array(
            flat_weights, dtype=np.float32
        ) if flat_weights else np.zeros(0, dtype=np.float32),
        "row_l2": row_l2,
    }


def sparse_dot(
    query: dict[int, float],
    array: dict[str, np.ndarray],
) -> list[tuple[int, float]]:
    """Trả [(row, cosine)] giảm dần điểm cho các doc khớp query term.
    Deterministic: tie-break theo row id tăng dần."""
    terms = array["terms"]
    if len(terms) == 0 or not query:
        return []

    q_norm = float(np.sqrt(sum(v * v for v in query.values())))
    if q_norm == 0.0:
        return []

    dots: dict[int, float] = {}
    starts = array["starts"]
    rows = array["rows"]
    weights = array["weights"]

    for term, q_weight in query.items():
        idx = int(np.searchsorted(terms, int(term), side="left"))
        if idx >= len(terms) or int(terms[idx]) != int(term):
            continue
        lo, hi = int(starts[idx]), int(starts[idx + 1])
        for r, w in zip(rows[lo:hi], weights[lo:hi]):
            r = int(r)
            dots[r] = dots.get(r, 0.0) + float(w) * q_weight

    scored = [
        (row, dot / (q_norm * float(array["row_l2"][row])))
        for row, dot in dots.items()
        if array["row_l2"][row] > 0.0
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    rank_lists: Sequence[Sequence[tuple[int, float]]],
    limit: int,
    k: int = RRF_K,
) -> list[tuple[int, float, dict[str, int]]]:
    """Gộp các list (row, score) bằng Reciprocal Rank Fusion.

    `rank_lists[i]` là top list của tín hiệu i (row, score). Trả về
    [(row, rrf_score, {f"s{i}": rank})] giảm rrf_score, tie-break row id.
    Row đứng thứ r (1-based) trong list i đóng góp 1/(k + r)."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for i, rank_list in enumerate(rank_lists):
        for rank, (row, _score) in enumerate(rank_list, start=1):
            key = f"s{i}"
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(row, {})[key] = rank

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        (row, score, ranks[row])
        for row, score in ordered[:limit]
    ]


# ---------------------------------------------------------------------------
# DatasetIndex: 1 dataset đã load
# ---------------------------------------------------------------------------


@dataclass
class DatasetIndex:
    dataset: str
    faiss_index: Any
    rows_meta: list[dict[str, Any]]
    row_texts: list[str]
    sparse: dict[str, np.ndarray]
    n_rows: int

    def search_dense(
        self, query_dense: np.ndarray, limit: int
    ) -> list[tuple[int, float]]:
        limit = min(limit, self.n_rows)
        if limit <= 0:
            return []
        _scores, ids = self.faiss_index.search(
            np.ascontiguousarray(
                query_dense, dtype=np.float32
            ).reshape(1, -1),
            limit,
        )
        return [
            (int(i), float(s))
            for i, s in zip(ids[0], _scores[0])
            if i >= 0
        ]

    def search_sparse(
        self, query_sparse: dict[int, float], limit: int
    ) -> list[tuple[int, float]]:
        found = sparse_dot(query_sparse, self.sparse)
        return found[:limit]


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """Build (hash-aware) + load + search các index BGE-M3 per dataset.

    Layout: data/vector_db/bge_m3/<dataset>/{dense.index, rows.json,
    sparse_inv.npz, meta.json}"""

    SUBDIR = "bge_m3"

    def __init__(
        self,
        processed_root: Path,
        index_root: Path,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.index_root = Path(index_root)
        self._cache: dict[str, DatasetIndex] = {}

    # -- Helpers -----------------------------------------------------------

    def _dataset_files(self, dataset: str) -> list[Path]:
        folder = self.processed_root / dataset
        if not folder.is_dir():
            raise ValueError(
                f"Dataset '{dataset}' không có trong {folder}"
            )
        files = sorted(folder.glob("*.jsonl"))
        if not files:
            raise ValueError(f"Không có file JSONL trong {folder}")
        return files

    def _dir(self, dataset: str) -> Path:
        return self.index_root / self.SUBDIR / dataset

    # -- Build -------------------------------------------------------------

    def build_dataset(
        self,
        dataset: str,
        force: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Build index cho 1 dataset. Hash-aware: skip nếu sha256 các
        JSONL nguồn + model không đổi (CLAUDE.md §9.3)."""
        files = self._dataset_files(dataset)
        current_hashes = {
            f.name: _sha256_file(f) for f in files
        }

        out_dir = self._dir(dataset)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "meta.json"

        if not force and meta_path.exists():
            meta = json.loads(meta_path.read_text("utf-8"))
            if (
                meta.get("model") == embedder.model_id()
                and meta.get("hashes") == current_hashes
                and (out_dir / "dense.index").exists()
                and (out_dir / "rows.json").exists()
                and (out_dir / "sparse_inv.npz").exists()
            ):
                if progress:
                    progress(
                        f"{dataset}: SKIP (hash không đổi, "
                        f"{meta.get('n_rows', 0)} vec)"
                    )
                return {
                    "dataset": dataset,
                    "skipped": True,
                    "n_rows": meta.get("n_rows", 0),
                }

        # -- Collect
        # Đã bỏ chunking (04/11): nguồn = data/processed/, mỗi record = 1
        # điều/khoản (text ĐẦY ĐỦ: điều + khoản + điểm). Không embed
        # riêng Điểm (điểm đã nằm trong text của khoản).
        records: list[dict[str, Any]] = []
        for f in files:
            for r in read_jsonl(f):
                text = build_embed_text(r)
                if not text.strip():
                    # Điều có khoản có text="" -> nội dung đã nằm ở khoản
                    # con; embed chuỗi rỗng là vô nghĩa.
                    continue
                r = dict(r)
                r["text"] = text
                records.append(r)

        texts = [str(r.get("text", "")) for r in records]
        rows_meta = [
            {k: r.get(k) for k in METADATA_FIELDS}
            | {"text": text}
            for r, text in zip(records, texts)
        ]

        # -- Embed (dense + sparse)
        if progress:
            progress(
                f"{dataset}: embed {len(texts)} records..."
            )
        dense, sparse = embedder.embed_texts(texts)

        # -- Persist dense (FAISS flat, inner product = cosine)
        faiss_index = faiss.IndexFlatIP(dense.shape[1])
        faiss_index.add(dense)
        faiss.write_index(
            faiss_index, str(out_dir / "dense.index")
        )

        with (out_dir / "rows.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(rows_meta, f, ensure_ascii=False)

        np.savez_compressed(
            out_dir / "sparse_inv.npz",
            **build_sparse_array(sparse),
        )

        meta = {
            "model": embedder.model_id(),
            "dim": embedder.dimension(),
            "n_rows": len(records),
            "hashes": current_hashes,
            "pipeline": "retrieval-v2-processed",
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if progress:
            progress(
                f"{dataset}: build xong ({len(records)} vec)"
            )
        return {
            "dataset": dataset,
            "skipped": False,
            "n_rows": len(records),
        }

    def build_all(
        self,
        datasets: Sequence[str] | None = None,
        force: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        if datasets is None:
            datasets = self.discover_datasets()
        return [
            self.build_dataset(ds, force=force, progress=progress)
            for ds in datasets
        ]

    def discover_datasets(self) -> list[str]:
        if not self.processed_root.is_dir():
            return []
        return sorted(
            d.name
            for d in self.processed_root.iterdir()
            if d.is_dir()
        )

    # -- Load --------------------------------------------------------------

    def load_dataset(self, dataset: str) -> DatasetIndex:
        if dataset in self._cache:
            return self._cache[dataset]

        folder = self._dir(dataset)
        npz = np.load(str(folder / "sparse_inv.npz"))
        rows_meta = json.loads(
            (folder / "rows.json").read_text("utf-8")
        )
        index = DatasetIndex(
            dataset=dataset,
            faiss_index=faiss.read_index(
                str(folder / "dense.index")
            ),
            rows_meta=rows_meta,
            row_texts=[m["text"] for m in rows_meta],
            sparse={
                "terms": npz["terms"],
                "starts": npz["starts"],
                "rows": npz["rows"],
                "weights": npz["weights"],
                "row_l2": npz["row_l2"],
            },
            n_rows=len(rows_meta),
        )
        self._cache[dataset] = index
        return index

    def loaded_datasets(self) -> list[str]:
        root = self.index_root / self.SUBDIR
        if not root.is_dir():
            return []
        return sorted(
            d.name
            for d in root.iterdir()
            if (d / "meta.json").exists()
        )

    # -- Search ------------------------------------------------------------

    def search(
        self,
        query: str,
        dataset: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        rerank_fn: Callable[
            [str, list[dict[str, Any]]],
            list[dict[str, Any]],
        ]
        | None = None,
    ) -> list[dict[str, Any]]:
        """Search hybrid (BGE-M3 dense + sparse, RRF fusion).

        - `dataset`: search trong 1 dataset, hoặc None = toàn bộ
          (câu OOD) — mỗi kết quả có field `dataset`.
        - `rerank_fn(query, candidates) -> list` (nếu có + rerank_top_n
          > 0): xếp lại top_k + rerank_top_n candidate, trả top_k;
          score cuối = điểm rerank.
        Kết quả: {storage_id, text, dataset, metadata, score, rank,
        dense_rank, sparse_rank, rrf_score}.
        """
        if dataset is not None:
            targets: dict[str, DatasetIndex] = {
                dataset: self.load_dataset(dataset)
            }
        else:
            available = self.loaded_datasets()
            if not available:
                raise ValueError(
                    "Chưa có index nào. "
                    "Chạy scripts/build_index.py trước."
                )
            targets = {
                ds: self.load_dataset(ds) for ds in available
            }

        q_dense, q_sparse = embedder.embed_query(query)
        per_signal = max(CANDIDATE_LIMIT, top_k + rerank_top_n)

        # Mỗi dataset: 2 tín hiệu (dense, sparse); row id toàn cục
        # (offset) để RRF gộp không va chạm giữa các dataset.
        row_owner: dict[int, tuple[str, int]] = {}
        g_list: list[Sequence[tuple[int, float]]] = []
        # Điểm cosine DENSE thật của từng row (dùng làm confidence).
        dense_score: dict[int, float] = {}

        offset = 0
        for ds, idx in targets.items():
            d_res = idx.search_dense(q_dense, per_signal)
            s_res = idx.search_sparse(q_sparse, per_signal)
            for row, _s in d_res:
                row_owner.setdefault(offset + row, (ds, row))
            for row, _s in s_res:
                row_owner.setdefault(offset + row, (ds, row))
            for row, s in d_res:
                g_row = offset + row
                if g_row not in dense_score or s > dense_score[g_row]:
                    dense_score[g_row] = s
            g_list.append(
                [(offset + r, s) for r, s in d_res]
            )
            g_list.append(
                [(offset + r, s) for r, s in s_res]
            )
            offset += idx.n_rows

        fused = rrf_fuse(
            g_list,
            limit=max(top_k + rerank_top_n, top_k),
        )

        # Trong g_list: ds thứ p (0-based) có dense = s{2p}, sparse = s{2p+1}
        ds_position = {ds: i for i, ds in enumerate(targets)}

        results: list[dict[str, Any]] = []
        for g_row, rrf_score, signal_ranks in fused:
            if g_row not in row_owner:
                continue
            ds, row = row_owner[g_row]
            idx = targets[ds]
            meta = idx.rows_meta[row]
            pos = ds_position[ds]
            dense_rank = signal_ranks.get(f"s{2 * pos}")
            sparse_rank = signal_ranks.get(f"s{2 * pos + 1}")
            results.append(
                {
                    "storage_id": meta.get("storage_id"),
                    "text": idx.row_texts[row],
                    "dataset": ds,
                    "metadata": {
                        "document_number": meta.get(
                            "document_number"
                        ),
                        "article_number": meta.get(
                            "article_number"
                        ),
                        "clause_number": meta.get("clause_number"),
                        "record_type": meta.get("record_type"),
                        "citation": meta.get("citation"),
                        "effective_date": meta.get("effective_date"),
                        "amended_by": meta.get(
                            "amended_by", []
                        ),
                        "document_amended_by": meta.get(
                            "document_amended_by", []
                        ),
                    },
                    "rrf_score": rrf_score,
                    "dense_rank": dense_rank,
                    "sparse_rank": sparse_rank,
                    "score": rrf_score,
                    # Cosine dense THẬT — dùng làm confidence fast-track.
                    # `score` (RRF) chỉ phản ánh thứ hạng, không phản ánh
                    # độ khớp thực sự.
                    "dense_score": dense_score.get(g_row, 0.0),
                }
            )

        if rerank_fn is not None and rerank_top_n > 0 and results:
            pool = results[: top_k + rerank_top_n]
            reranked = rerank_fn(query, pool)
            results = list(reranked)[:top_k]
        else:
            results = results[:top_k]

        for i, r in enumerate(results, start=1):
            r["rank"] = i
        return results


__all__ = [
    "DatasetIndex",
    "VectorStore",
    "build_embed_text",
    "rrf_fuse",
    "sparse_dot",
    "RRF_K",
    "DEFAULT_TOP_K",
    "DEFAULT_RERANK_TOP_N",
    "METADATA_FIELDS",
]
