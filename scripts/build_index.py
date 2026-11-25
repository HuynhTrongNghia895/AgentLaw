from __future__ import annotations

"""Build (hoặc cập nhật) index FAISS + BGE-M3 cho retrieval.

Cách dùng (từ thư mục gốc repo, dùng Python của .venv):

    .venv/Scripts/python scripts/build_index.py
    .venv/Scripts/python scripts/build_index.py --dataset thue_gtgt
    .venv/Scripts/python scripts/build_index.py --force

Hash-aware (CLAUDE.md §9.3): dataset nào có data không đổi (sha256 các
JSONL không đổi) sẽ bị SKIP, không embed lại. Chỉ build mới khi data
thay đổi hoặc --force.

Nguồn: data/processed/ (đã bỏ chunking 04/11) — mỗi record = 1 điều/khoản.

Kết quả:
    data/vector_db/bge_m3/<dataset>/{dense.index, rows.json,
                                     sparse_inv.npz, meta.json}
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "retrieval"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from vector_store import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build index FAISS + BGE-M3 (hash-aware)"
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Tên dataset (doanh_nghiep, quan_ly_thue, thue_gtgt, "
             "thue_tndn) hoặc 'all' (mặc định).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua kiểm tra hash, build lại toàn bộ.",
    )
    args = parser.parse_args()

    processed_root = PROJECT_ROOT / "data" / "processed"
    index_root = PROJECT_ROOT / "data" / "vector_db"

    datasets = (
        None
        if args.dataset == "all"
        else [args.dataset]
    )

    store = VectorStore(
        processed_root=processed_root,
        index_root=index_root,
    )

    def progress(msg: str) -> None:
        print(f"  {msg}")

    print("Build index BGE-M3 ...")
    started = time.time()
    reports = store.build_all(
        datasets=datasets,
        force=args.force,
        progress=progress,
    )

    print()
    print(f"{'dataset':20} {'trạng thái':10} {'vec':>6}")
    print("-" * 42)
    for r in reports:
        state = "SKIP" if r["skipped"] else "XOAY"
        print(
            f"{r['dataset']:20} {state:10} {r['n_rows']:>6}"
        )
    print("-" * 42)
    print(f"Tổng thời gian: {time.time() - started:.1f}s")

    skipped = sum(1 for r in reports if r["skipped"])
    built = len(reports) - skipped
    print(f"Build mới: {built}, skip (hash không đổi): {skipped}")

    # Load lại để chắc chắn index hợp lệ (không crash khi đọc ngược).
    print()
    for r in reports:
        if not r["skipped"]:
            idx = store.load_dataset(r["dataset"])
            print(
                f"  load verify {r['dataset']}: "
                f"{idx.n_rows} rows, "
                f"n_terms sparse = "
                f"{len(idx.sparse['terms']) if len(idx.sparse['terms']) else 0}"
            )


if __name__ == "__main__":
    main()
