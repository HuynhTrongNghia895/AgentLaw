from __future__ import annotations

"""Build đồ thị quan hệ pháp lý vào Neo4j từ 2 CSV metadata.

Cách dùng (từ thư mục gốc repo):

    .venv/Scripts/python scripts/build_graph.py
    .venv/Scripts/python scripts/build_graph.py --force   (xóa + build lại)

Nguồn sự thật: data/metadata/DocumentRelations.csv (20 quan hệ,
verification_status=verified) + DocumentRegistry.csv (21 văn bản).
Chỉ dùng quan hệ verified - không suy diễn (CLAUDE.md §6.4).

Idempotent (MERGE): chạy 2 lần -> graph không đổi. --force xóa
toàn bộ node Document/Provision rồi build lại.

Kiểm tra parse fail: nếu provision nào KHÔNG parse được, STOP và in
danh sách (không build nửa vời lên DB).
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "graph"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from Neo4jBuilder import (  # noqa: E402
    Neo4jBuilder,
    read_relations,
    read_registry,
)
from Neo4jClient import Neo4jClient  # noqa: E402
from ProvisionParser import (  # noqa: E402
    parse_provision_list,
    provision_unparsed,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Neo4j graph pháp lý từ 2 CSV metadata"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Xóa graph (Document+Provision) rồi build lại.",
    )
    args = parser.parse_args()

    print("Kiểm tra kết nối Neo4j ...")
    client = Neo4jClient()
    if not client.ping():
        print(
            "LỖI: không nối được Neo4j. "
            "Đảm bảo service đang chạy (sc query neo4j = RUNNING), "
            "kiểm tra .env (NEO4J_URI/USERNAME/PASSWORD). "
            "Thử: http://localhost:7474"
        )
        return 1

    registry = read_registry(PROJECT_ROOT)
    relations = read_relations(PROJECT_ROOT)
    print(
        f"Đã đọc {len(registry)} văn bản (registry), "
        f"{len(relations)} quan hệ (relations)."
    )

    # -- Kiểm tra parse TRƯỚC (không bẩy nửa vời lên DB)
    unparsed_all: list[str] = []
    for rel in relations:
        for side in ("from", "to"):
            refs = parse_provision_list(rel.get(f"{side}_provision"))
            unparsed_all.extend(
                provision_unparsed(refs)
            )
    if unparsed_all:
        print(
            f"\nDỪNG: {len(unparsed_all)} provision KHÔNG parse "
            "được (không bịa số điều/khoản). Sửa parser rồi chạy lại:"
        )
        for u in unparsed_all:
            print(f"  - {u!r}")
        return 2

    print("Parse OK (0 provision lỗi).")

    # -- Build
    builder = Neo4jBuilder(client)
    started = time.time()
    if args.force:
        print("--force: xóa graph cũ ...")
        builder.clear()
    report = builder.build(registry, relations)
    elapsed = time.time() - started

    print(
        f"\nBuild xong trong {elapsed:.1f}s "
        f"(force={args.force}):"
    )
    for k, v in report.items():
        print(f"  {k}: {v}")

    # -- Verify: stats + 1 query nghiệp vụ
    stats = client.document_graph_stats()
    print("\nStats từ DB (verify):")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    demo = client.amendments_of("67/2025/QH15", "4")
    print(
        "\nDemo query - Điều 4 Luật TNDN (67/2025/QH15) bị "
        "sửa bởi:"
    )
    if demo:
        for row in demo:
            print(
                f"  {row['relation_type']} bởi "
                f"{row['by_document']} "
                f"(khoản {row['clauses']}, "
                f"hiệu lực {row['effective_date']})"
            )
    else:
        print("  (trống - chưa có quan hệ AMENDS cho điều 4)")

    status = client.provision_status(
        "67/2025/QH15", "4"
    )
    print(
        f"\nprovision_status(Điều 4) = {status['status']} "
        f"(amended={status['is_amended']}, "
        f"repealed={status['is_repealed']})"
    )

    # -- Demo time-travel (hiệu lực 3 lớp, phương án A)
    print("\n--- Demo governing_provision (time-travel) ---")
    demos = [
        # (so_hieu, article, clause, as_of, mô tả kỳ vọng)
        ("09/2026/QH16", "3", None, "2025-12-31",
         "Điều 3 (09/2026) hiệu lực riêng 01/01/2026 -> CHƯA"),
        ("09/2026/QH16", "3", None, "2026-02-01",
         "đã hiệu lực, bản gốc (chưa có quan hệ sửa hiệu lực <=T)"),
        ("48/2024/QH15", "5", "25", "2025-09-01",
         "khoản 25 Điều 5 (48/2024) hiệu lực riêng 01/01/2026 -> CHƯA"),
        ("48/2024/QH15", "5", None, "2025-09-01",
         "Điều 5 nói chung (48/2024) hiệu lực 01/07/2025 -> đã hiệu lực"),
    ]
    for so, art, cl, asof, desc in demos:
        g = client.governing_provision(
            so, art, as_of_date=asof, clause=cl
        )
        cl_s = f"khoản {cl}" if cl else "cả điều"
        print(
            f"  {so} Điều {art} ({cl_s}) tại {asof}: "
            f"{g['status']} "
            f"(in_effect={g['provision_in_effect_at_T']}, "
            f"ngày={g['effective_date_used']} "
            f"[{g['effective_date_source']}])  # {desc}"
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
