from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (SRC_ROOT, SRC_ROOT / "utils", SRC_ROOT / "graph"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from DocxIngestionPipeline import DocxIngestionPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DOCX -> clean -> parse pháp lý -> processed JSONL -> chunked JSONL"
        )
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help=(
            "Tên file DOCX đúng như cột 'Tên File' trong DocumentRegistry.csv. "
            "Có thể truyền nhiều lần. Bỏ trống để chạy toàn bộ rule ready."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["doanh_nghiep", "quan_ly_thue", "thue_gtgt", "thue_tndn"],
        help="Chỉ xử lý một dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Xử lý lại dù DOCX và cấu hình không thay đổi.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ parse và validate, không ghi đè JSONL.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Lưu báo cáo JSON vào đường dẫn chỉ định.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline = DocxIngestionPipeline(PROJECT_ROOT)
    results, failures = pipeline.ingest(
        filenames=args.file,
        dataset=args.dataset,
        force=args.force,
        dry_run=args.dry_run,
    )

    for result in results:
        marker = {
            "success": "OK",
            "validated": "VALID",
            "skipped": "SKIP",
        }.get(result.status, result.status.upper())
        print(
            f"[{marker}] {result.document_number} | {result.dataset_target} | "
            f"records={result.records} | chunks={result.chunks} | "
            f"raw/clean={result.raw_blocks}/{result.cleaned_blocks}"
        )

    for failure in failures:
        print(
            f"[LỖI] {failure['document_number']} | "
            f"{failure['dataset_target']} | {failure['error']}"
        )

    report = {
        "summary": {
            "success": sum(r.status == "success" for r in results),
            "validated": sum(r.status == "validated" for r in results),
            "skipped": sum(r.status == "skipped" for r in results),
            "failed": len(failures),
            "records": sum(r.records for r in results),
            "chunks": sum(r.chunks for r in results),
        },
        "results": [result.to_dict() for result in results],
        "failures": failures,
    }

    print("\n===== TỔNG KẾT =====")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.json_report:
        report_path = args.json_report
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Báo cáo: {report_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
