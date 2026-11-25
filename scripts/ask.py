"""CLI hỏi pháp luật — entry point cho người dùng (P7).

Dùng:
    .venv\\Scripts\\python scripts\\ask.py "Điều 49 Luật Doanh nghiệp sửa đổi bởi văn bản nào?"
    .venv\\Scripts\\python scripts\\ask.py --trace --evidence "..."

Không cần set PYTHONPATH — script tự chèn các đường dẫn src/.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hỏi pháp luật (Legal Agent + Graphiti + Web Agent)"
    )
    ap.add_argument("question", nargs="+", help="câu hỏi")
    ap.add_argument(
        "--web", action="store_true",
        help="bật Web Agent (cần TAVILY_API_KEY trong .env)",
    )
    ap.add_argument(
        "--trace", action="store_true", help="in các bước đã đi qua"
    )
    ap.add_argument(
        "--evidence", action="store_true", help="in danh sách căn cứ"
    )
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        ap.print_help()
        return 1

    from agents.Orchestrator import Orchestrator

    orch = Orchestrator(use_web=args.web)
    result = orch.run(question)

    print("=" * 72)
    print(f"CÂU HỎI: {question}")
    print("=" * 72)

    if result.get("follow_up"):
        print(f"\n[HỎI LẠI] {result['follow_up']}")

    answer = result.get("answer") or ""
    if answer:
        print(f"\nTRẢ LỜI:\n{answer}")
    else:
        print("\n(không có câu trả lời)")

    if args.evidence:
        ev = result.get("evidence") or []
        print(f"\n--- CĂN CỨ ({len(ev)}) ---")
        for i, e in enumerate(ev, 1):
            src = e.get("source", "internal")
            cite = e.get("citation", "")
            print(f"[{i}] ({src}) {cite}")

    if args.trace:
        print(f"\n--- TRACE --- route={result.get('route')}")
        for step in result.get("trace") or []:
            print(f"  • {step}")

    cr = result.get("conflict_report")
    if cr:
        print(f"\n--- XUNG ĐỘT --- {cr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
