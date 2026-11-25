"""Entry CLI cho KG 4 tầng pháp lý (mới, thay các lệnh seed GĐ3).

Cách dùng (từ gốc repo):
    .venv\\Scripts\\python scripts\\build_kg.py smoke   # 3 vb thử + 4 case time-travel
    .venv\\Scripts\\python scripts\\build_kg.py all     # seed toàn bộ 22 vb
    .venv\\Scripts\\python scripts\\build_kg.py check   # chỉ kiểm temporal (không seed)

Mỗi lần chạy in report JSON ra data\\logs\\kg_build_<UTC>.json
(số node/cạnh thật + kết quả 4 case + danh sách missing).
Chạy lại bao nhiêu lần cũng được (MERGE idempotent, wipe phần kind4tier trước).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.kg.build import check_temporal, seed_all, smoke  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"


def _report(rep: dict, cmd: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rep["command"] = cmd
    rep["run_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = LOG_DIR / f"kg_build_{datetime.now(timezone.utc):%Y%m%dT%H%M}.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\nreport -> {path.relative_to(PROJECT_ROOT)}")

    fails = [k for k, v in rep.get("temporal", {}).items() if v.get("pass") is False]
    if fails:
        print(f"FAIL: {fails}")
        sys.exit(1)


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "smoke":
        _report(smoke(), "smoke")
    elif cmd == "all":
        _report(seed_all(), "all")
    elif cmd == "check":
        _report({"temporal": check_temporal()}, "check")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
