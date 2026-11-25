"""So baseline_pre.json vs baseline_post.json (regression tracer tích hợp).

Chỉ so các key HÀNH VI: route, answer, follow_up, choices,
evidence_node_ids, metadata, llm_calls, prompt_hashes.
KHÔNG so elapsed_s (không so sánh latency giữa 2 chế độ).

Exit 0 = không khác biệt; exit 1 = có diff (in ra).

Cách chạy:
    .venv\\Scripts\\python scripts\\diff_baselines.py
        -> so baseline_pre.json (trước sửa) vs baseline_post.json (sau sửa,
           tracer=None)
    .venv\\Scripts\\python scripts\\diff_baselines.py <left.json> <right.json>
        -> so 2 file tuỳ ý (vd baseline_post.json vs
           baseline_post_traced.json = tracer ON vs tracer OFF)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEFT = ROOT / "evaluation" / "results" / "baseline_pre.json"
DEFAULT_RIGHT = ROOT / "evaluation" / "results" / "baseline_post.json"

BEHAVIOR_KEYS = [
    "route", "answer", "follow_up", "choices", "evidence_node_ids",
    "metadata", "llm_calls", "prompt_hashes",
]


def main() -> int:
    if len(sys.argv) >= 3:
        left_path, right_path = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        left_path, right_path = DEFAULT_LEFT, DEFAULT_RIGHT
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    diffs = 0
    for cid in left:
        p = {k: left[cid][k] for k in BEHAVIOR_KEYS if k in left[cid]}
        q = {k: right.get(cid, {}).get(k) for k in p}
        for k, v in p.items():
            if q[k] != v:
                diffs += 1
                print(f"DIFF {cid}.{k}:\n  left : {v!r}\n  right: {q[k]!r}")
    extra = set(right) - set(left)
    if extra:
        print(f"WARNING: case mới trong right: {sorted(extra)}")
    label = f"{left_path.name} -> {right_path.name}"
    if diffs == 0 and not extra:
        print(
            f"OK: {len(left)} case ({', '.join(BEHAVIOR_KEYS)}) "
            f"không đổi {label}."
        )
        return 0
    print(f"FAILED: {diffs} khác biệt hành vi {label}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
