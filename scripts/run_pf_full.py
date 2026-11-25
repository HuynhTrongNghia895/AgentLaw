"""P-F — (Tùy chọn) Bộ 65 câu · chạy MỘT LẺN (user).

Phạm vi P-F (chốt lại): **CHỈ bộ 65 câu** (retrieval thuần + OOD, NoLLM).
2 test `--runslow` (test_graph/test_retrieval) KHÔNG còn dùng — BỎ.
Cần môi trường: BGE embed/rerank model ~4.5GB (cache sẵn) + FAISS;
Không cần Neo4j (NoLLM path dùng pipeline retrieval nội bộ).

Chạy (Windows PowerShell, từ root dự án):
    .venv\\Scripts\\python scripts/run_pf_full.py
    (bằng .venv\\Scripts\\python evaluation/EvaluatePipeline.py)

Tùy chọn (không thuộc P-F): benchmark Qwen thật S1–S9+OOD_10+REF_12
chạy riêng bằng `evaluation/runner_benchmark.py` (~15-25 phút).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(title: str, cmd: list[str]) -> int:
    print(f"\n{'=' * 78}\n### {title}\n    $ {' '.join(cmd)}\n{'=' * 78}",
          flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    print(f"[P-F] {title} -> exit {rc}", flush=True)
    return rc


def main() -> int:
    # Bộ 65 câu (NoLLM retrieval thuần + OOD).
    rc = _run(
        "P-F Bộ 65 câu (retrieval NoLLM + OOD)",
        [PY, "evaluation/EvaluatePipeline.py"],
    )
    print(
        f"\n[P-F] Xong: 65-cau=exit{rc}. "
        "(Cần BGE model cache + FAISS; không cần Neo4j. "
        "Benchmark Qwen thật (tuỳ chọn) chạy riêng bằng "
        "evaluation/runner_benchmark.py.)"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
