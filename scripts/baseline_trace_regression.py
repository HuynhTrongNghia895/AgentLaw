"""Baseline regression cho tích hợp TraceRecorder (Giai đoạn 2).

Chạy 4 kịch bản đại diện (S1 fast-track, S5 nội dung đa ý, S7 đa ý + OOD,
S8 quan hệ) qua Orchestrator thật (FAISS + B1 + reranker thật) với **stub
LLM deterministic** (không Qwen, không kết quả ngẫu nhiên làm golden).

Snapshot lưu: route, answer, follow_up, choices, evidence node_ids,
metadata, số lần gọi LLM, hash prompt theo thứ tự.

Trước khi sửa code:  .venv\\Scripts\\python scripts\\baseline_trace_regression.py --save evaluation\\results\\baseline_pre.json
Sau khi sửa:         .venv\\Scripts\\python scripts\\baseline_trace_regression.py --save evaluation\\results\\baseline_post.json
Rồi so 2 file (chỉ các key hành vi, KHÔNG so elapsed_s).

Lưu ý: kết quả phụ thuộc trạng thái Neo4j (B1) và reranker (cache model)
TẠI LÚC CHẠY. Chạy 2 snapshot trong cùng điều kiện (cùng phiên, cùng
trạng thái dịch vụ) để diff có nghĩa.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "utils",
    PROJECT_ROOT / "src" / "retrieval",
    PROJECT_ROOT / "src" / "graph",
    PROJECT_ROOT / "src" / "agents",
    PROJECT_ROOT / "src" / "graph" / "kg",
    PROJECT_ROOT / "evaluation",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
)

CASES = [
    {
        "id": "S1",
        "question": (
            "Khoản 8 Điều 4 Luật Quản lý thuế 108/2025/QH15 quy định gì "
            "về kỳ tính thuế?"
        ),
    },
    {
        "id": "S5",
        "question": (
            "Theo Luật Quản lý thuế 108/2025/QH15: người nộp thuế là ai "
            "và kỳ tính thuế là gì?"
        ),
    },
    {
        "id": "S7",
        "question": (
            "Theo Luật Quản lý thuế 108/2025/QH15: người nộp thuế là ai, "
            "và chạy đèn đỏ bị phạt bao nhiêu?"
        ),
    },
    {
        "id": "S8",
        "question": (
            "Nghị định 168/2025 về đăng ký doanh nghiệp còn hiệu lực "
            "không, và đã bị văn bản nào sửa đổi?"
        ),
    },
]


class StubLLM:
    """LLM deterministic: không kết quả ngẫu nhiên, số call đếm được.

    Trả lời theo loại prompt (refiner / coverage / synthesis) — cùng input
    luôn cùng output, nên snapshot có thể so sánh lại sau tích hợp.
    `calls` / `prompt_hashes` là instance-attr (đếm trên đúng instance pass
    vào agent).
    """

    name = "stub"

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_hashes: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompt_hashes.append(
            hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:16]
        )
        p = prompt or ""
        if "=== JSON ===" in p and "VĂN BẢN CÓ SẴN" in p:  # refiner
            # Trả "vague" + câu chuẩn hóa RÁC -> pipeline NO-EVIDENCE.
            # Đây là HÀNH VI STUB (không phải bug pipeline) — CHỖ GIỮ
            # NGUYÊN để regression baseline (pre/post/traced, cùng stub)
            # so được 0-diff. "Cứu" NO-EVIDENCE làm bằng Qwen thật
            # (scripts/trace_diagnostic.py), không bằng sửa stub.
            # (audit P-A: S5/S7/S8 NO-EVIDENCE trong baseline = artifact
            #  stub refiner, KHÔNG phải lỗi hệ thống.)
            return (
                '{"kind":"vague","question":"vấn đề pháp lý",'
                '"legal_group":null,"intent":"hieu_luc",'
                '"missing":["văn bản"],"choices":[],'
                '"clarifying_question":null,"confidence":0.5}'
            )
        if "=== CĂN CỨ ĐÃ TÌM ĐƯỢC ===" in p:  # coverage (decomposer)
            return json.dumps({
                "intents": [
                    "người nộp thuế là ai theo 108/2025/QH15",
                    "chạy đèn đỏ bị phạt bao nhiêu",
                ],
                "uncovered": ["người nộp thuế là ai theo 108/2025/QH15"],
                "clarify": None,
                "confidence": 0.9,
            })
        return "STUB-ANSWER: câu trả lời thử nghiệm (không Qwen)."


def main() -> int:
    save_to: str | None = None
    if "--save" in sys.argv:
        save_to = sys.argv[sys.argv.index("--save") + 1]
    tracer_on = "--tracer" in sys.argv  # bật TraceRecorder (chế độ diagnostic)

    from agents.LegalRetrieval import LegalRetrievalAgent
    from agents.Orchestrator import Orchestrator

    tracer = None
    if tracer_on:
        from agents.tracing import TraceRecorder

        tracer = TraceRecorder(trace_id="regression-on")

    t0 = time.perf_counter()
    stub = StubLLM()
    agent = LegalRetrievalAgent(llm=stub, tracer=tracer)
    orch = Orchestrator(legal=agent, use_web=False)
    print(f"[init agent+orchestrator] {time.perf_counter() - t0:.0f}s"
          + (" [tracer ON]" if tracer_on else ""))

    out: dict[str, dict] = {}
    for c in CASES:
        stub.calls = 0
        stub.prompt_hashes = []
        if tracer is not None:
            tracer.spans = []  # 1 trace sạch per case (regression)
            tracer.llm_calls = []
        t1 = time.perf_counter()
        res = orch.run(c["question"])
        elapsed = time.perf_counter() - t1
        ev = [str(e.get("node_id") or "") for e in (res.get("evidence") or [])]
        entry = {
            "route": res.get("route", ""),
            "answer": res.get("answer") or "",
            "follow_up": res.get("follow_up") or "",
            "choices": res.get("choices") or [],
            "evidence_node_ids": ev,
            "metadata": res.get("metadata") or {},
            "llm_calls": stub.calls,
            "prompt_hashes": stub.prompt_hashes,
            "elapsed_s": round(elapsed, 1),
        }
        out[c["id"]] = entry
        if tracer is not None:
            # Artifact chẩn đoán (KHÔNG nằm trong BEHAVIOR_KEYS của
            # diff_baselines.py): chứng minh tracer thật sự ghi được.
            out[c["id"]]["trace_summary"] = tracer.summary()
        print(
            f"{c['id']}: route={entry['route']} llm_calls={entry['llm_calls']} "
            f"ev={ev[:3]} ({elapsed:.0f}s)"
        )

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[baseline] saved -> {save_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
