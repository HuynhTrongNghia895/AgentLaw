"""P-A: chạy traced path với Qwen THẬT (Ollama qwen3:8b) + điều tra
NO-EVIDENCE của S5/S7/S8 trong baseline stub.

Với MỖI case:
- tracer bật (diagnostic mode), trace_id = scn-<id>,
- in answer (tắt thinking + trunc), route, evidence, spans,
- đối chiếu giả thuyết "NO-EVIDENCE là OOD gate chặn (do stub refiner
  thay câu bằng 'vấn đề pháp lý' + anchor S1 pin được nên không bị
  chặn) bằng cách xem có span ood_gate không.

Qwen: 30-90s/call (S5/S7 có thể 2 call). Chạy nền.
"""

from __future__ import annotations

import io
import json
import re
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
    PROJECT_ROOT / "evaluation",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
)

CASES = [
    ("S1", "Khoản 8 Điều 4 Luật Quản lý thuế 108/2025/QH15 quy định gì về kỳ tính thuế?"),
    ("S5", "Theo Luật Quản lý thuế 108/2025/QH15: người nộp thuế là ai và kỳ tính thuế là gì?"),
    ("S7", "Theo Luật Quản lý thuế 108/2025/QH15: người nộp thuế là ai, và chạy đèn đỏ bị phạt bao nhiêu?"),
    ("S8", "Nghị định 168/2025 về đăng ký doanh nghiệp còn hiệu lực không, và đã bị văn bản nào sửa đổi?"),
]

THINK = re.compile(r"^\s*<\|think\|>.*", re.DOTALL)


def _clean(text: str) -> str:
    m = THINK.search(text or "")
    return (text[m.end():] if m else text).strip()


def main() -> int:
    import argparse
    import urllib.request

    # Chỉ chạy subset case (--ids S7 S8) để tiết kiệm Qwen khi verify.
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="chỉ chạy case nào (vd: 'S7 S8')")
    ap.add_argument("--out", default="", help="đường dẫn JSON output")
    args, _ = ap.parse_known_args()

    cases = CASES
    if args.ids:
        ids = args.ids.split()
        cases = [c for c in CASES if c[0] in ids]
        if not cases:
            print(f"[!] không có case nào khớp --ids {ids} trong "
                  f"{[c[0] for c in CASES]}")
            return 1

    # Xác nhận Ollama (P-A requirement: confirm Qwen3:8b).
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=5
        ) as resp:
            tags = json.loads(resp.read())
            names = [m.get("name") for m in tags.get("models", [])]
            print(f"[ollama] models = {names}")
    except Exception as exc:
        print(f"[ollama] KHÔNG GỌI ĐƯỢC: {exc}")
        return 1

    from agents.LegalRetrieval import LegalRetrievalAgent
    from agents.Orchestrator import Orchestrator
    from agents.tracing import TraceRecorder

    rec = TraceRecorder(trace_id="diag-real")
    agent = LegalRetrievalAgent(tracer=rec)
    try:
        active = agent.llm.active_backend()
    except AttributeError:
        active = f"(inner {getattr(agent.llm, '_inner', None) and type(agent.llm._inner).__name__} không có active_backend)"
    print(f"[llm] backend = {agent.llm.name} (active: {active})")

    results = []
    for cid, q in CASES:
        rec.spans = []
        rec.llm_calls = []
        t0 = time.perf_counter()
        case_err = ""
        try:
            res = Orchestrator(legal=agent, use_web=False, tracer=rec).run(q)
            dt = time.perf_counter() - t0
        except Exception as exc:
            # 1 case crash không chặn các case khác — ghi lỗi, tiếp tục.
            dt = time.perf_counter() - t0
            res = {"route": "", "evidence": [], "answer": ""}
            case_err = f"{type(exc).__name__}: {exc}"
            print(f"[{cid}] EXCEPTION ({dt:.0f}s): {case_err}")
            results.append({
                "id": cid, "route": "", "n_evidence": 0, "evidence_node_ids": [],
                "error": case_err, "elapsed_s": round(dt, 1),
            })
            continue
        ev = [str(e.get("node_id") or "") for e in (res.get("evidence") or [])]
        ans = _clean(res.get("answer") or "")
        span_dicts = [s.to_dict() for s in rec.spans]
        layers = sorted({s["layer"] for s in span_dicts})
        ood = [s for s in span_dicts if s["layer"] == "ood_gate"]
        n_ev = len(ev)
        print(
            f"\n=== {cid} ({dt:.0f}s) route={res.get('route')} ev={n_ev} "
            f"ood_gate={'CÓ' if ood else 'không'} layers={layers}"
        )
        print(f"  evidence[:4]: {ev[:4]}")
        print(f"  answer: {ans[:300]}{'...' if len(ans) > 300 else ''}")
        for s in span_dicts:
            if s["layer"] in ("ood_gate", "coverage", "rerank", "merge", "faiss"):
                print(
                    f"  span {s['layer']} [{s['state']}] "
                    f"{s['reason']} {s['data']}"
                )
        for c in [x.to_dict() for x in rec.llm_calls]:
            print(f"  llm_call {c['purpose']} {c['in_chars']}in/"
                  f"{c['out_chars']}out {c['status']}"
                  + (f" ERR:{c['error'][:120]}" if c["status"] == "error" else ""))
        call_dicts = [x.to_dict() for x in rec.llm_calls]
        results.append({
            "id": cid, "route": res.get("route"), "n_evidence": n_ev,
            "evidence_node_ids": ev,
            "ood_gate": bool(ood), "llm_calls": len(rec.llm_calls),
            "llm_status": [c["status"] for c in call_dicts],
            "llm_calls_detail": call_dicts,
            "spans": span_dicts,
            "answer": ans, "answer_head": ans[:200],
            "elapsed_s": round(dt, 1),
        })

    out = Path(args.out) if args.out else PROJECT_ROOT / "evaluation" / "results" / "trace_diagnostic_pA.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"cases": results, "trace_id": rec.trace_id},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[diagnostic] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
