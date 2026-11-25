"""TraceRecorder + TracedLLM (Giai đoạn 2) — quan sát pipeline thật.

Nguyên tắc thiết kế (đã duyệt):
- Tracing phải quan sát lần gọi LLM THẬT (Qwen/ollama), KHÔNG tạo thêm
  LLM call chỉ để ghi log.
- Lỗi của recorder không được lan ra pipeline, không che exception LLM.
- Khi `tracer=None` (mặc định), không tạo wrapper, không phát span,
  zero overhead — pipeline chạy y hệt trước khi có tracing.

5 trạng thái SPAN (không gộp):
- OK          : layer chạy, có kết quả (≠ rỗng).
- SKIPPED     : chủ động không chạy (có lý do ghi rõ).
- EMPTY       : chạy thật, kết quả = 0 (không lỗi).
- FAILED      : chạy và bị lỗi thật (exception) hoặc bị chặn.
- INSUFFICIENT: chạy OK nhưng bằng chứng chưa đủ để trả (riêng coverage).

Không thay đổi prompt, thứ tự gọi, số lần gọi LLM hay logic nghiệp vụ
khi tracing bật.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

from .llm import BaseLLM

# 5 trạng thái span (không gộp).
OK = "OK"
SKIPPED = "SKIPPED"
EMPTY = "EMPTY"
FAILED = "FAILED"
INSUFFICIENT = "INSUFFICIENT"


@dataclass
class LLMCall:
    """1 lần gọi LLM (thành công hoặc thất bại) được quan sát."""

    purpose: str
    in_chars: int
    out_chars: int = 0
    duration_ms: int = 0
    status: str = "ok"          # "ok" | "error"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "in_chars": self.in_chars,
            "out_chars": self.out_chars,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class Span:
    """1 tầng pipeline (1 intent). State ∈ {OK,SKIPPED,EMPTY,FAILED,INSUFFICIENT}."""

    layer: str
    intent: str | None = None
    turn: int = 0
    state: str = OK
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "intent": self.intent,
            "turn": self.turn,
            "state": self.state,
            "reason": self.reason,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }


_seq = itertools.count(1)


class TraceRecorder:
    """Ghi spans + LLM calls theo trace_id xuyên suốt các lượt (multi-turn).

    Tạo 1 lần mỗi scenario/conversation; ID giữ nguyên qua các lượt
    clarification (không tạo ID mới mỗi `.run()`, không dùng chung giữa
    các scenario).
    """

    def __init__(self, trace_id: str | None = None, purpose: str = "agent") -> None:
        if trace_id is None:
            trace_id = f"trc_{int(time.time() * 1000)}_{next(_seq)}"
        self.trace_id = trace_id
        self.purpose = purpose
        self.spans: list[Span] = []
        self.llm_calls: list[LLMCall] = []
        self.errors: list[str] = []      # lỗi nội bộ của recorder (không lan ra)
        self._turn = 0
        self._intent: str | None = None

    # -- Lifecycle (multi-turn) --------------------------------------

    def start_turn(self, turn: int) -> None:
        """Đánh dấu lượt hội thoại; span kế thừa turn này."""
        self._turn = turn

    def set_intent(self, intent: str | None) -> None:
        """Span thuộc intent nào (None = pipeline chung, không phải intent con)."""
        self._intent = intent

    # -- Ghi span ------------------------------------------------------

    def emit(
        self,
        layer: str,
        state: str = OK,
        reason: str = "",
        data: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Phát 1 span. LỖI CỦA RECORDER (nếu có) được nuốt riêng,
        KHÔNG lan ra pipeline (yêu cầu #5)."""
        try:
            self.spans.append(Span(
                layer=layer,
                intent=self._intent,
                turn=self._turn,
                state=state,
                reason=reason,
                data=data or {},
                duration_ms=duration_ms,
            ))
        except Exception as exc:  # noqa: BLE001 — không bao giờ phá pipeline
            self.errors.append(f"span {layer}: {exc}")

    # -- Kết xuất ------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        by_layer: dict[str, list[dict[str, Any]]] = {}
        for s in self.spans:
            d = s.to_dict()
            by_layer.setdefault(s.layer, []).append(d)
        return {
            "trace_id": self.trace_id,
            "n_turns": self._turn + 1 if self._turn >= 0 else 0,
            "n_llm_calls": len(self.llm_calls),
            "n_spans": len(self.spans),
            "recorder_errors": list(self.errors),
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "spans_by_layer": by_layer,
        }


class TracedLLM(BaseLLM):
    """Bọc `BaseLLM` để ghi lại MỌI lần gọi thật (thành công + thất bại).

    - Forward ĐỦ interface BaseLLM: `available()`, `complete()`, class
      attr `name` (decomposer nhận diện stub qua `llm.name in ("none","")`),
      và `active_backend()` (nếu inner có).
    - Không sửa prompt, không đổi kết quả: `complete()` gọi inner, log,
      trả nguyên giá trị inner.
    - Inner LỖI: ghi 1 LLMCall (status="error") rồi **nâng exception
      NGUYÊN** — không che, không nuốt.
    - Lỗi recorder không làm thay đổi kết quả hay che lỗi LLM.
    """

    def __init__(self, inner: BaseLLM, recorder: TraceRecorder,
                 purpose: str = "llm") -> None:
        self._inner = inner
        self._rec = recorder
        self._purpose = purpose
        self._purpose_override: str | None = None
        # Forward attribute (không hardcode "base"): rewriter check
        # `llm.name in ("none","")` với NoLLM — phải giữ đúng giá trị inner.
        self.name = getattr(inner, "name", "traced")  # type: ignore[assignment]

    # -- Interface BaseLLM (forward đầy đủ) ----------------------------

    def available(self) -> bool:
        return self._inner.available()

    def active_backend(self) -> Any:  # type: ignore[no-redef]
        """Forward CHỈ khi inner có (FallbackLLM). Backend đơn (OllamaLLM,
        GeminiLLM, stub) KHÔNG có method này -> không được hardcode forward
        (bug audit P-A: forward vô điều kiện sẽ AttributeError khi gọi)."""
        fn = getattr(self._inner, "active_backend", None)
        if fn is None:
            raise AttributeError(
                f"inner LLM {type(self._inner).__name__} không có "
                "active_backend() (chỉ FallbackLLM có)"
            )
        return fn()

    def set_purpose(self, purpose: str) -> None:
        """Override no-op của BaseLLM: nhớ nhãn cho LLM call kế tiếp
        (mặc định "llm"). Không ảnh hưởng pipeline (tracer-only)."""
        self._purpose_override = purpose
        inner_setter = getattr(self._inner, "set_purpose", None)
        if callable(inner_setter):
            try:
                inner_setter(purpose)
            except Exception:
                pass

    def complete(self, prompt: str) -> str:
        rec = self._rec
        t0 = time.perf_counter()
        purpose = getattr(self, "_purpose_override", None) or self._purpose
        self._purpose_override = None  # reset sau khi đọc (không rò rỉ sang call sau)
        try:
            out = self._inner.complete(prompt)
        except Exception as exc:
            # Ghi thất bại (nếu được) rồi NÂNG exception NGUYÊN — không che.
            try:
                rec.llm_calls.append(LLMCall(
                    purpose=purpose,
                    in_chars=len(prompt or ""),
                    out_chars=0,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                ))
            except Exception as rec_exc:  # noqa: BLE001
                rec.errors.append(f"llm log: {rec_exc}")
            raise
        try:
            rec.llm_calls.append(LLMCall(
                purpose=purpose,
                in_chars=len(prompt or ""),
                out_chars=len(out or ""),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                status="ok",
            ))
        except Exception as rec_exc:  # noqa: BLE001 — lỗi recorder không
            # lan ra, không đổi kết quả, không che lỗi LLM.
            rec.errors.append(f"llm log: {rec_exc}")
        return out


def coverage_state(gap_meta: dict[str, Any]) -> tuple[str, str]:
    """Ánh xạ meta của QueryDecomposer.fill -> (trạng thái span, lý do).

    Nguyên tắc fidelity (audit P-A): span không được KAI điều chưa xảy ra.
    - KHÔNG có "stop_reason" trong meta -> decomposer NO-OP (câu đơn,
      should_check=False, không LLM coverage) -> SKIPPED (chưa kiểm tra
      coverage), không phải INSUFFICIENT (bug fidelity đã phát hiện:
      S1/S5 Qwen thật trước fix gán INSUFFICIENT "unknown" cho đường
      no-op đơn ý).
    - "covered" -> OK (mọi ý đã đủ căn cứ, coverage đã CHẠY).
    - "max_rounds"/"budget_evidence" -> INSUFFICIENT (chạy thật, hết
      ngân sách, vẫn còn ý thiếu căn cứ).
    - "coverage_unparsed" -> FAILED (chạy thật nhưng parse LLM lỗi,
      dừng an toàn).
    """
    if "stop_reason" not in (gap_meta or {}):
        return SKIPPED, "câu đơn (lọc rẻ) — không kiểm tra coverage, không LLM"
    stop = str(gap_meta.get("stop_reason") or "")
    if stop == "covered":
        return OK, "mọi ý con đã đủ căn cứ (coverage đã chạy)"
    if stop in ("max_rounds", "budget_evidence"):
        return INSUFFICIENT, (
            f"hết ngân sách ({stop}), vẫn còn ý con thiếu căn cứ"
        )
    if stop == "coverage_unparsed":
        return FAILED, "dừng an toàn: không parse được JSON coverage của LLM"
    return INSUFFICIENT, f"chưa đủ căn cứ (stop_reason={stop or 'unknown'})"


__all__ = [
    "FAILED",
    "INSUFFICIENT",
    "LLMCall",
    "OK",
    "SKIPPED",
    "EMPTY",
    "Span",
    "TracedLLM",
    "TraceRecorder",
    "coverage_state",
]
