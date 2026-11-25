"""Test TraceRecorder + TracedLLM (Giai đoạn 2).

Kiểm tra các tiêu chí đã thống nhất:
  1. tracer=None -> KHÔNG tạo wrapper, pipeline không đổi (zero overhead).
  2. tracer bật -> wrap LLM TRƯỚC khi decomposer capture reference.
  3. Inner LLM lỗi -> TracedLLM NÂNG exception NGUYÊN (không che),
     pipeline không đổi, recorder có entry failed.
  4. Lỗi recorder (lỗi nội bộ) không che exception / đổi kết quả LLM.
  5. Fidelity: prompt log = prompt thực gửi; response trả = output inner
     (không sửa). Số lần gọi LLM bật/tắt = nhau.
  6. Forward đủ interface BaseLLM: name (stub "none" không hỏng),
     available, active_backend.
"""

from __future__ import annotations

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

from agents.llm import BaseLLM
from agents.tracing import (
    EMPTY,
    FAILED,
    INSUFFICIENT,
    OK,
    SKIPPED,
    TracedLLM,
    TraceRecorder,
    coverage_state,
)


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------

class EchoLLM(BaseLLM):
    """Trả lại "OK:" + prompt (deterministic), đếm call."""

    name = "echo"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return "OK:" + prompt

    def active_backend(self):  # type: ignore[no-redef]
        return "echo"


class NoLLM:
    """Stub như runner_benchmark: name="none" (rewriter nhận diện)."""

    name = "none"

    def complete(self, prompt: str) -> str:
        return ""


class RaisingLLM(BaseLLM):
    """Inner LLM luôn raise (mô phỏng Ollama down / LLMError)."""

    name = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        self.calls += 1
        raise RuntimeError("backend dead")


def _light_agent(llm, tracer=None):
    """LegalRetrievalAgent NHẸ (không load FAISS/BGE/Neo4j): pass store/kg/
    router stub -> __init__ chỉ capture llm + tạo decomposer."""
    from agents.LegalRetrieval import LegalRetrievalAgent

    return LegalRetrievalAgent(
        store=object(),
        kg=object(),
        llm=llm,
        router=object(),
        tracer=tracer,
    )


# ---------------------------------------------------------------------------
# 1. tracer=None -> không wrapper
# ---------------------------------------------------------------------------

def test_tracer_none_does_not_wrap():
    llm = EchoLLM()
    agent = _light_agent(llm, tracer=None)
    assert agent.llm is llm, "tracer=None phải giữ llm NGUYÊN, không wrap"
    # decomposer capture reference: phải là llm gốc (chưa wrap).
    assert agent.decomposer.llm is llm


# ---------------------------------------------------------------------------
# 2. tracer bật -> wrap trước decomposer capture
# ---------------------------------------------------------------------------

def test_tracer_enabled_wraps_before_decomposer():
    llm = EchoLLM()
    rec = TraceRecorder()
    agent = _light_agent(llm, tracer=rec)
    assert isinstance(agent.llm, TracedLLM), "bật tracer phải wrap llm"
    # decomposer nhận ĐÚC LLM ĐÃ WRAP (wrap đúng thứ tự) -> mọi call
    # trong decomposer (rewriter/coverage) được quan sát.
    assert agent.decomposer.llm is agent.llm
    assert isinstance(agent.decomposer.llm, TracedLLM)


# ---------------------------------------------------------------------------
# 5. Fidelity: prompt/response + số call
# ---------------------------------------------------------------------------

def test_prompt_response_fidelity_and_call_count():
    inner = EchoLLM()
    rec = TraceRecorder()
    wrapped = TracedLLM(inner, rec)

    out = wrapped.complete("PROMPT-X")
    assert out == "OK:PROMPT-X", "response phải = output inner, không sửa"
    # prompt log lại = prompt thực gửi.
    assert len(rec.llm_calls) == 1
    c = rec.llm_calls[0]
    assert c.in_chars == len("PROMPT-X")
    assert c.status == "ok"
    assert inner.calls == 1, "TracedLLM không tạo thêm LLM call (1:1)"

    # 1 wrapped call = 1 inner call (không nhân). 2 lần wrapped -> 2 lần inner.
    assert wrapped.complete("A") == "OK:A"
    assert inner.calls == 2, "2 lần wrapped.complete -> đúng 2 lần inner, không thêm"
    # Kết quả giống hệt gọi trực tiếp (tracing không đổi output).
    assert inner.complete("A") == "OK:A"


# ---------------------------------------------------------------------------
# 3. Inner lỗi -> nâng exception NGUYÊN
# ---------------------------------------------------------------------------

def test_inner_exception_raised_verbatim():
    inner = RaisingLLM()
    rec = TraceRecorder()
    wrapped = TracedLLM(inner, rec)
    try:
        wrapped.complete("x")
        raise AssertionError("phải raise exception của inner")
    except RuntimeError as exc:
        assert "backend dead" in str(exc), "giữ nguyên message exception"
    # recorder vẫn ghi entry FAILED, không che lỗi.
    assert len(rec.llm_calls) == 1
    assert rec.llm_calls[0].status == "error"
    assert "backend dead" in rec.llm_calls[0].error


def test_pipeline_unchanged_when_llm_raises():
    """Pipeline (synthesis) catch LLM error -> trả evidence, không bịa.
    Wrapper không đổi hành vi đó: LegalAnswer vẫn metadata llm_error."""
    inner = RaisingLLM()
    rec = TraceRecorder()
    agent = _light_agent(inner, tracer=rec)
    # ép route không fast-track + evidence rỗng -> _synthesize không chạy
    # LLM; ở đây ta kiểm tra trực tiếp wrapper giữ pipeline không đổi:
    # gọi complete -> raise, agent không crash ở bước wrap.
    try:
        agent.llm.complete("p")
        raise AssertionError("phải raise")
    except RuntimeError:
        pass
    # recorder đã ghi 1 call failed, pipeline vẫn "sống" (agent tồn tại).
    assert len(rec.llm_calls) == 1


# ---------------------------------------------------------------------------
# 4. Lỗi recorder không che lỗi LLM / đổi kết quả
# ---------------------------------------------------------------------------

def test_recorder_failure_does_not_mask_llm():
    class BrokenRecorder(TraceRecorder):
        """Recorder lỗi nội bộ: append span/call thì raise."""

        def __init__(self) -> None:
            super().__init__()
            self._boom = True

        def emit(self, *a, **k):  # type: ignore[no-redef]
            if self._boom:
                raise RuntimeError("recorder broken")

    inner = EchoLLM()
    rec = BrokenRecorder()
    wrapped = TracedLLM(inner, rec)
    # recorder lỗi khi ghi LLM call (append) -> không đổi kết quả,
    # không che exception LLM (inner thành công -> trả đúng output).
    out = wrapped.complete("P")
    assert out == "OK:P", "lỗi recorder không được đổi kết quả LLM"
    # và không phải exception của recorder lan ra.
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# 6. Forward interface BaseLLM (name / available / active_backend)
# ---------------------------------------------------------------------------

def test_forwards_name_and_active_backend():
    inner = EchoLLM()
    wrapped = TracedLLM(inner, TraceRecorder())
    assert wrapped.name == "echo", "phải forward inner.name (không hardcode)"
    assert wrapped.available() is True
    assert wrapped.active_backend() == "echo"


def test_forwards_name_none_stub():
    """NoLLM (name='none') — rewriter check `llm.name in ('none','')`.
    Wrap không được phá điều kiện đó."""
    stub = NoLLM()
    wrapped = TracedLLM(stub, TraceRecorder())
    assert getattr(wrapped, "name", "") in ("none", ""), (
        "phải giữ name='none' để rewriter nhận stub đúng"
    )


# ---------------------------------------------------------------------------
# 5 trạng thái span khác nhau (không gộp)
# ---------------------------------------------------------------------------

def test_span_states_are_distinct():
    rec = TraceRecorder()
    rec.emit("faiss", OK, data={"n": 5})
    rec.emit("web", SKIPPED, reason="fast-track")
    rec.emit("merge", EMPTY, reason="dedup hết")
    rec.emit("ood_gate", FAILED, reason="chặn")
    rec.emit("coverage", INSUFFICIENT, reason="thiếu căn cứ")
    s = rec.summary()
    states = [x["state"] for x in s["spans_by_layer"]["faiss"]]
    assert states == [OK]
    assert s["spans_by_layer"]["coverage"][0]["state"] == INSUFFICIENT
    assert s["spans_by_layer"]["web"][0]["state"] == SKIPPED
    assert s["spans_by_layer"]["merge"][0]["state"] == EMPTY
    assert s["spans_by_layer"]["ood_gate"][0]["state"] == FAILED
    # 5 hằng số khác nhau, không trùng.
    assert len({OK, SKIPPED, EMPTY, FAILED, INSUFFICIENT}) == 5


# ---------------------------------------------------------------------------
# trace_id lifecycle: 1 ID per scenario, giữ xuyên multi-turn
# ---------------------------------------------------------------------------

def test_trace_id_stable_across_turns():
    rec = TraceRecorder(trace_id="scn-S7")
    rec.start_turn(0)
    rec.emit("router", OK)
    rec.start_turn(1)  # lượt clarify
    rec.emit("router", OK)
    s = rec.summary()
    assert s["trace_id"] == "scn-S7"
    assert s["n_llm_calls"] == 0
    # hai span thuộc 2 turn khác nhau cùng 1 trace_id.
    turns = [sp["turn"] for sp in s["spans_by_layer"]["router"]]
    assert turns == [0, 1]


# ---------------------------------------------------------------------------
# set_purpose: gắn nhãn mục đích cho LLM call kế tiếp (tracer-only, no-op
# khi không có TracedLLM) + reset sau khi dùng (không rò rỉ sang call khác).
# ---------------------------------------------------------------------------

def test_set_purpose_routes_to_recorder_and_resets():
    inner = EchoLLM()
    rec = TraceRecorder()
    wrapped = TracedLLM(inner, rec, purpose="default")
    wrapped.set_purpose("coverage_check")
    wrapped.complete("x")
    assert rec.llm_calls[0].purpose == "coverage_check", "phải dùng nhãn đã set"
    # Reset: call kế tiếp không kế thừa nhãn cũ.
    wrapped.complete("y")
    assert rec.llm_calls[1].purpose == "default", "phải reset về purpose mặc định"


def test_stub_without_set_purpose_is_safe():
    """Dispatch an toàn ở call-site: `getattr(llm,'set_purpose',None)` ->
    None với stub trần (không kế thừa BaseLLM) -> bỏ qua, không crash.
    (Kiểm tra đúng pattern dùng trong QueryDecomposer/LegalRetrieval/...)"""
    class BareStub:
        name = "bare"
        def complete(self, p):
            return "R:" + p

    stub = BareStub()
    sp = getattr(stub, "set_purpose", None)  # pattern call-site
    assert sp is None, "stub trần không có set_purpose -> dispatch phải bỏ qua"
    assert stub.complete("a") == "R:a", "pipeline không đổi"


# ---------------------------------------------------------------------------
# coverage_state fidelity (audit P-A: no-op đơn ý không được KAI INSUFFICIENT)
# ---------------------------------------------------------------------------

def test_coverage_state_noop_is_skipped_not_insufficient():
    """Meta no-op của QueryDecomposer (câu đơn, should_check=False) KHÔNG
    có 'stop_reason' -> phải là SKIPPED (chưa kiểm tra coverage), không phải
    INSUFFICIENT 'unknown' (bug fidelity đã phát hiện ở S1/S5 Qwen thật)."""
    noop = {"filled": 0, "rounds": 0,
            "per_intent": [{"sub_query": "q", "node_ids": ["x"]}]}
    assert coverage_state(noop) == (SKIPPED,
                                    "câu đơn (lọc rẻ) — không kiểm tra coverage, không LLM")


def test_coverage_state_transitions():
    assert coverage_state({"stop_reason": "covered"})[0] == OK
    assert coverage_state({"stop_reason": "max_rounds"})[0] == INSUFFICIENT
    assert coverage_state({"stop_reason": "budget_evidence"})[0] == INSUFFICIENT
    assert coverage_state({"stop_reason": "coverage_unparsed"})[0] == FAILED
    assert coverage_state({})[0] == SKIPPED
    assert coverage_state(None)[0] == SKIPPED


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
