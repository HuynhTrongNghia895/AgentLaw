from agents import llm


def test_default_ollama_model_is_qwen3_8b_unless_explicitly_overridden():
    expected = __import__("os").environ.get("OLLAMA_MODEL") or "qwen3:8b"
    assert llm.OLLAMA_MODEL == expected


def test_gemini_fallback_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ENABLE_GEMINI_FALLBACK", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=True)
    local_only = llm.FallbackLLM()
    assert [backend.name for backend in local_only.backends] == ["ollama"]

    monkeypatch.setenv("ENABLE_GEMINI_FALLBACK", "true")
    opt_in = llm.FallbackLLM()
    assert [backend.name for backend in opt_in.backends] == ["ollama", "gemini"]


def test_openai_compatible_selected_without_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "shopaikey")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    selected = llm.FallbackLLM()
    assert [backend.name for backend in selected.backends] == ["openai-compatible"]
    assert isinstance(selected.backends[0], llm.OpenAICompatibleLLM)


def test_openai_compatible_payload_uses_purpose_limits_and_effort(monkeypatch):
    monkeypatch.setenv("LUNA_CONFLICT_EFFORT", "low")
    backend = llm.OpenAICompatibleLLM(api_key="x", model="gpt-6-luna")
    backend.set_purpose("conflict_resolution")
    payload = backend._payload("question")
    assert payload["model"] == "gpt-6-luna"
    assert payload["max_tokens"] == 4096
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["messages"] == [{"role": "user", "content": "question"}]


def test_fallback_forwards_purpose_to_ollama_backend():
    class Backend:
        name = "stub"

        def __init__(self):
            self.purpose = ""

        def available(self):
            return True

        def set_purpose(self, purpose):
            self.purpose = purpose

        def complete(self, _prompt):
            return "ok"

    backend = Backend()
    fallback = llm.FallbackLLM([backend])
    fallback.set_purpose("synthesis_per_intent")
    assert fallback.complete("x") == "ok"
    assert backend.purpose == "synthesis_per_intent"
