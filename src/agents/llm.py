"""LLM backends for AgentLaw agents."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV = PROJECT_ROOT / ".env"
if _ENV.exists():
    load_dotenv(_ENV, override=False)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120"))
OPENAI_MAX_RETRIES = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "1")))
OPENAI_SEND_REASONING = os.getenv("OPENAI_SEND_REASONING", "true").lower() in {"1", "true", "yes", "on"}


class LLMError(RuntimeError):
    """LLM request or configuration failure."""


class BaseLLM:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def set_purpose(self, purpose: str) -> None:
        return None


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible Chat Completions backend (ShopAIKey/Luna)."""

    name = "openai-compatible"

    _EFFORTS = {
        "decomposition_plan": os.getenv("LUNA_PLANNER_EFFORT", "none"),
        "sub_query_rewrite": os.getenv("LUNA_TOOL_EFFORT", "none"),
        "clarified_intent_rewrite": os.getenv("LUNA_TOOL_EFFORT", "none"),
        "synthesis": os.getenv("LUNA_SYNTHESIS_EFFORT", "low"),
        "synthesis_per_intent": os.getenv("LUNA_SYNTHESIS_EFFORT", "low"),
        "web_summary": os.getenv("LUNA_WEB_SUMMARY_EFFORT", "low"),
        "conflict_resolution": os.getenv("LUNA_CONFLICT_EFFORT", "low"),
    }
    _MAX_TOKENS = {
        "decomposition_plan": 2048,
        "sub_query_rewrite": 2048,
        "clarified_intent_rewrite": 2048,
        "synthesis": 4096,
        "synthesis_per_intent": 4096,
        "web_summary": 2048,
        "conflict_resolution": 4096,
    }

    def __init__(self, api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL,
                 model: str = OPENAI_MODEL, timeout: float = OPENAI_TIMEOUT,
                 max_retries: int = OPENAI_MAX_RETRIES) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self._purpose = "llm"
        self.last_usage: dict[str, Any] = {}

    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def set_purpose(self, purpose: str) -> None:
        self._purpose = purpose or "llm"

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._MAX_TOKENS.get(self._purpose, 2048),
            "temperature": 0,
        }
        if OPENAI_SEND_REASONING:
            payload["reasoning"] = {"effort": self._EFFORTS.get(self._purpose, "low")}
        return payload

    def complete(self, prompt: str) -> str:
        if not self.available():
            raise LLMError("OpenAI-compatible backend chưa được cấu hình đầy đủ")
        body = json.dumps(self._payload(prompt), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") or []
                message = choices[0].get("message") if choices else None
                text = (message or {}).get("content") or ""
                self.last_usage = data.get("usage") or {}
                if not text.strip():
                    raise LLMError("OpenAI-compatible backend trả về nội dung rỗng")
                return text.strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = LLMError(f"Lỗi OpenAI-compatible HTTP {exc.code}: {detail}")
                if exc.code in {400, 401, 402, 403, 404}:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(1.0 * (attempt + 1), 2.0))
        if isinstance(last_error, LLMError):
            raise last_error
        raise LLMError(f"Lỗi gọi OpenAI-compatible backend: {last_error}") from last_error


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL,
                 timeout: float = OLLAMA_TIMEOUT) -> None:
        self.model, self.base_url, self.timeout = model, base_url, timeout
        self._checked, self._ok, self._purpose = False, False, "llm"

    def available(self) -> bool:
        if self._checked:
            return self._ok
        self._checked = True
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                self._ok = response.status == 200
        except Exception:
            self._ok = False
        return self._ok

    def set_purpose(self, purpose: str) -> None:
        self._purpose = purpose

    def complete(self, prompt: str) -> str:
        if not self.available():
            raise LLMError(f"Ollama không khả dụng tại {self.base_url}")
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.2, "num_ctx": 8192,
                               "num_predict": 1024 if self._purpose in {"synthesis", "synthesis_per_intent", "conflict_resolution"} else 512}}
        if self.model.lower().startswith("qwen3"):
            payload["think"] = False
        try:
            req = urllib.request.Request(f"{self.base_url}/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                text = (json.loads(response.read().decode()).get("response") or "").strip()
            if not text:
                raise LLMError("Ollama trả về nội dung rỗng")
            return text
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Lỗi gọi Ollama: {exc}") from exc


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, model: str = GEMINI_MODEL) -> None:
        self.model = model

    def available(self) -> bool:
        try:
            from LLMClient import load_api_key
            load_api_key()
            return True
        except Exception:
            return False

    def complete(self, prompt: str) -> str:
        try:
            from LLMClient import generate_answer
            return generate_answer(prompt, temperature=0.2, max_output_tokens=2048)
        except Exception as exc:
            raise LLMError(f"Lỗi gọi Gemini: {exc}") from exc


class FallbackLLM(BaseLLM):
    name = "fallback"

    def __init__(self, backends: list[BaseLLM] | None = None) -> None:
        if backends is not None:
            self.backends = backends
        elif os.getenv("LLM_PROVIDER", "").strip().lower() in {"openai", "shopaikey", "openai-compatible"}:
            self.backends = [OpenAICompatibleLLM()]
        else:
            self.backends = [OllamaLLM()]
            if os.getenv("ENABLE_GEMINI_FALLBACK", "false").lower() in {"1", "true", "yes", "on"}:
                self.backends.append(GeminiLLM())

    def available(self) -> bool:
        return any(backend.available() for backend in self.backends)

    def set_purpose(self, purpose: str) -> None:
        for backend in self.backends:
            setter = getattr(backend, "set_purpose", None)
            if callable(setter):
                setter(purpose)

    def complete(self, prompt: str) -> str:
        errors: list[str] = []
        for backend in self.backends:
            if not backend.available():
                errors.append(f"{backend.name}: không khả dụng")
                continue
            try:
                return backend.complete(prompt)
            except LLMError as exc:
                errors.append(str(exc))
        raise LLMError("Không backend LLM nào trả lời được: " + "; ".join(errors))

    def active_backend(self) -> str | None:
        for backend in self.backends:
            if backend.available():
                return backend.name
        return None


def get_llm() -> FallbackLLM:
    return FallbackLLM()


__all__ = ["BaseLLM", "FallbackLLM", "GeminiLLM", "LLMError", "OllamaLLM", "OpenAICompatibleLLM", "get_llm"]
