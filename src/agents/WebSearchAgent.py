"""WebSearch Agent (P7) — bổ sung nguồn pháp luật từ web.

Ràng buộc đã chốt (PROJECT_CONTEXT §4, quyết định #9, #12):
  - **Domain-restriction bắt buộc**: chỉ `vbpl.vn`, `chinhphu.vn`,
    `thuvienphapluat.vn`.
  - **Chỉ gọi khi fast-track không đủ** (quyết định #12) — orchestrator
    quyết định, agent này chỉ thực thi.
  - Nội dung web là **DỮ LIỆU KHÔNG TIN CẬY** — không được gọi LLM với
    nội dung này như mệnh lệnh.
  - Kiểm tra tình trạng hiệu lực của văn bản tìm được trước khi trả
    (agent LLM tóm tắt; nếu không rõ -> ghi "nguồn không nêu rõ").

Tavily chưa cài trong môi trường này → module degrade gracefully:
`search()` trả về danh sách rỗng + `available=False`, orchestrator biết
và bỏ qua nhánh web (không lỗi, không chết).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV = PROJECT_ROOT / ".env"
if _ENV.exists():
    load_dotenv(_ENV)

# Domain-restriction bắt buộc (quyết định #9)
ALLOWED_DOMAINS: tuple[str, ...] = (
    "vbpl.vn",
    "chinhphu.vn",
    "thuvienphapluat.vn",
)

MAX_RESULTS = 5
MAX_CONTENT_CHARS = 4000
MAX_SOURCE_BYTES = 2_000_000
MAX_SOURCE_TEXT_CHARS = 150_000


@dataclass
class WebSource:
    """1 nguồn web đã lấy."""

    title: str = ""
    url: str = ""
    content: str = ""
    date: str = ""
    score: float = 0.0
    authority_tier: str = "unknown"
    source_text: str = field(default="", repr=False)
    full_text_fetched: bool = False
    fetch_status: str = "not_attempted"
    legal_extraction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "date": self.date,
            "score": self.score,
            "authority_tier": self.authority_tier,
            "full_text_fetched": self.full_text_fetched,
            "fetch_status": self.fetch_status,
            "legal_extraction": self.legal_extraction,
        }

    def to_prompt_block(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "date": self.date,
            "authority_tier": self.authority_tier,
            "content": self.content,
        }


@dataclass
class WebResult:
    """Kết quả Web Agent."""

    sources: list[WebSource] = field(default_factory=list)
    summary: str = ""
    available: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "summary": self.summary,
            "available": self.available,
            "reason": self.reason,
        }


class WebSearchAgent:
    """Agent tra cứu web có giới hạn domain."""

    def __init__(
        self,
        api_key: str | None = None,
        domains: tuple[str, ...] = ALLOWED_DOMAINS,
        max_results: int = MAX_RESULTS,
        llm: Any = None,
        source_fetcher: Any = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("TAVILY_API_KEY", "")
            or ""
        ).strip()
        self.domains = domains
        self.max_results = max_results
        self.llm = llm
        self.source_fetcher = source_fetcher
        self._client: Any = None
        self._client_failed = False

    # -- Hạ tầng -----------------------------------------------------------

    def available(self) -> bool:
        """Có dùng được Tavily không?"""
        if not self.api_key:
            return False
        if self._client_failed:
            return False
        if self._client is None:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=self.api_key)
            except Exception:
                # tavily chưa cài -> degrade, không lỗi
                self._client_failed = True
                return False
        return True

    # -- Tìm kiếm ----------------------------------------------------------

    def search(self, question: str) -> WebResult:
        """Tìm nguồn web cho câu hỏi (chỉ trong domain cho phép)."""
        question = (question or "").strip()
        if not question:
            return WebResult(
                available=False, reason="Câu hỏi rỗng"
            )
        if not self.available():
            return WebResult(
                available=False,
                reason=(
                    "Chưa cấu hình TAVILY_API_KEY / tavily chưa cài -> "
                    "bỏ qua nhánh web (không ảnh hưởng nhánh nội bộ)"
                ),
            )

        try:
            raw = self._client.search(
                query=question,
                include_domains=list(self.domains),
                max_results=self.max_results,
                search_depth="basic",
            )
        except Exception as exc:
            return WebResult(
                available=False,
                reason=f"Tavily lỗi: {exc}",
            )

        sources: list[WebSource] = []
        for item in (raw or {}).get("results", []):
            url = str(item.get("url") or "")
            # Chốt chặn cuối: chỉ nhận URL thuộc domain được phép.
            if not self._domain_allowed(url):
                continue
            content = str(item.get("content") or "")
            sources.append(WebSource(
                title=str(item.get("title") or ""),
                url=url,
                content=content[:MAX_CONTENT_CHARS],
                date=str(
                    item.get("published_date")
                    or item.get("date")
                    or ""
                ),
                score=float(item.get("score") or 0.0),
                authority_tier=(
                    "official" if self._official_domain(url) else "secondary"
                ),
            ))

        if not sources:
            return WebResult(
                available=False,
                reason="Không có nguồn nào trong domain được phép",
            )

        return WebResult(sources=sources, available=True)

    @staticmethod
    def _domain_allowed(url: str) -> bool:
        """URL có thuộc domain được phép không?

        Parse HOST thật (không so chuỗi) — nếu so chuỗi thì domain
        giống ('vbpl.vn.evil.com') hay subdomain-nhái ('notvbpl.vn')
        đều LỌT QUA -> hỏng domain-restriction (quyết định #9).
        """
        if not url:
            return False
        try:
            host = urlparse(url if "//" in url else f"//{url}").hostname
        except ValueError:
            return False
        if not host:
            return False
        host = host.lower().rstrip(".")
        for d in ALLOWED_DOMAINS:
            d = d.lower()
            if host == d or host.endswith(f".{d}"):
                return True
        return False

    @staticmethod
    def _official_domain(url: str) -> bool:
        """Phân biệt domain cơ quan nhà nước với nguồn tra cứu thứ cấp."""
        try:
            host = urlparse(url if "//" in url else f"//{url}").hostname
        except ValueError:
            return False
        if not host:
            return False
        host = host.lower().rstrip(".")
        official_domains = ("vbpl.vn", "chinhphu.vn")
        return any(host == d or host.endswith(f".{d}") for d in official_domains)

    # -- Tóm tắt -----------------------------------------------------------

    def summarize(self, question: str, result: WebResult) -> WebResult:
        """Tóm tắt nguồn web (LLM) + kiểm tra tình trạng hiệu lực."""
        if not result.available or not result.sources:
            return result
        if self.llm is None:
            result.summary = (
                "Chưa tóm tắt (thiếu LLM). "
                f"{len(result.sources)} nguồn đã lấy từ domain được phép."
            )
            return result

        from .prompts import build_web_prompt

        prompt = build_web_prompt(
            question,
            [s.to_prompt_block() for s in result.sources],
        )
        try:
            _sp = getattr(self.llm, "set_purpose", None)
            if _sp:
                _sp("web_summary")
            result.summary = self.llm.complete(prompt)
        except Exception as exc:
            result.summary = (
                f"Không tóm tắt được (LLM lỗi: {exc}). "
                f"{len(result.sources)} nguồn đã lấy."
            )
        return result

    def run(self, question: str) -> WebResult:
        """Search, fetch official pages, and extract source-backed fields."""
        result = self.search(question)
        if not result.available:
            return result
        from .LegalSourceExtractor import extract_legal_evidence, html_to_text

        for source in result.sources:
            if source.authority_tier == "official":
                try:
                    raw = (self.source_fetcher or self._fetch_source_text)(source.url)
                    source_text = html_to_text(raw) if "<" in raw and ">" in raw else raw
                    source.source_text = source_text[:MAX_SOURCE_TEXT_CHARS]
                    source.full_text_fetched = bool(source_text) and len(source_text) <= MAX_SOURCE_TEXT_CHARS
                    source.fetch_status = (
                        "fetched" if source.full_text_fetched else
                        "truncated" if source_text else "empty"
                    )
                except Exception as exc:
                    source.fetch_status = f"unavailable: {type(exc).__name__}"
            else:
                source.fetch_status = "secondary_snippet_only"
            source.legal_extraction = extract_legal_evidence(
                source.source_text or source.content,
                title=source.title,
                url=source.url,
                official=source.authority_tier == "official",
                full_text=source.full_text_fetched,
            )
        if os.getenv("WEB_SUMMARY_ENABLED", "false").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            return self.summarize(question, result)
        return result

    def _fetch_source_text(self, url: str) -> str:
        """Fetch allowlisted HTML/text pages; unsupported formats fail closed."""
        if not self._domain_allowed(url):
            raise ValueError("source domain is not allowed")
        request = Request(url, headers={"User-Agent": "AgentLaw/1.0 legal-source-reader"})
        with urlopen(request, timeout=8) as response:
            if not self._official_domain(response.geturl()):
                raise ValueError("official source redirected outside official domains")
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
                raise ValueError("unsupported source format")
            data = response.read(MAX_SOURCE_BYTES + 1)
            if len(data) > MAX_SOURCE_BYTES:
                raise ValueError("source exceeds size limit")
            charset_match = re.search(r"charset=([\w-]+)", content_type)
            charset = charset_match.group(1) if charset_match else "utf-8"
            return data.decode(charset, errors="replace")


__all__ = [
    "ALLOWED_DOMAINS",
    "MAX_CONTENT_CHARS",
    "MAX_RESULTS",
    "WebResult",
    "WebSearchAgent",
    "WebSource",
]
