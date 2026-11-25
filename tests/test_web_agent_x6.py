"""Test Web Agent chạy thật (X6) — OFFLINE, không gọi network.

Bao phủ phần "hạ tầng" của X6 (phần gọi Tavily thật kiểm tra bằng case
study thủ công, không để trong test để test không tốn phí + không phụ
thuộc mạng):

  - thiếu API key -> available() = False, search() trả sources rỗng kèm
    lý do, KHÔNG raise (degrade an toàn, không chết nhánh nội bộ)
  - domain-restriction: chỉ nhận URL thuộc 3 domain được phép
  - ConflictResolver ("Thẩm phán"): ưu tiên theo NGÀY BAN HÀNH
    (deterministic, không để LLM quyết), ghi "DB nội bộ cần cập nhật"
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

from agents.ConflictResolver import (  # noqa: E402
    DB_UPDATE_NOTE,
    ConflictResolver,
    SourceInfo,
    detect_conflicts,
)
from agents.WebSearchAgent import (  # noqa: E402
    ALLOWED_DOMAINS,
    WebSearchAgent,
)


# ---------------------------------------------------------------------------
# Hạ tầng: thiếu key -> degrade, không lỗi
# ---------------------------------------------------------------------------
def test_no_api_key_means_unavailable():
    agent = WebSearchAgent(api_key="", llm=None)
    # api_key="" -> fallback os.getenv; monkeypatch để chắc chắn rỗng.
    agent.api_key = ""
    agent._client_failed = True
    assert agent.available() is False


def test_search_without_key_returns_empty_with_reason():
    """Không có key -> sources rỗng + nêu lý do, KHÔNG raise."""
    agent = WebSearchAgent(api_key="", llm=None)
    agent.api_key = ""
    agent._client_failed = True
    result = agent.search("Luật Doanh nghiệp 2020 Điều 213")
    assert result.sources == []
    assert result.available is False
    assert result.reason  # có giải thích lý do bỏ qua


def test_empty_question_is_rejected():
    agent = WebSearchAgent(api_key="x", llm=None)
    result = agent.search("   ")
    assert result.sources == []
    assert result.available is False


# ---------------------------------------------------------------------------
# Domain-restriction (quyết định #9)
# ---------------------------------------------------------------------------
def test_allowed_domains_are_the_three_expected():
    assert ALLOWED_DOMAINS == (
        "vbpl.vn",
        "chinhphu.vn",
        "thuvienphapluat.vn",
    )


def test_domain_allowed_accepts_legal_domains():
    for url in (
        "https://vbpl.vn/abc",
        "https://chinhphu.vn/abc",
        "https://thuvienphapluat.vn/abc",
        "http://vbpl.vn/abc",
    ):
        assert WebSearchAgent._domain_allowed(url) is True, url


def test_domain_allowed_rejects_other_domains():
    for url in (
        "https://example.com/abc",
        "https://facebook.com/abc",
        "https://vbpl.vn.evil.com/abc",  # look-alike domain
        "",
    ):
        assert WebSearchAgent._domain_allowed(url) is False, url


# ---------------------------------------------------------------------------
# Coordinator không suy ra ưu tiên pháp lý từ ngày ban hành.
# ---------------------------------------------------------------------------
def _internal(date: str) -> list[SourceInfo]:
    return [
        SourceInfo(
            so_hieu="59/2020/QH14",
            ngay_ban_hanh=date,
            origin="internal",
            title="Luật Doanh nghiệp",
        )
    ]


def _web(date: str) -> list[SourceInfo]:
    return [
        SourceInfo(
            so_hieu="100/2026/NĐ-CP",
            ngay_ban_hanh=date,
            origin="web",
            title="Văn bản web",
        )
    ]


def test_web_newer_issue_date_does_not_select_source():
    report = detect_conflicts(_internal("2020-06-17"), _web("2026-01-01"))
    assert report.has_conflict is False
    assert report.preferred_origin is None
    assert report.resolution_status == "unresolved"


def test_web_older_issue_date_does_not_select_internal():
    report = detect_conflicts(_internal("2025-07-01"), _web("2020-06-17"))
    assert report.has_conflict is False
    assert report.preferred_origin is None


def test_resolve_status_without_structured_proof_abstains():
    resolver = ConflictResolver(llm=None)
    out = resolver.resolve(
        question="Điều 213 còn hiệu lực không?",
        internal_block="Nội dung nội bộ",
        web_block="Nội dung web",
        internal_sources=_internal("2020-06-17"),
        web_sources=_web("2026-01-01"),
    )
    assert out["needs_legal_review"] is True
    assert out["conflict_report"]["preferred_origin"] is None
    assert "Chưa đủ căn cứ" in out["answer"]
    assert out["llm_error"] is False


def test_resolve_with_empty_sources_does_not_crash():
    resolver = ConflictResolver(llm=None)
    out = resolver.resolve(
        question="câu hỏi",
        internal_block="",
        web_block="",
        internal_sources=[],
        web_sources=[],
    )
    assert out["answer"]
    assert out["conflict_report"] is not None


# ---------------------------------------------------------------------------
# Không đụng network trong test: search() với key giả phải degrade
# ---------------------------------------------------------------------------
def test_fake_key_search_degrades_instead_of_raising():
    """Key sai -> client lỗi -> trả sources rỗng + reason, KHÔNG raise."""
    agent = WebSearchAgent(api_key="tvly-invalid-key-for-test", llm=None)
    agent._client = None
    agent._client_failed = False
    result = agent.search("Luật Doanh nghiệp 2020 Điều 213")
    # Không assert về số nguồn (mạng có thể không có) — chỉ assert KHÔNG
    # raise và có lý do khi không lấy được nguồn nào.
    assert isinstance(result.sources, list)
    if not result.sources:
        assert result.reason
