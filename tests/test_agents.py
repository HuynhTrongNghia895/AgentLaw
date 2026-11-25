"""Test P7 — 3 Agent (LegalRetrieval / WebSearchAgent / ConflictResolver).

Tập trung phần DETERMINISTIC (không cần LLM, FAISS hay Neo4j thật):
  - hiệu lực web (domain-restriction)
  - phát hiện xung đột + quy tắc ưu tiên ngày mới hơn
  - routing của Orchestrator (khi nào gọi web)
  - LegalRetrieval: suy ra node_id từ metadata, hỏi ngược user
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
    ConflictResolver,
    SourceInfo,
    collect_sources,
    detect_conflicts,
    extract_dates,
    extract_so_hieu,
)
from agents.LegalRetrieval import LegalAnswer, LegalRetrievalAgent  # noqa: E402
from agents.Orchestrator import Orchestrator, OrchestratorState  # noqa: E402
from agents.WebSearchAgent import (  # noqa: E402
    ALLOWED_DOMAINS,
    WebSearchAgent,
    WebSource,
)
from retrieval.QueryUnderstanding import (  # noqa: E402
    INTENT_HIEU_LUC,
    INTENT_TRA_CUU,
    understand,
)


class StubLLM:
    def __init__(self, reply: str = "stub") -> None:
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.reply


# ---------------------------------------------------------------------------
# WebSearchAgent — domain-restriction (quyết định #9)
# ---------------------------------------------------------------------------
def test_allowed_domains_exactly_three():
    assert ALLOWED_DOMAINS == (
        "vbpl.vn", "chinhphu.vn", "thuvienphapluat.vn"
    )


def test_domain_allowed_accepts_allowed():
    ok = WebSearchAgent._domain_allowed
    assert ok("https://vbpl.vn/chung-cuc-dau-tu")
    assert ok("https://chinhphu.vn/van-ban")
    assert ok("https://thuvienphapluat.vn/van-ban/luat-doanh-nghiep")
    assert ok("http://vbpl.vn/x")


def test_domain_allowed_rejects_others():
    ok = WebSearchAgent._domain_allowed
    assert not ok("https://google.com/search?q=luat")
    assert not ok("https://example.com/luat")
    assert not ok("https://fakevbpl.vn.evil.com/x")
    assert not ok("")


def test_official_domain_rejects_lookalike_and_secondary_domains():
    classify = WebSearchAgent._official_domain
    assert classify("https://vbpl.vn/x") is True
    assert classify("https://sub.chinhphu.vn/x") is True
    assert classify("https://thuvienphapluat.vn/x") is False
    assert classify("https://vbpl.vn.evil.example/x") is False


def test_web_agent_fetches_official_full_text_and_extracts_verified_amendment_offline():
    source_html = """
    <html><body>
      <p>Nghị định số 2/2025/NĐ-CP</p>
      <p>Điều 2. Sửa đổi, bổ sung một số điều của Nghị định số 1/2024/NĐ-CP.</p>
      <p>Bổ sung khoản 7 vào sau khoản 6 Điều 4 của Nghị định số 1/2024/NĐ-CP.</p>
      <p>Điều 3. Điều khoản thi hành.</p>
      <p>Nghị định này có hiệu lực thi hành từ ngày 1 tháng 1 năm 2025.</p>
    </body></html>
    """

    class SearchClient:
        def search(self, **_kwargs):
            return {"results": [{
                "title": "Nghị định 2/2025/NĐ-CP",
                "url": "https://vbpl.vn/doc/2-2025",
                "content": "Kết quả tìm kiếm rút gọn.",
                "published_date": "2025-01-01",
            }]}

    agent = WebSearchAgent(
        api_key="fixture", source_fetcher=lambda _url: source_html,
    )
    agent._client = SearchClient()
    result = agent.run("Nghị định 2/2025/NĐ-CP sửa điều nào của 1/2024/NĐ-CP?")
    source = result.sources[0]

    assert source.full_text_fetched is True
    assert source.fetch_status == "fetched"
    assert source.legal_extraction["effective_date"] == "2025-01-01"
    relation = source.legal_extraction["relations"][0]
    assert relation["relation_type"] == "AMENDS"
    assert relation["status"] == "verified"
    proof = relation["verified_relationship"]
    assert proof["to_document"] == "1/2024/NĐ-CP"
    assert proof["affected_provisions"][0]["article"] == "4"
    assert proof["affected_provisions"][0]["clause"] == "7"


def test_web_extraction_does_not_verify_snippet_or_secondary_source():
    from agents.LegalSourceExtractor import extract_legal_evidence
    data = extract_legal_evidence(
        "Bổ sung khoản 7 vào sau khoản 6 Điều 4 của Nghị định số 1/2024/NĐ-CP.",
        title="Nghị định 2/2025/NĐ-CP", url="https://thuvienphapluat.vn/x",
        official=False, full_text=False,
    )
    assert data["relations"][0]["status"] == "candidate"
    assert "verified_relationship" not in data["relations"][0]


def test_vbpl_json_ld_and_next_payload_are_decoded_for_extraction():
    from agents.LegalSourceExtractor import html_to_text, extract_legal_evidence

    html = r"""
    <html><body>
      <script type="application/ld+json">
      {"@type":"Legislation","legislationIdentifier":"296/2026/NĐ-CP",
       "legislationDate":"2026-07-23T00:00:00","legislationLegalForce":"InForce"}
      </script>
      <script>
      \u003cp class="prov-article"\u003eĐiều 22. Điều khoản thi hành\u003c/p\u003e
      \u003cp class="prov-clause"\u003e1. Nghị định này có hiệu lực thi hành từ ngày 23 tháng 7 năm 2026.\u003c/p\u003e
      \u003cp class="prov-article"\u003eĐiều 19. Xử lý chuyển tiếp\u003c/p\u003e
      \u003cp class="prov-clause"\u003e9. Bãi bỏ Điều 37, khoản 2 Điều 52, Điều 111, khoản 3 Điều 124 của Nghị định số 168/2025/NĐ-CP.\u003c/p\u003e
      </script>
    </body></html>
    """
    text = html_to_text(html)
    data = extract_legal_evidence(
        text,
        title="Nghị định 296/2026/NĐ-CP",
        url="https://vbpl.vn/van-ban/chi-tiet/296",
        official=True,
        full_text=True,
    )

    assert "Điều 22" in text
    assert "23 tháng 7 năm 2026" in text
    assert data["effective_date"] == "2026-07-23"
    assert data["document_effective_status"] == "in_force"
    relation = next(r for r in data["relations"] if r["relation_type"] == "REPEALS")
    assert relation["status"] == "verified"
    assert relation["to_document"] == "168/2025/NĐ-CP"


def test_web_summary_is_optional_and_does_not_replace_structured_extraction(monkeypatch):
    from agents.WebSearchAgent import WebSearchAgent, WebResult, WebSource

    monkeypatch.setenv("WEB_SUMMARY_ENABLED", "false")

    class StubLLM:
        name = "stub"

        def set_purpose(self, purpose):
            raise AssertionError(f"summary should be skipped: {purpose}")

        def complete(self, _prompt):
            raise AssertionError("summary LLM should not run")

    agent = WebSearchAgent(api_key="fixture", llm=StubLLM())
    class SearchClient:
        def search(self, **_kwargs):
            return {"results": [{
                "title": "Nghị định 296/2026/NĐ-CP",
                "url": "https://vbpl.vn/van-ban/296",
                "content": "Nghị định này có hiệu lực thi hành từ ngày 23 tháng 7 năm 2026.",
            }]}

    agent._client = SearchClient()
    agent.source_fetcher = lambda _url: "Nghị định này có hiệu lực thi hành từ ngày 23 tháng 7 năm 2026."
    returned = agent.run("Nghị định 296/2026/NĐ-CP hiện còn hiệu lực không?")
    assert returned.sources
    assert returned.sources[0].legal_extraction["effective_date"] == "2026-07-23"


def test_web_agent_degrades_without_api_key(monkeypatch):
    """Không có TAVILY_API_KEY -> available=False, KHÔNG raise.

    X6: máy dev ĐÃ có key trong `.env` -> phải xoá biến môi trường,
    nếu không test này sẽ phụ thuộc cấu hình cá nhân.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = WebSearchAgent(api_key="")
    agent.api_key = ""          # chặn fallback os.getenv
    agent._client_failed = True  # và chặn fallback tạo client
    assert agent.available() is False
    result = agent.search("quy định về hộ kinh doanh")
    assert result.available is False
    assert result.sources == []
    assert "TAVILY" in result.reason or "tavily" in result.reason


def test_web_agent_empty_question():
    agent = WebSearchAgent(api_key="")
    result = agent.search("   ")
    assert result.available is False


# ---------------------------------------------------------------------------
# ConflictResolver — trích xuất metadata
# ---------------------------------------------------------------------------
def test_extract_so_hieu():
    text = (
        "Theo Nghị định 168/2025/NĐ-CP và Luật Doanh nghiệp "
        "59/2020/QH14, tổ chức..."
    )
    sos = extract_so_hieu(text)
    assert "168/2025/NĐ-CP" in sos
    assert "59/2020/QH14" in sos


def test_extract_dates_iso():
    dates = extract_dates("Ngày ban hành: 01/01/2025 ngày khác 15-03-2026")
    assert "2025-01-01" in dates
    assert "2026-03-15" in dates


def test_extract_dates_ignores_invalid():
    dates = extract_dates("Ngày ban hành: 32/13/2025")
    assert dates == []


def test_collect_sources_uses_registry_first():
    registry = {"59/2020/QH14": "2020-06-17"}
    sources = collect_sources(
        ["Văn bản 59/2020/QH14 ban hành ngày 01/01/2000"],
        origin="internal",
        registry_dates=registry,
    )
    assert len(sources) == 1
    assert sources[0].so_hieu == "59/2020/QH14"
    # Ngày từ registry (đáng tin) được ưu tiên ngày đào trong text.
    assert sources[0].ngay_ban_hanh == "2020-06-17"


# ---------------------------------------------------------------------------
# ConflictResolver — ngày ban hành không phải căn cứ chọn phiên bản.
# ---------------------------------------------------------------------------
# Kịch bản thật: DB nội bộ mới nhất là 296/2026/NĐ-CP (23/07/2026).
# Web có thể tìm được văn bản MỚI HƠN (chưa có trong DB) -> xung đột,
# ưu tiên web + ghi chú "DB nội bộ cần cập nhật".
INTERNAL_NEWEST = SourceInfo("296/2026/NĐ-CP", "2026-07-23", "internal")
INTERNAL_OLD = SourceInfo("59/2020/QH14", "2020-06-17", "internal")


def test_issue_date_does_not_select_internal_source():
    web = SourceInfo("67/2025/QH15", "2025-07-01", "web")
    report = detect_conflicts([INTERNAL_NEWEST], [web])
    assert report.has_conflict is False
    assert report.preferred_origin is None
    assert report.resolution_status == "unresolved"


def test_issue_date_does_not_select_web_source():
    web = SourceInfo("310/2026/NĐ-CP", "2026-08-15", "web")
    report = detect_conflicts([INTERNAL_NEWEST], [web])
    assert report.has_conflict is False
    assert report.preferred_origin is None
    assert report.resolution_status == "unresolved"


def test_same_provision_status_disagreement_is_unresolved():
    internal = SourceInfo(
        "168/2025/NĐ-CP", "2025-06-25", "internal",
        article="100", status_claim="active",
    )
    web = SourceInfo(
        "168/2025/NĐ-CP", "2026-06-25", "web",
        article="100", status_claim="repealed", official=True,
    )
    report = detect_conflicts([internal], [web])
    assert report.has_conflict is True
    assert report.preferred_origin is None
    assert report.resolution_status == "unresolved"


def test_conflict_web_only():
    report = detect_conflicts([], [SourceInfo("168/2025/NĐ-CP", "2025-06-25", "web")])
    assert report.has_conflict is False
    assert report.preferred_origin is None


def test_conflict_no_web_dates_no_conflict():
    """Web không nêu ngày -> KHÔNG suy đoán xung đột."""
    internal = [SourceInfo("59/2020/QH14", "2020-06-17", "internal")]
    web = [SourceInfo("168/2025/NĐ-CP", "", "web")]
    report = detect_conflicts(internal, web)
    assert report.has_conflict is False
    assert report.preferred_origin is None


def test_legal_status_conflict_abstains_without_verified_resolution():
    llm = StubLLM("không được dùng khi thiếu legal resolution")
    resolver = ConflictResolver(llm=llm)
    out = resolver.resolve(
        "Điều 100 còn hiệu lực không?", "internal provision", "official web provision",
        [SourceInfo("168/2025/NĐ-CP", "2025-06-25", "internal", article="100")],
        [SourceInfo("168/2025/NĐ-CP", "2026-06-25", "web", article="100", url="https://vbpl.vn/x", official=True)],
    )
    assert out["needs_legal_review"] is True
    assert out["conflict_report"]["preferred_origin"] is None
    assert "Chưa đủ căn cứ" in out["answer"]
    assert llm.calls == []


def test_conflict_resolver_compares_only_sources_within_each_intent():
    class RecordingLLM:
        def __init__(self):
            self.prompts = []
        def complete(self, prompt):
            self.prompts.append(prompt)
            return "answer scoped to this intent"

    llm = RecordingLLM()
    resolver = ConflictResolver(llm=llm)
    active = SourceInfo(
        "168/2025/NĐ-CP", "2025-06-25", "internal", article="100",
        clause="1", status_claim="active",
    )
    repealed = SourceInfo(
        "168/2025/NĐ-CP", "2025-06-25", "web", article="100",
        clause="1", status_claim="repealed", url="https://vbpl.vn/x",
        official=True,
    )
    result = resolver.resolve_per_intent([
        {"intent_id": "i1", "query": "intent A", "internal_block": "A",
         "internal_sources": [active], "web_sources": []},
        {"intent_id": "i2", "query": "intent B", "web_block": "B",
         "internal_sources": [], "web_sources": [repealed]},
    ])
    assert len(llm.prompts) == 2
    assert result["conflict_report"]["conflicts"] == []
    assert [item["conflict_report"]["intent_id"] for item in result["intent_results"]] == ["i1", "i2"]
    assert result["intent_results"][0]["comparison"]["status"] == "not_verified"
    assert result["intent_results"][0]["comparison"]["publication_dates"] == ["2025-06-25"]
    assert result["intent_results"][0]["comparison"]["effective_dates"] == []


def test_relationship_resolution_fails_closed_without_official_provision_proof():
    llm = StubLLM("must not answer from a graph edge alone")
    out = ConflictResolver(llm=llm).resolve(
        "Văn bản B sửa đổi điều nào của văn bản A?", "KG: B AMENDS A",
        "Web result mentions both documents",
        internal_sources=[SourceInfo("1/2024/QH15", origin="internal")],
        web_sources=[SourceInfo("2/2025/QH15", origin="web", url="https://vbpl.vn/x", official=True)],
    )
    assert out["needs_legal_review"] is True
    assert out["conflict_report"]["resolution_status"] == "unresolved"
    assert llm.calls == []


def test_relationship_resolution_accepts_explicit_official_provision_proof():
    llm = StubLLM("B bổ sung khoản 2 Điều 4 của A")
    proof = {
        "verified": True,
        "relation_type": "AMENDS",
        "from_document": "2/2025/QH15",
        "to_document": "1/2024/QH15",
        "effective_date": "2025-01-01",
        "relationship_checked": True,
        "affected_provisions": [{
            "document_number": "1/2024/QH15", "article": "4", "clause": "2",
            "scope": "clause_added", "quote": "Bổ sung khoản 2 Điều 4.",
        }],
        "source": {
            "official": True, "url": "https://vbpl.vn/x",
            "citation": "Điều 2", "excerpt": "Bổ sung khoản 2 Điều 4.",
            "effective_date_excerpt": "có hiệu lực từ ngày 01 tháng 01 năm 2025.",
            "source_sha256": "a" * 64,
        },
    }
    out = ConflictResolver(llm=llm).resolve(
        "Văn bản B bổ sung khoản nào của văn bản A?", "source A", "source B",
        internal_sources=[SourceInfo("1/2024/QH15", origin="internal")],
        web_sources=[SourceInfo("2/2025/QH15", origin="web", url="https://vbpl.vn/x", official=True)],
        verified_relationship=proof,
    )
    assert llm.calls
    assert out["answer"] == "B bổ sung khoản 2 Điều 4 của A"


def test_boolean_verified_flag_without_primary_source_proof_is_rejected():
    llm = StubLLM("should not be used")
    out = ConflictResolver(llm=llm).resolve(
        "Nghị định 168/2025/NĐ-CP còn hiệu lực không?",
        "internal source", "",
        internal_sources=[SourceInfo("168/2025/NĐ-CP", origin="internal")],
        verified_resolution={"verified": True},
    )
    assert out["needs_legal_review"] is True
    assert out["conflict_report"]["resolution_status"] == "unresolved"
    assert llm.calls == []


def test_official_source_flag_does_not_trust_lookalike_domain():
    source = SourceInfo(
        so_hieu="168/2025/NĐ-CP", origin="web",
        url="https://vbpl.vn.attacker.example/document", official=True,
    )
    assert ConflictResolver._official_source(source) is False


def test_resolver_without_llm_returns_evidence():
    resolver = ConflictResolver(llm=None)
    out = resolver.resolve(
        "câu hỏi",
        internal_block="[Nguồn 1] nội dung",
        web_block="",
    )
    assert out["llm_error"] is False
    assert "nội dung" in out["answer"]


def test_resolver_no_evidence_message():
    resolver = ConflictResolver(llm=None)
    out = resolver.resolve("câu hỏi", internal_block="", web_block="")
    assert "Không tìm thấy" in out["answer"]


# ---------------------------------------------------------------------------
# Orchestrator — quy tắc gọi web (quyết định #12)
# ---------------------------------------------------------------------------
def test_orchestrator_skips_web_on_fast_track():
    state = OrchestratorState(question="q", route="fast_track")
    assert Orchestrator._needs_web(state) is False


def test_orchestrator_calls_web_for_hieu_luc():
    u = understand("Điều 100 NĐ 168/2025 còn hiệu lực không?")
    state = OrchestratorState(question="q", understanding=u, route="vector")
    assert u.intent == INTENT_HIEU_LUC
    assert Orchestrator._needs_web(state) is True


def test_orchestrator_calls_web_when_no_evidence():
    state = OrchestratorState(question="q", route="vector")
    assert Orchestrator._needs_web(state) is True


# ---------------------------------------------------------------------------
# LegalRetrieval — suy ra node_id từ metadata
# ---------------------------------------------------------------------------
def test_node_ids_from_hits():
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    hits = [
        {"metadata": {"document_number": "59/2020/QH14", "article_number": "49"}},
        {"metadata": {
            "document_number": "59/2020/QH14", "article_number": "49",
            "clause_number": "1",
        }},
        {"metadata": {"document_number": "168/2025/NĐ-CP", "article_number": "100"}},
        {"metadata": {}},
    ]
    ids = agent._node_ids_from_hits(hits)  # noqa: SLF001
    assert ids == [
        "59_2020_QH14_D49",
        "59_2020_QH14_D49_K1",
        "168_2025_NĐ_CP_D100",
    ]


def test_follow_up_asks_for_dataset_when_unclear():
    u = understand("quy định về thời hạn này")
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    q = agent._follow_up_question(u)  # noqa: SLF001
    assert q  # có câu hỏi làm rõ
    assert "?" in q or "。" in q


def test_follow_up_asks_for_provision_when_doc_known():
    u = understand("quy định trong Luật Doanh nghiệp về đăng ký doanh nghiệp")
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    q = agent._follow_up_question(u)  # noqa: SLF001
    assert q


def test_version_resolution_passes_exact_query_time_and_provision():
    from types import SimpleNamespace

    calls = []

    class FakeKG:
        def governing(self, so, article, *, as_of_date, clause=None):
            calls.append((so, article, as_of_date, clause))
            return {
                "status": "active", "provision_in_effect_at_T": True,
                "effective_date_source": "registry",
                "effective_date_used": "2025-01-01",
            }

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.kg = FakeKG()
    understanding = understand(
        "Điều 4 khoản 7 của Nghị định 168/2025/NĐ-CP có hiệu lực ngày 01/09/2025 không?"
    )
    assert understanding.intent == "hieu_luc"
    _, result = agent._resolve_versions(  # noqa: SLF001
        [SimpleNamespace(metadata={
            "so_hieu": "168/2025/NĐ-CP", "dieu": "4", "khoan": "7",
        })],
        understanding,
    )
    assert calls == [("168/2025/NĐ-CP", "4", "2025-09-01", "7")]
    # KG chỉ là dữ liệu cấu trúc; không đủ để khẳng định pháp lý T2.
    assert result["verified"] is False
    assert result["status"] == "graph_resolved_needs_primary_source"


def test_version_resolution_uses_explicit_request_date_and_question_date_precedence():
    from types import SimpleNamespace

    calls = []

    class FakeKG:
        def governing(self, so, article, *, as_of_date, clause=None):
            calls.append((so, article, as_of_date, clause))
            return {
                "status": "active", "provision_in_effect_at_T": True,
                "effective_date_source": "registry",
                "effective_date_used": "2025-01-01",
            }

    evidence = [SimpleNamespace(metadata={
        "so_hieu": "296/2026/NĐ-CP", "dieu": "22", "khoan": "1",
    })]
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.kg = FakeKG()

    before = understand(
        "Trước ngày 23/07/2026, Điều 22 của 296/2026/NĐ-CP có hiệu lực không?"
    )
    _, result = agent._resolve_versions(  # noqa: SLF001
        evidence, before, as_of_date="2026-11-19"
    )
    assert calls[-1][2] == "2026-07-22"
    assert result["as_of_date"] == "2026-07-22"

    effective_date_question = understand(
        "296/2026/NĐ-CP có hiệu lực từ khi nào?"
    )
    _, result = agent._resolve_versions(  # noqa: SLF001
        evidence, effective_date_question, as_of_date="2026-11-19"
    )
    assert calls[-1][2] == "2026-11-19"
    assert result["as_of_date"] == "2026-11-19"


def test_version_resolution_rejects_invalid_explicit_query_time_without_today_fallback():
    from types import SimpleNamespace

    class FakeKG:
        def governing(self, *_args, **_kwargs):
            raise AssertionError("invalid query_time must fail before KG lookup")

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.kg = FakeKG()
    u = understand("Nghị định 168/2025/NĐ-CP còn hiệu lực không?")
    _, result = agent._resolve_versions(  # noqa: SLF001
        [SimpleNamespace(metadata={
            "so_hieu": "168/2025/NĐ-CP", "dieu": "4", "khoan": "1",
        })],
        u,
        as_of_date="2026-13-44",
    )
    assert result["status"] == "invalid_query_time"
    assert result["as_of_date"] is None


def test_time_travel_without_date_requests_clarification_before_pipeline():
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    understanding = understand("?i?u 4 tr??c ??y c? hi?u l?c kh?ng?")
    understanding.intent = "time_travel"
    result = agent.answer("?i?u 4 tr??c ??y c? hi?u l?c kh?ng?", understanding)
    assert result.route == "ask"
    assert result.metadata["version_resolution"]["status"] == "missing_query_time"
    assert result.follow_up


def test_decomposer_runs_after_empty_initial_retrieval_and_returns_clarification(monkeypatch):
    from types import SimpleNamespace
    from retrieval.QueryRouter import ROUTE_VECTOR
    import retrieval.AnchorVerifier as anchor_verifier

    class FakeRouter:
        def route(self, _question):
            return SimpleNamespace(
                route=ROUTE_VECTOR, confidence=0.0, target_node_ids=[],
                datasets=[], mode="vector", five_route="",
            )

    class EmptyKG:
        def evidence_for_many(self, _node_ids):
            return []

    class ClarifyingDecomposer:
        called = False

        def fill(self, question, evidence, runner):
            self.called = True
            assert evidence == []
            return [], {"clarify": "Bạn hỏi văn bản nào?", "per_intent": [
                {"sub_query": question, "node_ids": []}
            ]}

    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.router = FakeRouter()
    agent.kg = EmptyKG()
    agent.decomposer = ClarifyingDecomposer()
    agent.store = object()
    agent._route_classify_budget = 0
    agent.llm = None
    agent._vector_hits = lambda *_: []
    agent._enrich_with_graph_relations = lambda _u, ids: ids
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None

    result = agent.answer("Quy định đăng ký doanh nghiệp ra sao?")
    assert agent.decomposer.called is True
    assert result.follow_up == "Bạn hỏi văn bản nào?"


def test_pre_retrieval_planner_searches_only_resolved_intents(monkeypatch):
    from types import SimpleNamespace
    from retrieval.QueryRouter import ROUTE_VECTOR
    import retrieval.AnchorVerifier as anchor_verifier

    class FakeRouter:
        def route(self, _question):
            return SimpleNamespace(
                route=ROUTE_VECTOR, confidence=0.1, target_node_ids=[],
                datasets=[], mode="vector", five_route="",
            )

    class Planner:
        def plan(self, _question):
            return {"planned": True, "status": "planned", "intents": [
                {"sub_query": "ý đủ dữ liệu", "needs_clarification": False},
                {"sub_query": "ý thiếu văn bản", "needs_clarification": True,
                 "clarification": "Ý thứ hai thuộc văn bản nào?"},
            ]}

    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.router = FakeRouter()
    agent.decomposer = Planner()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    searched = []
    agent._evidence_for_subquestion = lambda sub: (
        searched.append(sub) or [SimpleNamespace(node_id="node_good")]
    )
    captured = {}

    def synthesize(_q, _u, decision, evidence, per_intent=None):
        captured["decision"] = decision
        captured["evidence"] = evidence
        captured["per_intent"] = per_intent
        return LegalAnswer(answer="partial answer")

    agent._synthesize = synthesize
    result = agent.answer("hai ý pháp lý đủ dài để nhận diện đây là câu hỏi nhiều ý")
    assert searched == ["ý đủ dữ liệu"]
    assert [item.node_id for item in captured["evidence"]] == ["node_good"]
    assert captured["per_intent"][0]["node_ids"] == ["node_good"]
    assert result.follow_up == "Ý thứ hai thuộc văn bản nào?"
    assert result.metadata["per_intent"][0]["intent_id"] == "intent-1"
    assert result.metadata["per_intent"][0]["evidence_status"] == "covered"
    assert result.metadata["per_intent"][1]["execution"] == "CLARIFY"


def test_multi_intent_planner_runs_before_whole_question_router(monkeypatch):
    from types import SimpleNamespace
    import retrieval.AnchorVerifier as anchor_verifier
    import agents.LegalRetrieval as legal_retrieval_module

    calls = []
    class Router:
        def route(self, query):
            calls.append(query)
            return SimpleNamespace(route="vector", confidence=0.1,
                                   target_node_ids=[], datasets=[],
                                   mode="vector", five_route="")
    class Planner:
        def plan(self, _question):
            return {"planned": True, "status": "planned", "intents": [
                {"intent_id": "intent-1", "sub_query": "ý một", "needs_clarification": False},
                {"intent_id": "intent-2", "sub_query": "ý hai", "needs_clarification": True,
                 "clarification": "Làm rõ ý hai?"},
                {"intent_id": "intent-3", "sub_query": "ý ba", "needs_clarification": True,
                 "clarification": "Làm rõ ý ba?"},
            ]}
    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    monkeypatch.setattr(legal_retrieval_module, "is_multi_intent", lambda _q: True)
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.router, agent.decomposer = Router(), Planner()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    agent._evidence_for_subquestion = lambda _q: []
    agent._synthesize = lambda *_a, **_k: LegalAnswer(answer="")
    result = agent.answer("hai ý pháp lý đủ dài để nhận diện đây là câu hỏi nhiều ý")
    assert calls == []
    assert result.metadata["planner_state"]["pending_intent_id"] == "intent-2"
    assert result.metadata["planner_state"]["pending_intent_ids"] == ["intent-2", "intent-3"]
    assert result.follow_up == "1) Làm rõ ý hai?\n2) Làm rõ ý ba?"


def test_planner_failure_is_execution_error_not_user_insufficient(monkeypatch):
    import agents.LegalRetrieval as legal_retrieval_module
    monkeypatch.setattr(legal_retrieval_module, "is_multi_intent", lambda _q: True)
    class Planner:
        def plan(self, _question):
            return {"planned": True, "status": "planner_unavailable", "intents": []}
    class Router:
        def route(self, _question):
            raise AssertionError("must not route after planner failure")
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.decomposer = Planner()
    agent.router = Router()
    result = agent.answer("ý A và ý B đủ dài để planner nhận biết câu nhiều ý")
    assert result.route == "error"
    assert result.metadata["execution"] == "ERROR"
    assert result.follow_up == ""


def test_planner_retrieval_marks_error_empty_and_success_distinct(monkeypatch):
    from types import SimpleNamespace
    import retrieval.AnchorVerifier as anchor_verifier

    class Router:
        def route(self, _q):
            return SimpleNamespace(route="vector", confidence=0.0,
                                   target_node_ids=[], datasets=[], mode="vector")
    class Planner:
        def plan(self, _q):
            return {"planned": True, "status": "planned", "intents": [
                {"intent_id": f"intent-{i}", "sub_query": q, "needs_clarification": False}
                for i, q in enumerate(("ý lỗi", "ý rỗng", "ý có evidence"), 1)
            ]}
    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.router, agent.decomposer = Router(), Planner()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    def retrieve(q):
        if q == "ý lỗi":
            raise RuntimeError("offline retrieval failure")
        if q == "ý rỗng":
            return []
        return [SimpleNamespace(node_id="n1")]
    agent._evidence_for_subquestion = retrieve
    captured = {}
    def synthesize(_q, _u, _decision, _evidence, per_intent=None):
        captured["per_intent"] = per_intent
        return LegalAnswer(answer="partial")
    agent._synthesize = synthesize
    agent.answer("câu gồm ý lỗi, ý rỗng và ý có evidence")
    assert [x["execution"] for x in captured["per_intent"]] == [
        "ERROR", "EMPTY", "SUCCESS"
    ]
    assert captured["per_intent"][0]["pending"] == "error"
    assert captured["per_intent"][1]["pending"] == "insufficient"


def test_planner_answer_is_applied_only_to_the_pending_intent(monkeypatch):
    from types import SimpleNamespace
    import retrieval.AnchorVerifier as anchor_verifier
    import agents.LegalRetrieval as legal_retrieval_module

    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    monkeypatch.setattr(legal_retrieval_module, "is_multi_intent", lambda _q: True)
    class Decomposer:
        def rewrite_clarified_intent(self, **kwargs):
            query = kwargs["original_sub_query"] + " — " + kwargs["user_answer"]
            return query, {
                "status": "ANCHORS_VALIDATED",
                "original_sub_query": kwargs["original_sub_query"],
                "rewritten_query": query,
                "semantic_review": "not_proven_automatically",
            }
    class Router:
        def route(self, _q): raise AssertionError("parent question must not be routed")
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.router = Router()
    agent.decomposer = Decomposer()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    searched = []
    def retrieve(query):
        searched.append(query)
        return [SimpleNamespace(node_id="node-" + str(len(searched)))]
    agent._evidence_for_subquestion = retrieve
    captured = {}
    def synthesize(_q, _u, _d, _e, per_intent=None):
        captured["intents"] = per_intent
        return LegalAnswer(answer="done")
    agent._synthesize = synthesize
    state = {
        "original_question": "ý A và ý B",
        "pending_intent_id": "intent-2",
        "intents": [
            {"intent_id": "intent-1", "sub_query": "ý A", "original_sub_query": "ý A",
             "needs_clarification": False},
            {"intent_id": "intent-2", "sub_query": "ý B", "original_sub_query": "ý B",
             "needs_clarification": True,
             "clarification": "Văn bản nào áp dụng cho B?"},
        ],
    }
    agent.answer("ý A và ý B", planner_state=state, user_answer="Nghị định X")
    assert searched[0] == "ý A"
    assert any(q == "ý B — Nghị định X" for q in searched)
    assert all("Nghị định X" not in item["sub_query"] for item in captured["intents"][:1])


def test_multi_intent_clarifications_are_asked_together_and_answers_map_by_id(monkeypatch):
    from types import SimpleNamespace
    import retrieval.AnchorVerifier as anchor_verifier
    import agents.LegalRetrieval as legal_retrieval_module

    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    monkeypatch.setattr(legal_retrieval_module, "is_multi_intent", lambda _q: True)
    class Decomposer:
        def rewrite_clarified_intent(self, **kwargs):
            query = f"{kwargs['original_sub_query']} — {kwargs['user_answer']}"
            return query, {"status": "ANCHORS_VALIDATED",
                           "original_sub_query": kwargs["original_sub_query"],
                           "rewritten_query": query}
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.decomposer = Decomposer()
    agent.router = SimpleNamespace(route=lambda _q: (_ for _ in ()).throw(
        AssertionError("must not route the parent query")))
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    queries = []
    agent._evidence_for_subquestion = lambda query: (
        queries.append(query) or [SimpleNamespace(node_id=f"node-{len(queries)}")]
    )
    agent._synthesize = lambda *_a, **_k: LegalAnswer(answer="done")
    state = {
        "original_question": "ý A và ý B",
        "pending_intent_id": "intent-A",
        "pending_intent_ids": ["intent-A", "intent-B"],
        "intents": [
            {"intent_id": "intent-A", "original_sub_query": "ý A",
             "sub_query": "ý A", "needs_clarification": True,
             "clarification": "A cần văn bản nào?"},
            {"intent_id": "intent-B", "original_sub_query": "ý B",
             "sub_query": "ý B", "needs_clarification": True,
             "clarification": "B cần thời điểm nào?"},
        ],
    }
    result = agent.answer(
        "ý A và ý B", planner_state=state,
        user_answer="1) Luật 59/2020/QH14\n2) ngày 01/01/2026",
    )
    assert queries == ["ý A — Luật 59/2020/QH14", "ý B — ngày 01/01/2026"]
    assert result.metadata["planner_rewrites"][0]["original_sub_query"] == "ý A"
    assert result.metadata["planner_rewrites"][1]["original_sub_query"] == "ý B"


def test_retrieval_error_verdict_is_not_insufficient():
    assert LegalRetrievalAgent._intent_verdict(
        {"status": "stopped", "execution": "ERROR", "pending": "error"},
        [], False,
    ) == "FAILED"


def test_shared_evidence_keeps_all_intent_provenance():
    from types import SimpleNamespace
    from graph.kg.materialize import Evidence
    from retrieval.QueryUnderstanding import understand

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.top_k, agent.rerank_fn, agent.tracer = 5, None, None
    agent._default_rerank_fn = lambda: None
    agent._five_route_for_intent = lambda _q: ("semantic", [])
    agent._needs_relation = lambda *_: False
    agent._status_block = lambda *_: ""
    agent._rerank_intent = lambda _q, pool, _pins: (list(pool), 0.0)
    class LLM:
        def complete(self, _prompt): return "Tôi đã tách hai ý."
    agent.llm = LLM()
    evidence = Evidence(node_id="shared", citation="Nguồn", text="Nội dung", kind="article")
    result = agent._synthesize_per_intent(
        "ý A và ý B", understand("ý A và ý B"),
        SimpleNamespace(route="vector", confidence=0, five_route="full", datasets=[]),
        [evidence],
        [
            {"sub_query": "ý A", "node_ids": ["shared"], "status": "covered", "execution": "SUCCESS"},
            {"sub_query": "ý B", "node_ids": ["shared"], "status": "covered", "execution": "SUCCESS"},
        ],
    )
    assert result.evidence[0]["metadata"]["intent_indices"] == [0, 1]
    assert result.primary_evidence[0]["metadata"]["intent_indices"] == [0, 1]


def test_multi_intent_status_resolution_is_scoped_per_intent():
    from types import SimpleNamespace
    from graph.kg.materialize import Evidence
    from retrieval.QueryUnderstanding import understand

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.top_k, agent.rerank_fn, agent.tracer = 5, None, None
    agent._default_rerank_fn = lambda: None
    agent._five_route_for_intent = lambda _q: ("semantic", [])
    agent._needs_relation = lambda *_: False
    agent.last_gap_meta = {}
    agent._rerank_intent = lambda _q, pool, _pins: (list(pool), 0.5)
    resolved = []
    def resolve(evs, u, **_kwargs):
        resolved.append((u.intent, [e.node_id for e in evs]))
        return "GRAPH STATUS: unverified", {"verified": False, "status": "graph_resolved_needs_primary_source", "items": []}
    agent._resolve_versions = resolve
    agent._status_block = lambda *_: ""
    class LLM:
        prompt = ""
        def complete(self, prompt): self.prompt = prompt; return "Không thể kết luận."
    llm = LLM(); agent.llm = llm
    e1 = Evidence(node_id="n1", citation="A", text="A", kind="article")
    e2 = Evidence(node_id="n2", citation="B", text="B", kind="article")
    result = agent._synthesize_per_intent(
        "nội dung A và hiệu lực B", understand("nội dung A"),
        SimpleNamespace(route="vector", confidence=0, five_route="full", datasets=[]),
        [e1, e2],
        [
            {"sub_query": "nội dung A", "node_ids": ["n1"], "status": "covered", "execution": "SUCCESS"},
            {"sub_query": "hiệu lực B tại 01/01/2025", "node_ids": ["n2"], "status": "covered", "execution": "SUCCESS"},
        ],
    )
    assert resolved == [("time_travel", ["n2"])]
    assert "TRẠNG THÁI CHƯA ĐƯỢC XÁC MINH" in llm.prompt
    assert result.metadata["version_resolution"]["by_intent"]["1"]["verified"] is False


def test_full_route_fails_closed_when_planner_cannot_parse(monkeypatch):
    from types import SimpleNamespace
    from retrieval.QueryRouter import ROUTE_VECTOR
    import retrieval.AnchorVerifier as anchor_verifier

    class FakeRouter:
        def route(self, _question):
            return SimpleNamespace(
                route=ROUTE_VECTOR, confidence=0.1, target_node_ids=[],
                datasets=[], mode="vector", five_route="",
            )

    class FailedPlanner:
        def plan(self, _question):
            return {"planned": True, "status": "planner_unavailable", "intents": []}

    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: None)
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.router = FakeRouter()
    agent.decomposer = FailedPlanner()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    agent._vector_hits = lambda *_: (_ for _ in ()).throw(AssertionError("full route must not use broad recall"))
    result = agent.answer(
        "Hộ kinh doanh đăng ký thuế thế nào, và mức thuế GTGT bao nhiêu?"
    )
    assert result.route == "error"
    assert result.metadata["execution"] == "ERROR"
    assert "planner" in result.answer.lower()


def test_per_intent_synthesis_never_falls_back_to_other_intent_evidence():
    from types import SimpleNamespace
    from graph.kg.materialize import Evidence
    from retrieval.QueryUnderstanding import understand

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.top_k = 5
    agent.rerank_fn = None
    agent._default_rerank_fn = lambda: None
    agent._five_route_for_intent = lambda _q: ("semantic", [])
    agent._needs_relation = lambda *_: False
    agent._status_block = lambda *_: ""
    agent.tracer = None
    agent.last_gap_meta = {}

    seen_pools = []
    def rerank(_query, pool, _pins):
        seen_pools.append([e.node_id for e in pool])
        return list(pool), 0.0
    agent._rerank_intent = rerank

    class AnswerLLM:
        def complete(self, _prompt):
            return "Hai ý đã được tách riêng."
    agent.llm = AnswerLLM()

    evidence = Evidence(
        node_id="n_primary", citation="Nguồn A", text="nội dung", kind="article"
    )
    result = agent._synthesize_per_intent(
        "ý A và ý B", understand("ý A và ý B"),
        SimpleNamespace(route="vector", confidence=0.2, five_route="full", datasets=[]),
        [evidence],
        [
            {"sub_query": "ý A", "node_ids": ["n_primary"], "status": "covered", "execution": "OK"},
            {"sub_query": "ý B", "node_ids": [], "status": "stopped", "execution": "EMPTY", "pending": "insufficient"},
        ],
    )
    assert seen_pools == [["n_primary"], []]
    assert "ý B" in result.follow_up


def test_verified_exact_route_fetches_anchor_without_vector_search(monkeypatch):
    from types import SimpleNamespace
    from retrieval.QueryRouter import ROUTE_VECTOR
    import retrieval.AnchorVerifier as anchor_verifier

    class FakeRouter:
        def route(self, _question):
            return SimpleNamespace(
                route=ROUTE_VECTOR, confidence=0.1, target_node_ids=[],
                datasets=[], mode="vector", five_route="",
            )

    class ExactResult:
        route = "exact"
        checks = []
        reason = "verified exact anchor"

        @staticmethod
        def pin_node_ids():
            return ["296_2026_ND_CP_D22"]

    class ExactKG:
        looked_up = []

        def evidence_for_many(self, node_ids):
            self.looked_up.extend(node_ids)
            return [SimpleNamespace(node_id=node_ids[0])]

    monkeypatch.setattr(anchor_verifier, "classify_query", lambda *a, **k: ExactResult())
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.router = FakeRouter()
    agent.kg = ExactKG()
    agent.store = object()
    agent._route_classify_budget = 0
    agent.llm = None
    agent._vector_hits = lambda *_: (_ for _ in ()).throw(AssertionError("vector bypassed exact"))
    agent._synthesize = lambda _q, _u, _d, evidence: LegalAnswer(
        answer="exact", evidence=[{"node_id": e.node_id} for e in evidence]
    )
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None

    result = agent.answer("Nội dung Điều 22 của Nghị định 296/2026/NĐ-CP là gì?")
    assert agent.kg.looked_up == ["296_2026_ND_CP_D22"]
    assert result.evidence == [{"node_id": "296_2026_ND_CP_D22"}]


def test_unranked_evidence_fusion_is_bounded_and_keeps_anchor():
    from types import SimpleNamespace

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.rerank_fn = None
    agent._default_rerank_fn = lambda: None
    agent.top_k = 2
    internal = [SimpleNamespace(node_id=f"n{i}", text="") for i in range(3)]
    graph = [{"node_id": "g1", "text": "relation"}]
    kept, kept_graph, _score = agent._final_rerank(  # noqa: SLF001
        "q", internal, graph, anchor_node_ids=["n2"]
    )
    assert [item.node_id for item in kept] == ["n2", "n0"]
    assert kept_graph == []


def test_reranking_preserves_primary_vs_graph_context_provenance():
    from types import SimpleNamespace

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.top_k = 2
    primary = SimpleNamespace(node_id="p1", text="primary clause")
    graph_context = {"source": "graphiti", "text": "relation context"}
    agent.rerank_fn = lambda _q, pool: [pool[1], pool[0]]
    internal, context, _score = agent._final_rerank(  # noqa: SLF001
        "q", [primary], [graph_context]
    )
    assert [e.node_id for e in internal] == ["p1"]
    assert context == [graph_context]


def test_synthesis_rejects_document_number_absent_from_evidence():
    validate = LegalRetrievalAgent._unsupported_document_citations
    evidence = [{
        "citation": "Nghị định 168/2025/NĐ-CP Điều 4",
        "text": "Nội dung Điều 4.",
        "metadata": {"so_hieu": "168/2025/NĐ-CP"},
    }]
    assert validate("Theo Nghị định 310/2026/NĐ-CP ...", evidence) == [
        "310/2026/NĐ-CP"
    ]
    assert validate("Theo 168/2025/NĐ-CP ...", evidence) == []


def test_synthesis_rejects_cited_provision_not_in_primary_evidence():
    validate = LegalRetrievalAgent._unsupported_provision_citations
    evidence = [{
        "citation": "Điều 4 Khoản 1 (168/2025/NĐ-CP)",
        "kind": "clause",
        "metadata": {"so_hieu": "168/2025/NĐ-CP", "dieu": "4", "khoan": "1"},
    }]
    assert validate("Theo Điều 22 Nghị định 168/2025/NĐ-CP ...", evidence) == [
        "168/2025/NĐ-CP Điều 22"
    ]
    assert validate("Theo Điều 4 Khoản 1 Nghị định 168/2025/NĐ-CP ...", evidence) == []


def test_claim_evidence_ledger_never_marks_claims_as_fully_proven():
    review = LegalRetrievalAgent._claim_evidence_review
    evidence = [{
        "node_id": "168_2025_D4_K1",
        "citation": "Điều 4 Khoản 1 (168/2025/NĐ-CP)",
        "metadata": {"so_hieu": "168/2025/NĐ-CP", "dieu": "4", "khoan": "1"},
    }]
    claims = review(
        "Điều 4 Khoản 1 Nghị định 168/2025/NĐ-CP quy định việc đăng ký.",
        evidence,
    )
    assert claims[0]["matched_primary_evidence"][0]["node_id"] == "168_2025_D4_K1"
    assert claims[0]["status"] == "human_review_required"


def test_claim_evidence_ledger_does_not_infer_clauses_from_numbered_prose():
    review = LegalRetrievalAgent._claim_evidence_review
    evidence = [{
        "node_id": "168_2025_D4",
        "citation": "Điều 4 (168/2025/NĐ-CP)",
        "metadata": {"so_hieu": "168/2025/NĐ-CP", "dieu": "4", "khoan": ""},
    }]
    claims = review(
        "Điều 4 Nghị định 168/2025/NĐ-CP quy định các khoản 1, 2, 3, 4.",
        evidence,
    )
    assert claims[0]["cited_articles"] == ["4"]
    assert claims[0]["cited_clauses"] == []
    assert claims[0]["matched_primary_evidence"]


def test_anchor_semantic_route_filters_vector_hits_to_anchored_document():
    from types import SimpleNamespace

    class Store:
        def search(self, *_args, **_kwargs):
            return [
                {"metadata": {"document_number": "168/2025/NĐ-CP"}},
                {"metadata": {"document_number": "59/2020/QH14"}},
            ]

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.store = Store()
    agent.top_k = 5
    agent.rerank_top_n = 10
    agent.rerank_fn = None
    decision = SimpleNamespace(datasets=[], five_route="anchor_semantic")
    hits = agent._vector_hits(
        "Nghị định 168/2025/NĐ-CP quy định các trường hợp nào?", decision
    )
    assert [h["metadata"]["document_number"] for h in hits] == [
        "168/2025/NĐ-CP"
    ]


# ---------------------------------------------------------------------------
# F5b per-intent synthesize (S6) — regression: string phải vào TEXT builder
# ---------------------------------------------------------------------------

def test_per_intent_synthesize_routes_string_to_text_builder():
    """S6 crash repro. `_synthesize_per_intent` tự ghép context_block DÃY
    (header từng ý + OOD label). Lỗi cũ: truyền string đó cho builder LIST
    (`utils.BuildPrompt.build_legal_prompt`) -> bên trong gọi
    `format_evidence_block(string)` -> iterate từng ký tự ->
    `'str' object has no attribute 'citation'`. Sửa: route về TEXT builder
    (`agents.prompts.build_legal_prompt`).

    Test chạy OFFLINE: không Qwen, không tải reranker (cache=None -> None)."""
    from graph.kg.materialize import Evidence  # noqa: E402
    from retrieval.QueryRouter import ROUTE_VECTOR, RouteDecision  # noqa: E402

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    # Trạng thái tối thiểu cho nhánh per-intent.
    agent.top_k = 5
    agent.rerank_fn = None
    agent._rerank_fn_cache = None  # _default_rerank_fn -> None, KHÔNG import
    agent.graphiti = None
    agent._graph_relation_keys = set()

    class _Node:
        def __init__(self, nid, so_hieu, dieu) -> None:
            self.node_id, self.so_hieu, self.dieu = nid, so_hieu, dieu

    class _FakeKG:
        _N = {
            "108_2025_QH15_D2": _Node("108_2025_QH15_D2", "108/2025/QH15", "2"),
            "108_2025_QH15_D4": _Node("108_2025_QH15_D4", "108/2025/QH15", "4"),
        }

        def node(self, nid):
            return self._N.get(nid)

        def status_of(self, so_hieu, dieu):
            return {"status": "hiệu lực", "relations": []}

    agent.kg = _FakeKG()

    captured: dict[str, str] = {}

    class _FakeLLM:
        name = "fake"

        def complete(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "ĐÁNH GIÁ GIẢ: trả lời từng ý."

    agent.llm = _FakeLLM()

    u = understand("Điều 2 và Điều 4 của 108/2025/QH15 quy định gì?")
    decision = RouteDecision(route=ROUTE_VECTOR, understanding=u, confidence=0.5)
    evs = [
        Evidence(node_id="108_2025_QH15_D2",
                 citation="108/2025/QH15 Điều 2", text="người nộp thuế..."),
        Evidence(node_id="108_2025_QH15_D4",
                 citation="108/2025/QH15 Điều 4", text="kỳ tính thuế..."),
    ]
    per_intent = [
        {"sub_query": "Điều 2 của 108/2025/QH15",
         "node_ids": ["108_2025_QH15_D2"]},
        {"sub_query": "Điều 4 của 108/2025/QH15",
         "node_ids": ["108_2025_QH15_D4"]},
    ]

    ans = agent._synthesize(
        "Điều 2 và Điều 4 của 108/2025/QH15 quy định gì?",
        u, decision, evs, per_intent=per_intent,
    )

    # Không crash (lỗi cũ: AttributeError ở build_legal_prompt).
    assert ans.answer == "ĐÁNH GIÁ GIẢ: trả lời từng ý."
    # Chứng minh context_block DÃY (header + citation từng ý) vào LLM thật:
    assert "=== Ý CON: Điều 2" in captured["prompt"]
    assert "=== Ý CON: Điều 4" in captured["prompt"]
    assert "108/2025/QH15 Điều 2" in captured["prompt"]
    assert "108/2025/QH15 Điều 4" in captured["prompt"]
    # evidence trả về vẫn là 2 node (provenance flat cho grader).
    assert {e["node_id"] for e in ans.evidence} == {
        "108_2025_QH15_D2", "108_2025_QH15_D4"
    }


def test_rerank_trace_keeps_full_order_before_top_k_for_metrics():
    from types import SimpleNamespace
    from agents.tracing import TraceRecorder

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)  # noqa: SLF001
    agent.top_k = 1
    agent.rerank_fn = lambda _query, items: [
        {**items[1], "score": 0.9}, {**items[0], "score": 0.7},
    ]
    agent.tracer = TraceRecorder()
    agent._active_intent_id = "intent-1"
    pool = [SimpleNamespace(node_id="node-a", text="A"),
            SimpleNamespace(node_id="node-b", text="B")]

    returned, _ = agent._rerank_intent("query", pool)
    span = agent.tracer.summary()["spans_by_layer"]["retrieval_rank"][0]
    assert span["intent"] == "intent-1"
    assert span["data"]["node_ids_ranked"] == ["node-b", "node-a"]
    assert span["data"]["ranking_stage"] == "post_rerank_pre_topk"
    assert [item.node_id for item in returned] == ["node-b"]


def test_exact_temporal_retrieval_expands_effective_amendment_source():
    from types import SimpleNamespace

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.router = SimpleNamespace(route=lambda _q: SimpleNamespace(route="vector", five_route=""))
    agent._five_route_for_intent = lambda _q: ("exact", ["168_2025_NĐ_CP_D4"])
    captured = {}
    def expand(_u, ids, **kwargs):
        captured.update(kwargs)
        return list(ids) + ["296_2026_NĐ_CP_D1_K2"]
    agent._enrich_with_graph_relations = expand
    agent.kg = SimpleNamespace(evidence_for_many=lambda ids: list(ids))
    agent.tracer = None
    agent._query_time = "2026-11-21"

    evidence_ids = agent._evidence_for_subquestion(
        "Điều 4 Nghị định 168/2025/NĐ-CP từ ngày 23/07/2026 được bổ sung gì?"
    )
    assert captured["as_of_date"] == "2026-07-23"
    assert captured["relation_types"] == {"AMENDS", "REPEALS"}
    assert "296_2026_NĐ_CP_D1_K2" in evidence_ids


def test_version_resolver_uses_each_intents_normalized_target_date():
    from types import SimpleNamespace
    from retrieval.QueryUnderstanding import parse_temporal_constraint, understand

    calls = []
    class KG:
        def governing(self, doc, article, as_of_date=None, clause=None):
            calls.append((doc, article, as_of_date, clause))
            return {"status": "amended", "provision_in_effect_at_T": True,
                    "effective_date_source": "document", "effective_date_used": "2025-01-01",
                    "governing_amendments": []}
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.kg = KG()
    query = "Điều 4 Nghị định 168/2025/NĐ-CP từ ngày 23/07/2026 được bổ sung gì?"
    constraint = parse_temporal_constraint(query)
    _, result = agent._resolve_versions(
        [SimpleNamespace(metadata={"so_hieu": "168/2025/NĐ-CP", "dieu": "4"})],
        understand(query), as_of_date="2026-11-21", temporal_constraint=constraint,
    )
    assert calls == [("168/2025/NĐ-CP", "4", "2026-07-23", None)]
    assert result["temporal_constraint"]["operator"] == "FROM"
    assert result["as_of_date"] == "2026-07-23"


def test_per_intent_citation_guard_preserves_supported_sibling_answer():
    from types import SimpleNamespace
    from graph.kg.materialize import Evidence
    from retrieval.QueryUnderstanding import understand

    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.top_k, agent.rerank_fn, agent.tracer = 5, None, None
    agent._default_rerank_fn = lambda: None
    agent._five_route_for_intent = lambda _q: ("semantic", [])
    agent._needs_relation = lambda *_: False
    agent._status_block = lambda *_: ""
    agent._rerank_intent = lambda _q, pool, _pins: (list(pool), 0.5)

    class LLM:
        def complete(self, _prompt):
            return (
                "[[INTENT:1]]\nÝ 1: Theo 108/2025/QH15 Điều 2 Khoản 1, sai nguồn.\n"
                "[[INTENT:2]]\nÝ 2: Theo 59/2020/QH14 Điều 3 Khoản 1, có căn cứ."
            )
    agent.llm = LLM()
    ev_a = Evidence(node_id="a", citation="Điều 3 Khoản 1 (59/2020/QH14)",
                    text="Căn cứ A", kind="clause",
                    metadata={"so_hieu": "59/2020/QH14", "dieu": "3", "khoan": "1"})
    ev_b = Evidence(node_id="b", citation="Điều 2 Khoản 1 (108/2025/QH15)",
                    text="Căn cứ B", kind="clause",
                    metadata={"so_hieu": "108/2025/QH15", "dieu": "2", "khoan": "1"})
    result = agent._synthesize_per_intent(
        "Hai ý", understand("Hai ý"),
        SimpleNamespace(route="vector", confidence=0, five_route="full", datasets=[]),
        [ev_a, ev_b],
        [
            {"intent_id": "i1", "intent_index": 0, "sub_query": "ý A", "node_ids": ["a"], "status": "covered", "execution": "SUCCESS"},
            {"intent_id": "i2", "intent_index": 1, "sub_query": "ý B", "node_ids": ["a"], "status": "covered", "execution": "SUCCESS"},
        ],
    )
    assert "Ý 1: Theo 108/2025/QH15 Điều 2 Khoản 1, sai nguồn." in result.answer
    assert "Ý 2: Theo 59/2020/QH14 Điều 3 Khoản 1, có căn cứ." in result.answer
    assert result.metadata["synthesis_validation"]["passed"] is False


def test_multi_intent_plan_forces_per_intent_path_with_one_active_record():
    from types import SimpleNamespace
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    expected = object()
    agent._synthesize_per_intent = lambda *_args: expected
    result = agent._synthesize(
        "question", understand("question"),
        SimpleNamespace(route="vector", confidence=0, five_route="full", datasets=[]),
        [], [{"intent_id": "ready", "_is_multi_intent_plan": True}],
    )
    assert result is expected


def test_intent_answer_parser_splits_unheaded_second_paragraph_and_saves_answer():
    from types import SimpleNamespace
    from graph.kg.materialize import Evidence
    from retrieval.QueryUnderstanding import understand
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.top_k, agent.rerank_fn, agent.tracer = 5, None, None
    agent._default_rerank_fn = lambda: None
    agent._five_route_for_intent = lambda _q: ("semantic", [])
    agent._needs_relation = lambda *_args: False
    agent._status_block = lambda *_args: ""
    agent._rerank_intent = lambda _q, pool, _pins: (list(pool), 0.8)
    class LLM:
        def complete(self, _prompt):
            return "\u00dd 1. First answer.\n\nSecond answer without a heading."
    agent.llm = LLM()
    evidence = [
        Evidence(node_id="n1", citation="source one", text="evidence one", kind="clause"),
        Evidence(node_id="n2", citation="source two", text="evidence two", kind="clause"),
    ]
    records = [
        {"intent_id": "i1", "intent_index": 0, "sub_query": "first question",
         "node_ids": ["n1"], "status": "covered", "execution": "SUCCESS"},
        {"intent_id": "i2", "intent_index": 1, "sub_query": "second question",
         "node_ids": ["n2"], "status": "covered", "execution": "SUCCESS"},
    ]
    result = agent._synthesize_per_intent(
        "both questions", understand("both questions"),
        SimpleNamespace(route="vector", confidence=0, five_route="full", datasets=[]),
        evidence, records,
    )
    assert records[0]["answer"].startswith("\u00dd 1:")
    assert records[1]["answer"].startswith("\u00dd 2:")
    assert "Second answer without a heading." in records[1]["answer"]
    assert result.metadata["per_intent_answers"]["i2"] == records[1]["answer"]


def test_out_of_domain_intent_skips_per_intent_retrieval():
    from types import SimpleNamespace
    from graph.kg.materialize import Evidence
    from retrieval.QueryUnderstanding import understand
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.top_k, agent.rerank_fn, agent.tracer = 5, None, None
    agent._default_rerank_fn = lambda: None
    agent._five_route_for_intent = lambda _q: ("semantic", [])
    agent._needs_relation = lambda *_args: False
    agent._status_block = lambda *_args: ""
    calls = []
    agent._rerank_intent = lambda q, pool, _pins: (calls.append(q) or list(pool), 0.8)
    class LLM:
        def complete(self, _prompt):
            return "[[INTENT:1]]\n? 1: Ngo?i ph?m vi.\n\n[[INTENT:2]]\n? 2: C? c?n c?."
    agent.llm = LLM()
    evidence = Evidence(node_id="n1", citation="source", text="text", kind="clause")
    result = agent._synthesize_per_intent(
        "weather and legal", understand("legal"),
        SimpleNamespace(route="vector", confidence=0, five_route="full", datasets=[]),
        [evidence],
        [
            {"intent_id": "ood", "intent_index": 0, "sub_query": "weather",
             "is_out_of_domain": True, "pending": "ood_gate", "node_ids": []},
            {"intent_id": "legal", "intent_index": 1, "sub_query": "legal",
             "node_ids": ["n1"], "status": "covered", "execution": "SUCCESS"},
        ],
    )
    assert calls == ["legal"]
    assert result.metadata["intent_verdicts_by_id"]["ood"] == "OOD"


def test_current_temporal_resolver_uses_query_time_and_keeps_current_operator():
    from types import SimpleNamespace
    from retrieval.QueryUnderstanding import parse_temporal_constraint, understand
    calls = []
    class KG:
        def governing(self, doc, article, *, as_of_date, clause=None):
            calls.append((doc, article, as_of_date, clause))
            return {"status": "active", "provision_in_effect_at_T": True,
                    "effective_date_source": "registry", "effective_date_used": "2025-01-01"}
    query = "\u0110i\u1ec1u 4 Ngh\u1ecb \u0111\u1ecbnh 168/2025/N\u0110-CP c\u00f2n hi\u1ec7u l\u1ef1c hi\u1ec7n nay kh\u00f4ng?"
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.kg = KG()
    _, result = agent._resolve_versions(
        [SimpleNamespace(metadata={"so_hieu": "168/2025/N\u0110-CP", "dieu": "4"})],
        understand(query), as_of_date="2026-11-21",
        temporal_constraint=parse_temporal_constraint(query),
    )
    assert calls == [("168/2025/N\u0110-CP", "4", "2026-11-21", None)]
    assert result["temporal_constraint"]["operator"] == "CURRENT"
    assert result["temporal_constraint"]["normalized_target_date"] == "2026-11-21"
    assert result["temporal_constraint"]["temporal_constraint_source"] == "query_understanding"


def test_answer_skips_retrieval_for_planner_ood_and_keeps_verdict(monkeypatch):
    from types import SimpleNamespace
    import agents.LegalRetrieval as module
    monkeypatch.setattr(module, "is_multi_intent", lambda _q: True)
    class Planner:
        def plan(self, _q):
            return {"planned": True, "status": "planned", "intents": [
                {"intent_id": "ood", "sub_query": "weather",
                 "is_out_of_domain": True, "needs_clarification": False},
                {"intent_id": "law", "sub_query": "legal question",
                 "is_out_of_domain": False, "needs_clarification": False},
            ]}
    class Router:
        def route(self, _q): raise AssertionError("must not route the combined question")
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.decomposer, agent.router = Planner(), Router()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    searched, captured = [], {}
    def retrieve(q):
        searched.append(q)
        return [SimpleNamespace(node_id="n1")]
    agent._evidence_for_subquestion = retrieve
    def synthesize(_q, _u, _d, _e, per_intent=None):
        captured["records"] = per_intent
        return LegalAnswer(answer="answer", metadata={"intent_verdicts_by_id": {
            "ood": "OOD", "law": "COVERED"}})
    agent._synthesize = synthesize
    result = agent.answer("combined question")
    assert searched == ["legal question"]
    ood = next(item for item in result.metadata["per_intent"] if item["intent_id"] == "ood")
    assert ood["status"] == "stopped" and ood["pending"] == "ood_gate"
    assert ood["intent_verdict"] == "OOD"


def test_one_ready_intent_still_uses_multi_intent_synth_when_sibling_needs_clarification(monkeypatch):
    from types import SimpleNamespace
    import agents.LegalRetrieval as module
    monkeypatch.setattr(module, "is_multi_intent", lambda _q: True)
    class Planner:
        def plan(self, _q):
            return {"planned": True, "status": "planned", "intents": [
                {"intent_id": "ready", "sub_query": "answerable intent",
                 "needs_clarification": False},
                {"intent_id": "pending", "sub_query": "unclear intent",
                 "needs_clarification": True, "clarification": "Clarify this intent"},
            ]}
    class Router:
        def route(self, _q): raise AssertionError("must not route combined query")
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.decomposer, agent.router = Planner(), Router()
    agent._route_classify_budget = 0
    agent.llm = None
    agent.anchor_verify_source = lambda: None
    agent._route_classify_llm = lambda: None
    agent._evidence_for_subquestion = lambda _q: [SimpleNamespace(node_id="n1")]
    captured = {}
    def synthesize(_q, _u, _d, _e, per_intent=None):
        captured["forced"] = per_intent[0].get("_is_multi_intent_plan")
        return LegalAnswer(answer="partial", metadata={})
    agent._synthesize = synthesize
    result = agent.answer("two intent query")
    assert captured["forced"] is True
    assert result.metadata["per_intent"][0].get("_is_multi_intent_plan") is None
    assert result.metadata["per_intent"][1]["pending"] == "clarification"
