"""ConflictResolver / Coordinator "Thẩm phán" (P7).

Nhiệm vụ: hợp nhất nguồn NỘI BỘ (FAISS+KG) và nguồn WEB, rồi quyết định
khi hai nguồn MÂU THUẪN.

Quy tắc:
  - Không suy ra ưu tiên hoặc hiệu lực từ ngày ban hành.
  - Chỉ so sánh khi xác định được cùng văn bản/provision và có status claim.
  - Thiếu quan hệ pháp lý, effective date hoặc nguồn chính thức -> abstain.
  - LLM chỉ diễn đạt căn cứ đã cung cấp; không tự giải quyết mâu thuẫn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib.parse import urlparse

from .prompts import NO_EVIDENCE_MESSAGE, build_conflict_prompt


# Ngày ban hành trong text: "Ngày ban hành: 01/01/2025" hoặc "01-01-2025"
_DATE_PATTERNS = (
    re.compile(r"Ngày ban hành[:\s]*(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
               re.IGNORECASE),
    re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"),
)
# Ngày ISO đã chuẩn hoá: "2026-08-15"
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# Số hiệu văn bản: "59/2020/QH14", "168/2025/NĐ-CP"
_SO_HIEU_RE = re.compile(
    r"\b(\d{1,3}/\d{4}/(?:QH\d+|NĐ[-‑]CP|NĐ[-‑]CP))\b"
)

DB_UPDATE_NOTE = "DB nội bộ cần cập nhật"


@dataclass
class SourceInfo:
    """Metadata 1 nguồn (nội bộ hoặc web) để so sánh."""

    so_hieu: str = ""
    ngay_ban_hanh: str = ""     # ISO yyyy-mm-dd
    origin: str = ""            # "internal" | "web"
    title: str = ""
    url: str = ""
    article: str = ""
    clause: str = ""
    status_claim: str = ""
    official: bool | None = None
    intent_id: str = ""
    query: str = ""
    content: str = ""
    effective_date: str = ""
    effective_status: str = ""
    relation_type: str = ""
    related_document: str = ""
    affected_provision: str = ""
    legal_extraction: dict[str, Any] = field(default_factory=dict)
    verified_relationships: list[dict[str, Any]] = field(default_factory=list)
    source_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "so_hieu": self.so_hieu,
            "ngay_ban_hanh": self.ngay_ban_hanh,
            "origin": self.origin,
            "title": self.title,
            "url": self.url,
            "article": self.article,
            "clause": self.clause,
            "status_claim": self.status_claim,
            "official": self.official,
            "intent_id": self.intent_id,
            "query": self.query,
            "effective_date": self.effective_date,
            "effective_status": self.effective_status,
            "relation_type": self.relation_type,
            "related_document": self.related_document,
            "affected_provision": self.affected_provision,
            "legal_extraction": self.legal_extraction,
            "verified_relationships": self.verified_relationships,
            "source_date": self.source_date,
            "content": self.content,
        }


@dataclass
class ConflictReport:
    """Kết quả phát hiện xung đột."""

    has_conflict: bool = False
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    preferred_origin: str | None = None
    resolution_status: str = "not_assessed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_conflict": self.has_conflict,
            "conflicts": self.conflicts,
            "notes": self.notes,
            "preferred_origin": self.preferred_origin,
            "resolution_status": self.resolution_status,
        }


def extract_dates(text: str) -> list[str]:
    """Rút các ngày ISO (yyyy-mm-dd) từ text."""
    out: list[str] = []
    # Ngày dd/mm/yyyy (Việt Nam)
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text or ""):
            d, mth, y = m.group(1), m.group(2), m.group(3)
            try:
                iso = date(int(y), int(mth), int(d)).isoformat()
            except ValueError:
                continue
            if iso not in out:
                out.append(iso)
    # Ngày ISO yyyy-mm-dd (đã chuẩn hoá, vd từ web hoặc metadata)
    for m in _ISO_DATE_RE.finditer(text or ""):
        try:
            iso = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            iso = iso.isoformat()
        except ValueError:
            continue
        if iso not in out:
            out.append(iso)
    return out


def extract_so_hieu(text: str) -> list[str]:
    """Rút số hiệu văn bản từ text."""
    out: list[str] = []
    for m in _SO_HIEU_RE.finditer(text or ""):
        sh = m.group(1).replace("NĐ‑", "NĐ-")
        if sh not in out:
            out.append(sh)
    return out


def collect_sources(
    texts: list[str],
    origin: str,
    registry_dates: dict[str, str] | None = None,
) -> list[SourceInfo]:
    """ Gom metadata nguồn từ nội dung (internal/web).

    Ưu tiên ngày từ registry (nội bộ, đáng tin) trước khi đào trong text.
    """
    registry_dates = registry_dates or {}
    out: list[SourceInfo] = []
    seen: set[str] = set()
    for text in texts:
        sos = extract_so_hieu(text)
        dates = extract_dates(text)
        for so in sos:
            if so in seen:
                continue
            seen.add(so)
            iso = registry_dates.get(so, "")
            if not iso and dates:
                iso = dates[0]
            out.append(SourceInfo(
                so_hieu=so,
                ngay_ban_hanh=iso,
                origin=origin,
                title=text[:120],
            ))
    return out


def detect_conflicts(
    internal: list[SourceInfo],
    web: list[SourceInfo],
) -> ConflictReport:
    """Chỉ nhận diện xung đột khi cùng provision có status claim rõ.

    Ngày ban hành không đủ căn cứ để chọn phiên bản hoặc kết luận hiệu lực.
    """
    report = ConflictReport()
    report.resolution_status = "unresolved"
    for i in internal:
        for w in web:
            same_target = (
                i.so_hieu == w.so_hieu
                and i.article == w.article
                and i.clause == w.clause
                and bool(i.article)
            )
            internal_status = i.effective_status or i.status_claim
            web_status = w.effective_status or w.status_claim
            if same_target and internal_status and web_status and internal_status != web_status:
                report.has_conflict = True
                report.conflicts.append({
                    "kind": "same_provision_status_disagreement",
                    "internal": i.to_dict(),
                    "web": w.to_dict(),
                    "resolution": "cần quan hệ sửa đổi/bãi bỏ và hiệu lực chính thức",
                })
            if same_target and i.effective_date and w.effective_date and i.effective_date != w.effective_date:
                report.has_conflict = True
                report.conflicts.append({
                    "kind": "same_provision_effective_date_disagreement",
                    "internal": i.to_dict(), "web": w.to_dict(),
                    "resolution": "cần đối chiếu ngày hiệu lực trong nguồn chính thức",
                })
    if report.has_conflict:
        report.notes.append(
            "Có mâu thuẫn trên cùng provision; chưa có căn cứ cấu trúc "
            "để giải quyết. Không chọn nguồn theo ngày ban hành."
        )
    elif internal and web:
        report.notes.append(
            "Chưa đủ metadata để so sánh cùng văn bản/điều khoản và trạng thái; "
            "không suy ra ưu tiên từ ngày ban hành."
        )
    return report


class ConflictResolver:
    """Hợp nhất nội bộ + web, quyết định nguồn ưu tiên, sinh câu trả lời."""

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def resolve_per_intent(self, intents: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve each intent against only its own internal/Web evidence."""
        results: list[dict[str, Any]] = []
        for index, item in enumerate(intents, start=1):
            intent_id = str(item.get("intent_id") or f"intent-{index}")
            query = str(item.get("query") or "").strip()
            status = str(item.get("status") or "ready").upper()
            if status in ("ERROR", "CLARIFY"):
                answer = str(item.get("status_message") or (
                    "Không thể xử lý ý này do lỗi truy xuất." if status == "ERROR"
                    else "Cần làm rõ ý này trước khi tra cứu."
                ))
                result = {
                    "answer": answer,
                    "conflict_report": {
                        "resolution_status": "not_attempted",
                        "preferred_origin": None,
                        "conflicts": [],
                        "notes": [f"Intent status: {status}"],
                    },
                    "llm_error": False,
                }
            else:
                result = self.resolve(
                    question=query,
                    internal_block=str(item.get("internal_block") or ""),
                    web_block=str(item.get("web_block") or ""),
                    internal_sources=list(item.get("internal_sources") or []),
                    web_sources=list(item.get("web_sources") or []),
                    verified_resolution=item.get("verified_resolution"),
                    verified_relationship=item.get("verified_relationship"),
                )
            report = dict(result.get("conflict_report") or {})
            report["intent_id"] = intent_id
            internal_sources = list(item.get("internal_sources") or [])
            web_sources = list(item.get("web_sources") or [])
            sources = internal_sources + web_sources
            verified_status = self._verified_primary_resolution(
                item.get("verified_resolution")
            )
            verified_relation = self._verified_relationship_resolution(
                item.get("verified_relationship")
            )
            comparison = {
                "status": "verified" if verified_status or verified_relation else "not_verified",
                "documents": sorted({s.so_hieu for s in sources if s.so_hieu}),
                "publication_dates": sorted({s.ngay_ban_hanh for s in sources if s.ngay_ban_hanh}),
                "source_dates": sorted({s.source_date for s in sources if s.source_date}),
                "effective_dates": sorted({s.effective_date for s in sources if s.effective_date}),
                "effective_statuses": sorted({
                    s.effective_status or s.status_claim
                    for s in sources if s.effective_status or s.status_claim
                }),
                "relations": [{
                    "relation_type": s.relation_type,
                    "related_document": s.related_document,
                    "affected_provision": s.affected_provision,
                    "intent_id": s.intent_id,
                } for s in sources if s.relation_type or s.related_document],
                "provisions": sorted({
                    f"{s.so_hieu} Điều {s.article} Khoản {s.clause}".strip()
                    for s in sources if s.so_hieu and s.article
                }),
                "as_of_date": (
                    (item.get("verified_resolution") or {}).get("as_of_date")
                    or (item.get("verified_relationship") or {}).get("as_of_date")
                ),
                "missing": [],
            }
            for key, label in (
                ("documents", "document_identity"),
                ("effective_dates", "effective_date"),
                ("effective_statuses", "effective_status"),
                ("relations", "legal_relationship"),
                ("provisions", "affected_provision"),
            ):
                if not comparison[key]:
                    comparison["missing"].append(label)
            if comparison["as_of_date"] is None and re.search(
                r"hiệu lực|thời điểm|ngày nào", query, re.IGNORECASE
            ):
                comparison["missing"].append("query_time")
            result.update({"intent_id": intent_id, "query": query,
                           "conflict_report": report,
                           "comparison": comparison})
            results.append(result)

        answer = "\n\n".join(
            f"Ý {i}. {item['query']}\n{item['answer']}"
            for i, item in enumerate(results, start=1)
        )
        return {
            "answer": answer,
            "intent_results": results,
            "conflict_report": {
                "resolution_status": "per_intent",
                "preferred_origin": None,
                "conflicts": [
                    conflict
                    for item in results
                    for conflict in item["conflict_report"].get("conflicts", [])
                ],
                "intents": [item["conflict_report"] for item in results],
            },
            "llm_error": any(bool(item.get("llm_error")) for item in results),
        }

    @staticmethod
    def _official_source(source: SourceInfo) -> bool:
        try:
            host = urlparse(source.url).hostname or ""
        except ValueError:
            return False
        host = host.lower().rstrip(".")
        domain_ok = any(
            host == domain or host.endswith("." + domain)
            for domain in ("vbpl.vn", "chinhphu.vn")
        )
        return source.official is True and domain_ok


    @classmethod
    def _verified_primary_resolution(cls, payload: dict[str, Any] | None) -> bool:
        """Accept only a provision/time result carrying explicit primary-source proof."""
        if not isinstance(payload, dict) or payload.get("verified") is not True:
            return False
        source = payload.get("source")
        provision = payload.get("provision")
        if not isinstance(source, dict) or not isinstance(provision, dict):
            return False
        try:
            host = urlparse(str(source.get("url") or "")).hostname or ""
        except ValueError:
            return False
        host = host.lower().rstrip(".")
        official = source.get("official") is True and any(
            host == domain or host.endswith("." + domain)
            for domain in ("vbpl.vn", "chinhphu.vn")
        )
        try:
            date.fromisoformat(str(payload.get("as_of_date") or ""))
            date.fromisoformat(str(payload.get("effective_date") or ""))
        except ValueError:
            return False
        return bool(
            official
            and source.get("citation")
            and source.get("excerpt")
            and source.get("effective_date_excerpt")
            and re.fullmatch(r"[0-9a-fA-F]{64}", str(source.get("source_sha256") or ""))
            and provision.get("document_number")
            and provision.get("article")
            and payload.get("document_effective_status")
            and payload.get("provision_effective_status")
            and payload.get("as_of_date")
            and payload.get("effective_date")
            and payload.get("relationship_checked") is True
        )

    @classmethod
    def _verified_relationship_resolution(
        cls, payload: dict[str, Any] | None
    ) -> bool:
        """Require official proof of the relation, exact provision and date."""
        if not isinstance(payload, dict) or payload.get("verified") is not True:
            return False
        source = payload.get("source")
        provisions = payload.get("affected_provisions")
        if not isinstance(source, dict) or not isinstance(provisions, list) or not provisions:
            return False
        try:
            host = (urlparse(str(source.get("url") or "")).hostname or "").lower().rstrip(".")
        except ValueError:
            return False
        official = source.get("official") is True and any(
            host == domain or host.endswith("." + domain)
            for domain in ("vbpl.vn", "chinhphu.vn")
        )
        try:
            date.fromisoformat(str(payload.get("effective_date") or ""))
        except ValueError:
            return False
        return bool(
            official
            and str(payload.get("relation_type") or "").upper()
            in {"AMENDS", "REPEALS", "GUIDES"}
            and payload.get("from_document")
            and payload.get("to_document")
            and source.get("citation")
            and source.get("excerpt")
            and payload.get("effective_date")
            and payload.get("relationship_checked") is True
            and all(
                isinstance(item, dict)
                and item.get("document_number")
                and item.get("article")
                and item.get("scope")
                and item.get("quote")
                for item in provisions
            )
        )

    def resolve(
        self,
        question: str,
        internal_block: str,
        web_block: str = "",
        internal_sources: list[SourceInfo] | None = None,
        web_sources: list[SourceInfo] | None = None,
        verified_resolution: dict[str, Any] | None = None,
        verified_relationship: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Hợp nhất 2 nguồn, trả {answer, conflict_report, ...}.

        Không tự chọn nguồn theo ngày ban hành. Status query cần bằng chứng
        cấu trúc về đúng provision và effective date; nếu chưa có thì abstain.
        """
        report = detect_conflicts(
            internal_sources or [], web_sources or []
        )

        legal_status_query = bool(re.search(
            r"hiệu lực|còn hiệu lực|hết hiệu lực|bãi bỏ|tại thời điểm|hiện hành",
            question or "", re.IGNORECASE,
        ))
        official_web = any(self._official_source(s) for s in (web_sources or []))
        verified_payload = verified_resolution
        resolution_verified = self._verified_primary_resolution(verified_payload)
        if legal_status_query and not resolution_verified:
            report.resolution_status = "unresolved"
            if web_block and not official_web:
                report.notes.append(
                    "Web evidence is not from a verified official domain."
                )
            report.notes.append(
                "Status/time answer requires a primary-source provision, effective-date, and relationship check."
            )
            return {
                "answer": (
                    "Ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9 \u0111\u01b0\u1ee3c x\u00e1c minh t\u1eeb ngu\u1ed3n ph\u00e1p l\u00fd ch\u00ednh th\u1ee9c "
                    "\u0111\u1ec3 k\u1ebft lu\u1eadn hi\u1ec7u l\u1ef1c t\u1ea1i th\u1eddi \u0111i\u1ec3m h\u1ecfi."
                ),
                "conflict_report": report.to_dict(),
                "llm_error": False,
                "needs_legal_review": True,
            }

        relationship_query = bool(re.search(
            r"sửa đổi|bãi bỏ|thay thế|bổ sung\s+(?:khoản|điều|điểm)|"
            r"văn bản hướng dẫn|hướng dẫn thi hành",
            question or "", re.IGNORECASE,
        ))
        relationship_verified = self._verified_relationship_resolution(
            verified_relationship
        )
        if relationship_query and not relationship_verified:
            report.resolution_status = "unresolved"
            report.notes.append(
                "Relationship query requires official proof of the relation, affected provision, and effective date."
            )
            return {
                "answer": (
                    "Chưa đủ căn cứ chính thức để xác nhận văn bản quan hệ với nhau "
                    "thế nào, điều khoản bị tác động và ngày áp dụng."
                ),
                "conflict_report": report.to_dict(),
                "llm_error": False,
                "needs_legal_review": True,
            }
        if relationship_query and relationship_verified:
            report.resolution_status = "resolved_by_primary_source"
        if legal_status_query:
            report.resolution_status = "resolved_by_primary_source"

        conflict_note = ""
        if report.notes:
            conflict_note = "\n".join(f"- {n}" for n in report.notes)
        if resolution_verified:
            conflict_note += (
                "\n- Official primary-source resolution: "
                + str(verified_payload)
            )

        if not internal_block and not web_block:
            return {
                "answer": NO_EVIDENCE_MESSAGE,
                "conflict_report": report.to_dict(),
                "llm_error": False,
            }

        if self.llm is None:
            return {
                "answer": "\n\n".join(
                    x for x in (internal_block, web_block) if x
                )[:4000],
                "conflict_report": report.to_dict(),
                "llm_error": False,
                "note": "Không có LLM -> trả nguyên văn evidence.",
            }

        prompt = build_conflict_prompt(
            question, internal_block, web_block, conflict_note
        )
        try:
            _sp = getattr(self.llm, "set_purpose", None)
            if _sp:
                _sp("conflict_resolution")
            answer = self.llm.complete(prompt)
        except Exception as exc:
            return {
                "answer": (
                    "Không giải quyết được các nguồn bằng LLM. "
                    "Chưa đưa ra kết luận pháp lý; bằng chứng cần được đối chiếu: "
                    f"{internal_block[:1000]} {web_block[:1000]}"
                ),
                "conflict_report": report.to_dict(),
                "llm_error": True,
            }

        supported_documents = {s.so_hieu for s in (internal_sources or []) + (web_sources or []) if s.so_hieu}
        cited_documents = set(extract_so_hieu(answer))
        unsupported_documents = sorted(cited_documents - supported_documents)
        if unsupported_documents:
            report.resolution_status = "unresolved"
            report.notes.append(
                "Coordinator output included document numbers absent from the supplied evidence."
            )
            return {
                "answer": (
                    "Chưa thể xác nhận câu trả lời vì có số hiệu văn bản "
                    "không xuất hiện trong evidence đã cung cấp."
                ),
                "conflict_report": report.to_dict(),
                "llm_error": False,
                "needs_legal_review": True,
                "unsupported_document_citations": unsupported_documents,
            }

        return {
            "answer": answer,
            "conflict_report": report.to_dict(),
            "llm_error": False,
        }


__all__ = [
    "ConflictReport",
    "ConflictResolver",
    "DB_UPDATE_NOTE",
    "SourceInfo",
    "collect_sources",
    "detect_conflicts",
    "extract_dates",
    "extract_so_hieu",
]
