"""Orchestrator c?p cao: Planner/Refiner -> LegalRetrieval -> Web -> synthesis.

Graph LangGraph th?c t? ch? bi?u di?n c?c node c?p Orchestrator:
    START -> internal (Planner + Router + internal retrieval)
                 -> web (theo intent c?n Web) -> synthesize -> END
                 -> synthesize -> END

Fast-track/vector l? chi?n l??c b?n trong LegalRetrieval, kh?ng ph?i node
c?a graph n?y. Web ???c quy?t ??nh theo intent khi metadata c? per_intent;
c?u ??n gi? quy t?c t??ng th?ch c?.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from retrieval.QueryUnderstanding import (
    INTENT_HIEU_LUC,
    INTENT_TIME_TRAVEL,
    RE_DOC_NUMBER,
    extract_anchors,
    understand,
)
from .ConflictResolver import (
    ConflictResolver,
    SourceInfo,
    collect_sources,
    extract_dates,
)
from .LegalRetrieval import LegalAnswer, LegalRetrievalAgent
from .WebSearchAgent import WebSearchAgent
from .QueryDecomposer import is_multi_intent

try:
    from retrieval.QueryRefiner import QueryRefiner, RefineResult
except ImportError:
    QueryRefiner = None  # type: ignore
    RefineResult = None  # type: ignore


def _build_allowed_documents() -> list[str]:
    """Danh sách vb "có sẵn" kèm mô tả ngắn cho refiner (rich-choices).

    Đọc registry (CSV, 22 văn bản, rẻ) -> mỗi mục dạng
    "59/2020/QH14 (Doanh nghiệp)" — refiner dùng mô tả này để viết
    lựa chọn ĐỌC HIỂU VỚI NGƯỜI KHÔNG RÀNH LUẬT (không chỉ số hiệu
    trần). Trả [] nếu registry thiếu -> refiner chạy như trước (degrade
    an toàn, không bịa, không crash).
    """
    try:
        from graph.kg.loader import load_registry
    except Exception:
        return []
    out: list[str] = []
    try:
        reg = load_registry()
    except Exception:
        return []
    for so_hieu, meta in reg.items():
        ten = (
            meta.get("nhom_pl")
            or meta.get("ten_van_ban")
            or ""
        ).strip()
        out.append(f"{so_hieu} ({ten})" if ten else so_hieu)
    return out


@dataclass
class OrchestratorState:
    """State đi qua đồ thị LangGraph."""

    question: str = ""
    understanding: Any = None
    decision: Any = None
    internal_answer: LegalAnswer | None = None
    internal_hits: list[dict[str, Any]] = field(default_factory=list)
    web_result: Any = None
    web_results: list[Any] = field(default_factory=list)
    conflict_report: Any = None
    intent_results: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    follow_up: str = ""
    route: str = ""
    trace: list[str] = field(default_factory=list)
    planner_state: dict[str, Any] | None = None
    planner_plan: dict[str, Any] | None = None
    user_answer: str = ""
    query_time: str = ""
    claim_evidence_review: dict[str, Any] | None = None

    def note(self, step: str) -> None:
        self.trace.append(step)


class Orchestrator:
    """�iều phối Legal Agent + Web Agent + ConflictResolver."""

    def __init__(
        self,
        legal: LegalRetrievalAgent | None = None,
        web: WebSearchAgent | None = None,
        resolver: ConflictResolver | None = None,
        use_web: bool = True,
        refiner: Any | None = None,
        tracer: Any = None,
    ) -> None:
        self.tracer = tracer
        # Tracer CHỈ truyền khi TỰ TẠO legal (legal=None) — legal inject
        # (test / runner dùng stub) giữ nguyên không wrap lại (tránh
        # double-wrap, tránh đụng stub của người gọi). Khi truyền,
        # LegalRetrievalAgent tự wrap llm trước khi decomposer capture.
        if legal is not None:
            self.legal = legal
        else:
            self.legal = LegalRetrievalAgent(tracer=tracer)
        self.web = web or WebSearchAgent(llm=self.legal.llm)
        self.resolver = resolver or ConflictResolver(llm=self.legal.llm)
        self.use_web = use_web
        # P7 X3: skill hỏi lại. None -> tự tạo (nếu import được).
        if refiner is not None:
            self.refiner = refiner
        elif QueryRefiner is not None:
            try:
                self.refiner = QueryRefiner(
                    llm=self.legal.llm,
                    allowed_documents=_build_allowed_documents(),
                )
            except Exception:
                self.refiner = None
        else:
            self.refiner = None

    # -- P7 X3: Refine (hỏi lại) ---------------------------------------

    def _refine(
        self,
        question: str,
        user_answer: str = "",
        conversation_context: str = "",
    ) -> "RefineResult | None":
        """Chạy QueryRefiner. Trả None nếu không dùng được (degrade)."""
        if self.refiner is None:
            return None
        try:
            return self.refiner.refine(
                question,
                user_answer=user_answer,
                conversation_context=conversation_context,
            )
        except Exception:
            # Degrade an toàn: không refine được -> vẫn cho pipeline chạy
            # (không chặn user).
            return None

    @staticmethod
    def _valid_planner_resume(
        planner_state: dict[str, Any] | None, user_answer: str
    ) -> bool:
        """Only resume an intent plan for a non-empty answer to its pending ID."""
        if not isinstance(planner_state, dict) or not str(user_answer or "").strip():
            return False
        if not str(planner_state.get("original_question") or "").strip():
            return False
        pending_ids = planner_state.get("pending_intent_ids")
        if not isinstance(pending_ids, list):
            pending_ids = [planner_state.get("pending_intent_id")]
        pending_list = [str(value) for value in pending_ids if str(value)]
        if len(pending_list) != len(set(pending_list)):
            return False
        pending_ids = set(pending_list)
        if not pending_ids:
            return False
        intents = planner_state.get("intents")
        if not isinstance(intents, list):
            return False
        matches = [
            item for item in intents
            if isinstance(item, dict)
            and str(item.get("intent_id") or "") in pending_ids
            and item.get("needs_clarification") is True
        ]
        return {str(item.get("intent_id") or "") for item in matches} == pending_ids

    @staticmethod
    def _audit_query_rewrite(source: str, rewritten: str) -> dict[str, Any]:
        """Check provision/date anchors without treating an LLM rewrite as truth."""
        before = extract_anchors(source)
        after = extract_anchors(rewritten)
        fields = ("doc_numbers", "articles", "clauses", "points", "dates")
        dropped = {
            field: [value for value in getattr(before, field)
                    if value not in getattr(after, field)]
            for field in fields
        }
        added = {
            field: [value for value in getattr(after, field)
                    if value not in getattr(before, field)]
            for field in fields
        }
        dropped = {key: value for key, value in dropped.items() if value}
        added = {key: value for key, value in added.items() if value}
        return {
            "status": "REJECTED" if dropped or added else "ANCHORS_VALIDATED",
            "dropped_anchors": dropped,
            "unsupported_anchors": added,
            "semantic_review": "not_proven_automatically",
        }

    @staticmethod
    def _resolve_context_document_reference(
        question: str, conversation_context: str
    ) -> tuple[str, str | None]:
        """Resolve an explicit document demonstrative from the latest user turn.

        Only a single formal document number in the most recent user message is
        considered. Ambiguous or implicit references are left for clarification.
        """
        reference = re.compile(
            r"\b(?:văn bản|luật|nghị định)\s+(?:đó|này)\b", re.IGNORECASE
        )
        if not reference.search(question or "") or not conversation_context:
            return question, None
        user_messages = re.findall(
            r"(?m)^Người dùng:\s*(.*)$", conversation_context
        )
        if not user_messages:
            return question, None
        doc_numbers = list(dict.fromkeys(
            match.group(1).upper() for match in RE_DOC_NUMBER.finditer(user_messages[-1])
        ))
        if len(doc_numbers) != 1:
            return question, None
        document = doc_numbers[0]
        resolved = reference.sub(lambda match: f"văn bản {document}", question)
        return resolved, document

    @staticmethod
    def _is_scoped_provision_lookup(question: str) -> bool:
        """A named topic asking where it is regulated is scoped without an article anchor."""
        if not re.search(
            r"\b(?:điều|khoản)\s+nào\b|\bquy định\s+(?:ở đâu|tại điều nào)\b",
            question or "", re.IGNORECASE,
        ):
            return False
        target = RE_DOC_NUMBER.sub(" ", question or "")
        target = re.sub(
            r"\b(?:văn bản|luật|nghị định|đó|này|vậy|được|quy định|trong|của|ở|tại|điều|khoản|nào|theo|cho|biết|hãy|thế|nào)\b",
            " ", target, flags=re.IGNORECASE,
        )
        target_words = re.findall(r"[^\W\d_]+", target, flags=re.UNICODE)
        return len(target_words) >= 2

    # -- Quyết định gọi web -------------------------------------------------

    @staticmethod
    def _temporal_verification(state: "OrchestratorState") -> dict[str, Any]:
        """Fix F: separate 'web was called' from 'claim was confirmed'.

        Returns {web_called, web_status, status} so the UI only shows
        "đã xác minh" when web actually returned a source. A time-travel
        answer without a confirmed web source is provisional, not confirmed."""
        intent = getattr(state.understanding, "intent", "")
        is_temporal = intent in (INTENT_HIEU_LUC, INTENT_TIME_TRAVEL)
        if not is_temporal:
            return {}
        results = getattr(state, "web_results", []) or []
        if not results and getattr(state, "web_result", None) is not None:
            results = [("", state.question, state.web_result)]
        web_called = bool(results)
        confirmed = False
        no_result = False
        unavailable = False
        for _intent_id, _query, result in results:
            if getattr(result, "available", None) is False:
                unavailable = True
                continue
            if result.sources:
                confirmed = True
            else:
                no_result = True
        if confirmed:
            web_status = "confirmed"
            status = "confirmed"
        elif unavailable and not confirmed:
            web_status = "unavailable"
            status = "unverified"
        elif no_result:
            web_status = "no_result"
            status = "provisional"
        else:
            web_status = "called"
            status = "provisional"
        return {"web_called": web_called, "web_status": web_status, "status": status}

    @staticmethod
    def _web_targets(state: OrchestratorState) -> list[tuple[str, str]]:
        """Return eligible per-intent Web queries, excluding clarification/errors."""
        internal = getattr(state, "internal_answer", None)
        metadata = _field(internal, "metadata", {}) or {}
        if _field(metadata, "execution") == "ERROR":
            return []
        per_intent = metadata.get("per_intent") or []
        targets: list[tuple[str, str]] = []
        if per_intent:
            for item in per_intent:
                if not isinstance(item, dict):
                    continue
                if item.get("is_out_of_domain") is True or str(
                    item.get("intent_verdict") or ""
                ).upper() == "OOD":
                    continue
                query = str(item.get("sub_query") or "").strip()
                if not query or item.get("pending") in ("error", "clarification"):
                    continue
                sub_intent = understand(query).intent
                needs_current = sub_intent in (INTENT_HIEU_LUC, INTENT_TIME_TRAVEL)
                needs_relation_source = bool(re.search(
                    r"sửa đổi|bãi bỏ|thay thế|bổ sung\s+(?:khoản|điều|điểm)|"
                    r"văn bản hướng dẫn|hướng dẫn thi hành",
                    query, re.IGNORECASE,
                ))
                execution = str(item.get("execution") or "").upper()
                if execution == "ERROR":
                    continue
                empty = execution == "EMPTY" or item.get("pending") == "insufficient"
                if needs_current or needs_relation_source or empty:
                    targets.append((str(item.get("intent_id") or ""), query))
            return targets

        intent = getattr(state.understanding, "intent", "")
        needs_current = intent in (INTENT_HIEU_LUC, INTENT_TIME_TRAVEL)
        needs_relation_source = bool(re.search(
            r"sửa đổi|bãi bỏ|thay thế|bổ sung\s+(?:khoản|điều|điểm)|"
            r"văn bản hướng dẫn|hướng dẫn thi hành",
            state.question, re.IGNORECASE,
        ))
        if needs_current or needs_relation_source:
            return [("", state.question)]
        if state.route == "fast_track":
            return []
        if state.internal_answer is None or not state.internal_answer.evidence:
            return [("", state.question)]
        return []

    @staticmethod
    def _needs_web(state: OrchestratorState) -> bool:
        """Compatibility predicate: Web is needed if any intent is eligible."""
        return bool(Orchestrator._web_targets(state))

    # -- Luồng -------------------------------------------------------------

    def _run_internal(self, state: OrchestratorState) -> OrchestratorState:
        state.note("internal: retrieve + evidence + LLM")
        if state.understanding is None:
            state.understanding = understand(state.question)
        if state.planner_state:
            answer = self.legal.answer(
                state.question, understanding=state.understanding,
                planner_state=state.planner_state,
                user_answer=state.user_answer,
                query_time=state.query_time,
            )
        else:
            answer = self.legal.answer(
                state.question, understanding=state.understanding,
                planner_plan=state.planner_plan,
                query_time=state.query_time,
            )
        state.internal_answer = answer
        state.route = answer.route
        state.follow_up = answer.follow_up
        return state

    def _run_web(self, state: OrchestratorState) -> OrchestratorState:
        targets = self._web_targets(state) or [("", state.question)]
        results = []
        for intent_id, query in targets:
            state.note(
                "web: search (domain-restricted)" if not intent_id else
                f"web: search (domain-restricted; intent={intent_id})"
            )
            result = self.web.run(query)
            results.append((intent_id, query, result))
        state.web_results = results
        # Keep the legacy single-result field for integrations that inspect it.
        state.web_result = results[0][2] if len(results) == 1 else None
        tracer = self.tracer
        if tracer is not None:
            from .tracing import OK, SKIPPED, EMPTY
            n_src = sum(len(result.sources or []) for _, _, result in results)
            unavailable = [r for _, _, r in results if r.available is False]
            if n_src:
                w_state, w_reason = OK, f"{n_src} nguồn web từ {len(results)} intent"
            elif unavailable:
                w_state, w_reason = SKIPPED, (unavailable[0].reason or "web không khả dụng (chưa cấu hình)")
            else:
                w_state, w_reason = EMPTY, "web chạy nhưng không trả nguồn"
            tracer.emit(
                "web", w_state, reason=w_reason,
                data={"n_sources": n_src, "intent_ids": [i for i, _, _ in results],
                      "available": all(r.available is not False for _, _, r in results)},
            )
        return state

    def _synthesize(
        self, state: OrchestratorState
    ) -> OrchestratorState:
        """Hợp nhất nội bộ + web (nếu có) -> câu trả lời cuối."""
        internal = state.internal_answer
        internal_meta = _field(internal, "metadata", {}) or {}
        if _field(internal_meta, "execution") == "ERROR":
            state.final_answer = str(_field(internal, "answer", "") or "")
            state.route = "error"
            return state
        internal_block = ""
        internal_sources: list[SourceInfo] = []
        primary_list: list[Any] = []
        context_list: list[Any] = []

        if internal is not None:
            from graph.kg.materialize import format_evidence_block

            # internal_answer có thể là object HOẶC dict (LangGraph
            # serialize state -> dict) -> đọc qua _field.
            ev_list = _field(internal, "evidence") or []
            primary_list = _field(internal, "primary_evidence") or [
                e for e in ev_list
                if _field(e, "evidence_class", "primary") == "primary"
            ]
            context_list = _field(internal, "legal_context") or [
                e for e in ev_list
                if _field(e, "evidence_class", "") == "legal_context"
            ]
            if primary_list or context_list:
                sections = []
                if primary_list:
                    sections.append(
                        "=== PRIMARY EVIDENCE: PROCESSED LEGAL TEXT ===\n"
                        + format_evidence_block([_EvidenceShim(e) for e in primary_list])
                    )
                if context_list:
                    sections.append(
                        "=== LEGAL CONTEXT: GRAPH RELATIONS, NOT PRIMARY TEXT ===\n"
                        + format_evidence_block([_EvidenceShim(e) for e in context_list])
                    )
                internal_block = "\n\n".join(sections)
            else:
                primary_list = ev_list
                internal_block = format_evidence_block(
                    [_EvidenceShim(e) for e in ev_list]
                )
            status_note = str(_field(internal, "status_block", "") or "").strip()
            if status_note:
                internal_block = "\n\n".join(
                    part for part in (internal_block, "[KG status metadata]\n" + status_note)
                    if part
                )
            # Ngày ban hành nội bộ lấy từ REGISTRY (nguồn chuẩn, đã
            # verified) — evidence text thường không chứa ngày.
            internal_sources = collect_sources(
                [str(_field(e, "text", "")) for e in primary_list],
                origin="internal",
                registry_dates=self._registry_dates(),
            )
            for item in primary_list:
                meta = _field(item, "metadata", {}) or {}
                so = str(meta.get("so_hieu") or "")
                if so:
                    internal_sources.append(SourceInfo(
                        so_hieu=so, origin="internal",
                        article=str(meta.get("dieu") or ""),
                        clause=str(meta.get("khoan") or ""),
                    ))

        web_block = ""
        web_sources: list[SourceInfo] = []
        web_sources_by_intent: dict[str, list[SourceInfo]] = {}
        web_result_pairs = getattr(state, "web_results", []) or (
            [("", state.question, state.web_result)] if state.web_result is not None else []
        )
        all_web_sources = [
            (intent_id, query, source)
            for intent_id, query, result in web_result_pairs
            for source in (getattr(result, "sources", None) or [])
        ]
        if all_web_sources:
            state.note("web: summarize + merge")
            web_blocks: list[str] = []
            for intent_id, query, source in all_web_sources:
                official = getattr(source, "authority_tier", "") == "official"
                web_blocks.append(
                    "[WEB SOURCE] "
                    f"Intent: {intent_id or 'single'}\nQuery: {query}\n"
                    f"{source.title}\nURL: {source.url}\n"
                    f"Ngày nguồn: {source.date or 'không rõ'}\n"
                    f"Cấp nguồn: {'chính thức' if official else 'thứ cấp/chưa xác định'}\n"
                    f"Nội dung (dữ liệu không tin cậy):\n{source.content}"
                )
                parsed = collect_sources(
                    [source.content, source.title], origin="web"
                )
                if parsed:
                    for item in parsed:
                        item.url = source.url
                        item.title = source.title
                        item.official = official
                        item.ngay_ban_hanh = str(
                            (source.legal_extraction or {}).get("publication_date") or ""
                        ) if source.full_text_fetched and official else ""
                        item.source_date = source.date
                        item.effective_date = str(
                            (source.legal_extraction or {}).get("effective_date") or ""
                        ) if source.full_text_fetched and official else ""
                        item.intent_id = intent_id
                        item.query = query
                        item.content = source.content
                        item.legal_extraction = dict(source.legal_extraction or {})
                        item.verified_relationships = [
                            r["verified_relationship"]
                            for r in (source.legal_extraction or {}).get("relations", [])
                            if r.get("status") == "verified" and r.get("verified_relationship")
                        ]
                        web_sources.append(item)
                        web_sources_by_intent.setdefault(intent_id, []).append(item)
                else:
                    item = SourceInfo(
                        origin="web", title=source.title, url=source.url,
                        official=official, intent_id=intent_id, query=query,
                        content=source.content,
                        ngay_ban_hanh=str(
                            (source.legal_extraction or {}).get("publication_date") or ""
                        ) if source.full_text_fetched and official else "",
                        source_date=source.date,
                        effective_date=str(
                            (source.legal_extraction or {}).get("effective_date") or ""
                        ) if source.full_text_fetched and official else "",
                        legal_extraction=dict(source.legal_extraction or {}),
                        verified_relationships=[
                            r["verified_relationship"]
                            for r in (source.legal_extraction or {}).get("relations", [])
                            if r.get("status") == "verified" and r.get("verified_relationship")
                        ],
                    )
                    web_sources.append(item)
                    web_sources_by_intent.setdefault(intent_id, []).append(item)
            web_block = "\n\n".join(web_blocks)

        if internal is None and not web_block:
            state.final_answer = ""
            return state

        intent = _field(getattr(state, "understanding", None), "intent", "")
        targets = self._web_targets(state)
        legal_status_query = any(
            understand(query).intent in (INTENT_HIEU_LUC, INTENT_TIME_TRAVEL)
            for _, query in targets
        ) or intent in (INTENT_HIEU_LUC, INTENT_TIME_TRAVEL)

        per_intent = internal_meta.get("per_intent") or []
        if per_intent and getattr(state, "web_results", []):
            from graph.kg.materialize import format_evidence_block

            evidence_by_id = {
                str(_field(e, "node_id", "")): e
                for e in primary_list if _field(e, "node_id", "")
            }
            version_by_intent = (
                (internal_meta.get("version_resolution") or {}).get("by_intent") or {}
            )
            payloads: list[dict[str, Any]] = []
            for intent_index, record in enumerate(per_intent):
                intent_id = str(record.get("intent_id") or "")
                query = str(record.get("sub_query") or "").strip()
                node_ids = [str(n) for n in (record.get("node_ids") or [])]
                local_evidence = [evidence_by_id[n] for n in node_ids if n in evidence_by_id]
                local_internal_block = (
                    format_evidence_block([_EvidenceShim(e) for e in local_evidence])
                    if local_evidence else ""
                )
                local_context = [
                    e for e in context_list
                    if (_field(e, "metadata", {}) or {}).get("intent_index") == intent_index
                ]
                if local_context:
                    local_internal_block += (
                        "\n\n=== GRAPH RELATION CONTEXT (not primary legal text) ===\n"
                        + format_evidence_block([_EvidenceShim(e) for e in local_context])
                    )
                local_internal_sources: list[SourceInfo] = collect_sources(
                    [str(_field(e, "text", "")) for e in local_evidence],
                    origin="internal", registry_dates=self._registry_dates(),
                )
                for evidence in local_evidence:
                    meta = _field(evidence, "metadata", {}) or {}
                    so = str(meta.get("so_hieu") or "")
                    if so:
                        local_internal_sources.append(SourceInfo(
                            so_hieu=so, origin="internal",
                            article=str(meta.get("dieu") or ""),
                            clause=str(meta.get("khoan") or ""),
                            intent_id=intent_id, query=query,
                            content=str(_field(evidence, "text", "") or ""),
                            effective_date=str(meta.get("effective_date") or meta.get("ngay_hieu_luc") or ""),
                            effective_status=str(meta.get("effective_status") or ""),
                            relation_type=str(meta.get("relation_type") or ""),
                            related_document=str(meta.get("related_document") or ""),
                            affected_provision=str(meta.get("affected_provision") or ""),
                        ))
                for source in local_internal_sources:
                    source.intent_id = intent_id
                    source.query = query

                local_web = web_sources_by_intent.get(intent_id, [])
                local_web_block = "\n\n".join(
                    f"[WEB SOURCE]\nIntent: {intent_id}\nQuery: {query}\n"
                    f"{source.title}\nURL: {source.url}\n"
                    f"Ngày nguồn: {source.ngay_ban_hanh or 'không rõ'}\n"
                    f"Cấp nguồn: {'chính thức' if source.official else 'chưa xác định'}\n"
                    f"Nội dung: {source.content}\nMetadata: {source.to_dict()}"
                    for source in local_web
                )
                execution = str(record.get("execution") or "").upper()
                pending = str(record.get("pending") or "").lower()
                status = (
                    "ERROR" if execution == "ERROR" or pending == "error" else
                    "CLARIFY" if execution == "CLARIFY" or pending == "clarification" else
                    "READY"
                )
                version_resolution = next((
                    value for value in version_by_intent.values()
                    if str(value.get("sub_query") or "") == query
                ), None)
                relation_proof = (
                    (internal_meta.get("verified_relationship_by_intent") or {})
                    .get(intent_id)
                )
                if not relation_proof:
                    found_proofs = {
                        (
                            proof.get("from_document"), proof.get("to_document"),
                            proof.get("relation_type"), proof.get("effective_date"),
                        ): proof
                        for source in local_web
                        for proof in source.verified_relationships
                    }
                    if len(found_proofs) == 1:
                        relation_proof = next(iter(found_proofs.values()))
                payloads.append({
                    "intent_id": intent_id, "query": query,
                    "status": status,
                    "status_message": (
                        str(record.get("clarification") or "Cần làm rõ ý này trước khi tra cứu.")
                        if status == "CLARIFY" else
                        f"Không thể truy xuất ý này do lỗi kỹ thuật ({query})."
                    ),
                    "internal_block": local_internal_block,
                    "web_block": local_web_block,
                    "internal_sources": local_internal_sources,
                    "web_sources": local_web,
                    "verified_resolution": version_resolution,
                    "verified_relationship": relation_proof,
                })

            merged = self.resolver.resolve_per_intent(payloads)
            state.final_answer = merged["answer"]
            state.conflict_report = merged["conflict_report"]
            state.intent_results = list(merged.get("intent_results") or [])
            claim_review = getattr(self.legal, "_claim_evidence_review", None)
            if callable(claim_review):
                state.claim_evidence_review = {
                    "status": "automatic_provision_linkage",
                    "claims": claim_review(state.final_answer, primary_list),
                }
            if self.tracer is not None:
                from .tracing import OK, EMPTY
                self.tracer.emit(
                    "synthesis", OK if state.final_answer else EMPTY,
                    reason="ConflictResolver xử lý riêng theo intent",
                    data={"intent_count": len(payloads),
                          "intent_ids": [x["intent_id"] for x in payloads],
                          "llm_error": merged.get("llm_error", False)},
                )
            return state

        if not web_block:
            # Chỉ có nội bộ -> trả thẳng câu trả lời của Legal Agent
            # (không gọi LLM lần 2 -> tiết kiệm token, tránh lệch).
            if legal_status_query:
                internal_answer = str(_field(internal, "answer", "") or "").strip()
                limitation = (
                    "Ch\u01b0a x\u00e1c minh \u0111\u01b0\u1ee3c hi\u1ec7u l\u1ef1c: n\u1ed9i dung tr\u00ean l\u00e0 c\u0103n c\u1ee9 n\u1ed9i b\u1ed9 "
                    "\u0111\u00e3 truy xu\u1ea5t; ch\u01b0a \u0111\u1ed1i chi\u1ebfu hi\u1ec7u l\u1ef1c "
                    "v\u1edbi ngu\u1ed3n ch\u00ednh th\u1ee9c tr\u00ean Web (T2)."
                )
                state.final_answer = (
                    f"{internal_answer}\n\n{limitation}" if internal_answer else limitation
                )
                state.conflict_report = {
                    "resolution_status": "unresolved",
                    "preferred_origin": None,
                    "notes": ["Thiếu đối chiếu nguồn chính thức cho câu hỏi hiệu lực/thời điểm."],
                }
                return state
            state.final_answer = internal.answer if internal else ""
            state.conflict_report = None
            return state

        state.note("conflict: resolve + synthesize")
        internal_meta = _field(internal, "metadata", {}) or {}
        merged = self.resolver.resolve(
            question=state.question,
            internal_block=internal_block,
            web_block=web_block,
            internal_sources=internal_sources,
            web_sources=web_sources,
            verified_resolution=internal_meta.get("verified_resolution"),
        )
        state.final_answer = merged["answer"]
        state.conflict_report = merged["conflict_report"]
        if self.tracer is not None:
            from .tracing import OK, EMPTY
            n_conf = len((merged.get("conflict_report") or {}).get("conflicts", []))
            self.tracer.emit(
                "synthesis",
                OK if state.final_answer else EMPTY,
                reason=f"merge nội bộ + web (conflict report: {n_conf} mâu thuẫn)",
                data={"n_conflicts": n_conf, "merged": bool(state.final_answer)},
            )
        return state

    # -- API ----------------------------------------------------------------

    def _registry_dates(self) -> dict[str, str]:
        """{so_hieu: ngay_ban_hanh (ISO)} từ DocumentRegistry.csv."""
        if getattr(self, "_reg_dates", None) is not None:
            return self._reg_dates
        out: dict[str, str] = {}
        try:
            from graph.kg.loader import load_registry

            for so, meta in load_registry().items():
                raw = str(meta.get("ngay_ban_hanh") or "").strip()
                if not raw:
                    continue
                dates = extract_dates(raw)
                if dates:
                    out[so] = dates[0]
        except Exception:
            out = {}
        self._reg_dates = out
        return out

    def run(
        self,
        question: str,
        user_answer: str = "",
        conversation_context: str = "",
        planner_state: dict[str, Any] | None = None,
        query_time: str = "",
    ) -> dict[str, Any]:
        """Trả lời 1 câu hỏi (đi qua đồ thị LangGraph nếu có).

        `user_answer` = câu user đã chọn/trả lời ở lượt hỏi lại trước
        (P7 X3). Rỗng -> lượt đầu.
        `conversation_context` = lược sử hội thoại các lượt trước (agent
        STATELESS: không tự nhớ, người gọi truyền lại). Rỗng -> hành vi
        1 lượt y hệt trước đây (65 câu / CLI không đổi).
        """
        question = (question or "").strip()
        original_user_question = question
        query_rewrite: dict[str, Any] | None = None
        if not question:
            return {
                "answer": "",
                "follow_up": "Bạn muốn tra cứu vấn đề pháp lý nào?",
                "route": "",
                "trace": [],
            }

        # P7 X3: refine (greeting / chuẩn hóa / hỏi lại) TRƯỚC khi
        # vào pipeline. Câu mơ hồ -> dừng, hỏi lại kèm lựa chọn.
        # conversation_context: cho LLM thấy các lượt trước -> không
        # hỏi lại thông tin user đã cung cấp.
        resume_planner = self._valid_planner_resume(planner_state, user_answer)
        contextual_reference_resolved = False
        if resume_planner:
            # A structured Planner clarification has a fixed target intent.
            # Do not let the general refiner reinterpret or reassign its answer.
            question = str(planner_state.get("original_question") or question)
            refined = None
        else:
            planner_state = None
            resolved_question, resolved_document = self._resolve_context_document_reference(
                question, conversation_context
            )
            if resolved_document and self._is_scoped_provision_lookup(resolved_question):
                question = resolved_question
                contextual_reference_resolved = True
            # Context is resolved before the multi-intent heuristic, so a
            # follow-up containing conjunctions cannot reopen an answered slot.
            refined = None

        planner_plan = None
        if not resume_planner:
            planner_fn = getattr(self.legal, "plan_question", None)
            if callable(planner_fn):
                planner_plan = planner_fn(question)
                intents = planner_plan.get("intents", []) if isinstance(planner_plan, dict) else []
                refined_records: list[tuple[str, str, Any, dict[str, Any]]] = []
                # Refine only after decomposition, one intent at a time. Questions
                # remain attached to their intent and are combined by LegalRetrieval.
                for intent_item in (
                    intents if len(intents) == 1 and not contextual_reference_resolved else []
                ):
                    # Planner clarification already belongs to this intent and
                    # includes its missing-information decision. Do not let the
                    # general Refiner replace it with a narrower or unrelated ask.
                    if intent_item.get("needs_clarification"):
                        continue
                    sub_query = str(intent_item.get("sub_query") or "").strip()
                    if not sub_query:
                        continue
                    refined_item = self._refine(
                        sub_query, user_answer="", conversation_context=conversation_context,
                    )
                    if refined_item is None:
                        continue
                    rewritten_item = refined_item.question or sub_query
                    audit_item = self._audit_query_rewrite(sub_query, rewritten_item)
                    refined_records.append((sub_query, rewritten_item, refined_item, audit_item))
                    if audit_item["status"] == "REJECTED":
                        intent_item["needs_clarification"] = True
                        intent_item["clarification"] = (
                            "Câu hỏi sau chuẩn hóa làm mất hoặc tự thêm thông tin định danh. "
                            "Vui lòng xác nhận lại văn bản/điều khoản cho ý này."
                        )
                    elif refined_item.needs_clarification:
                        intent_item["needs_clarification"] = True
                        intent_item["clarification"] = refined_item.follow_up or (
                            f"Bạn có thể làm rõ ý này: {sub_query}?"
                        )
                    else:
                        intent_item["sub_query"] = rewritten_item
                        intent_item["rewritten_sub_query"] = rewritten_item
                        intent_item["rewrite_audit"] = audit_item
                if len(intents) == 1 and refined_records:
                    source_q, rewritten_q, refined_item, audit_item = refined_records[0]
                    query_rewrite = {
                        "original_question": original_user_question,
                        "question_sent_to_refiner": source_q,
                        "user_answer": "",
                        "rewritten_question": rewritten_q,
                        "clarifying_question": refined_item.follow_up,
                        "source": refined_item.source,
                        **audit_item,
                    }
                    if audit_item["status"] == "REJECTED":
                        return {
                            "answer": "Câu truy vấn chuẩn hóa làm mất hoặc tự thêm thông tin định danh; chưa thực hiện tra cứu.",
                            "follow_up": "Vui lòng xác nhận lại văn bản, điều/khoản hoặc nội dung cần hỏi.",
                            "route": "refine", "evidence": [],
                            "query_rewrite": query_rewrite,
                            "trace": ["query_rewrite_rejected"],
                        }
                    if not refined_item.needs_clarification:
                        question = rewritten_q
                if not contextual_reference_resolved and not is_multi_intent(question):
                    pending = next((i for i in intents if i.get("needs_clarification")), None)
                    if pending:
                        planner_state = {
                            "original_question": question,
                            "intents": intents,
                            "pending_intent_id": str(pending.get("intent_id") or "intent-1"),
                            "pending_intent_ids": [str(pending.get("intent_id") or "intent-1")],
                        }
                        return {
                            "answer": str(pending.get("clarification") or "Cần làm rõ câu hỏi."),
                            "follow_up": str(pending.get("clarification") or ""),
                            "route": "ask", "evidence": [], "planner_state": planner_state,
                            "trace": ["planner", "clarification"],
                        }
            else:
                # Backward-compatible path for injected LegalRetrieval adapters
                # that predate the planning API.
                if not contextual_reference_resolved and not is_multi_intent(question):
                    refined = self._refine(
                        question, user_answer=user_answer,
                        conversation_context=conversation_context,
                    )
        if refined is not None:
            rewritten_question = refined.question or question
            rewrite_validation = self._audit_query_rewrite(
                "\n".join(x for x in (question, user_answer) if x),
                rewritten_question,
            )
            query_rewrite = {
                "original_question": original_user_question,
                "question_sent_to_refiner": question,
                "user_answer": user_answer,
                "rewritten_question": rewritten_question,
                "clarifying_question": refined.follow_up,
                "source": refined.source,
                **rewrite_validation,
            }
            if self.tracer is not None:
                from .tracing import FAILED, OK
                self.tracer.emit(
                    "query_rewrite",
                    OK if rewrite_validation["status"] == "ANCHORS_VALIDATED" else FAILED,
                    reason=f"refiner={refined.source}; {rewrite_validation['status']}",
                    data=query_rewrite,
                )
            if rewrite_validation["status"] == "REJECTED":
                return {
                    "answer": "Câu truy vấn chuẩn hóa làm mất hoặc tự thêm thông tin định danh; chưa thực hiện tra cứu.",
                    "follow_up": "Vui lòng xác nhận lại văn bản, điều/khoản hoặc mốc thời gian cần hỏi.",
                    "route": "refine",
                    "evidence": [],
                    "query_rewrite": query_rewrite,
                    "trace": ["query_rewrite_rejected"],
                }
        if refined is not None and refined.needs_clarification:
            return {
                "answer": refined.answer,
                "follow_up": refined.follow_up,
                "route": "refine",
                "evidence": [],
                "conflict_report": {},
                "web_available": None,
                "choices": refined.choices,
                "query_rewrite": query_rewrite,
                "trace": [
                    f"refine: {refined.kind} ({refined.source})"
                    + (" [ctx]" if conversation_context else "")
                ],
            }
        if refined is not None and refined.question:
            question = refined.question

        state = OrchestratorState(
            question=question, planner_state=planner_state,
            planner_plan=planner_plan, user_answer=user_answer,
            query_time=query_time,
        )
        if contextual_reference_resolved:
            state.note("context document reference resolved from latest user answer")

        if self._graph is not None:
            raw = self._graph.invoke(state)
            # LangGraph trả về dict; chuẩn hoá về object để code phía
            # dưới không phải phân biệt 2 kiểu.
            if isinstance(raw, dict):
                state = OrchestratorState(**raw)
            else:
                state = raw
        else:
            state = self._run_internal(state)
            if self.use_web and self._needs_web(state):
                state = self._run_web(state)
            state = self._synthesize(state)

        internal = state.internal_answer
        internal_metadata = _field(internal, "metadata", {}) or {}
        per_intent_output = [
            dict(item) for item in (_field(internal_metadata, "per_intent", []) or [])
            if isinstance(item, dict)
        ]
        verdicts_by_id = _field(internal_metadata, "intent_verdicts_by_id", {}) or {}
        for item in per_intent_output:
            if not item.get("intent_verdict"):
                verdict = verdicts_by_id.get(str(item.get("intent_id") or ""))
                if verdict:
                    item["intent_verdict"] = str(verdict).upper()
        return {
            "answer": state.final_answer,
            "follow_up": state.follow_up,
            "route": state.route,
            "evidence": _field(internal, "evidence") or [],
            "primary_evidence": _field(internal, "primary_evidence") or [],
            "legal_context": _field(internal, "legal_context") or [],
            "planner_state": _field(
                _field(internal, "metadata", {}) or {}, "planner_state"
            ),
            "planner_rewrites": _field(
                _field(internal, "metadata", {}) or {}, "planner_rewrites", []
            ),
            "planner_audit": _field(
                _field(internal, "metadata", {}) or {}, "planner_audit"
            ),
            "per_intent": per_intent_output,
            "claim_evidence_review": _field(internal_metadata, "claim_evidence_review"),
            "version_resolution": _field(
                _field(internal, "metadata", {}) or {}, "version_resolution"
            ),
            "execution_status": _field(
                _field(internal, "metadata", {}) or {}, "execution", "SUCCESS"
            ),
            "query_rewrite": query_rewrite,
            "conflict_report": _as_dict(state.conflict_report),
            "intent_results": state.intent_results,
            "claim_evidence_review": (
                state.claim_evidence_review
                or _field(
                    _field(internal, "metadata", {}) or {},
                    "claim_evidence_review",
                )
            ),
            "web_available": (
                all(result.available is not False for _, _, result in state.web_results)
                if state.web_results else _field(state.web_result, "available")
            ),
            "temporal_verification": self._temporal_verification(state),
            "web_sources": [
                {**source.to_dict(), "intent_id": intent_id, "query": query}
                for intent_id, query, result in state.web_results
                for source in (result.sources or [])
            ] if state.web_results else (
                [s.to_dict() for s in (state.web_result.sources or [])]
                if state.web_result is not None else []
            ),
            "trace": state.trace,
        }

    # -- LangGraph ---------------------------------------------------------

    @property
    def _graph(self) -> Any:
        """Dựng đồ thị LangGraph (1 lần, cache)."""
        if getattr(self, "_lg_graph", None) is not None:
            return self._lg_graph
        self._lg_graph = None
        try:
            from langgraph.graph import END, START, StateGraph

            builder = StateGraph(OrchestratorState)
            builder.add_node("internal", self._run_internal)
            builder.add_node("web", self._run_web)
            builder.add_node("synthesize", self._synthesize)

            builder.add_edge(START, "internal")
            builder.add_conditional_edges(
                "internal",
                self._route_after_internal,
                {"web": "web", "synthesize": "synthesize"},
            )
            builder.add_edge("web", "synthesize")
            builder.add_edge("synthesize", END)

            self._lg_graph = builder.compile()
        except Exception:
            self._lg_graph = None
        return self._lg_graph

    def _route_after_internal(
        self, state: OrchestratorState
    ) -> str:
        self._emit_web_decision(state)
        if self.use_web and self._needs_web(state):
            return "web"
        return "synthesize"

    def _emit_web_decision(self, state: OrchestratorState) -> None:
        """Trace: web được gọi hay skip + LÍ DO (yêu cầu #4 trace)."""
        tracer = self.tracer
        if tracer is None:
            return
        from .tracing import OK, SKIPPED

        go_web = self.use_web and self._needs_web(state)
        intent = _field(getattr(state, "understanding", None), "intent", "")
        route = _field(state, "route", "")
        internal = _field(state, "internal_answer", None)
        has_ev = bool(_field(internal, "evidence", []))
        if go_web:
            reason = (
                f"intent {intent} (đồ thị/thời gian, vector không biết)"
                if intent in (INTENT_HIEU_LUC, INTENT_TIME_TRAVEL)
                else "internal answer None / evidence rỗng"
            )
        elif not self.use_web:
            reason = "web tắt (use_web=False)"
        elif _field(_field(internal, "metadata", {}) or {}, "execution") == "ERROR":
            reason = "execution error: stop downstream Web call"
        elif route == "fast_track":
            reason = "fast-track: KG tất định, không cần web"
        else:
            reason = "nội bộ đủ evidence + intent không thời gian -> web skip"
        tracer.emit(
            "web_decision", OK if go_web else SKIPPED, reason=reason,
            data={"intent": intent, "route": route, "has_evidence": has_ev},
        )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Đọc field từ object HOẶC dict (LangGraph serialize -> dict)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Chuẩn hoá object/dict về dict (LangGraph serialize state -> dict)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return None


class _EvidenceShim:
    """Bọc dict evidence thành object có `.text` cho format_evidence_block."""

    __slots__ = ("text", "citation")

    def __init__(self, data: dict[str, Any]) -> None:
        self.text = str(data.get("text") or "")
        self.citation = str(data.get("citation") or "")


__all__ = ["Orchestrator", "OrchestratorState"]
