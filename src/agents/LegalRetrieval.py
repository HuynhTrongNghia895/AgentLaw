"""LegalRetrieval Agent (P7) — nối P1→P5.

Luồng:
    QueryUnderstanding (P3)
      -> QueryRouter (P4): fast-track | vector
      -> FAISS hybrid (BGE-M3) + Reranker (BGE-Reranker-M3)
      -> KG duyệt (Neo4j, P5): quan hệ sửa/bãi/hướng dẫn + hiệu lực
      -> materialize() evidence (P2) -> cây Điều/Khoản/Điểm + citation
      -> LLM sinh câu trả lời kèm căn cứ

Nguyên tắc:
  - KHÔNG đưa node "trần" cho LLM — luôn evidence có ranh giới + citation.
  - Không đủ căn cứ -> trả NO_EVIDENCE_MESSAGE, KHÔNG bịa.
  - Câu hỏi thiếu thông tin (không rõ văn bản/điều) -> `follow_up`
    (hỏi ngược user, trần mềm ~3 câu — quyết định #7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Any

from retrieval.QueryRouter import (
    ROUTE_FAST_TRACK,
    ROUTE_VECTOR,
    QueryRouter,
    RouteDecision,
    anchor_node_ids,
)
from retrieval.QueryUnderstanding import (
    QueryUnderstanding, understand, parse_temporal_constraint,
)
from retrieval.vector_store import VectorStore
from graph.kg.materialize import format_evidence_block
from utils.BuildPrompt import build_legal_prompt
from .KgAccess import KgAccess
from .llm import BaseLLM, get_llm
from .tracing import (
    EMPTY, FAILED, INSUFFICIENT, OK, SKIPPED, TracedLLM, TraceRecorder,
    coverage_state,
)
from .QueryDecomposer import (
    MAX_EVIDENCE,
    QueryDecomposer,
    VERDICT_COVERED,
    VERDICT_FAILED,
    VERDICT_INSUFFICIENT,
    VERDICT_OOD,
    default_sub_query_rewriter,
    format_intent_clarifications,
    is_multi_intent,
    parse_intent_clarification_answers,
)
from .prompts import (
    NO_EVIDENCE_MESSAGE,
    build_legal_prompt as build_legal_prompt_text,
)


# Số node trả cho LLM (quyết định #8: top-k = 5)
DEFAULT_TOP_K = 5
# Số ứng viên trước rerank
DEFAULT_RERANK_TOP_N = 10
# Trần mềm số câu hỏi ngược (quyết định #7)
MAX_FOLLOW_UP = 3

# Sentinel: chưa cache rerank_fn (khác với None = đã thử, không có).
_UNSET = object()

# Ngưỡng OOD (P7 X4). Điểm rerank top-1 (sigmoid 0..1) THẤP hơn ngưỡng
# này (VÀ câu không có anchor) -> xét là câu hỏi NGOÀI PHẠM VI DB ->
# chặn, không để LLM bịa.
#
# Dùng ngưỡng RERANK (đã chấm câu hỏi so với provision), KHÔNG dùng
# cosine FAISS top-1: câu QUAN HỆ có cosine ~0.0 nhưng vẫn trả lời
# được bằng đồ thị -> cosine không phải tín hiệu OOD.
#
# CHỐT BẰNG SỐ ĐO (không đoán) — đo trên 10 câu OOD của
# `evaluation/questions_test.jsonl` + 5 câu in-scope:
#     max OOD = 0.027   (câu OOD_005 "bị lừa chuyển tiền qua mạng")
#     min IN  = 0.258   (câu "Hộ kinh doanh nộp thuế GTGT theo kỳ nào?")
# -> ngưỡng 0.10 nằm GIỮA khoảng cách (không sát biên nào), chặn đúng
# 10/10 câu OOD mà không chặn nhầm câu in-scope.
OOD_REJECT_THRESHOLD = 0.10

OOD_NO_ANSWER_MESSAGE = (
    "Chưa tìm thấy quy định phù hợp trong dữ liệu hiện có. "
    "Cụm từ bạn hỏi không khớp với văn bản nào trong cơ sở dữ liệu."
)


@dataclass
class LegalAnswer:
    """Kết quả trả lời của Legal Agent."""

    answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    route: str = ""
    confidence: float = 0.0
    follow_up: str = ""
    status_block: str = ""
    dataset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    primary_evidence: list[dict[str, Any]] = field(default_factory=list)
    legal_context: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": self.evidence,
            "route": self.route,
            "confidence": round(self.confidence, 4),
            "follow_up": self.follow_up,
            "status_block": self.status_block,
            "dataset": self.dataset,
            "metadata": self.metadata,
            "primary_evidence": self.primary_evidence,
            "legal_context": self.legal_context,
        }


class LegalRetrievalAgent:
    """Agent tra cứu pháp luật nội bộ (FAISS + Neo4j + LLM)."""

    def __init__(
        self,
        store: VectorStore | None = None,
        kg: KgAccess | None = None,
        llm: BaseLLM | None = None,
        router: QueryRouter | None = None,
        graphiti: Any = None,
        rerank_fn: Any = None,
        top_k: int = DEFAULT_TOP_K,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        decomposer: Any = None,
        tracer: TraceRecorder | None = None,
    ) -> None:
        self.store = store or self._default_store()
        self.kg = kg or KgAccess()
        self.tracer = tracer
        if tracer is not None:
            # Wrap LLM ĐÚNG LÚC NÀY (trước khi decomposer capture
            # reference ở bên dưới) để mọi nơi dùng `self.llm` (decomposer,
            # rewriter, synthesis, graphiti, refiner...) đều được quan sát.
            # tracer=None -> giữ llm nguyên, KHÔNG tạo wrapper (yêu cầu:
            # tắt tracing = zero overhead, pipeline y hệt trước).
            self.llm = TracedLLM(llm or get_llm(), tracer)
        else:
            self.llm = llm or get_llm()
        self.router = router or QueryRouter(
            store=self.store,
            graph_lookup=self.kg.lookup,
        )
        # P6: Graphiti làm lớp đồng bộ tri thức MỚI (quan hệ sửa đổi /
        # văn bản mới chưa có trong KG). None -> sẽ tạo lazy, degrade
        # an toàn nếu Neo4j/Graphiti không sẵn sàng.
        self.graphiti = graphiti
        self._graphiti_cache: Any = None
        self.rerank_fn = rerank_fn
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        # X2: cache fn rerank chung (nội bộ + graphiti). _UNSET = chưa đọc.
        self._rerank_fn_cache: Any = _UNSET
        # P7: gap-filling cho câu nhiều ý (không "cắt N ý" cố định).
        # None -> tạo mặc định (dùng LLM của agent; no-op nếu LLM lỗi).
        # "Vẽ lại ý con": rewriter viết lại từng ý con thành câu tra đủ
        # ngữ cảnh (Qwen, dùng câu cha + evidence đã tìm) -> tìm đúng
        # điều/khoản. NoLLM (đường test 65 câu) -> rewriter tự trả nguyên
        # (fallback an toàn, không bịa, không crash) -> đường cũ không đổi.
        self.decomposer = decomposer or QueryDecomposer(
            llm=self.llm,
            sub_query_rewriter=default_sub_query_rewriter(self.llm),
        )
        # Meta lần gap-fill gần nhất (P7: để eval/đo quan sát).
        self.last_gap_meta: dict[str, Any] = {"filled": 0, "rounds": 0}
        # P7 fix OOD: keys (so_hieu, dieu) mà đồ thị khẳng định có quan
        # hệ sửa/bãi/hướng dẫn -> dùng để tránh chặn OOD oan.
        self._graph_relation_keys: set[tuple[str, str]] = set()

        # B1: cửa sổ "hỏi đồ thị cho MẤY FAISS đầu" (ai sửa provision
        # này?). Mặc định 5: provision gốc có thể đứng #4/#5 trong FAISS
        # (S2: 48/2024 D5 ở #5) -> top-3 bỏ sót, vb sửa không bao giờ về.
        # KHÔNG cap theo target/toàn cục: dedup giữ mỗi provision 1 lần,
        # trần thật là rerank top-5 (chấm hết pool rồi cắt 5) + MAX_EVIDENCE.
        self._enrich_window = 5
        # PC-1: ngân sách LLM call route_classify cho LƯỢT này (§14.1:
        # "CHỈ khi rule + reuse không phân loại được ... mới gọi 1 call
        # ngắn" — trần 1 call, không phải mỗi intent 1 call). Reset mỗi
        # lần answer(). Intent intent-classify sau khi hết ngân sách ->
        # ANCHOR_SEMANTIC an toàn (0 call).
        self._route_classify_budget = 1
        self._route_budget_used = 0

    def plan_question(self, question: str) -> dict[str, Any] | None:
        """Expose planning to Orchestrator so refinement happens after planning."""
        planner = getattr(self.decomposer, "plan", None)
        if not callable(planner):
            return None
        try:
            result = planner((question or "").strip())
        except Exception:
            return {"planned": True, "status": "planner_error", "intents": []}
        return result if isinstance(result, dict) else None

    # -- PC-1: pin verified (chống bịa nút vb chết, degrade an toàn) -----

    def _pin_ids(self, u: QueryUnderstanding) -> list[str]:
        """Node phải PIN (provision được hỏi, không cho rerank/OOD đẩy).

        Ưu tiên pin đã verify-by-level (`self._verified_pins`, bao gồm cả
        fallback article_body §4.3 đã verify). Rỗng -> degrade về anchor
        THÔ (đường cũ 0-diff: không LLM / fake KG / registry thiếu /
        câu chưa phân loại). Không bao giờ trả nút của vb chết (R3/R4/
        R12: pin_node_ids() chỉ gồm node ok)."""
        pins = list(getattr(self, "_verified_pins", []) or [])
        if pins:
            return pins
        return list(anchor_node_ids(u))

    # -- P7: gap-filling (câu nhiều ý) ----------------------------------

    def _evidence_for_subquestion(
        self, sub_question: str
    ) -> list[Any]:
        """Chạy pipeline cho 1 sub-question -> evidence.

        Dùng lại ĐÚNG pipeline hiện có (router -> FAISS -> B1 graph ->
        KG) để ý con có chất lượng ngang ý chính, không phải nhánh rẻ
        hơn. Bỏ qua fast-track/OOD vì đây là bước "lấp", còn gate
        chính sẽ chạy lại trên pool gộp.
        """
        sub_u = understand(sub_question)
        sub_decision = self.router.route(sub_question)
        five_route, verified_pins = self._five_route_for_intent(sub_question)
        sub_decision.five_route = five_route
        tracer = getattr(self, "tracer", None)
        if tracer is not None:
            tracer.set_intent(sub_question)
            tracer.emit("route_decision", OK,
                        reason="route selected for planned sub-question",
                        data={"five_route": five_route, "actual_route": five_route,
                              "router_route": sub_decision.route,
                              "intent_id": getattr(self, "_active_intent_id", ""),
                              "target_node_ids": list(verified_pins or [])})
            tracer.set_intent(None)
        temporal = parse_temporal_constraint(sub_question)
        version_sensitive = bool(temporal) or sub_u.intent == "hieu_luc"
        if five_route == "exact" and verified_pins:
            exact_ids = list(verified_pins)
            if version_sensitive:
                target_date = (
                    str(temporal.get("normalized_target_date")) if temporal
                    else str(getattr(self, "_query_time", "") or "")
                )
                exact_ids = self._enrich_with_graph_relations(
                    sub_u, exact_ids, as_of_date=target_date,
                    relation_types={"AMENDS", "REPEALS"},
                )
                if not getattr(self, "_last_graph_relations", []):
                    exact_ids = self._enrich_with_graph_relations(
                        sub_u, exact_ids, as_of_date=None,
                        relation_types={"AMENDS", "REPEALS"},
                    )
            return self.kg.evidence_for_many(exact_ids)
        if five_route == "anchor_unresolved":
            # Return only verified live anchors (e.g. article-body fallback).
            return self.kg.evidence_for_many(verified_pins)
        sub_hits = self._vector_hits(sub_question, sub_decision)
        sub_node_ids = self._node_ids_from_hits(sub_hits)
        for nid in verified_pins or anchor_node_ids(sub_u):
            if nid not in sub_node_ids:
                sub_node_ids.append(nid)
        sub_node_ids = self._enrich_with_graph_relations(
            sub_u, sub_node_ids,
            as_of_date=(str(temporal.get("normalized_target_date")) if temporal else None),
            relation_types=({"AMENDS", "REPEALS"} if version_sensitive else None),
        )
        return self.kg.evidence_for_many(sub_node_ids)

    def _graphiti_bridge(self) -> Any:
        """Lazy bridge; trả None nếu không dùng được (offline degrade).

        Cache trong `self._graphiti_cache` (KHÔNG cùng tên method, tránh
        instance-attribute shadow bound-method -> bug bridge không tạo).
        """
        if self.graphiti is not None:
            return self.graphiti
        cached = self._graphiti_cache
        if cached is None:
            try:
                from agents.GraphitiBridge import GraphitiBridge

                cached = GraphitiBridge(use_llm=False)
            except Exception:
                cached = False
        self._graphiti_cache = cached
        self.graphiti = cached or None
        return self.graphiti

    def graphiti_block(
        self, question: str, top_n: int = 5
    ) -> tuple[str, list[dict[str, Any]]]:
        """Trả (text_block, facts) từ graphiti.

        Ưu tiên EXACT (P7 #1): nếu câu hỏi có anchor (số hiệu vb + điều)
        -> tra cạnh graphiti bằng Cypher theo anchor, KHÔNG dùng vector.
        Không có anchor -> mới dùng vector search, giới hạn `top_n` fact.

        Degrade: bridge unavailable hay search lỗi -> ("", []).
        """
        bridge = self._graphiti_bridge()
        if bridge is None:
            return "", []

        results = self._graphiti_search(bridge, question, top_n)
        if not results:
            return "", []
        lines = []
        facts = []
        for r in results:
            src = r.get("source_node") or {}
            dst = r.get("target_node") or {}
            fact = (r.get("fact") or "").strip()
            name = r.get("name") or ""
            bit = f"- {name}: {src.get('citation', '?')} → {dst.get('citation', '?')}"
            if fact and fact != name:
                bit += f" ({fact})"
            lines.append(bit)
            facts.append({
                "source": "graphiti",
                "fact": fact,
                "name": name,
                "citation": bit,
                "text": (src.get("text") or dst.get("text") or "")[:400],
                "metadata": {
                    "valid_at": r.get("valid_at"),
                    "source_node": src.get("kg_id"),
                    "target_node": dst.get("kg_id"),
                    "match": r.get("match"),
                },
            })
        return "\n".join(lines), facts

    def _graphiti_search(
        self, bridge: Any, question: str, top_n: int
    ) -> list[dict[str, Any]]:
        """Chọn chiến lược search graphiti theo anchor của câu hỏi.

        - Có anchor (doc + điều) -> `search_exact` (Cypher, không embedding).
          Đây là đường chính cho câu hỏi pháp lý có số hiệu rõ.
        - Có anchor nhưng exact ra rỗng -> thử vector (cap nhỏ) để không bỏ
          sót thông tin.
        - Không có anchor -> vector search (cap nhỏ, tránh nhiễu).
        """
        try:
            u = understand(question)
        except Exception:
            u = None

        docs = list(u.anchors.doc_numbers) if u else []
        arts = list(u.anchors.articles) if u else []

        if docs or arts:
            try:
                exact = bridge.search_exact(
                    docs, arts, query=question, num_results=top_n
                )
            except Exception:
                exact = []
            if exact:
                return exact

        # Fallback: vector (giới hạn nhỏ để không nhiễu prompt).
        try:
            return bridge.search_with_nodes(question, num_results=top_n)
        except Exception:
            return []

    @staticmethod
    def _default_store() -> VectorStore:
        root = Path(__file__).resolve().parents[2]
        return VectorStore(
            processed_root=root / "data" / "processed",
            index_root=root / "data" / "vector_db",
        )

    # -- Hỏi ngược user ---------------------------------------------------

    @staticmethod
    def _follow_up_question(u: QueryUnderstanding) -> str:
        """Câu hỏi làm rõ, hỏi đúng thông tin đang thiếu.

        Không cố định số câu (quyết định #7) — hỏi đúng phần thiếu:
        nhóm pháp lý (dataset) chưa rõ, hay chưa rõ văn bản/điều.
        """
        if not u.legal_group:
            return (
                "Câu hỏi chưa rõ thuộc lĩnh vực nào (doanh nghiệp, thuế GTGT, "
                "thuế TNDN hay quản lý thuế). Bạn tra cứu về vấn đề gì?"
            )
        if not u.anchors.doc_numbers:
            return (
                "Bạn muốn tra cứu quy định nào cụ thể (số hiệu văn bản, hoặc "
                "điều/khoản) trong "
                f"{u.legal_group.replace('_', ' ')}?"
            )
        return (
            "Bạn muốn tra cứu điều/khoản nào trong "
            f"{', '.join(u.anchors.doc_numbers)}?"
        )

    # -- PC-1: 5-route (verify-by-level) + per-intent routing --------------

    def anchor_verify_source(self) -> Any | None:
        """Nguồn verify-by-level (PC-1): KgAccess thật (registry + cây
        node trong bộ nhớ — T1 cấu trúc, không chạm Neo4j).

        Fake/None (test) trả về nếu không đủ interface -> classify degrade
        an toàn (registry trống -> SEMANTIC, không bịa, không chặn oan)."""
        kg = self.kg
        reg_fn = getattr(kg, "registry", None)
        if reg_fn is None or not callable(reg_fn):
            return None
        return kg

    def _route_classify_llm(self) -> Any:
        """LLM dùng cho route_classify, GIỚI HẠN ngân sách §14.1: toàn
        lượt tối đa 1 call route_classify (chia sẻ giữa classify câu +
        per-intent, đếm qua `self._route_budget_used`). Hết ngân sách ->
        trả JSON an toàn (anchor_semantic), KHÔNG gọi LLM thật. Purpose
        khác (coverage/synthesis/rewriter) gọi LLM thật bình thường."""
        agent = self

        class _RouteBudgetLLM:
            def __init__(self, inner: Any) -> None:
                self._inner = inner
                self._purpose = ""

            @property
            def name(self) -> str:
                return getattr(self._inner, "name", "route")

            def available(self) -> bool:
                return self._inner.available()

            def active_backend(self) -> Any:  # type: ignore[no-redef]
                return self._inner.active_backend()

            def set_purpose(self, p: str) -> None:
                self._purpose = p
                sp = getattr(self._inner, "set_purpose", None)
                if sp:
                    sp(p)

            def complete(self, prompt: str) -> str:
                if self._purpose != "route_classify":
                    return self._inner.complete(prompt)
                used = getattr(agent, "_route_budget_used", 0)
                if used >= agent._route_classify_budget:
                    # Hết ngân sách -> safe default (không gọi LLM thật,
                    # không bịa EXACT; giữ anchor pin + semantic bổ sung).
                    return '{"route": "anchor_semantic"}'
                agent._route_budget_used = used + 1
                return self._inner.complete(prompt)

        return _RouteBudgetLLM(self.llm)

    def _five_route_for_intent(
        self, sub_query: str
    ) -> tuple[str, list[str]]:
        """PC-1 (R11/R12): phân loại 1 intent con (text sub_query đã
        rewriter) -> (five_route, pin_node_ids).

        Mỗi intent tự verify-by-level trên ĐÚNG anchor của nó — intent
        UNRESOLVED không kéo thụt intent khác (anti pull-down, F5b).
        Route-classify chia sẻ ngân sách toàn lượt (trần 1 call, §14.1).
        Lỗi bất kỳ -> ("", []) (degrade: không đổi behavior)."""
        try:
            from retrieval.AnchorVerifier import classify_query

            res = classify_query(
                sub_query, understand(sub_query),
                llm=self._route_classify_llm(),
                source=self.anchor_verify_source(),
            )
            return res.route, res.pin_node_ids()
        except Exception:
            return "", []

    # -- Thu thập evidence -------------------------------------------------

    def _vector_hits(
        self,
        question: str,
        decision: RouteDecision,
    ) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        dataset = decision.datasets[0] if decision.datasets else None
        hits = self.store.search(
            question,
            dataset=dataset,
            top_k=self.top_k,
            rerank_top_n=self.rerank_top_n,
            rerank_fn=self.rerank_fn,
        )
        if getattr(decision, "five_route", "") == "anchor_semantic":
            anchors = understand(question).anchors.doc_numbers
            if anchors:
                normalized = {"".join(x.upper().split()) for x in anchors}
                hits = [
                    hit for hit in hits
                    if "".join(str((hit.get("metadata") or {}).get(
                        "document_number") or "").upper().split()) in normalized
                ]
        return hits

    def _node_ids_from_hits(
        self, hits: list[dict[str, Any]]
    ) -> list[str]:
        """Suy ra node_id KG từ metadata của hit vector.

        Metadata có document_number + article_number (+ clause_number) ->
        dựng node id theo đúng convention của loader (`_vb_key`).
        """
        from graph.kg.loader import _vb_key

        out: list[str] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            so = str(meta.get("document_number") or "").strip()
            if not so:
                continue
            key = _vb_key(so)
            dieu = str(meta.get("article_number") or "").strip()
            khoan = str(meta.get("clause_number") or "").strip()
            if dieu and khoan:
                nid = f"{key}_D{dieu}_K{khoan}"
            elif dieu:
                nid = f"{key}_D{dieu}"
            else:
                continue
            if nid not in out:
                out.append(nid)
        return out

    def _enrich_with_graph_relations(
        self,
        u: QueryUnderstanding,
        node_ids: list[str],
        *,
        as_of_date: str | None = None,
        relation_types: set[str] | None = None,
    ) -> list[str]:
        """B1: kéo thêm provision của "văn bản sửa" vào pool.

        Với MẤY node FAISS tốt nhất (cửa sổ `self._enrich_window`,
        mặc định 5), hỏi đồ thị: "điều này bị AMENDS/REPEALS/GUIDES
        bởi đâu?" rồi thêm provision phía "văn bản sửa"
        (by_document/by_article) vào list node_id. Nhờ đó câu hỏi
        "quy định MỚI / hiện hành" được trả bằng đúng văn bản sửa,
        không phải văn bản gốc (vector không phân biệt thời gian).

        Vì sao top-5 chứ không top-3 (S2): provision gốc có thể đứng
        #4/#5 trong FAISS (S2: 48/2024 D5 ở #5) -> cửa sổ top-3 bỏ
        sót, vb sửa (149/2025) không bao giờ về. Mở rộng cửa sổ phủ
        được cả hit sâu.

        An toàn / rẻ:
          - Neo4j tắt / lỗi -> trả nguyên list (không enrich).
          - Mỗi provision (so_hieu+dieu) chỉ query 1 lần (dedup).
          - Mỗi provision "văn bản sửa" chỉ thêm 1 lần (dedup seen).
          - KHÔNG cap theo target/toàn cục: trần thật là rerank
            top-5 (chấm hết pool rồi cắt 5) + MAX_EVIDENCE.
        """
        if not node_ids:
            return node_ids

        from graph.kg.loader import _vb_key

        seen = set(node_ids)
        targets: set[tuple[str, str]] = set()
        # Ghi nhớ node NÀO có quan hệ sửa/bãi/hướng dẫn trong đồ thị ->
        # OOD gate (X4) tin bằng chứng đồ thị này (câu chắc chắn có
        # trong DB) thay vì chỉ tin điểm rerank.
        # (reset tại đây: mỗi lần enrich làm sạch lại bộ keys này.)
        self._graph_relation_keys = set()
        self._last_graph_relations = []
        self._last_graph_relation_node_ids: list[str] = []
        # Chỉ cân nhắc top-<window> FAISS hit (những node đầu của pool).
        for nid in node_ids[: self._enrich_window]:
            node = self.kg.node(nid)
            if node is None or not node.so_hieu or not node.dieu:
                continue
            key = (node.so_hieu, node.dieu)
            if key in targets:
                continue
            targets.add(key)

        for so_hieu, dieu in targets:
            for rel in self.kg.relations_of(
                so_hieu, article=dieu, as_of_date=as_of_date
            ):
                if relation_types and str(rel.get("relation_type") or "").upper() not in relation_types:
                    continue
                # Có quan hệ thật trong đồ thị -> câu hỏi chắc chắn có
                # trong DB (dù reranker chấm thấp). Ghi nhớ cho OOD gate.
                self._graph_relation_keys.add((so_hieu, dieu))
                by_doc = str(rel.get("by_document") or "").strip()
                by_art = str(rel.get("by_article") or "").strip()
                by_khoan = str(rel.get("by_clause") or "").strip()
                relation = {
                    "from": by_doc,
                    "relation_type": str(rel.get("relation_type") or ""),
                    "to": so_hieu,
                    "from_article": by_art,
                    "from_clause": by_khoan,
                    "to_article": dieu,
                    # Preserve the target provision granularity from the KG
                    # relation property (for example, "khoản 2"). Without
                    # this field the acceptance grader can only see document
                    # and article level and would miss clause-level errors.
                    "to_clauses": str(rel.get("clauses") or ""),
                    "relation_id": rel.get("relation_id"),
                    "effective_date": rel.get("effective_date"),
                }
                relation_key = (
                    str(relation.get("relation_id") or ""),
                    relation["from"], relation["relation_type"],
                    relation["to"], relation["from_article"],
                    relation["from_clause"], relation["to_article"],
                    relation["to_clauses"],
                )
                if relation_key not in {
                    (
                        str(existing.get("relation_id") or ""),
                        existing.get("from", ""), existing.get("relation_type", ""),
                        existing.get("to", ""), existing.get("from_article", ""),
                        existing.get("from_clause", ""), existing.get("to_article", ""),
                        existing.get("to_clauses", ""),
                    )
                    for existing in self._last_graph_relations
                }:
                    self._last_graph_relations.append(relation)
                if not by_doc or not by_art:
                    continue
                tkey = _vb_key(by_doc)
                cand = (
                    f"{tkey}_D{by_art}_K{by_khoan}"
                    if by_khoan
                    else f"{tkey}_D{by_art}"
                )
                if cand not in getattr(self, "_last_graph_relation_node_ids", []):
                    self._last_graph_relation_node_ids.append(cand)
                if not by_khoan:
                    clause_children = self.kg.children(cand)
                    relation_text = str(rel.get("clauses") or "")
                    matched_children: list[Any] = []
                    for child in clause_children:
                        child_text = str(getattr(child, "text", "") or "")
                        child_clause = str(getattr(child, "khoan", "") or "")
                        if child_clause and relation_text:
                            clauses = re.findall(r"\d+", relation_text)
                            if clauses and any(
                                re.search(
                                    rf"(?:khoản|khoan)\s*{re.escape(clause)}",
                                    child_text,
                                    re.I,
                                ) or re.search(
                                    rf"^Khoản\s*{re.escape(clause)}",
                                    child_text,
                                    re.I,
                                )
                                for clause in clauses
                            ):
                                matched_children.append(child)
                    for child in matched_children:
                        child_id = str(getattr(child, "node_id", "") or "")
                        if child_id and child_id not in seen:
                            seen.add(child_id)
                            node_ids.append(child_id)
                            if child_id not in getattr(self, "_last_graph_relation_node_ids", []):
                                self._last_graph_relation_node_ids.append(child_id)
                if cand in seen:
                    continue
                seen.add(cand)
                node_ids.append(cand)
        tracer = getattr(self, "tracer", None)
        if tracer is not None:
            tracer.set_intent(str(getattr(self, "_active_intent_id", "") or "") or None)
            tracer.emit(
                "kg_relations", OK if self._last_graph_relations else EMPTY,
                reason="relations returned by KG lookup",
                data={"actual_relations": list(self._last_graph_relations),
                      "intent_id": str(getattr(self, "_active_intent_id", "") or "")},
            )
            tracer.set_intent(None)
        return node_ids

    def _status_block(
        self, node_ids: list[str], u: QueryUnderstanding
    ) -> str:
        """Tóm tắt trạng thái hiệu lực từ đồ thị (P5)."""
        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for nid in node_ids:
            node = self.kg.node(nid)
            if node is None or not node.dieu:
                continue
            key = (node.so_hieu, node.dieu)
            if key in seen:
                continue
            seen.add(key)
            st = self.kg.status_of(node.so_hieu, node.dieu)
            status = st.get("status", "unknown")
            bits = [f"{node.so_hieu} Điều {node.dieu}: {status}"]
            for rel in st.get("relations", [])[:3]:
                bits.append(
                    f"  - {rel.get('relation_type')} bởi "
                    f"{rel.get('by_document')} (hiệu lực "
                    f"{rel.get('effective_date') or 'không rõ'})"
                )
            lines.append("\n".join(bits))
        return "\n".join(lines)

    def _resolve_versions(
        self, evidences: list[Any], u: QueryUnderstanding, *,
        as_of_date: str | None = None,
        temporal_constraint: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Consult Neo4j's provision resolver; never treat graph absence as proof.

        The result is explicitly marked unverified until its relation/effective
        date has been checked against an official primary source by a caller.
        """
        dates = list(getattr(u.anchors, "dates", []) or [])
        temporal_value = temporal_constraint or parse_temporal_constraint(
            str(getattr(u, "query", "") or "")
        )
        temporal = dict(temporal_value) if temporal_value else None
        as_of = None
        query_text = str(getattr(u, "query", "") or "")
        if temporal:
            if temporal.get("operator") == "CURRENT":
                as_of = str(as_of_date or "").strip()
                if as_of:
                    temporal["normalized_target_date"] = as_of
                    temporal["resolution_date_source"] = "query_time"
                    temporal["temporal_constraint_source"] = (
                        temporal.get("temporal_constraint_source") or "query_time_current"
                    )
            else:
                as_of = str(temporal.get("normalized_target_date") or "")
        elif u.intent in {"time_travel", "hieu_luc"} and dates:
            try:
                d, m, y = dates[0].split("/")
                as_of = date(int(y), int(m), int(d)).isoformat()
                temporal = {
                    "operator": "AT", "raw_date": as_of,
                    "normalized_target_date": as_of, "inclusive": True,
                    "temporal_constraint_source": "query_date_fallback",
                }
            except (ValueError, TypeError):
                return "", {"verified": False, "status": "invalid_query_time",
                             "as_of_date": None, "items": [], "temporal_constraint": temporal}
        elif u.intent == "time_travel":
            # Time-travel bắt buộc cần mốc pháp lý trong câu hỏi.
            return "", {
                "verified": False, "status": "missing_query_time",
                "as_of_date": None, "items": [], "temporal_constraint": temporal,
            }

        # Không có mốc trong câu hỏi -> chỉ dùng ngày tiếp nhận được caller
        # truyền tường minh. Resolver không tự lấy ngày hiện tại vì sẽ làm
        # mất tính tái lập và có thể che query_time sai định dạng.
        if not as_of:
            qt = str(as_of_date or "").strip()
            if not qt:
                return "", {
                    "verified": False, "status": "missing_query_time",
                    "as_of_date": None, "items": [],
                }
            try:
                parsed = date.fromisoformat(qt)
                if parsed.isoformat() != qt:
                    raise ValueError("query_time must be YYYY-MM-DD")
                as_of = parsed.isoformat()
            except (ValueError, TypeError):
                return "", {
                    "verified": False, "status": "invalid_query_time",
                    "as_of_date": None, "items": [],
                }

        if temporal is None and u.intent == "hieu_luc":
            temporal = {
                "operator": "AT", "raw_date": None,
                "normalized_target_date": as_of, "inclusive": True,
                "temporal_constraint_source": "query_time_current",
            }

        rows: list[dict[str, Any]] = []
        lines: list[str] = []
        governing = getattr(self.kg, "governing", None)
        if not callable(governing):
            return "", {
                "verified": False,
                "status": "governing_lookup_unavailable",
                "as_of_date": as_of,
                "items": [],
                "reason": "KG adapter does not expose provision-level governing lookup.",
            }
        seen: set[tuple[str, str, str]] = set()
        for ev in evidences:
            meta = getattr(ev, "metadata", {}) or {}
            so = str(meta.get("so_hieu") or "")
            article = str(meta.get("dieu") or "")
            clause = str(meta.get("khoan") or "") or None
            key = (so, article, clause or "")
            if not so or not article or key in seen:
                continue
            seen.add(key)
            result = governing(
                so, article, as_of_date=as_of, clause=clause
            )
            rows.append(result)
            lines.append(
                f"{so} Điều {article}"
                + (f" Khoản {clause}" if clause else "")
                + f": graph_status={result.get('status', 'unknown')}; "
                + f"as_of={as_of}; effective_date="
                + f"{result.get('effective_date_used') or 'unknown'}; "
                + f"source={result.get('effective_date_source') or 'unknown'}"
            )
        all_graph_resolved = bool(rows) and all(
            r.get("status") != "unknown"
            and r.get("provision_in_effect_at_T") is not None
            and r.get("effective_date_source") != "unknown"
            for r in rows
        )
        return "\n".join(lines), {
            "verified": False,
            "status": "graph_resolved_needs_primary_source"
            if all_graph_resolved else "unknown",
            "as_of_date": as_of,
            "items": rows,
            "temporal_constraint": temporal,
            "reason": "KG status is structural evidence, not T2 legal verification.",
        }

    # -- Trả lời -----------------------------------------------------------

    def answer(
        self,
        question: str,
        understanding: QueryUnderstanding | None = None,
        planner_state: dict[str, Any] | None = None,
        user_answer: str = "",
        planner_plan: dict[str, Any] | None = None,
        query_time: str = "",
    ) -> LegalAnswer:
        """Trả lời 1 câu hỏi pháp luật.

        `query_time` (ISO yyyy-mm-dd): mốc hệ thống tiếp nhận câu hỏi,
        dùng làm `as_of_date` khi câu hỏi KHÔNG nêu mốc thời gian pháp
        lý của riêng mình. Ngày người dùng nêu trong câu hỏi vẫn ưu tiên.
        """
        question = (question or "").strip()
        if not question:
            return LegalAnswer(
                answer=NO_EVIDENCE_MESSAGE,
                follow_up="Bạn muốn tra cứu vấn đề pháp lý nào?",
            )

        u = understanding or understand(question)
        # Reset bằng chứng đồ thị của câu này (B1 sẽ ghi vào).
        self._graph_relation_keys = set()
        # PC-1: reset ngân sách route_classify (toàn lượt = 1 call) +
        # pin verified (rỗng tới khi classify chạy).
        self._route_budget_used = 0
        self._verified_pins: list[str] = []
        # Mốc thời gian hệ thống tiếp nhận câu hỏi (fallback as_of khi câu
        # hỏi không nêu mốc pháp lý riêng; ngày trong câu luôn ưu tiên).
        # Capture the request date at the entry point. A supplied harness date
        # stays authoritative; real calls without one use receipt date here,
        # rather than letting the resolver silently choose a date downstream.
        self._query_time = str(query_time or date.today().isoformat()).strip()

        # Tracing (chỉ khi bật; tắt = không phát span, pipeline không đổi).
        tracer = getattr(self, "tracer", None)

        # Multi-intent planner runs before any whole-question/vector retrieval.
        # Each resolved intent owns only its own retrieved node IDs; a missing
        # or ambiguous intent is clarified independently and is not searched.
        planner = getattr(getattr(self, "decomposer", None), "plan", None)
        planner_rewrites: list[dict[str, Any]] = []
        if planner_state:
            planned_intents = [dict(item) for item in planner_state.get("intents", [])]
            pending_ids = planner_state.get("pending_intent_ids")
            if not isinstance(pending_ids, list):
                pending_ids = [planner_state.get("pending_intent_id")]
            pending_ids = [str(value) for value in pending_ids if str(value)]
            if len(pending_ids) != len(set(pending_ids)):
                return LegalAnswer(
                    answer="Không thể khôi phục các ý đang chờ do trạng thái Planner bị trùng ID.",
                    route="error",
                    metadata={"planner_state_error": "duplicate_pending_intent_id",
                              "execution": "ERROR"},
                )
            answers = parse_intent_clarification_answers(pending_ids, user_answer)
            pending_by_id = {
                str(item.get("intent_id") or ""): item
                for item in planned_intents
                if item.get("needs_clarification") is True
            }
            if not pending_ids or not set(pending_ids).issubset(pending_by_id):
                return LegalAnswer(
                    answer=NO_EVIDENCE_MESSAGE, route="error",
                    follow_up="Không thể khôi phục đúng các ý đang chờ làm rõ.",
                    metadata={"planner_state_error": "pending_intent_not_found",
                              "execution": "ERROR"},
                )
            if not answers:
                return LegalAnswer(
                    answer="Chưa ánh xạ được câu trả lời vào từng ý đang chờ.",
                    route="ask",
                    follow_up=format_intent_clarifications(planned_intents),
                    metadata={"planner_state": planner_state,
                              "planner_state_error": "clarification_answer_unmapped"},
                )
            for intent_id, response in answers.items():
                item = pending_by_id[intent_id]
                original_sub_query = str(
                    item.get("original_sub_query") or item.get("sub_query") or ""
                ).strip()
                rewriter = getattr(self.decomposer, "rewrite_clarified_intent", None)
                if not callable(rewriter):
                    rewritten, audit = "", {
                        "status": "ERROR", "reason": "intent_rewriter_unavailable",
                        "intent_id": intent_id,
                    }
                else:
                    rewritten, audit = rewriter(
                        original_question=str(planner_state.get("original_question") or question),
                        original_sub_query=original_sub_query,
                        clarification=str(item.get("clarification") or ""),
                        user_answer=response,
                    )
                    audit["intent_id"] = intent_id
                planner_rewrites.append(audit)
                if tracer is not None:
                    tracer.set_intent(intent_id)
                    tracer.emit(
                        "planner_rewrite",
                        OK if audit.get("status") == "ANCHORS_VALIDATED" else FAILED,
                        reason=str(audit.get("status") or "rewrite failed"),
                        data=audit,
                    )
                    tracer.set_intent(None)
                if audit.get("status") == "ERROR":
                    return LegalAnswer(
                        answer="Chưa thể xử lý ý đã làm rõ do lỗi khi chuẩn hóa câu truy vấn.",
                        route="error",
                        metadata={"execution": "ERROR", "planner_rewrites": planner_rewrites,
                                  "planner_state": planner_state},
                    )
                if audit.get("status") == "REJECTED":
                    item["clarification"] = (
                        "Câu truy vấn sau chuẩn hóa không giữ đúng anchor đã nêu. "
                        "Vui lòng xác nhận lại số hiệu văn bản/điều khoản cho ý này."
                    )
                    item["needs_clarification"] = True
                    item["rewrite_audit"] = audit
                    continue
                item["original_sub_query"] = original_sub_query
                item["clarification_answer"] = response
                item["rewritten_sub_query"] = rewritten
                item["sub_query"] = rewritten
                item["needs_clarification"] = False
                item["clarification"] = ""
            plan = {"planned": True, "status": "planned", "intents": planned_intents}
        else:
            plan = planner_plan if isinstance(planner_plan, dict) else (
                planner(question) if callable(planner) else None
            )
        planned_intents = (
            plan.get("intents", []) if isinstance(plan, dict) else []
        )
        seen_intent_ids: set[str] = set()
        for index, intent in enumerate(planned_intents, start=1):
            intent_id = str(intent.get("intent_id") or f"intent-{index}")
            if intent_id in seen_intent_ids:
                intent_id = f"intent-{index}"
            intent["intent_id"] = intent_id
            seen_intent_ids.add(intent_id)
        if tracer is not None:
            tracer.set_intent(None)
            planner_status = (plan or {}).get("status", "unavailable") if isinstance(plan, dict) else "unavailable"
            planner_failed = planner_status in {"llm_error", "parse_error", "invalid_plan", "planner_error"}
            tracer.emit(
                "planner", FAILED if planner_failed else (OK if isinstance(plan, dict) and planned_intents else EMPTY),
                reason=str((plan or {}).get("status", "planner unavailable"))
                if isinstance(plan, dict) else "planner unavailable",
                data={
                    "planner_status": (plan or {}).get("status", "unavailable")
                    if isinstance(plan, dict) else "unavailable",
                    "planner_diagnostic": (plan or {}).get("planner_diagnostic")
                    if isinstance(plan, dict) else None,
                    "intent_count": len(planned_intents),
                    "intents": [{
                        "intent_id": str(item.get("intent_id") or ""),
                        "sub_query": str(item.get("sub_query") or ""),
                        "source_clause": str(item.get("source_clause") or ""),
                        "temporal_constraint": item.get("temporal_constraint"),
                        "temporal_expression": str(item.get("temporal_expression") or ""),
                        "is_out_of_domain": bool(item.get("is_out_of_domain")),
                        "needs_clarification": bool(item.get("needs_clarification")),
                        "clarification": str(item.get("clarification") or ""),
                    } for item in planned_intents],
                },
            )
        def current_planner_state() -> dict[str, Any]:
            pending = [
                item for item in planned_intents
                if item.get("needs_clarification")
                and str(item.get("clarification") or "").strip()
            ]
            pending_ids = [str(item.get("intent_id") or "") for item in pending]
            return {
                "original_question": str(
                    (planner_state or {}).get("original_question") or question
                ),
                "intents": planned_intents,
                "pending_intent_id": pending_ids[0] if pending_ids else "",
                "pending_intent_ids": pending_ids,
            }

        def pending_follow_up() -> str:
            return format_intent_clarifications(planned_intents)

        def current_planner_audit() -> dict[str, Any]:
            return {
                "original_question": str(
                    (planner_state or {}).get("original_question") or question
                ),
                "intents": [{
                    "intent_id": str(item.get("intent_id") or ""),
                    "original_sub_query": str(
                        item.get("original_sub_query") or item.get("sub_query") or ""
                    ),
                    "clarification": str(item.get("clarification") or ""),
                    "clarification_answer": str(item.get("clarification_answer") or ""),
                    "rewritten_sub_query": str(item.get("rewritten_sub_query") or ""),
                    "needs_clarification": bool(item.get("needs_clarification")),
                } for item in planned_intents],
            }

        multi_planned = bool(
            isinstance(plan, dict) and plan.get("status") == "planned"
            and len(planned_intents) >= 2
        )
        if multi_planned:
            # Do not route or search the whole multi-intent question. Each
            # sub-question is routed independently inside retrieval below.
            decision = RouteDecision(
                route=ROUTE_VECTOR, understanding=u,
                reason="multi-intent plan; route each sub-question",
                five_route="full",
            )
        if (
            is_multi_intent(question)
            and (not isinstance(plan, dict) or plan.get("status") != "planned")
        ):
            return LegalAnswer(
                answer=(
                    "Chưa thể xử lý câu hỏi nhiều ý do Planner không tạo được "
                    "kế hoạch phân rã hợp lệ. Vui lòng thử lại sau."
                ),
                route="error",
                metadata={"five_route": "full", "planner": plan,
                          "planner_status": (plan or {}).get("status")
                          if isinstance(plan, dict) else "unavailable",
                          "execution": "ERROR"},
            )
        if not multi_planned:
            # Planner/fail-closed check precedes any Router confidence lookup.
            question_temporal = parse_temporal_constraint(question)
            if (u.intent == "time_travel" and not u.anchors.dates
                    and (question_temporal or {}).get("operator") != "CURRENT"):
                if tracer is not None:
                    tracer.set_intent(question)
                    tracer.emit("legal_version", INSUFFICIENT,
                                reason="query time missing; clarification required",
                                data={"actual_version": {
                                    "as_of_date": None,
                                    "status": "missing_query_time",
                                    "items": [], "verified": False,
                                }})
                    tracer.set_intent(None)
                return LegalAnswer(
                    answer="Chưa rõ mốc thời điểm cần áp dụng quy định.",
                    route="ask",
                    follow_up="Bạn muốn xác định hiệu lực tại ngày/tháng/năm nào?",
                    metadata={"version_resolution": {
                        "verified": False, "status": "missing_query_time",
                    }},
                )
            decision = self.router.route(question)
        if (
            isinstance(plan, dict) and plan.get("planned")
            and plan.get("status") == "planned"
            and planned_intents
        ):
            decision.five_route = "full"
            active: list[dict[str, Any]] = []
            all_evidence: list[Any] = []
            seen_nodes: set[str] = set()
            clarifications: list[str] = []
            remaining = MAX_EVIDENCE
            for intent_index, intent in enumerate(planned_intents):
                sub_query = str(intent.get("sub_query") or "").strip()
                if not sub_query:
                    continue
                if intent.get("is_out_of_domain") is True:
                    active.append({
                        "intent_id": str(intent.get("intent_id") or ""),
                        "intent_index": intent_index,
                        "sub_query": sub_query,
                        "original_sub_query": str(intent.get("original_sub_query") or sub_query),
                        "source_clause": str(intent.get("source_clause") or ""),
                        "node_ids": [], "status": "stopped",
                        "execution": "SKIPPED", "evidence_status": "ood",
                        "pending": "ood_gate", "intent_verdict": "OOD",
                        "is_out_of_domain": True,
                    })
                    continue
                sub_understanding = understand(sub_query)
                intent_temporal = (
                    intent.get("temporal_constraint")
                    or parse_temporal_constraint(sub_query)
                )
                if (sub_understanding.intent == "time_travel"
                        and not sub_understanding.anchors.dates
                        and (intent_temporal or {}).get("operator") != "CURRENT"):
                    if tracer is not None:
                        tracer.set_intent(str(intent.get("intent_id") or sub_query))
                        tracer.emit("legal_version", INSUFFICIENT,
                                    reason="query time missing; clarification required",
                                    data={"actual_version": {
                                        "as_of_date": None,
                                        "status": "missing_query_time",
                                        "items": [], "verified": False,
                                    }, "intent_id": str(intent.get("intent_id") or "")})
                        tracer.set_intent(None)
                    intent["needs_clarification"] = True
                    intent["clarification"] = (
                        "Bạn muốn xác định hiệu lực của ý này tại ngày/tháng/năm nào?"
                    )
                    clarifications.append(intent["clarification"])
                    continue
                if intent.get("needs_clarification"):
                    clarification = str(intent.get("clarification") or "").strip()
                    if clarification:
                        clarifications.append(clarification)
                    continue
                try:
                    self._last_graph_relations = []
                    self._active_intent_id = str(intent.get("intent_id") or "")
                    rank_spans_before = sum(
                        1 for span in (getattr(tracer, "spans", []) if tracer is not None else [])
                        if getattr(span, "layer", "") == "retrieval_rank"
                        and getattr(span, "intent", None) == self._active_intent_id
                    )
                    if tracer is not None:
                        tracer.set_intent(self._active_intent_id or None)
                    found = self._evidence_for_subquestion(sub_query)
                    execution = "SUCCESS" if found else "EMPTY"
                    if tracer is not None:
                        rank_span_count = sum(
                            1 for span in getattr(tracer, "spans", [])
                            if getattr(span, "layer", "") == "retrieval_rank"
                            and getattr(span, "intent", None) == self._active_intent_id
                        )
                        if rank_span_count == rank_spans_before:
                            tracer.set_intent(self._active_intent_id or None)
                            tracer.emit("retrieval_rank", OK if found else EMPTY,
                                        reason="exact/direct retrieval result order",
                                        data={"node_ids_ranked": [
                                            str(getattr(ev, "node_id", "") or "") for ev in found
                                            if getattr(ev, "node_id", None)],
                                            "ranking_stage": "direct_retrieval_result",
                                            "top_k": self.top_k,
                                            "intent_id": self._active_intent_id})
                except Exception:
                    found, execution = [], "ERROR"
                finally:
                    self._active_intent_id = ""
                    if tracer is not None:
                        tracer.set_intent(None)
                if tracer is not None:
                    tracer.set_intent(str(intent.get("intent_id") or ""))
                    tracer.emit(
                        "kg_relations",
                        OK if self._last_graph_relations else EMPTY,
                        reason="relations returned by KG lookup for this intent",
                        data={"actual_relations": list(self._last_graph_relations),
                              "intent_id": str(intent.get("intent_id") or "")},
                    )
                    tracer.set_intent(None)
                local_evidence: list[Any] = []
                for ev in found:
                    nid = str(getattr(ev, "node_id", "") or "")
                    if remaining <= 0:
                        break
                    if nid and nid not in seen_nodes:
                        seen_nodes.add(nid)
                        all_evidence.append(ev)
                        local_evidence.append(ev)
                        remaining -= 1
                    elif nid and nid in seen_nodes:
                        local_evidence.append(ev)
                active.append({
                    "intent_id": str(intent.get("intent_id") or ""),
                    "intent_index": intent_index,
                    "sub_query": sub_query,
                    "original_sub_query": str(intent.get("original_sub_query") or sub_query),
                    "original_sub_question": str(intent.get("original_sub_question") or intent.get("original_sub_query") or sub_query),
                    "source_clause": str(intent.get("source_clause") or ""),
                    "rewritten_sub_query": str(intent.get("rewritten_sub_query") or ""),
                    "graph_node_ids": list(getattr(self, "_last_graph_relation_node_ids", [])),
                    "temporal_constraint": intent_temporal,
                    "temporal_expression": str(
                        intent.get("temporal_expression")
                        or (intent_temporal or {}).get("temporal_expression") or ""
                    ),
                    "node_ids": [
                        str(getattr(ev, "node_id", ""))
                        for ev in local_evidence if getattr(ev, "node_id", None)
                    ],
                    "status": "covered" if local_evidence else "stopped",
                    "execution": execution,
                    "evidence_status": (
                        "covered" if local_evidence else
                        "error" if execution == "ERROR" else "empty"
                    ),
                    "pending": None if local_evidence else (
                        "error" if execution == "ERROR" else "insufficient"
                    ),
                })

            if not active:
                pending_intent = next((
                    item for item in planned_intents
                    if item.get("needs_clarification")
                    and str(item.get("clarification") or "").strip()
                ), None)
                clarification_records = [{
                    "intent_id": str(item.get("intent_id") or ""),
                    "intent_index": index,
                    "sub_query": str(item.get("sub_query") or ""),
                    "node_ids": [], "status": "clarification",
                    "execution": "CLARIFY", "evidence_status": "clarification",
                    "pending": "clarification",
                    "clarification": str(item.get("clarification") or ""),
                } for index, item in enumerate(planned_intents)
                    if item.get("needs_clarification")]
                metadata = {"five_route": "full", "planner": plan,
                            "planner_audit": current_planner_audit()}
                if clarification_records:
                    metadata["per_intent"] = clarification_records
                if pending_intent:
                    metadata["planner_state"] = current_planner_state()
                if planner_rewrites:
                    metadata["planner_rewrites"] = planner_rewrites
                return LegalAnswer(
                    answer="Cần làm rõ một hoặc nhiều ý trước khi tra cứu.",
                    route="ask",
                    follow_up=(pending_follow_up()
                               if pending_intent else ""),
                    metadata=metadata,
                )
            if not all_evidence:
                metadata = {"five_route": "full", "planner": plan,
                            "planner_audit": current_planner_audit(),
                             "per_intent": active + [{
                                "intent_id": str(item.get("intent_id") or ""),
                                 "intent_index": index,
                                "sub_query": str(item.get("sub_query") or ""),
                                "node_ids": [], "status": "clarification",
                                "execution": "CLARIFY", "evidence_status": "clarification",
                                "pending": "clarification",
                                "clarification": str(item.get("clarification") or ""),
                             } for index, item in enumerate(planned_intents)
                                if item.get("needs_clarification")]}
                if clarifications:
                    metadata["planner_state"] = current_planner_state()
                if planner_rewrites:
                    metadata["planner_rewrites"] = planner_rewrites
                return LegalAnswer(
                    answer=NO_EVIDENCE_MESSAGE,
                    route=decision.route,
                    follow_up=(pending_follow_up() if clarifications else "") or
                        "Chưa tìm thấy đủ căn cứ cho các ý đã nêu; vui lòng bổ sung văn bản hoặc điều khoản.",
                    metadata=metadata,
                )
            if multi_planned:
                for record in active:
                    record["_is_multi_intent_plan"] = True
            planned_answer = self._synthesize(
                question, u, decision, all_evidence, per_intent=active
            )
            if clarifications:
                planned_answer.follow_up = pending_follow_up()
                planned_answer.metadata["clarifications"] = [
                    item.get("clarification") for item in planned_intents
                    if item.get("needs_clarification") and item.get("clarification")
                ]
                planned_answer.metadata["planner_state"] = current_planner_state()
            if planner_rewrites:
                planned_answer.metadata["planner_rewrites"] = planner_rewrites
            planned_answer.metadata["planner"] = plan
            planned_answer.metadata["planner_audit"] = current_planner_audit()
            verdict_by_query = planned_answer.metadata.get("intent_verdicts") or {}
            verdict_by_id = planned_answer.metadata.get("intent_verdicts_by_id") or {}
            if not isinstance(verdict_by_query, dict):
                verdict_by_query = {}
            if not isinstance(verdict_by_id, dict):
                verdict_by_id = {}
            for record in active:
                record.pop("_is_multi_intent_plan", None)
                record_answer = (
                    planned_answer.metadata.get("per_intent_answers") or {}
                ).get(str(record.get("intent_id") or ""))
                if record_answer is not None:
                    record["answer"] = str(record_answer)
                verdict = str(
                    verdict_by_id.get(str(record.get("intent_id") or ""))
                    or verdict_by_query.get(record.get("sub_query") or "") or ""
                ).upper()
                if verdict:
                    record["intent_verdict"] = verdict
                if verdict == "INSUFFICIENT":
                    record["evidence_status"] = "insufficient"
                    record["pending"] = "insufficient"
                elif verdict in ("FAILED", "ERROR"):
                    record["evidence_status"] = "error"
                    record["pending"] = "error"
                    record["execution"] = "ERROR"
                elif verdict == "OOD":
                    record["evidence_status"] = "ood"
                    record["pending"] = "ood_gate"
            # Preserve the retrieval outcome contract after synthesis, including
            # successful and mixed (answered + clarification) multi-intent turns.
            planned_answer.metadata["per_intent"] = active + [{
                "intent_id": str(item.get("intent_id") or ""),
                "intent_index": index,
                "sub_query": str(item.get("sub_query") or ""),
                "node_ids": [], "status": "clarification",
                "execution": "CLARIFY", "evidence_status": "clarification",
                "pending": "clarification",
                "clarification": str(item.get("clarification") or ""),
            } for index, item in enumerate(planned_intents) if item.get("needs_clarification")]
            return planned_answer

        # PC-1 (§14.1): phân loại 5-route + verify-by-level (rule-first).
        # Chạy TRƯỚC fast-track/vector để (a) ghi span route_decision,
        # (b) chặn ANCHOR_UNRESOLVED (anchor chết đã verify) -> follow-up,
        # KHÔNG semantic-search thay bằng node gần nghĩa. Lỗi -> None (degrade:
        # đường cũ không đổi). Câu nhiều ý (FULL) chỉ ghi nhãn, per-intent
        # verify qua `_five_route_for_intent` (R11/R12), không early-return.
        from retrieval.AnchorVerifier import (
            ROUTE_ANCHOR_UNRESOLVED,
            classify_query,
        )

        five_route_res = None
        try:
            five_route_res = classify_query(
                question, u,
                llm=self._route_classify_llm(),
                source=self.anchor_verify_source(),
            )
        except Exception:
            five_route_res = None
        decision.five_route = (
            "full" if is_multi_intent(question) else
            (five_route_res.route if five_route_res else "")
        )
        # PC-1: pin verified (nút đã verify hợp lệ — có thể rỗng; không
        # bao giờ chứa nút của vb đã chết). Rỗng = degrade về anchor thô
        # (đường cũ), giữ behavior không LLM/fake-KG.
        self._verified_pins = (
            five_route_res.pin_node_ids() if five_route_res else []
        )
        if tracer is not None:
            tracer.emit(
                "router", OK,
                reason=f"route={decision.route} mode={getattr(decision, 'mode', '-')}",
                data={
                    "route": decision.route,
                    "confidence": round(decision.confidence, 4),
                    "target_node_ids": list(decision.target_node_ids or []),
                },
            )
            tracer.emit(
                "route_decision",
                OK if five_route_res is not None else SKIPPED,
                reason=(five_route_res.reason if five_route_res else "degrade"),
                data={
                    "five_route": decision.five_route,
                    "is_full": is_multi_intent(question),
                    "llm_calls_added": (
                        five_route_res.llm_calls_added if five_route_res else 0
                    ),
                    "n_checks": len(five_route_res.checks) if five_route_res else 0,
                    "pins": five_route_res.pin_node_ids() if five_route_res else [],
                },
            )

        # PC-1: ANCHOR_UNRESOLVED (câu ĐƠN, anchor đã verify là CHẾT:
        # vb không trong registry / điều-khoản-điểm không có trong KG /
        # thiếu vb) -> trả follow-up, KHÔNG semantic-search thay.
        # Câu NHIỀU Ý (FULL) không early-return: mỗi intent tự verify
        # (_five_route_for_intent), intent chết KHÔNG kéo thụt intent lành.
        if (
            five_route_res is not None
            and five_route_res.route == ROUTE_ANCHOR_UNRESOLVED
            and not is_multi_intent(question)
        ):
            pins = five_route_res.pin_node_ids()
            evidences = self.kg.evidence_for_many(pins) if pins else []
            if tracer is not None:
                tracer.emit(
                    "anchor_verify", FAILED,
                    reason=five_route_res.reason,
                    data={
                        "missing": [
                            {"vb": c.vb, "level": c.level,
                             "missing": c.missing, "ok": c.ok}
                            for c in five_route_res.checks
                        ],
                        "n_pins": len(evidences),
                    },
                )
            if evidences:
                # Fallback article_body §4.3 đã verify (node cha, vb hợp
                # lệ trong câu đa-vb) -> trả node cha + follow-up phần
                # chết (không bịa điều/khoản con của vb chết).
                ans = self._synthesize(question, u, decision, evidences)
                ans.follow_up = five_route_res.follow_up or ans.follow_up
                return ans
            return LegalAnswer(
                answer=(
                    "Không tra được điều/khoản bạn chỉ định trong dữ liệu "
                    "hiện có (văn bản hoặc điều/khoản chưa có trong cơ sở "
                    "dữ liệu). " + OOD_NO_ANSWER_MESSAGE
                ),
                evidence=[],
                route=decision.route,
                confidence=decision.confidence,
                follow_up=five_route_res.follow_up
                or "Bạn có thể nêu rõ số hiệu văn bản hoặc điều/khoản "
                   "cần tra không?",
                dataset=(
                    decision.datasets[0] if decision.datasets else None
                ),
                metadata={"anchor_unresolved": True, "five_route": "anchor_unresolved"},
            )

        # A verified EXACT route fetches the anchored provision directly.
        # The legacy two-way router can still label it vector; broad recall
        # would dilute an exact provision query.
        from retrieval.AnchorVerifier import ROUTE_EXACT
        if (
            five_route_res is not None
            and five_route_res.route == ROUTE_EXACT
            and self._verified_pins
        ):
            exact_evidences = self.kg.evidence_for_many(self._verified_pins)
            if tracer is not None:
                tracer.emit(
                    "kg_lookup", OK if exact_evidences else EMPTY,
                    reason="verified EXACT: direct provision lookup",
                    data={"n_nodes": len(self._verified_pins),
                          "n_evidence": len(exact_evidences)},
                )
            if exact_evidences:
                return self._synthesize(
                    question, u, decision, exact_evidences
                )

        # 1) Fast-track: tra KG tất định, không search vector.
        if decision.route == ROUTE_FAST_TRACK:
            node_ids = decision.target_node_ids
            evidences = self.kg.evidence_for_many(node_ids)
            if tracer is not None:
                tracer.emit(
                    "kg_lookup",
                    OK if evidences else EMPTY,
                    reason="fast-track: tra KG tất định",
                    data={"n_nodes": len(node_ids), "n_evidence": len(evidences)},
                )
            if evidences:
                return self._synthesize(
                    question, u, decision, evidences
                )

        # 2) Nhánh đầy đủ: vector -> rerank -> KG -> evidence.
        #    NOTE (P7 X5): KHÔNG early-return khi legal_group=None.
        #    Thiết kế gốc (docstring detect_legal_group): "None = search
        #    tất cả". Dò group từ text keyword rất kém: ~20/65 câu
        #    in-scope (VD "Cơ sở dữ liệu thương mại là gì?") không
        #    trúng keyword -> bị chặn sớm, 0 evidence -> trượt hit.
        #    Câu ngoài phạm vi được chặn bởi OOD gate (X4) + follow-up
        #    khi evidence rỗng, nên không cần chặn ở đây.
        hits = self._vector_hits(question, decision)
        node_ids = self._node_ids_from_hits(hits)
        if tracer is not None:
            tracer.emit(
                "faiss", OK if hits else EMPTY,
                reason="vector recall (hybrid dense+sparse + RRF)",
                data={"n_hits": len(hits)},
            )

        # PC-1: pin verified (anchor đã verify-by-level hợp lệ) vào pool.
        # Rỗng -> degrade về anchor thô (đường cũ 0-diff khi không LLM/
        # fake KG / registry thiếu). Nút của vb chết KHÔNG bao giờ ở đây
        # (pin_node_ids() chỉ trả node ok) -> chống bịa (R3/R4/R12).
        pins = self._verified_pins or list(anchor_node_ids(u))
        for nid in pins:
            if nid not in node_ids:
                node_ids.append(nid)

        # B1 (P7): graph-enrichment — đối với hit vector tốt nhất, quét
        # các quan hệ sửa/bãi/hướng dẫn (AMENDS/REPEALS/GUIDES) có hướng
        # TỪ văn bản mới VỀ điều đang hiệu lực, rồi kéo provision của
        # "văn bản sửa" vào pool. Vector chỉ biết nội dung, không biết
        # thời gian, nên các câu hỏi "quy định mới" hay "hiện hành"
        # được giải quyết bằng đồ thị (Bước 2 của quy trình hòa hợp:
        # vector tìm node -> graph duyệt các nút lân cận -> rerank gộp).
        node_ids = self._enrich_with_graph_relations(
            u, node_ids
        )
        if tracer is not None:
            tracer.emit(
                "graph_b1", OK,
                reason="mở rộng quan hệ AMENDS/REPEALS/GUIDES (B1)",
                data={"pool_after_b1": len(node_ids)},
            )

        evidences = self.kg.evidence_for_many(node_ids)
        if tracer is not None:
            tracer.emit(
                "merge", OK if evidences else EMPTY,
                reason="merge/dedup: materialize node -> evidence (dedup theo node_id)",
                data={"pool": len(node_ids), "n_evidence": len(evidences)},
            )
        # P7: gap-filling — câu NHIỀU Ý: kiểm tra độ phủ, lấp ý chưa
        # có căn cứ bằng cách chạy lại pipeline cho ý đó. Câu đơn ->
        # lọc rẻ chặn trước, no-op (số eval không đổi).
        evidences, gap_meta = self.decomposer.fill(
            question, evidences, self._evidence_for_subquestion
        )
        self.last_gap_meta = gap_meta  # P7: để eval/đo quan sát
        if tracer is not None:
            # Audit P-A: ánh xạ fidelity (câu đơn no-op = SKIPPED, không
            # được KAI "INSUFFICIENT unknown" — bug trước fix).
            cov_state, cov_reason = coverage_state(gap_meta)
            tracer.emit(
                "coverage", cov_state, reason=cov_reason,
                data={
                    "stop_reason": gap_meta.get("stop_reason", ""),
                    "filled": gap_meta.get("filled", 0),
                    "rounds": gap_meta.get("rounds", 0),
                    "n_intents": len(gap_meta.get("per_intent") or []),
                },
            )

        if not evidences:
            return LegalAnswer(
                answer=NO_EVIDENCE_MESSAGE,
                route=decision.route,
                confidence=decision.confidence,
                follow_up=(
                    gap_meta.get("clarify")
                    or "Chua tim thay quy dinh phu hop trong du lieu hien co. "
                       "Ban hay neu ro so hieu van ban hoac dieu/khoan can tra."
                ),
                dataset=decision.datasets[0] if decision.datasets else None,
                metadata={"decomposition": gap_meta},
            )

        answer = self._synthesize(
            question, u, decision, evidences,
            per_intent=gap_meta.get("per_intent"),
        )
        if gap_meta.get("clarify"):
            answer.follow_up = str(gap_meta["clarify"])
            answer.metadata["decomposition_clarify"] = str(gap_meta["clarify"])
        return answer

    def _needs_relation(
        self, evidences: list[Any], u: QueryUnderstanding
    ) -> bool:
        """Có cần lớp Graphiti (quan hệ) không?

        Cần trong 3 trường hợp:
          1. intent là quan hệ (hieu_luc / time_travel / so_sanh)
             -> vector không biết thời gian, bắt buộc đồ thị.
          2. câu hỏi chứa từ khóa quan hệ (sửa/bãi bỏ/hướng dẫn).
          3. nội bộ trả không ra evidence (graphiti làm fallback).
        Ngược lại (câu nội dung thuần + đã có evidence) -> KHÔNG cần,
        tránh fact quan hệ chen vào làm LLM bối rối.
        """
        # 1+2: intent đồ thị hoặc từ khóa quan hệ trong câu hỏi.
        if u.intent in ("hieu_luc", "time_travel", "so_sanh"):
            return True
        if self._has_relation_keyword(u.query):
            return True
        # 3: nội bộ rỗng -> dùng graphiti làm fallback.
        if not evidences:
            return True
        return False

    @staticmethod
    def _has_relation_keyword(query: str) -> bool:
        """Từ khóa quan hệ (đồng bộ với bộ trong GraphitiBridge)."""
        try:
            from agents.GraphitiBridge import GraphitiBridge

            keys: list[str] = []
            for rel_keys in GraphitiBridge.RELATION_KEYWORDS.values():
                keys.extend(rel_keys)
            keys.extend(("hiệu lực", "còn hiệu lực", "phạm vi áp dụng"))
            q = (query or "").lower()
            return any(k in q for k in keys)
        except Exception:
            return False

    def _final_rerank(
        self,
        question: str,
        evidences: list[Any],
        graphiti_facts: list[dict[str, Any]],
        anchor_node_ids: list[str] | None = None,
    ) -> tuple[list[Any], list[dict[str, Any]], float]:
        """X2: rerank CHUNG trên pool đã gộp (nội bộ + graphiti).

        Vì sao: reranker chỉ xếp hạng ứng viên FAISS; fact Graphiti
        trước đây được "nhét" thêm vào prompt mà KHÔNG qua chấm điểm
        -> LLM thấy văn bản lạ (vd 296/2026) nổi lên và trả sai, dù
        fact đúng đã có trong pool.

        Cơ chế: gom 2 nguồn thành 1 list candidate (mỗi candidate có
        `text`), chấm bằng reranker, giữ top-K CHUNG. Sau đó tách lại
        về 2 nhóm (nội bộ / graphiti) để dựng prompt như cũ.

        - Node ANCHOR (điều/khoản người dùng nêu đích danh) LUÔN được
          giữ lại, kể cả khi reranker đẩy xuống: đó là provision được
          hỏi, không được bỏ vì có đoạn kháy về mặt ngữ nghĩa.
        - Không có reranker (thiếu model / lỗi) -> giữ nguyên, KHÔNG
          làm hỏng luồng (degrade an toàn, quyết định #12).
        - Trả thêm điểm rerank CAO NHẤT (để phán OOD, X4).
        """
        if not evidences and not graphiti_facts:
            return evidences, graphiti_facts, 0.0

        rerank_fn = self.rerank_fn or self._default_rerank_fn()
        if rerank_fn is None:
            # Even without a reranker, bound the combined prompt evidence.
            # Preserve exact user anchors first, then retain source order.
            pinned = []
            pinned_ids: set[str] = set()
            for ev in evidences:
                nid = getattr(ev, "node_id", None) or ""
                if nid in anchor_node_ids and nid not in pinned_ids:
                    pinned.append(ev)
                    pinned_ids.add(nid)
            remaining = max(0, self.top_k - len(pinned))
            internal = []
            graph = []
            for ev in evidences:
                nid = getattr(ev, "node_id", None) or ""
                if nid not in pinned_ids and len(internal) + len(graph) < remaining:
                    internal.append(ev)
            for fact in graphiti_facts:
                if len(internal) + len(graph) >= remaining:
                    break
                graph.append(fact)
            return pinned + internal, graph, 0.0

        anchor_ids = set(anchor_node_ids or [])

        # Gom pool: mỗi item mang "_origin" để tách lại sau.
        pool: list[dict[str, Any]] = []
        for ev in evidences:
            pool.append({
                "text": getattr(ev, "text", "") or "",
                "_origin": "internal",
                "_ev": ev,
            })
        for gf in graphiti_facts:
            pool.append({
                "text": gf.get("text") or gf.get("fact") or "",
                "_origin": "graphiti",
                "_gf": gf,
            })

        try:
            ranked = rerank_fn(question, pool)
        except Exception:
            return evidences, graphiti_facts, 0.0

        if not ranked:
            return evidences, graphiti_facts, 0.0

        best_score = max(float(r.get("score") or 0.0) for r in ranked)

        # Giữ top-K chung (K = top_k của agent).
        ranked = ranked[: self.top_k]

        out_internal: list[Any] = []
        out_graphiti: list[dict[str, Any]] = []
        for item in ranked:
            origin = item.get("_origin")
            if origin == "internal" and item.get("_ev") is not None:
                out_internal.append(item["_ev"])
            elif origin == "graphiti" and item.get("_gf") is not None:
                out_graphiti.append(item["_gf"])

        # PIN node anchor lên ĐẦU: user đã nêu đích danh provision
        # (vd "Khoản 3 Điều 12 NĐ 168/2025") -> provision đó phải ở
        # đầu prompt, dù reranker xếp nó sau provision cùng nội dung
        # (provision của văn bản sửa đổi thường gần nghĩa hơn).
        # Ngân sách top-k tính CẢ pin: pin dùng h slot -> còn
        # top_k - h slot cho phần reranked (trần top-k quyết định #8).
        pinned: list[Any] = []
        pinned_ids: set[str] = set()
        if anchor_ids:
            for item in ranked:
                ev = item.get("_ev")
                if ev is None:
                    continue
                nid = getattr(ev, "node_id", None) or ""
                if nid in anchor_ids and nid not in pinned_ids:
                    pinned.append(ev)
                    pinned_ids.add(nid)
            # Anchor có trong evidence mà reranker cắt -> vẫn pin.
            for ev in evidences:
                nid = getattr(ev, "node_id", None) or ""
                if nid in anchor_ids and nid not in pinned_ids:
                    pinned.append(ev)
                    pinned_ids.add(nid)

        out_internal: list[Any] = []
        out_graphiti: list[dict[str, Any]] = []
        seen_ids: set[str] = set(pinned_ids)
        for item in ranked[: max(0, self.top_k - len(pinned))]:
            origin = item.get("_origin")
            if origin == "internal":
                ev = item.get("_ev")
                if ev is None:
                    continue
                nid = getattr(ev, "node_id", None) or ""
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                out_internal.append(ev)
            elif origin == "graphiti":
                gf = item.get("_gf")
                if gf is not None:
                    out_graphiti.append(gf)

        return pinned + out_internal, out_graphiti, best_score

    def _ood_answer(
        self,
        u: QueryUnderstanding,
        decision: RouteDecision,
        rerank_score: float,
    ) -> LegalAnswer:
        """Trả lời an toàn khi câu hỏi ngoài phạm vi DB (P7 X4).

        Không để LLM sinh từ evidence mơ hồ -> giảm ảo giác. Cho
        `follow_up` để dẫn user sang QueryRefiner (hỏi lại).
        """
        return LegalAnswer(
            answer=OOD_NO_ANSWER_MESSAGE,
            evidence=[],
            route=decision.route,
            confidence=decision.confidence,
            follow_up=(
                "Bạn có thể nêu rõ số hiệu văn bản hoặc điều/khoản "
                "cần tra không?"
            ),
            dataset=u.legal_group,
            metadata={
                "ood": True,
                "rerank_score": round(rerank_score, 4),
            },
        )

    def _default_rerank_fn(self) -> Any | None:
        """Lấy reranker (BGE-Reranker-M3). None nếu không dùng được."""
        if self._rerank_fn_cache is not _UNSET:
            return self._rerank_fn_cache
        fn = None
        try:
            from retrieval.Reranker import rerank_for_search

            fn = rerank_for_search
        except Exception:
            fn = None
        self._rerank_fn_cache = fn
        return fn

    def _synthesize(
        self,
        question: str,
        u: QueryUnderstanding,
        decision: RouteDecision,
        evidences: list[Any],
        per_intent: list[dict[str, Any]] | None = None,
    ) -> LegalAnswer:
        """Ghép evidence -> prompt -> LLM -> câu trả lời kèm căn cứ.

        F5b (mỗi ý = 1 câu đơn): nếu `per_intent` có >= 2 ý con -> xếp
        hạng RIÊNG theo từng ý (rerank bằng câu con của ý đó, top-K/ý,
        gộp có provenance) rồi mới LLM. 1 ý (câu đơn) -> nhánh cũ y hệt
        (không đổi 1 bit).
        """
        is_multi_intent_plan = any(
            bool(item.get("_is_multi_intent_plan")) for item in (per_intent or [])
        )
        if per_intent and (is_multi_intent_plan or len(per_intent) >= 2):
            return self._synthesize_per_intent(
                question, u, decision, evidences, per_intent
            )

        # Graphiti CHỈ khi câu hỏi đụng quan hệ (sửa/bãi bỏ/hướng dẫn/
        # hiệu lực) HOẶC nội bộ không trả lời được (fallback).
        # Câu nội dung thuần đã có đủ evidence -> BỎ graphiti (tránh
        # fact quan hệ chen vào làm LLM bối rối).
        if self._needs_relation(evidences, u):
            graphiti_text, graphiti_facts = self.graphiti_block(question)
        else:
            graphiti_text, graphiti_facts = "", []

        # X2: GỘP (nội bộ + graphiti) -> reranker chấm CHUNG 1 thang ->
        # cắt top-K. Nhờ vậy LLM không bị "mồi câu" (vd evidence 296/2026
        # nổi lên vì không được chấm điểm so với fact sửa đúng).
        # Anchor = provision NGƯỜI DÙNG NÊU ĐÍCH DANH (số hiệu + điều).
        # Pin provision đó lên đầu prompt, cả 2 kiểu câu hỏi.
        evidences, graphiti_facts, rerank_score = self._final_rerank(
            question,
            evidences,
            graphiti_facts,
            # PC-1: pin verified (degrade về anchor thô khi chưa phân loại /
            # source không đủ interface — giữ đường cũ 0-diff).
            anchor_node_ids=self._pin_ids(u),
        )
        # PC-1 (R12): OOD gate câu đơn dùng pin verified: vb CHẾT (đã
        # verify) không còn trong pin -> gate hoạt động lại được; vb hợp
        # lệ vẫn bypass như cũ. _pin_ids degrade về anchor thô khi
        # _verified_pins rỗng (đường không LLM / fake KG).
        pins_single = self._pin_ids(u)
        tracer = getattr(self, "tracer", None)
        if tracer is not None:
            tracer.emit(
                "rerank", OK,
                reason="rerank CHUNG (nội bộ + graphiti), pin anchor đầu pool",
                data={
                    "n_internal": len(evidences),
                    "n_graphiti": len(graphiti_facts),
                    "best_score": round(rerank_score, 4),
                },
            )

        # X4: OOD check. Reranker (đã chấm cả pool) không cho điểm nào
        # cao -> câu hỏi ngoài phạm vi DB (không có provision liên
        # quan) -> CHẶN, trả "chưa có trong DB", KHÔNG bảo LLM bịa.
        # Câu đã có anchor (user nêu đích danh provision) thì BỎ qua:
        # anchor là bằng chứng đã có dữ liệu, không gọi là OOD.
        #
        # (P7 fix cũ — ĐÃ REVERT): từng thêm điều kiện "không có bằng
        # chứng đồ thị" để cứu DN_051. Nhưng điều kiện đó RỘNG QUÁ: B1
        # tìm được quan hệ trên provision gần nghĩa NGẪU NHIÊN (câu OOD
        # vẫn có provision trong DB gần nghĩa) -> gate mở oan -> OOD
        # tụt 100% xuống 40%. Cạnh cấp khoản (build mới) đã cứu DN_051
        # theo cách đúng: B1 kéo đúng K24 (khoản ngắn) -> rerank chấm
        # cao -> tự qua ngưỡng. Nên trả về gate nguyên trạng, chỉ dựa
        # vào điểm rerank.
        if (
            not list(pins_single)
            and rerank_score > 0.0
            and rerank_score < OOD_REJECT_THRESHOLD
        ):
            if tracer is not None:
                tracer.emit(
                    "ood_gate", FAILED,
                    reason=(
                        "chặn OOD: rerank top-1 "
                        f"{round(rerank_score, 4)} < {OOD_REJECT_THRESHOLD}, "
                        "không có anchor -> không bịa, trả 'chưa có trong DB'"
                    ),
                    data={"best_score": round(rerank_score, 4)},
                )
            return self._ood_answer(u, decision, rerank_score)

        status_block = self._status_block(
            [e.node_id for e in evidences], u
        )
        version_resolution = None
        if u.intent in ("hieu_luc", "time_travel"):
            status_block, version_resolution = self._resolve_versions(
                evidences, u, as_of_date=self._query_time
            )
            if tracer is not None:
                tracer.set_intent(question)
                tracer.emit(
                    "legal_version",
                    OK if version_resolution.get("items") else EMPTY,
                    reason=str(version_resolution.get("status") or "version result recorded"),
                    data={"actual_version": {
                        "as_of_date": version_resolution.get("as_of_date"),
                        "status": version_resolution.get("status"),
                        "items": version_resolution.get("items", []),
                        "verified": bool(version_resolution.get("verified")),
                    }},
                )
                if tracer is not None:
                    tracer.set_intent(None)
            if not version_resolution.get("verified"):
                return LegalAnswer(
                    answer=(
                        "Chưa đủ căn cứ để kết luận hiệu lực theo thời điểm hỏi. "
                        "KG chỉ cung cấp trạng thái/các quan hệ đã ingest; cần "
                        "đối chiếu văn bản chính thức và phạm vi điều khoản."
                    ),
                    evidence=(
                        self._tag_evidence_group(evidences, "primary", "processed_kb")
                        + self._tag_evidence_group(graphiti_facts, "legal_context", "graphiti")
                    ),
                    primary_evidence=self._tag_evidence_group(
                        evidences, "primary", "processed_kb"
                    ),
                    legal_context=self._tag_evidence_group(
                        graphiti_facts, "legal_context", "graphiti"
                    ),
                    route=decision.route,
                    confidence=decision.confidence,
                    status_block=status_block,
                    dataset=u.legal_group,
                    metadata={
                        "five_route": decision.five_route,
                        "version_resolution": version_resolution,
                    },
                )
        # X7: BuildPrompt là call site duy nhất — nhận thẳng list Evidence
        # (không phải context string), tự ghép block + status + graphiti.
        prompt = build_legal_prompt(
            question, evidences, status_block, graphiti_text
        )

        primary_evidence = self._tag_evidence_group(
            evidences, "primary", "processed_kb"
        )
        legal_context = self._tag_evidence_group(
            graphiti_facts, "legal_context", "graphiti"
        )
        all_evidence = primary_evidence + legal_context

        try:
            _sp = getattr(self.llm, "set_purpose", None)
            if _sp:
                _sp("synthesis")
            text = self.llm.complete(prompt)
        except Exception as exc:
            # LLM lỗi -> vẫn trả evidence để người dùng tự đọc, KHÔNG
            # bịa nội dung (quyết định #12: trả lời phải grounded).
            fallback_text = self._grounded_fallback_text(
                1, question, primary_evidence
            )
            return LegalAnswer(
                answer=fallback_text,
                evidence=all_evidence,
                primary_evidence=primary_evidence,
                legal_context=legal_context,
                route=decision.route,
                confidence=decision.confidence,
                status_block=status_block,
                dataset=u.legal_group,
                metadata={
                    "llm_error": True,
                    "claim_evidence_review": {
                        "status": "automatic_provision_linkage",
                        "claims": self._claim_evidence_review(
                            fallback_text, primary_evidence
                        ),
                    },
                },
            )

        unsupported = (
            self._unsupported_document_citations(text, all_evidence)
            + self._unsupported_provision_citations(text, primary_evidence)
        )
        if unsupported:
            return LegalAnswer(
                answer=(
                    "Chưa thể dùng câu trả lời được tạo vì có số hiệu văn bản "
                    "không xuất hiện trong nguồn đã truy xuất. Cần kiểm tra lại nguồn."
                ),
                evidence=all_evidence,
                primary_evidence=primary_evidence,
                legal_context=legal_context,
                route=decision.route,
                confidence=decision.confidence,
                status_block=status_block,
                dataset=u.legal_group,
                metadata={"synthesis_validation": {
                    "passed": False,
                    "unsupported_document_citations": unsupported,
                }},
            )

        return LegalAnswer(
            answer=text,
            evidence=all_evidence,
            primary_evidence=primary_evidence,
            legal_context=legal_context,
            route=decision.route,
            confidence=decision.confidence,
            status_block=status_block,
            dataset=u.legal_group,
            metadata={
                "five_route": decision.five_route,
                "pins": list(getattr(self, "_verified_pins", []) or []),
                "synthesis_validation": {"passed": True},
                "claim_evidence_review": {
                    "status": "automatic_provision_linkage",
                    "claims": self._claim_evidence_review(text, primary_evidence),
                },
                **({"version_resolution": version_resolution}
                   if version_resolution is not None else {}),
            },
        )

    @staticmethod
    def _unsupported_document_citations(
        answer: str, evidence: list[dict[str, Any]]
    ) -> list[str]:
        """Reject explicit legal document numbers absent from retrieved sources.

        This is a narrow deterministic citation check, not claim-level
        faithfulness verification.
        """
        pattern = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ0-9-]+\b", re.I)

        def canonical(value: str) -> str:
            return re.sub(r"\s+", "", value).upper()

        source_text = []
        for item in evidence:
            source_text.append(str(item.get("citation") or ""))
            source_text.append(str(item.get("text") or ""))
            metadata = item.get("metadata") or {}
            source_text.append(str(metadata.get("so_hieu") or ""))
        allowed = {
            canonical(match.group(0))
            for match in pattern.finditer(" ".join(source_text))
        }
        cited = {canonical(match.group(0)) for match in pattern.finditer(answer or "")}
        return sorted(cited - allowed)

    @staticmethod
    def _tag_evidence_group(
        items: list[Any], evidence_class: str, origin: str
    ) -> list[dict[str, Any]]:
        tagged: list[dict[str, Any]] = []
        for item in items:
            data = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            data["evidence_class"] = evidence_class
            metadata = data.get("metadata") or {}
            data["provenance"] = {
                "origin": origin,
                "node_id": data.get("node_id") or metadata.get("source_node"),
                "citation": data.get("citation"),
                "source_node": metadata.get("source_node"),
                "target_node": metadata.get("target_node"),
            }
            tagged.append(data)
        return tagged

    @staticmethod
    def _unsupported_provision_citations(
        answer: str, primary_evidence: list[dict[str, Any]]
    ) -> list[str]:
        """Check only explicit document/article/clause citation pairs."""
        doc_re = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ0-9-]+\b", re.I)
        article_re = re.compile(r"(?:điều|dieu)\s*(\d+[a-z]?)\b", re.I)
        clause_re = re.compile(r"(?:khoản|khoan)\s*(\d+[a-z]?)\b", re.I)

        def norm(value: str) -> str:
            return re.sub(r"\s+", "", value).upper()

        provision_index: dict[str, set[tuple[str, str]]] = {}
        for item in primary_evidence:
            metadata = item.get("metadata") or {}
            doc_ids = [str(metadata.get("so_hieu") or "")]
            citation = str(item.get("citation") or "")
            doc_ids.extend(m.group(0) for m in doc_re.finditer(citation))
            articles = {str(metadata.get("dieu") or "").strip()}
            articles.update(m.group(1) for m in article_re.finditer(citation))
            clauses = {str(metadata.get("khoan") or "").strip()}
            clauses = {c for c in clauses if c}
            kind = str(item.get("kind") or item.get("node_type") or "")
            for doc_id in doc_ids:
                if not doc_id:
                    continue
                for article in articles - {""}:
                    provision_index.setdefault(norm(doc_id), set()).add(
                        (article, "*")
                    )
                    for clause in clauses:
                        provision_index[norm(doc_id)].add((article, clause))

        unsupported: list[str] = []
        for match in doc_re.finditer(answer or ""):
            doc_id = norm(match.group(0))
            left = max(
                answer.rfind("\n", 0, match.start()),
                answer.rfind(";", 0, match.start()),
                answer.rfind(".", 0, match.start()),
                answer.rfind("]", 0, match.start()),
            )
            right_candidates = [
                pos for pos in (
                    answer.find("\n", match.end()),
                    answer.find(";", match.end()),
                    answer.find(".", match.end()),
                    answer.find("[", match.end()),
                ) if pos >= 0
            ]
            right = min(right_candidates) if right_candidates else len(answer)
            window = answer[max(0, left + 1):right]
            articles = [m.group(1) for m in article_re.finditer(window)]
            clauses = [m.group(1) for m in clause_re.finditer(window)]
            for article in articles:
                allowed = provision_index.get(doc_id, set())
                if not any(a == article for a, _c in allowed):
                    unsupported.append(f"{match.group(0)} Điều {article}")
                elif clauses and not any(
                    a == article and (c == "*" or c in clauses)
                    for a, c in allowed
                ):
                    unsupported.append(
                        f"{match.group(0)} Điều {article} Khoản {clauses[0]}"
                    )
        return sorted(set(unsupported))

    @staticmethod
    def _normalize_per_intent_citations(
        section: str, ranked: list[Any]
    ) -> str:
        """Chỉ chuẩn hóa document/article/clause có đúng evidence local."""
        normalized = section or ""
        doc_re = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ0-9-]+\b", re.I)
        for evidence in ranked:
            metadata = (
                evidence.metadata if hasattr(evidence, "metadata")
                else (evidence.get("metadata") or {}) if isinstance(evidence, dict)
                else {}
            )
            doc = str(metadata.get("so_hieu") or "")
            article = str(metadata.get("dieu") or "")
            clause = str(metadata.get("khoan") or "")
            if not doc or not article:
                continue
            pattern_doc = re.escape(doc.replace(" ", ""))
            pattern_doc = pattern_doc.replace(r"/", r"\\s*/\\s*")
            article_patterns = (
                rf"Điều\s+{re.escape(article)}\s*\(\s*{pattern_doc}\s*\)",
                rf"Điều\s+{re.escape(article)}\s*{pattern_doc}",
            )
            exact = (
                f"Điều {article} Khoản {clause} ({doc})"
                if clause else f"Điều {article} ({doc})"
            )
            for pattern in article_patterns:
                if clause:
                    normalized = re.sub(
                        pattern,
                        lambda _m: exact,
                        normalized,
                        flags=re.I,
                    )
        return normalized

    @staticmethod
    def _claim_evidence_review(
        answer: str, primary_evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build a review ledger; citation matches are not proof of entailment."""
        doc_re = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ0-9-]+\b", re.I)
        article_re = re.compile(r"(?:điều|dieu)\s*(\d+)\b", re.I)
        clause_re = re.compile(r"(?:khoản|khoan)\s*(\d+)\b", re.I)
        explicit_clause_re = re.compile(
            r"(?:\[\s*)?(?:Điều|dieu)\s*(\d+)\s*"
            r"(?:Khoản|khoan)\s*(\d+)\s*(?:\(?[^)\n]*?\)?)?",
            re.I,
        )
        sentences = [
            s.strip(" -\t") for s in re.split(r"(?<=[.!?])\s+|\n+", answer or "")
            if s.strip(" -\t")
        ]
        ledger = []
        for sentence in sentences:
            docs = [m.group(0) for m in doc_re.finditer(sentence)]
            articles = [m.group(1) for m in article_re.finditer(sentence)]
            clauses: list[str] = []
            for dm in doc_re.finditer(sentence):
                doc_start = max(0, dm.start() - 80)
                doc_end = min(len(sentence), dm.end() + 80)
                window = sentence[doc_start:doc_end]
                clauses.extend(
                    cm.group(2)
                    for cm in explicit_clause_re.finditer(window)
                )
            clauses = list(dict.fromkeys(clauses))
            matched = []
            for item in primary_evidence:
                metadata = item.get("metadata") or {}
                citation = str(item.get("citation") or "")
                so = str(metadata.get("so_hieu") or "")
                article = str(metadata.get("dieu") or "")
                clause = str(metadata.get("khoan") or "")
                doc_matches = bool(docs) and any(
                    d.replace(" ", "").upper()
                    in (so + " " + citation).replace(" ", "").upper()
                    for d in docs
                )
                article_only_claim = bool(articles) and not docs
                article_matches = bool(articles) and article in articles
                clause_matches = (not clauses) or (clause in clauses)
                if (
                    (doc_matches or (article_only_claim and article_matches))
                    and (not articles or article_matches)
                    and clause_matches
                ):
                    matched.append({
                        "node_id": item.get("node_id"),
                        "citation": citation,
                        "intent_indices": (metadata.get("intent_indices")
                                           or ([metadata["intent_index"]]
                                               if metadata.get("intent_index") is not None else [])),
                    })
            ledger.append({
                "claim_text": sentence,
                "cited_documents": docs,
                "cited_articles": articles,
                "cited_clauses": clauses,
                "matched_primary_evidence": matched,
                "link_status": "linked" if matched else "unlinked",
                "status": ("human_review_required" if matched else "unlinked")
                if docs or articles else "no_explicit_citation",
            })
        return ledger

    # -- F5b: rerank + tổng hợp theo TỪNG Ý CON (mỗi ý = 1 câu đơn) ------

    def _rerank_intent(
        self,
        sub_query: str,
        pool: list[Any],
        anchor_ids: list[str] | None = None,
    ) -> tuple[list[Any], float]:
        """F5b: xếp hạng pool evidence RIÊNG bằng câu con (sub_query) của ý.

        Điểm khác vs `_final_rerank`: signal là **câu con của ý đó** (sắc,
        đúng điều/khoản cần tìm), KHÔNG phải câu gộp trừu tượng. Trả
        (top-K của ý, điểm rerank cao nhất). Không reranker / lỗi -> giữ
        nguyên thứ tự (degrade an toàn, không crash, không bịa).
        """
        anchor_ids = set(anchor_ids or [])
        tracer = getattr(self, "tracer", None)
        intent_id = str(getattr(self, "_active_intent_id", "") or "")

        def emit_rank(node_ids: list[str], stage: str, state: str = OK) -> None:
            if tracer is None:
                return
            tracer.set_intent(intent_id or None)
            tracer.emit("retrieval_rank", state,
                        reason="ranked node IDs captured for retrieval metrics",
                        data={"node_ids_ranked": node_ids, "ranking_stage": stage,
                              "top_k": self.top_k, "intent_id": intent_id})

        if not pool:
            emit_rank([], "empty_pool", EMPTY)
            return [], 0.0
        rerank_fn = self.rerank_fn or self._default_rerank_fn()
        if rerank_fn is None:
            emit_rank([str(getattr(e, "node_id", "") or "") for e in pool if getattr(e, "node_id", None)],
                      "fallback_pool_order")
            return list(pool[: self.top_k]), 0.0
        items = [
            {"text": getattr(e, "text", "") or "", "_ev": e} for e in pool
        ]
        try:
            ranked = rerank_fn(sub_query, items)
        except Exception:
            emit_rank([str(getattr(e, "node_id", "") or "") for e in pool if getattr(e, "node_id", None)],
                      "reranker_error_pool_order", FAILED)
            return list(pool[: self.top_k]), 0.0
        if not ranked:
            emit_rank([str(getattr(e, "node_id", "") or "") for e in pool if getattr(e, "node_id", None)],
                      "empty_reranker_pool_order")
            return list(pool[: self.top_k]), 0.0
        emit_rank([str(getattr(item.get("_ev"), "node_id", "") or "")
                   for item in ranked if item.get("_ev") is not None
                   and getattr(item.get("_ev"), "node_id", None)],
                  "post_rerank_pre_topk")
        best = max(float(r.get("score") or 0.0) for r in ranked)
        ranked_top = ranked[: self.top_k]
        # Pin anchor của ý con (sub_query nêu đích danh điều/khoản) lên đầu.
        pinned: list[Any] = []
        pinned_ids: set[str] = set()
        if anchor_ids:
            for it in ranked_top:
                ev = it.get("_ev")
                if ev is None:
                    continue
                nid = getattr(ev, "node_id", None) or ""
                if nid in anchor_ids and nid not in pinned_ids:
                    pinned.append(ev)
                    pinned_ids.add(nid)
            for e in pool:  # anchor trong pool mà reranker cắt -> vẫn pin.
                nid = getattr(e, "node_id", None) or ""
                if nid in anchor_ids and nid not in pinned_ids:
                    pinned.append(e)
                    pinned_ids.add(nid)
        out: list[Any] = []
        seen: set[str] = set()
        for ev in pinned + [it.get("_ev") for it in ranked_top]:
            if ev is None:
                continue
            nid = getattr(ev, "node_id", None) or ""
            if nid in seen:
                continue
            seen.add(nid)
            out.append(ev)
        return out[: self.top_k], best

    @staticmethod
    def _intent_verdict(
        it: dict[str, Any], intent_anchors: list[str], intent_ood: bool
    ) -> str:
        """P-B (B1 + điều chỉnh #3): intent_verdict (KẾT LUẬN CUỐI) của 1 ý.

        Hai trục riêng: `execution` (OK/EMPTY/FAILED, trung gian) là INPUT;
        verdict (COVERED/OOD/INSUFFICIENT/FAILED) là KẾT LUẬN. KHÔNG đánh
        đồng. OOD CHỈ khi gate F5b `intent_ood` (đầy đủ: không anchor +
        rerank best<ngưỡng) — KHÔNG dùng riêng rerank score (điều chỉnh #3).

        Ưu tiên (§13.3):
          1. COVERED (required evidence đã đủ) -> COVERED, BỀN (không bị lỗi
             bước phụ xóa — điều chỉnh #3).
          2. execution = FAILED (lỗi THỰC khi fill) -> FAILED.
          3. status = "stopped" (no-progress) + pending:
             - "ood_gate": gate F5b xác nhận -> OOD; không xác nhận ->
               INSUFFICIENT (B1: không tự kết luận OOD).
             - "insufficient" (required node có thật trong KG mà chưa kéo
               về) -> INSUFFICIENT (KHÔNG OOD — provision có thật).
             - "failed" -> FAILED.
          4. intent chưa stopped nhưng vẫn không covered (open) -> INSUFFICIENT.
        """
        from agents.QueryDecomposer import (
            VERDICT_COVERED,
            VERDICT_FAILED,
            VERDICT_INSUFFICIENT,
            VERDICT_OOD,
        )

        status = str(it.get("status") or "")
        execution = str(it.get("execution") or "")
        pending = it.get("pending")
        # (1) COVERED: required evidence đủ trong pool.
        if status == "covered" and execution not in ("FAILED", "ERROR"):
            return VERDICT_COVERED
        # (2) Lỗi THỰC khi fill -> không kết luận được (điều chỉnh #3:
        #     không đánh đồng "không tìm thấy" (EMPTY) với FAILED).
        if execution in ("FAILED", "ERROR") or pending in ("failed", "error"):
            return VERDICT_FAILED
        # (3) stopped (no-progress):
        if status == "stopped":
            if pending == "ood_gate":
                # B1: OOD CHỈ khi gate F5b xác nhận (intent_ood), còn không
                # -> INSUFFICIENT (căn cứ chưa đủ để khẳng định "không có").
                return VERDICT_OOD if intent_ood else VERDICT_INSUFFICIENT
            if pending == "insufficient":
                # Retrieval rỗng ở nhánh pre-planned cũng ghi pending=
                # "insufficient" trước khi chạy gate theo intent. Nếu gate
                # xác nhận OOD và intent không có legal anchor, giữ verdict
                # OOD; nhưng provision được nêu đích danh vẫn là INSUFFICIENT.
                if intent_ood and not intent_anchors:
                    return VERDICT_OOD
                return VERDICT_INSUFFICIENT
            # pending rỗng/khác -> mặc định an toàn INSUFFICIENT (không bịa OOD).
            return VERDICT_INSUFFICIENT
        # (4) open (chưa stopped, chưa covered): intent hợp lệ, chưa đủ căn cứ.
        # (CÓ anchor nhưng chưa covered vẫn INSUFFICIENT, không OOD — vì
        #  anchor = người dùng nêu provision, chắc chắn có trong DB.)
        return VERDICT_INSUFFICIENT

    def _synthesis_budget(
        self,
        ranked: list[Any],
        pin_ids: list[str],
        graph_node_ids: list[str],
        evidences: list[Any],
    ) -> list[Any]:
        """Build the bounded prompt evidence: pins + graph + 3 semantic nodes."""
        by_id = {
            str(getattr(e, "node_id", "") or ""): e
            for e in list(evidences or []) + list(ranked or [])
            if getattr(e, "node_id", None)
        }
        pinned: list[Any] = []
        seen: set[str] = set()
        for nid in pin_ids:
            e = by_id.get(str(nid))
            if e is not None and str(nid) not in seen:
                seen.add(str(nid))
                pinned.append(e)
        graph: list[Any] = []
        for nid in graph_node_ids:
            e = by_id.get(str(nid))
            if e is not None and str(nid) not in seen:
                seen.add(str(nid))
                graph.append(e)
        semantic: list[Any] = []
        for e in ranked:
            nid = str(getattr(e, "node_id", "") or "")
            if nid and nid not in seen:
                seen.add(nid)
                semantic.append(e)
                if len(semantic) >= 3:
                    break
        return pinned + graph + semantic

    @staticmethod
    def _grounded_fallback_text(
        position: int,
        sub_q: str,
        evidence: list[Any],
    ) -> str:
        lines = [f"[[INTENT:{position}]]", f"{sub_q}", "Căn cứ đã truy xuất:"]
        for e in evidence:
            if isinstance(e, dict):
                citation = str(e.get("citation", "") or "").strip()
                text = str(e.get("text", "") or "").strip()
            else:
                citation = str(getattr(e, "citation", "") or "").strip()
                text = str(getattr(e, "text", "") or "").strip()
            if citation:
                lines.append(f"- {citation}")
            if text:
                lines.append(text[:900])
        lines.append("Kết luận pháp lý chưa được xác minh; cần đối chiếu nguồn chính thức.")
        return "\n".join(lines)

    def _synthesize_per_intent(
        self,
        question: str,
        u: QueryUnderstanding,
        decision: RouteDecision,
        evidences: list[Any],
        per_intent: list[dict[str, Any]],
    ) -> LegalAnswer:
        """F5b: mỗi ý con rerank RIÊNG bằng câu con của nó, top-K/ý, gộp
        có provenance; OOD **theo từng ý** (không kéo thụt cả câu).

        - evidence pool truyền vào = pool gộp từ `fill()` (mỗi ý đã gắn
          `node_ids` = seed + evidence ý đó tự sinh). Lấy lại pool của ý
          bằng `node_ids`, rerank bằng `sub_query` (signal sắc của ý).
        - Ý OOD (không anchor + điểm rerank thấp) -> đánh dấu, LLM được
          nhắc "nói rõ chưa có trong dữ liệu", KHÔNG bịa.
        - Gộp các top-K/ý (dedup node_id) -> prompt từng ý + citation từng ý.
        """
        from retrieval.AnchorVerifier import ROUTE_ANCHOR_UNRESOLVED

        # 5-element: (sub_q, ranked, best, intent_ood, intent_unresolved)
        per_intent_result: list[tuple[str, list[Any], float, bool, bool]] = []
        merged: list[Any] = []
        merged_ids: set[str] = set()
        context_by_intent: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        verdict_by_intent: dict[str, str] = {}
        pin_ids_by_intent: dict[str, list[str]] = {}
        tracer = getattr(self, "tracer", None)

        for it in per_intent:
            sub_q = str(it.get("sub_query") or question).strip() or question
            if it.get("is_out_of_domain") is True or it.get("pending") == "ood_gate":
                verdict_by_intent[sub_q] = "OOD"
                per_intent_result.append((sub_q, [], 0.0, True, False))
                if tracer is not None:
                    intent_id = str(it.get("intent_id") or sub_q)
                    tracer.set_intent(intent_id)
                    tracer.emit(
                        "intent_verdict", OK,
                        reason="planner marked this intent out of domain",
                        data={"intent_id": intent_id,
                              "intent_index": int(it.get("intent_index", 0)),
                              "intent_verdict": "OOD"},
                    )
                    tracer.set_intent(None)
                continue
            ids = {str(x) for x in (it.get("node_ids") or [])}
            pool = [
                e for e in evidences
                if (getattr(e, "node_id", None) or "") in ids
            ]
            sub_u = understand(sub_q)
            intent_anchors = list(anchor_node_ids(sub_u))
            # PC-1 (R11/R12): MỖI intent tự verify-by-level (5-route).
            # intent UNRESOLVED (anchor chết đã verify: vb/điều/khoản không
            # có trong KG) KHÔNG kéo thụt intent lành — pool per-intent giữ
            # nguyên (F5b). Verified pin chỉ dùng khi intent resolve được;
            # intent unresolved -> KHÔNG pin anchor bịa (chống bịa node).
            i_five, i_pins = self._five_route_for_intent(sub_q)
            intent_unresolved = i_five == ROUTE_ANCHOR_UNRESOLVED
            pin_ids: list[str] = []
            if not intent_unresolved:
                pin_ids = list(intent_anchors)
                for p in i_pins:
                    if p not in pin_ids:
                        pin_ids.append(p)
            pin_ids_by_intent[str(it.get("intent_id") or sub_q)] = list(pin_ids)
            ranked, best = self._rerank_intent(sub_q, pool, pin_ids)
            if self._needs_relation(ranked, sub_u):
                context_by_intent[sub_q] = self.graphiti_block(sub_q)
            else:
                context_by_intent[sub_q] = ("", [])
            # OOD theo từng ý (giống gate câu đơn, nhưng riêng intent).
            # DÙNG intent_anchors GỐC (không phải pin_ids) cho gate:
            # intent unresolved CÓ anchor bịa -> không gọi là OOD (đúng
            # §14.1: UNRESOLVED ≠ OOD, là follow-up).
            intent_ood = (
                not intent_anchors
                and best > 0.0
                and best < OOD_REJECT_THRESHOLD
            )
            # P-B (B1 + #3): finalize intent_verdict (kết luận cuối) từ
            # decomposer (status/execution/pending) + gate F5b + execution.
            # KHÔNG đổi prompt/đáp (đó là P-C) — verdict chỉ ghi trace.
            verdict = self._intent_verdict(it, intent_anchors, intent_ood)
            verdict_by_intent[sub_q] = verdict
            if tracer is not None:
                intent_id = str(it.get("intent_id") or "")
                tracer.set_intent(intent_id or sub_q)
                # B2: chỉ GHI NHẬN tín hiệu thô (collector cho P-D), không
                # gọi là "xác suất", không thêm LLM call. OOD = gate F5b.
                tracer.emit(
                    "rerank_per_intent",
                    OK if pool else EMPTY,
                    reason=(
                        "rerank riêng intent"
                        + (" (provenance rỗng -> dùng cả pool)" if not ids else "")
                    ),
                    data={
                        "intent_id": intent_id,
                        "intent_index": int(it.get("intent_index", 0)),
                        "pool": len(pool),
                        "ranked": len(ranked),
                        "best": round(best, 4),
                        "ood": intent_ood,
                        # PC-1: 5-route + unresolved (không đánh đồng OOD).
                        "five_route": i_five,
                        "unresolved": intent_unresolved,
                        # Trục verdict + execution (không đánh đồng).
                        "verdict": verdict,
                        "execution": str(it.get("execution") or ""),
                        "status": str(it.get("status") or ""),
                        # B2 signals (6 tín hiệu thô, KHÔNG composite):
                        "signals": {
                            "rerank_best_score": round(best, 4),
                            "ood_decision": intent_ood,
                            "coverage_state": str(it.get("status") or ""),
                            "retrieval_state": str(it.get("execution") or ""),
                            "clarify_state": bool(
                                getattr(self, "last_gap_meta", {}).get("clarify")
                            ),
                            "evidence_count": len(ranked),
                        },
                    },
                )
            for e in ranked:
                nid = getattr(e, "node_id", None) or ""
                if nid in merged_ids:
                    continue
                merged_ids.add(nid)
                merged.append(e)
            per_intent_result.append(
                (sub_q, ranked, best, intent_ood, intent_unresolved)
            )
        if tracer is not None:
            tracer.set_intent(None)  # về pipeline chung

        if not merged:
            verdicts = [
                ("OOD" if it.get("is_out_of_domain") is True
                 or it.get("pending") == "ood_gate"
                 else self._intent_verdict(it, [], False))
                for it in per_intent
            ]
            if any(v == "FAILED" for v in verdicts):
                return LegalAnswer(
                    answer="Không thể hoàn tất truy xuất cho một hoặc nhiều ý do lỗi kỹ thuật. Chưa có kết luận cho các ý đó.",
                    route=decision.route,
                    metadata={"intent_verdicts": verdicts,
                          "intent_verdicts_by_id": {
                              str(item.get("intent_id") or f"intent-{index + 1}"): verdicts[index]
                              for index, item in enumerate(per_intent)
                          },
                          "per_intent_answers": {},
                              "retrieval_status": "ERROR",
                              "five_route": decision.five_route},
                )
            if any(v == "INSUFFICIENT" for v in verdicts):
                return LegalAnswer(
                    answer=NO_EVIDENCE_MESSAGE,
                    route=decision.route,
                    follow_up=(
                        "Chưa đủ căn cứ cho một hoặc nhiều ý con. "
                        "Hãy bổ sung văn bản, điều khoản hoặc dữ kiện cần tra."
                    ),
                    metadata={"intent_verdicts": verdicts,
                              "intent_verdicts_by_id": {
                                  str(item.get("intent_id") or f"intent-{index + 1}"): verdicts[index]
                                  for index, item in enumerate(per_intent)
                              },
                              "per_intent_answers": {},
                              "five_route": decision.five_route},
                )
            if tracer is not None:
                tracer.emit(
                    "ood_gate", FAILED,
                    reason="mọi intent đều OOD / pool rỗng -> không bịa",
                    data={},
                )
            answer = self._ood_answer(u, decision, 0.0)
            answer.metadata.update({
                "intent_verdicts": verdicts,
                "intent_verdicts_by_id": {
                    str(item.get("intent_id") or f"intent-{index + 1}"): verdicts[index]
                    for index, item in enumerate(per_intent)
                },
                "per_intent_answers": {},
            })
            return answer

        # Resolve status using only each intent's ranked pool and query time.
        version_resolution: dict[str, Any] = {"by_intent": {}}
        for idx, (sub_q, ranked, _best, _ood, _unres) in enumerate(per_intent_result):
            sub_u = understand(sub_q)
            intent_item = per_intent[idx] if idx < len(per_intent) else {}
            temporal_constraint = (
                intent_item.get("temporal_constraint")
                or parse_temporal_constraint(sub_q)
            )
            local_status = self._status_block(
                [getattr(e, "node_id", None) for e in ranked], sub_u
            )
            local_version = None
            if temporal_constraint or sub_u.intent == "hieu_luc":
                local_status, local_version = self._resolve_versions(
                    ranked, sub_u, as_of_date=getattr(self, "_query_time", None),
                    temporal_constraint=temporal_constraint,
                )
                version_resolution["by_intent"][str(idx)] = {
                    "sub_query": sub_q, **local_version,
                }
                intent_id = str(intent_item.get("intent_id") or f"intent-{idx + 1}")
                if tracer is not None:
                    tracer.set_intent(intent_id)
                    tracer.emit(
                        "legal_version",
                        OK if local_version.get("items") else EMPTY,
                        reason=str(local_version.get("status") or "version result recorded"),
                        data={"actual_version": {
                            "as_of_date": local_version.get("as_of_date"),
                            "status": local_version.get("status"),
                            "items": local_version.get("items", []),
                            "verified": bool(local_version.get("verified")),
                        }, "intent_id": intent_id, "intent_index": idx},
                    )
                    tracer.set_intent(None)
            prior_context, prior_facts = context_by_intent.get(sub_q, ("", []))
            if local_status:
                prior_context = (prior_context + "\n[Legal status for this intent]\n" + local_status).strip()
            if local_version and not local_version.get("verified"):
                prior_context = (prior_context + f"\n[Ý {idx + 1}: TRẠNG THÁI CHƯA ĐƯỢC XÁC MINH BẰNG NGUỒN PHÁP LÝ CHÍNH THỨC; KHÔNG KẾT LUẬN]").strip()
            context_by_intent[sub_q] = (prior_context, prior_facts)

        # Prompt: mỗi ý 1 section + citation riêng; ý OOD/UNRESOLVED
        # nói rõ "không có trong dữ liệu" (không bịa).
        blocks = [
            "Câu hỏi gồm NHIỀU Ý CON dưới đây. Hãy trả lời TỪNG Ý CON, mỗi ý "
            "kèm căn cứ pháp lý (citation) CỦA Ý ĐÓ. Ý con nào có nhãn "
            "(KHÔNG CÓ CĂN CỨ TRONG DỮ LIỆU) -> nói rõ chưa tìm được quy "
            "định trong dữ liệu, TUYỆT ĐỐI KHÔNG bịa. Ý con nào có nhãn "
            "(ANCHOR CHƯA XÁC ĐỊNH ĐƯỢC) -> điều/khoản/văn bản được nêu "
            "không có trong dữ liệu, nói rõ điều đó + hỏi lại người dùng "
            "nêu rõ thông tin, TUYỆT ĐỐI KHÔNG bịa provision."
        ]
        for position, (sub_q, ranked, _best, intent_ood, intent_unres) in enumerate(per_intent_result, start=1):
            header = f"=== Ý CON: {sub_q}"
            if intent_ood:
                header += " (KHÔNG CÓ CĂN CỨ TRONG DỮ LIỆU)"
            elif intent_unres:
                header += " (ANCHOR CHƯA XÁC ĐỊNH ĐƯỢC)"
            elif verdict_by_intent.get(sub_q) == "INSUFFICIENT":
                header += " (INSUFFICIENT: CHƯA ĐỦ EVIDENCE)"
            elif verdict_by_intent.get(sub_q) == "FAILED":
                header += " (FAILED: RETRIEVAL ERROR)"
            blocks.append(f"[[INTENT:{position}]]\n{header}")
            blocks.append(
                format_evidence_block(ranked) if ranked
                else "(không có căn cứ nào phù hợp)"
            )
            context_text = context_by_intent.get(sub_q, ("", []))[0]
            if context_text:
                blocks.extend([
                    "Legal Context (Graphiti; chỉ tham khảo quan hệ, không phải primary evidence):",
                    context_text,
                ])
        context_block = "\n\n".join(blocks)
        context_block += (
            "\n\nĐỊNH DẠNG BẮT BUỘC: trả lời từng ý riêng, giữ nguyên đúng một dòng "
            "marker [[INTENT:1]], [[INTENT:2]], ... ở đầu phần tương ứng; "
            "không gộp hoặc bỏ marker."
        )
        required_citation_lines: list[str] = []
        for position, (_sub_q, ranked, _best, _ood, _unres) in enumerate(per_intent_result, start=1):
            citations = list(dict.fromkeys(
                str(getattr(e, "citation", "") or "")
                for e in ranked
                if getattr(e, "citation", None)
            ))
            if citations:
                required_citation_lines.append(
                    f"[[INTENT:{position}]] PHẢI CÓ CITATION TRONG NỘI BỘ Ý NÀY: "
                    + " | ".join(citations)
                )
        if required_citation_lines:
            context_block += (
                "\n\nPER-INTENT CITATION LOCK:\n"
                + "\n".join(required_citation_lines)
                + "\nKhông viện dẫn provision từ intent khác. "
                  "Nếu citation khóa không xuất hiện trong evidence của intent đó, "
                  "không được bịa hoặc suy ra citation thay thế."
            )
        context_block += (
            "\n\nLEGAL PROVENANCE RULES: For each intent, state each retrieved provision "
            "separately with its own document number, article, clause, and citation. "
            "Do not merge original provision text with amendment text or present an amendment "
            "as replacing the whole original provision unless the source explicitly proves "
            "that scope. Keep graph status/effective-date findings separate from legal text; "
            "a graph relation alone is not official-source (T2) verification."
        )
        prov: dict[str, list[int]] = {}
        for idx, (_sq, ranked, _b, _o, _u) in enumerate(per_intent_result):
            for e in ranked:
                nid = getattr(e, "node_id", None) or ""
                if nid:
                    prov.setdefault(nid, []).append(idx)
        all_evidence: list[dict[str, Any]] = []
        for e in merged:
            d = e.to_dict()
            d["metadata"] = {
                **(d.get("metadata") or {}),
                "intent_indices": prov.get(getattr(e, "node_id", None) or "", []),
            }
            all_evidence.append(d)
        for idx, (_sub_q, (_context_text, facts)) in enumerate(context_by_intent.items()):
            for fact in self._tag_evidence_group(
                facts, "legal_context", "graphiti"
            ):
                fact["metadata"] = {
                    **(fact.get("metadata") or {}), "intent_index": idx,
                }
                all_evidence.append(fact)
        primary_evidence = self._tag_evidence_group(
            merged, "primary", "processed_kb"
        )
        for item in primary_evidence:
            nid = str(item.get("node_id") or "")
            item.setdefault("metadata", {})["intent_indices"] = prov.get(nid, [])
        legal_context = [e for e in all_evidence if e.get("evidence_class") == "legal_context"]

        status_block = ""
        # Per-intent status is embedded only in that intent's section above.
        status_block = ""
        # X7: context_block DÃY ĐÃ ghép thủ công (header từng ý + OOD label)
        # -> gọi bản TEXT builder (agents.prompts), KHÔNG phải bản LIST
        # (utils.BuildPrompt tự format_evidence_block). Truyền string vào
        # bản list sẽ bị iterate từng ký tự -> crash ev.citation.
        intent_sections: list[str] = []
        budget_by_intent: dict[int, list[Any]] = {}
        for position, (sub_q, ranked, _best, intent_ood, intent_unresolved) in enumerate(
            per_intent_result, start=1
        ):
            verdict = verdict_by_intent.get(sub_q, "")
            has_local_evidence = bool(ranked)
            if intent_ood or intent_unresolved or (
                not has_local_evidence
                and verdict in {VERDICT_INSUFFICIENT, VERDICT_FAILED}
            ):
                if intent_ood:
                    section = f"[[INTENT:{position}]]\n{NO_EVIDENCE_MESSAGE}"
                elif intent_unresolved:
                    section = (
                        f"[[INTENT:{position}]]\nKhông tìm thấy provision đã nêu trong dữ liệu. "
                        "Vui lòng nêu rõ văn bản, điều, khoản hoặc điểm cần tra."
                    )
                else:
                    section = (
                        f"[[INTENT:{position}]]\n{NO_EVIDENCE_MESSAGE}"
                        if verdict == VERDICT_INSUFFICIENT
                        else f"[[INTENT:{position}]]\nKhông thể truy xuất ý này do lỗi kỹ thuật."
                    )
                intent_sections.append(section)
                continue

            intent_index = position - 1
            intent_record = per_intent[intent_index] if intent_index < len(per_intent) else {}
            pin_ids = list(pin_ids_by_intent.get(str(intent_record.get("intent_id") or ""), []))
            graph_node_ids = list(intent_record.get("graph_node_ids") or [])
            budget = self._synthesis_budget(
                ranked, pin_ids, graph_node_ids, evidences
            )
            budget_by_intent[position] = list(budget)
            intent_context = context_by_intent.get(sub_q, ("", []))[0]
            block = (
                f"[[INTENT:{position}]]\n=== Ý CON ===\n{sub_q}\n\n"
                f"=== CĂN CỨ RIÊNG CỦA Ý ===\n"
                f"{format_evidence_block(budget) if budget else '(không có căn cứ)'}"
            )
            if intent_context:
                block += (
                    "\n\n=== NGỮ CẢNH QUAN HỆ / HIỆU LỰC ===\n"
                    + intent_context
                    + "\nChỉ dùng làm context; không thay thế primary evidence."
                )
            prompt = build_legal_prompt_text(
                sub_q, block, "", ""
            )
            try:
                _sp = getattr(self.llm, "set_purpose", None)
                if _sp:
                    _sp("synthesis_per_intent")
                answer = self.llm.complete(prompt)
                answer = str(answer or "").strip()
                if not answer:
                    raise RuntimeError("empty synthesis output")
            except Exception:
                answer = self._grounded_fallback_text(
                    position, sub_q, budget
                )
            generic_answer = re.search(
                r"(?i)chưa thể xác nhận riêng ý này|không thể xác nhận riêng ý này|"
                r"citation không khớp evidence",
                answer,
            )
            if generic_answer and budget:
                answer = self._grounded_fallback_text(
                    position, sub_q, budget
                )
            if not answer.lstrip().startswith(f"[[INTENT:{position}]]"):
                answer = f"[[INTENT:{position}]]\n{answer}"
            intent_sections.append(answer)

        text = "\n\n".join(intent_sections)

        marker = re.compile(r"(?m)^\s*\[\[INTENT:(\d+)\]\]\s*$")
        marker_matches = list(marker.finditer(text or ""))
        sections: dict[int, str] = {}
        if marker_matches:
            for idx, found in enumerate(marker_matches):
                end = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(text)
                sections[int(found.group(1))] = text[found.end():end].strip()
        else:
            # Backward-compatible parser for the prior numbered answer format.
            numbered = list(re.finditer(r"(?im)^\s*\u00dd\s*(\d+)[.)]\s*", text or ""))
            for idx, found in enumerate(numbered):
                end = numbered[idx + 1].start() if idx + 1 < len(numbered) else len(text)
                sections[int(found.group(1))] = text[found.start():end].strip()
            if len(sections) != len(per_intent_result):
                paragraphs = [
                    part.strip() for part in re.split(r"\n\s*\n+", text or "")
                    if part.strip()
                ]
                if len(paragraphs) == len(per_intent_result):
                    sections = {}
                    for position, paragraph in enumerate(paragraphs, start=1):
                        paragraph = re.sub(
                            r"(?im)^\s*[\u00dd\u00fd]\s*\d+[.)]\s*", "", paragraph
                        ).strip()
                        sections[position] = f"\u00dd {position}: {paragraph}"

        unsupported_by_intent: dict[str, list[str]] = {}
        validated_sections: list[str] = []
        sections_parsed = bool(sections)
        if sections_parsed:
            for position, (sub_q, ranked, _best, _ood, _unres) in enumerate(per_intent_result, start=1):
                section = sections.get(position)
                if not section:
                    section = f"Ý {position}: Chưa tạo được câu trả lời riêng cho ý này."
                    unsupported_by_intent[str(position)] = ["missing_intent_section"]
                else:
                    intent_budget = budget_by_intent.get(position, ranked)
                    section = self._normalize_per_intent_citations(
                        section, intent_budget
                    )
                    local_primary = self._tag_evidence_group(intent_budget, "primary", "processed_kb")
                    local_all = self._tag_evidence_group(intent_budget, "primary", "processed_kb")
                    unsupported = (
                        self._unsupported_document_citations(section, local_all)
                        + self._unsupported_provision_citations(section, local_primary)
                    )
                    if unsupported:
                        unsupported_by_intent[str(position)] = unsupported
                validated_sections.append(section)
                if position <= len(per_intent):
                    per_intent[position - 1]["answer"] = section
            text = "\n\n".join(validated_sections)
        else:
            unsupported = (
                self._unsupported_document_citations(text, all_evidence)
                + self._unsupported_provision_citations(text, primary_evidence)
            )
            if unsupported:
                text = "Chưa thể kiểm tra citation riêng từng intent vì câu trả lời thiếu marker phân tách ý."
                unsupported_by_intent["unparsed"] = unsupported

        # PC-1 (R12): intent UNRESOLVED (anchor chết đã verify) -> follow-up
        # hỏi lại phần thiếu, KHÔNG bịa provision. Phần intent lành vẫn được
        # trả (không pull-down). Chỉ gộp follow-up khi CÓ intent unresolved.
        unres = [sq for (sq, _r, _b, _o, un) in per_intent_result if un]
        insufficient = [
            sq for sq, verdict in verdict_by_intent.items()
            if verdict == "INSUFFICIENT"
        ]
        failed_intents = [
            sq for sq, verdict in verdict_by_intent.items()
            if verdict == "FAILED"
        ]
        unres_follow = (
            "MỘT số ý con chưa xác định được điều/khoản/văn bản nêu trong "
            "dữ liệu (điểm đánh dấu ANCHOR CHƯA XÁC ĐỊNH ĐƯỢC phía trên). "
            "Bạn có thể nêu rõ lại số hiệu văn bản hoặc điều/khoản cho từng "
            "ý đó không?"
            if unres
            else ""
        )
        if insufficient:
            insufficient_follow = (
                "Chưa đủ căn cứ cho các ý sau: " + "; ".join(insufficient)
                + ". Vui lòng bổ sung văn bản, điều khoản hoặc dữ kiện cần tra."
            )
            unres_follow = "\n".join(
                x for x in (unres_follow, insufficient_follow) if x
            )
        if failed_intents:
            failed_follow = (
                "Chưa thể xử lý do lỗi kỹ thuật khi truy xuất các ý: "
                + "; ".join(failed_intents) + ". Vui lòng thử lại sau."
            )
            unres_follow = "\n".join(
                x for x in (unres_follow, failed_follow) if x
            )
        return LegalAnswer(
            answer=text,
            evidence=all_evidence,
            primary_evidence=primary_evidence,
            legal_context=legal_context,
            route=decision.route,
            confidence=decision.confidence,
            follow_up=unres_follow,
            status_block=status_block,
            dataset=u.legal_group,
            metadata={
                "five_route": decision.five_route,
                "unresolved_intents": unres,
                "intent_verdicts": verdict_by_intent,
                "intent_verdicts_by_id": {
                    str(item.get("intent_id") or f"intent-{index + 1}"):
                    verdict_by_intent.get(str(item.get("sub_query") or ""), "")
                    for index, item in enumerate(per_intent)
                },
                "per_intent_answers": {
                    str(item.get("intent_id") or f"intent-{index + 1}"):
                    str(item.get("answer") or "")
                    for index, item in enumerate(per_intent)
                },
                "synthesis_validation": {
                    "passed": not unsupported_by_intent,
                    "per_intent_unsupported": unsupported_by_intent,
                },
                "claim_evidence_review": {
                    "status": "automatic_provision_linkage",
                    "claims": self._claim_evidence_review(text, primary_evidence),
                },
                "version_resolution": version_resolution,
            },
        )

__all__ = [
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_TOP_K",
    "LegalAnswer",
    "LegalRetrievalAgent",
    "MAX_FOLLOW_UP",
]