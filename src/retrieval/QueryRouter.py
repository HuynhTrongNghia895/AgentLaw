"""QueryRouter — định tuyến câu hỏi (P4, 04/11).

Kiến trúc 3 tầng (đã chốt trong PROJECT_CONTEXT §3 P4):

    TẦNG 0 — FAST-TRACK
        Có anchor (điều/khoản/điểm/số hiệu vb) VÀ confidence FAISS top-1
        ≥ ngưỡng  ->  tra KG TẤT ĐỊNH, bỏ qua Graphiti + Web + LLM.
        Nhanh (~0.5s), không tốn token, không gọi LLM.

    TẦNG 1 — RECALL (FAISS/BGE-M3)
        Câu mơ hồ -> embed query -> hybrid dense+sparse -> RRF -> Reranker.
        Đây là "cửa vào": tìm node nào liên quan.

    TẦNG 2 — ĐỒ THỊ (Neo4j + Graphiti)
        Từ node tìm được -> duyệt quan hệ (sửa đổi / bãi bỏ / hiệu lực /
        anh-em cùng khoản) -> bổ sung ngữ cảnh.

    TẦNG 3 — WEB (chỉ khi fast-track không đủ)
        Domain-restriction: vbpl.vn / chinhphu.vn / thuvienphapluat.vn

Nguyên tắc: **không bao giờ đưa node "trần" cho LLM** — luôn đi kèm ngữ
cảnh dọc (materialize) + citation + tình trạng hiệu lực.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from QueryUnderstanding import (
    INTENT_TRA_CUU,
    QueryUnderstanding,
    understand,
)

try:
    from . import embedder
except ImportError:
    import embedder


# ---------------------------------------------------------------------------
# Ngưỡng fast-track
# ---------------------------------------------------------------------------
# Ngưỡng confidence để fast-track — CHỐT BẰNG SỐ ĐO (không đoán).
#
# Nguồn: evaluation/fasttrack_sweep.py (65 câu answerable, 13 câu đủ
# điều kiện fast-track). Điểm dùng làm ngưỡng = `dense_score` (cosine
# thật của BGE-M3), KHÔNG phải `score` (RRF fusion ~0.03, hằng số theo
# rank -> sweep không chọn được ngưỡng nào).
#
#   mốc   FT rate   sai%    chính xác
#   0.50    69.2%   66.7%     33.3%
#   0.55    61.5%   75.0%     25.0%
#   0.60    23.1%   33.3%     66.7%
#   0.65    15.4%    0.0%    100.0%
#   0.70    15.4%    0.0%    100.0%   <== chọn
#   0.75     7.7%    0.0%    100.0%
#   0.80+    0.0%     -         -        (fast-track không bao giờ chạy)
#
# Chọn 0.70: sai 0% VÀ fast-track nhiều nhất trong nhóm sai ≈ 0.
# Lưu ý: 0.80 trở lên -> fast-track không bao giờ chạy (vô dụng).
FAST_TRACK_THRESHOLD = 0.70

# Số node trả về cho LLM
DEFAULT_TOP_K = 5
# Số ứng viên trước rerank
DEFAULT_RERANK_TOP_N = 10


# ---------------------------------------------------------------------------
# Kết quả định tuyến
# ---------------------------------------------------------------------------
ROUTE_FAST_TRACK = "fast_track"
ROUTE_VECTOR = "vector"
ROUTE_GRAPH = "graph"
ROUTE_ASK = "ask"


@dataclass
class RouteDecision:
    """Quyết định định tuyến cho 1 câu hỏi."""

    route: str
    understanding: QueryUnderstanding
    reason: str = ""
    confidence: float = 0.0
    target_node_ids: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    # PC-1 (§14.1): nhãn 5-route (exact / anchor_semantic / semantic /
    # full / anchor_unresolved) do AnchorVerifier phân loại (rule-first +
    # verify-by-level). "" = chưa phân loại (backward-compat).
    five_route: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "five_route": self.five_route,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "target_node_ids": self.target_node_ids,
            "datasets": self.datasets,
            **self.understanding.to_dict(),
        }


# ---------------------------------------------------------------------------
# Resolve anchor -> node id trong KG
# ---------------------------------------------------------------------------
def vb_key(so_hieu: str) -> str:
    """'59/2020/QH14' -> '59_2020_QH14' (khớp loader._vb_key)."""
    return so_hieu.replace("/", "_").replace("NĐ-CP", "NĐ_CP")


def anchor_node_ids(
    understanding: QueryUnderstanding,
) -> list[str]:
    """Sinh node id từ anchor (điều/khoản/điểm + số hiệu vb).

    - Có số hiệu vb -> node id chính xác.
    - Không có số hiệu vb -> không sinh id (phải tra KG để tìm văn bản
      chứa điều đó) -> trả rỗng, caller lo.
    """
    docs = understanding.anchors.doc_numbers
    if not docs:
        return []
    out: list[str] = []
    for so_hieu in docs:
        key = vb_key(so_hieu)
        a = understanding.anchors.articles
        k = understanding.anchors.clauses
        p = understanding.anchors.points
        if a and k and p:
            # Có khoản/điểm: parser chưa tách cặp (điều, khoản) khi câu có
            # NHIỀU điều + NHIỀU khoản -> giữ a[0]/k[0]/p[0] (1 node), không
            # làm cross-product (bịa cặp sai). S6 là "chỉ điều, không khoản"
            # nên rơi vào nhánh elif a bên dưới.
            out.append(f"{key}_D{a[0]}_K{k[0]}_P{p[0]}")
        elif a and k:
            out.append(f"{key}_D{a[0]}_K{k[0]}")
        elif a:
            # F3-lớp1 (S6): "Điều 2 và Điều 4" -> articles=["2","4"]; dùng
            # a[0] chỉ sinh D2, bỏ mất D4 -> câu đa điều mất nửa answer.
            # Loop tất cả điều: mỗi điều 1 node (không kèm khoản).
            for art in a:
                out.append(f"{key}_D{art}")
    return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class QueryRouter:
    """Định tuyến câu hỏi -> fast-track / vector / graph / ask.

    Args:
        store: VectorStore (để tính confidence fast-track).
        graph_lookup: hàm `graph_lookup(node_ids) -> list[dict]` trả node
            + cây con từ Neo4j. Nếu None -> fast-track bị tắt (không an
            toàn trả lời khi không tra được KG).
        confidence_fn: hàm `(query, dataset) -> float` trả điểm FAISS
            top-1. Mặc định dùng store.search.
        fast_track_threshold: ngưỡng confidence.
    """

    def __init__(
        self,
        store: Any | None = None,
        graph_lookup: Callable[[list[str]], list[dict[str, Any]]] | None = None,
        confidence_fn: Callable[[str, str | None], float] | None = None,
        fast_track_threshold: float = FAST_TRACK_THRESHOLD,
    ) -> None:
        self.store = store
        self.graph_lookup = graph_lookup
        self.confidence_fn = confidence_fn or self._default_confidence
        self.fast_track_threshold = fast_track_threshold

    # -- Helpers -----------------------------------------------------------

    def _default_confidence(
        self, query: str, dataset: str | None
    ) -> float:
        """Điểm FAISS top-1 (max cosine) của câu hỏi.

        Dùng `dense_score` (cosine thật của BGE-M3), KHÔNG dùng `score`
        — `score` là RRF fusion score (~0.03, chỉ phản ánh thứ hạng nên
        hằng số theo rank, không phản ánh độ khớp). Cosine trên text
        pháp lý tiếng Việt nằm khoảng 0.5-0.8; ngưỡng chốt bằng sweep.
        """
        if self.store is None:
            return 0.0
        try:
            hits = self.store.search(
                query, dataset=dataset, top_k=1, rerank_top_n=0
            )
        except Exception:
            return 0.0
        if not hits:
            return 0.0
        return float(hits[0].get("dense_score", 0.0))

    def _datasets_for(self, understanding: QueryUnderstanding) -> list[str]:
        """Dataset cần search (1 nhóm PL, hoặc [] = tất cả).

        A1 (P7): chỉ giới hạn tủ khi nhóm PL suy ra từ **SỐ HIỆU VĂN BẢN**
        (registry — nguồn đáng tin, user nêu trực tiếp). Nhóm suy ra từ
        **KEYWORD** (nguồn đoán) KHÔNG được dùng để giới hạn:
        "doanh nghiệp" có thể xuất hiện trong câu thuộc thuế TNDN ->
        chọn sai tủ -> provision đúng bị bỏ ngoài cửa -> rerank thấp
        -> OOD gate chặn nhầm. 4 tủ nhỏ, lục hết rẻ hơn, X2 rerank
        sẽ xếp lại đúng chỗ.
        """
        try:
            from QueryUnderstanding import _registry_dataset
        except ImportError:
            from retrieval.QueryUnderstanding import _registry_dataset

        for so_hieu in understanding.anchors.doc_numbers:
            group = _registry_dataset(so_hieu)
            if group:
                return [group]
        return []

    # -- Route -------------------------------------------------------------

    def route(self, query: str) -> RouteDecision:
        """Quyết định nhánh xử lý cho câu hỏi.

        3 nhóm (P7 X1):
          1. CÂU QUAN HỆ (is_relation: sửa/bãi bỏ/hiệu lực/hướng dẫn)
             + có anchor (số hiệu vb + điều) -> FAST-TRACK VÔ ĐIỀU KIỆN:
             không so cosine FAISS (cosine câu quan hệ luôn ~0.0, so
             ngưỡng thì fast-track không bao giờ bật). Đáp án nằm trong
             đồ thị (AMENDS/REPEALS/GUIDES) -> Cypher 0 embedding.
          2. CÂU NỘI DUNG (tra_cuu/chinh_xac) + có anchor -> fast-track
             CHỈ khi cosine FAISS ≥ ngưỡng (nhánh này mới cần độ tin
             cậy, vì đáp án nằm trong text provision).
          3. Không anchor / không tra KG được -> vector recall.
        """
        u = understand(query)
        datasets = self._datasets_for(u)
        node_ids = anchor_node_ids(u)

        # 1) Câu quan hệ + anchor đủ -> đồ thị, bỏ qua vector hoàn toàn.
        if u.is_relation and node_ids and self.graph_lookup is not None:
            return RouteDecision(
                route=ROUTE_FAST_TRACK,
                understanding=u,
                reason=(
                    f"câu quan hệ + anchor {node_ids} -> đồ thị tất định "
                    f"(không so cosine: cosine câu quan hệ luôn thấp)"
                ),
                confidence=1.0,
                target_node_ids=node_ids,
                datasets=datasets,
            )

        # 2) Fast-track câu nội dung: có anchor + confidence đủ cao.
        if u.can_fast_track:
            if node_ids and self.graph_lookup is not None:
                conf = self.confidence_fn(query, datasets[0] if datasets else None)
                if conf >= self.fast_track_threshold:
                    return RouteDecision(
                        route=ROUTE_FAST_TRACK,
                        understanding=u,
                        reason=(
                            f"có anchor {node_ids} + confidence "
                            f"{conf:.3f} ≥ {self.fast_track_threshold}"
                        ),
                        confidence=conf,
                        target_node_ids=node_ids,
                        datasets=datasets,
                    )
                return RouteDecision(
                    route=ROUTE_VECTOR,
                    understanding=u,
                    reason=(
                        f"có anchor nhưng confidence {conf:.3f} < "
                        f"{self.fast_track_threshold} -> đi nhánh đầy đủ"
                    ),
                    confidence=conf,
                    target_node_ids=node_ids,
                    datasets=datasets,
                )

        # 3) Có anchor nhưng không tra KG được -> vẫn đi vector (an toàn).
        # 4) Không có anchor -> hỏi ngược user nếu mơ hồ, hoặc vector.
        if u.should_ask_web and u.intent == INTENT_TRA_CUU:
            # Chưa có anchor: vẫn thử vector trước (câu OOD vẫn trả lời
            # được nếu có nội dung liên quan). Agent P7 sẽ hỏi ngược khi
            # kết quả rỗng / điểm thấp.
            return RouteDecision(
                route=ROUTE_VECTOR,
                understanding=u,
                reason="mơ hồ, không có anchor -> vector recall (có thể hỏi ngược sau)",
                datasets=datasets,
            )

        return RouteDecision(
            route=ROUTE_VECTOR,
            understanding=u,
            reason=f"ý định '{u.intent}' -> vector recall + đồ thị",
            datasets=datasets,
        )

    # -- Execute -----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        rerank_fn: Callable[
            [str, list[dict[str, Any]]], list[dict[str, Any]]
        ]
        | None = None,
        decision: RouteDecision | None = None,
    ) -> tuple[list[dict[str, Any]], RouteDecision]:
        """Chạy nhánh đã định tuyến, trả (kết quả, quyết định).

        - fast_track -> lấy node từ KG (không search vector).
        - vector      -> VectorStore.search (hybrid + rerank).
        """
        d = decision or self.route(query)

        if d.route == ROUTE_FAST_TRACK and self.graph_lookup is not None:
            nodes = self.graph_lookup(d.target_node_ids)
            return (
                [
                    {
                        "node_id": n.get("node_id"),
                        "text": n.get("text", ""),
                        "metadata": n.get("metadata", {}),
                        "source": "graph",
                        "score": 1.0,
                    }
                    for n in nodes
                ],
                d,
            )

        if self.store is None:
            return [], d

        dataset = d.datasets[0] if d.datasets else None
        hits = self.store.search(
            query,
            dataset=dataset,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            rerank_fn=rerank_fn,
        )
        for h in hits:
            h["source"] = "vector"
        return hits, d


__all__ = [
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_TOP_K",
    "FAST_TRACK_THRESHOLD",
    "QueryRouter",
    "ROUTE_ASK",
    "ROUTE_FAST_TRACK",
    "ROUTE_GRAPH",
    "ROUTE_VECTOR",
    "RouteDecision",
    "anchor_node_ids",
    "vb_key",
]
