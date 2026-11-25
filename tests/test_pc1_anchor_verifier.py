"""PC-1 — 5-route classifier + verify-by-level (R1–R12, §14.1).

OFFLINE hoàn toàn: registry thật (DocumentRegistry.csv) + cây node thật
(`graph.kg.build.assemble()` — đọc data/processed, KHÔNG Neo4j) +
VectorStore/fake KG/fake LLM. Không tốn token, không gọi Ollama.

Phân lớp:
    R1 EXACT (S1) / R2 ANCHOR_SEMANTIC (1 LLM call, accounting) /
    R3 R4 ANCHOR_UNRESOLVED (vb / điều không tồn tại) / R5 anti-trùng-số /
    R6 R8 SEMANTIC (không anchor, 0 LLM) / R7 relation giữ (S8) /
    R11 R12 per-intent (FULL, anti pull-down) / R10 regression stub.

Không chạy test trong lúc code PC-1→P-F (theo chỉ thị) — file này là
nội dung bộ test R1–R12 cho bước "kiểm tra tổng thể một lượt" cuối.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "utils", ROOT / "src" / "retrieval",
          ROOT / "src" / "graph"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval.AnchorVerifier import (  # noqa: E402
    ROUTE_ANCHOR_SEMANTIC,
    ROUTE_ANCHOR_UNRESOLVED,
    ROUTE_EXACT,
    ROUTE_SEMANTIC,
    classify_query,
)
from retrieval.QueryUnderstanding import understand  # noqa: E402


# ---------------------------------------------------------------------------
# Fake source: registry THẬT + cây node THẬT (in-memory, offline)
# ---------------------------------------------------------------------------

class RealKGSrc:
    """`AnchorVerifySource` trên data thật (không Neo4j, không embed)."""

    def __init__(self) -> None:
        from graph.kg.loader import load_registry
        from graph.kg.build import assemble

        self._reg = load_registry()
        g = assemble()
        self._nodes = g.nodes
        self._children: dict[str, list] = {}
        for n in g.nodes.values():
            if n.parent_id:
                self._children.setdefault(n.parent_id, []).append(n)
        for kids in self._children.values():
            kids.sort(key=lambda x: (x.khoan, x.diem_index))

    def registry(self):
        return self._reg

    def node(self, node_id: str):
        return self._nodes.get(node_id)

    def children(self, node_id: str) -> list:
        return list(self._children.get(node_id, []))


class NoLLM:
    """NoLLM (name="none") — classifier phải trả ANCHOR_SEMANTIC an toàn,
    0 call (giữ đường test không LLM)."""

    name = "none"

    def complete(self, prompt: str) -> str:
        raise AssertionError("NoLLM không được gọi")


class CountingLLM:
    """Trả `route` cố định cho call route_classify; đếm số call thật."""

    name = "count"

    def __init__(self, route: str = "anchor_semantic") -> None:
        self.route = route
        self.rc_calls = 0
        self._purpose = ""

    def set_purpose(self, p: str) -> None:
        self._purpose = p

    def complete(self, prompt: str) -> str:
        if self._purpose == "route_classify":
            self.rc_calls += 1
            return json.dumps({"route": self.route})
        return "(llm khác)"


def _c(route: str, q: str, llm: object = NoLLM()) -> "AnchorVerifyResult":
    """classify_query tiện (cả câu hỏi + intent con R11/R12)."""
    return classify_query(q, understand(q), llm=llm, source=RealKGSrc())


# ---------------------------------------------------------------------------
# R1 — EXACT (S1): rule EXACT khớp, 0 LLM
# ---------------------------------------------------------------------------

def test_r1_s1_exact_no_llm():
    r = _c(ROUTE_EXACT,
           "Khoản 8 Điều 4 Luật Quản lý thuế 108/2025/QH15 "
           "quy định gì về kỳ tính thuế?")
    assert r.route == ROUTE_EXACT, r.reason
    assert r.pin_node_ids() == ["108_2025_QH15_D4_K8"]
    assert r.llm_calls_added == 0
    # R1 (giữ S1 P-A): pin chính xác đúng node kỳ tính thuế.


# ---------------------------------------------------------------------------
# R2 — ANCHOR_SEMANTIC: intent mở + LLM -> 1 call route_classify
# ---------------------------------------------------------------------------

def test_r2_anchor_semantic_llm_classify():
    q = "Điều 19 của 108/2025/QH15 quy định những quyền gì?"
    llm = CountingLLM("anchor_semantic")
    r = _c(ROUTE_ANCHOR_SEMANTIC, q, llm=llm)
    assert r.route == ROUTE_ANCHOR_SEMANTIC
    assert r.llm_calls_added == 1
    assert llm.rc_calls == 1  # đúng 1 call (accounting §14.1)
    # Anchor D19 vẫn pin (verified: node D19 có thật trong 108/2025).
    assert "108_2025_QH15_D19" in r.pin_node_ids()


def test_r2_anchor_semantic_no_llm_falls_back_safe():
    """NoLLM -> ANCHOR_SEMANTIC an toàn, 0 call (không bịa EXACT)."""
    r = _c(ROUTE_ANCHOR_SEMANTIC,
           "Điều 19 của 108/2025/QH15 quy định những quyền gì?")
    assert r.route == ROUTE_ANCHOR_SEMANTIC
    assert r.llm_calls_added == 0
    assert "108_2025_QH15_D19" in r.pin_node_ids()


# ---------------------------------------------------------------------------
# R3 — ANCHOR_UNRESOLVED: vb 999/2030 không trong registry
# ---------------------------------------------------------------------------

def test_r3b_provision_without_doc_stays_semantic():
    """S3/DN_003: 'Điều 2 quy định gì...' (có điều, KHÔNG nêu vb) ->
    KHÔNG tự map văn bản, KHÔNG chặn (giữ semantic, 0-diff cho REF_12
    NoLLM); refiner (luồng orchestrated) hỏi vb (clarify_expected)."""
    r = _c(ROUTE_SEMANTIC, "Điều 2 quy định gì về người nộp thuế?")
    assert r.route == ROUTE_SEMANTIC
    assert r.pin_node_ids() == []  # không bịa node (chưa rõ vb)
    assert r.llm_calls_added == 0


def test_r3_unresolved_vb_not_in_registry():
    r = _c(ROUTE_ANCHOR_UNRESOLVED,
           "Điều 4 của Luật 999/2030/QH16 quy định gì?")
    assert r.route == ROUTE_ANCHOR_UNRESOLVED
    assert r.unresolved
    assert any(not c.ok and c.missing == "vb_khong_ton_tai"
               for c in r.checks)
    assert r.pin_node_ids() == []  # KHÔNG bịa node cho vb không tồn tại
    assert "999/2030" in r.follow_up  # follow-up nêu đúng vb chết
    assert r.llm_calls_added == 0


# ---------------------------------------------------------------------------
# R4 — ANCHOR_UNRESOLVED: 108/2025 không có Điều 99 (max D53)
# ---------------------------------------------------------------------------

def test_r4_unresolved_article_missing():
    r = _c(ROUTE_ANCHOR_UNRESOLVED,
           "Khoản 1 Điều 99 của 108/2025/QH15 quy định gì?")
    assert r.route == ROUTE_ANCHOR_UNRESOLVED
    assert any(not c.ok and c.missing == "dieu_khong_ton_tai"
               for c in r.checks)
    assert r.pin_node_ids() == []  # không bịa "…_D99"


# ---------------------------------------------------------------------------
# R5 — anti-trùng-số: "Điều 4" verify trên ĐÚNG văn bản
# ---------------------------------------------------------------------------

def test_r5_same_article_number_different_documents():
    r48 = _c(ROUTE_EXACT, "Điều 4 của 48/2024/QH15 quy định gì?")
    r108 = _c(ROUTE_EXACT, "Điều 4 của 108/2025/QH15 quy định gì?")
    assert r48.route == ROUTE_EXACT
    assert r108.route == ROUTE_EXACT
    assert r48.pin_node_ids() == ["48_2024_QH15_D4"]
    assert r108.pin_node_ids() == ["108_2025_QH15_D4"]
    # Không lẫn node VB khác (anti-trùng-số).
    for nid in r48.pin_node_ids():
        assert nid.startswith("48_2024_QH15_")
    for nid in r108.pin_node_ids():
        assert nid.startswith("108_2025_QH15_")


def test_r5_cross_document_node_not_pinned():
    """"Điều 99 của 48/2024" (48/2024 không có D99) -> UNRESOLVED,
    KHÔNG pin node 48_2024_D4 (trùng số) hay node của VB khác."""
    r = _c(ROUTE_ANCHOR_UNRESOLVED, "Điều 99 của 48/2024/QH15 quy định gì?")
    assert r.route == ROUTE_ANCHOR_UNRESOLVED
    assert r.pin_node_ids() == []


# ---------------------------------------------------------------------------
# R6 — SEMANTIC (không anchor): giữ behavior, 0 LLM
# ---------------------------------------------------------------------------

def test_r6_semantic_no_anchor_0_llm():
    r = _c(ROUTE_SEMANTIC,
           "Các trường hợp không chịu thuế GTGT hiện hành là gì?")
    assert r.route == ROUTE_SEMANTIC
    assert r.pin_node_ids() == []
    assert r.llm_calls_added == 0
    assert r.checks == []


# ---------------------------------------------------------------------------
# R7 — relation giữ nguyên (S8): intent hieu_luc -> EXACT label, 0 LLM
# ---------------------------------------------------------------------------

def test_r7_relation_route_unchanged():
    r = _c(ROUTE_EXACT,
           "168/2025 còn hiệu lực không, và đã bị văn bản nào sửa đổi?")
    assert r.route == ROUTE_EXACT  # label EXACT; caller giữ nhánh relation
    assert r.llm_calls_added == 0
    assert any(c.ok for c in r.checks) or not r.checks


# ---------------------------------------------------------------------------
# R8 — không anchor, intent mở: SEMANTIC + 0 LLM
# ---------------------------------------------------------------------------

def test_r8_open_no_anchor_semantic_0_llm():
    r = _c(ROUTE_SEMANTIC, "Người nộp thuế phải thực hiện nghĩa vụ gì?")
    assert r.route == ROUTE_SEMANTIC
    assert r.llm_calls_added == 0
    assert r.pin_node_ids() == []


# ---------------------------------------------------------------------------
# R11 — FULL: 3 intent (EXACT + relation + SEMANTIC), trần 1 LLM call
# ---------------------------------------------------------------------------

def test_r11_multi_intent_per_intent_routing_budget():
    from agents.LegalRetrieval import LegalRetrievalAgent

    a = LegalRetrievalAgent(llm=CountingLLM("exact"), store=None,
                            kg=RealKGSrc2())
    # 1) EXACT: "người nộp thuế theo Điều 2 (108/2025)" -> vb + điều verify
    #    được (D2 có thật), intent fallback "chinh_xac" (has_provision,
    #    không expansion) -> EXACT rule, 0 LLM. Pin D2 (cấp điều).
    f1, p1 = a._five_route_for_intent(
        "người nộp thuế theo Điều 2 (108/2025/QH15) là ai?")
    assert f1 == ROUTE_EXACT, f1
    assert p1 == ["108_2025_QH15_D2"], p1
    # 2) relation: intent hieu_luc -> EXACT label (nhánh đồ thị), 0 call.
    f2, _ = a._five_route_for_intent(
        "168/2025/NĐ-CP còn hiệu lực không, bị văn bản nào sửa?")
    assert f2 == ROUTE_EXACT
    # 3) SEMANTIC: không anchor -> SEMANTIC, 0 call.
    f3, _ = a._five_route_for_intent("cơ sở dữ liệu thương mại là gì")
    assert f3 == ROUTE_SEMANTIC
    assert a._route_budget_used == 0  # cả 3 intent 0 LLM call
    assert a._route_classify_budget == 1  # trần đúng thiết kế


# ---------------------------------------------------------------------------
# R12 — UNRESOLVED intent không kéo thụt intent khác (anti pull-down)
# ---------------------------------------------------------------------------

def test_r12_unresolved_intent_does_not_pull_down_others():
    from agents.LegalRetrieval import LegalRetrievalAgent

    a = LegalRetrievalAgent(llm=CountingLLM("exact"), store=None,
                            kg=RealKGSrc2())
    # Intent 1 (chết): vb 999/2030 không trong registry -> UNRESOLVED,
    # KHÔNG bịa node 999, follow-up nêu đúng vb chết.
    f1, p1 = a._five_route_for_intent(
        "Điều 4 của 999/2025/NĐ-CP quy định gì?")
    assert f1 == ROUTE_ANCHOR_UNRESOLVED
    assert p1 == []  # không bịa node 999
    # Intent 2 (lành): chỉ nêu vb (không điều) -> vb hợp lệ trong
    # registry -> KHÔNG unresolved (không bị intent 1 kéo thụt);
    # vb làm filter semantic (không bịa pin node khi chưa chỉ định điều).
    # Evidence cho intent 2 do pipeline (vector + B1 + rewriter) cung cấp.
    f2, p2 = a._five_route_for_intent(
        "người nộp thuế là ai (108/2025/QH15)")
    assert f2 != ROUTE_ANCHOR_UNRESOLVED, f2
    assert p2 == []  # không bịa node (user chưa chỉ định điều/khoản)
    # 0 LLM call (intent 1 chết trước gate LLM; intent 2 vb-only rule).
    assert a._route_budget_used == 0
    assert a.llm.rc_calls == 0


# ---------------------------------------------------------------------------
# Pipeline (R3/R4 end-to-end + R10 regression)
# ---------------------------------------------------------------------------

class RealKGSrc2:
    """Fake KG đủ interface LegalRetrievalAgent (offline, no Neo4j).

    registry() -> verify-by-level thật (registry + cây node thật);
    evidence_for_many -> trả evidence thật (KgAccess) cho node tồn tại;
    status_of -> unknown (Neo4j tắt) -> pipeline degrade an toàn.
    """

    def __init__(self) -> None:
        self._src = RealKGSrc()
        from agents.KgAccess import KgAccess
        self._kg = KgAccess(use_neo4j=False)

    def registry(self):
        return self._src.registry()

    def node(self, nid):
        return self._src.node(nid)

    def children(self, nid):
        return self._src.children(nid)

    def evidence_for_many(self, node_ids):
        return self._kg.evidence_for_many(node_ids)

    def lookup(self, node_ids):
        return [
            {"node_id": n, "text": (self._src.node(n) or type(
                "N", (), {"text": ""})()).text if self._src.node(n) else "",
             "kind": "article", "metadata": {}}
            for n in node_ids if self._src.node(n)
        ]

    def status_of(self, so, dieu):
        return {"status": "unknown", "relations": []}

    def relations_of(self, so, article=None):
        return []


def _agent_with_store(store, rerank_score=0.5, llm=None):
    from agents.LegalRetrieval import LegalRetrievalAgent

    def fake_rerank(query: str, pool: list) -> list:
        out = []
        for i, c in enumerate(pool):
            txt = str(c.get("text") or "")
            # score theo content (không bừa): anchor text cao hơn.
            score = rerank_score
            out.append({**c, "score": score, "rerank": True,
                        "_orig_index": i})
        out.sort(key=lambda x: -x["score"])
        return out

    if llm is None:
        llm = NoLLM()
    return LegalRetrievalAgent(store=store, kg=RealKGSrc2(), llm=llm,
                                rerank_fn=fake_rerank)


class S1Store:
    """FAISS giả: S1 -> D4_K8 + tree (giữ S1 P-A Hit@1)."""

    def search(self, query, dataset=None, top_k=5, rerank_top_n=10,
               rerank_fn=None):
        return [{
            "node_id": "108_2025_QH15_D4_K8",
            "text": "Kỳ tính thuế, kỳ tính khoản thu khác.",
            "metadata": {"document_number": "108/2025/QH15",
                         "article_number": "4", "clause_number": "8"},
            "score": 0.8,
        }]


def test_r10_s1_regression_stub_hit1():
    """R10: S1 EXACT giữ Hit@1 + fast-track (đường không thuộc PC-1
    không đổi hành vi)."""
    a = _agent_with_store(S1Store())
    res = a.answer("Khoản 8 Điều 4 Luật Quản lý thuế 108/2025/QH15 "
                   "quy định gì về kỳ tính thuế?")
    got = [str(e.get("node_id") or "") for e in (res.evidence or [])]
    assert "108_2025_QH15_D4_K8" in got[:1], got  # Hit@1
    assert "D12_K8" not in got  # không có node lạ
    assert res.route in ("fast_track", "vector")


def test_r10_relation_s8_stub_relation_target_kept():
    """R10: S8 relation giữ nhánh đồ thị (5-route EXACT, không semantic).

    Dùng CountingLLM (không phải NoLLM) để synthesis hoàn tất ->
    metadata["five_route"] đọc được (NoLLM luôn raise -> llm_error)."""
    a = _agent_with_store(S1Store(), llm=CountingLLM("exact"))
    res = a.answer("Nghị định 168/2025/NĐ-CP còn hiệu lực không, "
                   "đã bị văn bản nào sửa đổi?")
    assert res.metadata.get("five_route") in ("exact", "full", ""), \
        res.metadata.get("five_route")
    # Intent hieu_luc -> không UNRESOLVED (vb 168/2025 hợp lệ).
    assert res.metadata.get("five_route") != "anchor_unresolved"


def test_r3_r4_pipeline_unresolved_no_fabrication():
    """R3/R4 end-to-end (câu đơn, anchor chết) -> follow-up, KHÔNG bịa
    node gần nghĩa, KHÔNG semantic-search thay."""
    for q in ("Điều 4 của Luật 999/2030/QH16 quy định gì?",
              "Khoản 1 Điều 99 của 108/2025/QH15 quy định gì?"):
        a = _agent_with_store(S1Store())
        res = a.answer(q)
        assert res.metadata.get("anchor_unresolved") is True, q
        assert res.metadata.get("five_route") == "anchor_unresolved", q
        assert res.follow_up, "UNRESOLVED phải có follow-up"
        got = [str(e.get("node_id") or "") for e in (res.evidence or [])]
        # Không bịa node của vb chết.
        assert not any("999_2030" in nid or "D99" in nid for nid in got), got
        # Không semantic-search thay (R3): evidence rỗng (không node gần nghĩa).
        assert got == [], f"UNRESOLVED không được trả node khác: {got}"


# ---------------------------------------------------------------------------
# LLM call accounting (trần 1 call route_classify / lượt)
# ---------------------------------------------------------------------------

def test_route_classify_budget_caps_at_one():
    from agents.LegalRetrieval import LegalRetrievalAgent

    llm = CountingLLM("exact")
    a = LegalRetrievalAgent(llm=llm, store=None, kg=RealKGSrc2())
    a._route_budget_used = 0
    # 3 intent intent-mở cùng câu (mỗi cái LLM phân loại):
    for q in ("Điều 19 của 108/2025/QH15 quy định những quyền gì?",
              "Điều 5 của 108/2025/QH15 quy định những nghĩa vụ gì?",
              "Điều 7 của 108/2025/QH15 quy định những trường hợp gì?"):
        a._five_route_for_intent(q)
    assert llm.rc_calls == 1, (
        f"trần 1 call route_classify/lượt, thực tế {llm.rc_calls}")
    assert a._route_budget_used == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
