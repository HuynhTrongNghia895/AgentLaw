"""Test materialize (P2) + QueryUnderstanding (P3) — không cần Neo4j/FAISS."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "graph",
    PROJECT_ROOT / "src" / "retrieval",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from graph.kg.loader import KgNode  # noqa: E402
from graph.kg.materialize import (  # noqa: E402
    format_evidence_block,
    materialize,
    node_citation,
)
from QueryUnderstanding import (  # noqa: E402
    detect_intent,
    detect_legal_group,
    extract_anchors,
    understand,
)


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------
def _article(**kw) -> KgNode:
    base = dict(
        node_id="168_2025_NĐ_CP_D100",
        so_hieu="168/2025/NĐ-CP",
        kind="article",
        dieu="100",
        title="Hồ sơ, trình tự, thủ tục đăng ký thay đổi",
        text="",
        implicit=True,
    )
    base.update(kw)
    return KgNode(**base)


def _clause(**kw) -> KgNode:
    base = dict(
        node_id="168_2025_NĐ_CP_D100_K1",
        so_hieu="168/2025/NĐ-CP",
        kind="clause",
        dieu="100",
        khoan="1",
        text="Chủ hộ kinh doanh có trách nhiệm đăng ký thay đổi.",
        parent_id="168_2025_NĐ_CP_D100",
    )
    base.update(kw)
    return KgNode(**base)


def _point(marker: str, text: str) -> KgNode:
    return KgNode(
        node_id=f"168_2025_NĐ_CP_D100_K1_P{marker}",
        so_hieu="168/2025/NĐ-CP",
        kind="point",
        dieu="100",
        khoan="1",
        diem=marker,
        diem_index=ord(marker) - 96,
        text=text,
        parent_id="168_2025_NĐ_CP_D100_K1",
    )


def test_node_citation_formats():
    assert node_citation(_article()) == "Điều 100 (168/2025/NĐ-CP)"
    assert node_citation(_clause()) == "Điều 100 Khoản 1 (168/2025/NĐ-CP)"
    assert node_citation(_point("a", "x")) == (
        "Điều 100 Khoản 1 Điểm a (168/2025/NĐ-CP)"
    )


def test_materialize_clause_includes_points():
    """Khoản -> evidence phải có câu dẫn + các điểm con, giữ ranh giới."""
    clause = _clause()
    children = {
        clause.node_id: [
            _point("a", "Tên hộ kinh doanh;"),
            _point("b", "Địa chỉ trụ sở;"),
        ]
    }
    ev = materialize(clause, children=children)
    assert "Khoản 1." in ev.text
    assert "a) Tên hộ kinh doanh;" in ev.text
    assert "b) Địa chỉ trụ sở;" in ev.text
    assert ev.citation == "Điều 100 Khoản 1 (168/2025/NĐ-CP)"


def test_materialize_article_includes_clause_and_point_subtree_verbatim():
    article = _article()
    clause = _clause(text="")
    point = _point("a", "Nguyên văn điểm a.")
    children = {
        article.node_id: [clause],
        clause.node_id: [point],
    }
    ev = materialize(article, children=children, include_status=False)
    assert "Khoản 1." in ev.text
    assert "a) Nguyên văn điểm a." in ev.text


def test_materialize_empty_node_and_subtree_has_no_body_text():
    article = _article(title="", text="")
    clause = _clause(text="")
    point = _point("a", "")
    children = {article.node_id: [clause], clause.node_id: [point]}
    ev = materialize(article, children=children, include_status=False)
    assert ev.text.strip() == "Điều 100"


def test_materialize_point_is_standalone():
    ev = materialize(_point("a", "Tên hộ kinh doanh;"))
    assert ev.text.startswith("Điểm a.")
    assert "Tên hộ kinh doanh;" in ev.text


def test_materialize_adds_effective_note():
    clause = _clause(provision_effective_date="2026-07-01")
    ev = materialize(clause, doc_meta={"effective_date": "2026-01-01"})
    assert "2026-01-01" in ev.text
    assert "2026-07-01" in ev.text
    assert "2026-07-01" in ev.metadata["effective_note"]


def test_materialize_without_status_skips_note():
    ev = materialize(_clause(), doc_meta={"effective_date": "2026-01-01"},
                     include_status=False)
    assert "Hiệu lực" not in ev.text


def test_format_evidence_block_has_citations():
    ev = materialize(_clause())
    block = format_evidence_block([ev])
    assert "[Điều 100 Khoản 1 (168/2025/NĐ-CP)]" in block
    assert "Khoản 1." in block


# ---------------------------------------------------------------------------
# QueryUnderstanding
# ---------------------------------------------------------------------------
def test_extract_anchors_full():
    a = extract_anchors("Điều 4 khoản 23 điểm a của 59/2020/QH14")
    assert a.doc_numbers == ["59/2020/QH14"]
    assert a.articles == ["4"]
    assert a.clauses == ["23"]
    assert a.points == ["a"]
    assert a.has_provision and a.has_doc


def test_extract_anchors_short_doc_number():
    """"NĐ 168/2025" -> tra registry ra số hiệu đầy đủ."""
    u = understand("Điều 100 NĐ 168/2025 còn hiệu lực không?")
    assert u.anchors.doc_numbers == ["168/2025/NĐ-CP"]


def test_detect_legal_group_by_keyword():
    assert detect_legal_group("Thuế TNDN tính thu nhập doanh nghiệp") == "thue_tndn"
    assert detect_legal_group("hộ kinh doanh đăng ký thay đổi") == "doanh_nghiep"
    assert detect_legal_group("hóa đơn giá trị gia tăng") == "thue_gtgt"


def test_detect_legal_group_by_document():
    """Văn bản được nhắc -> lấy dataset từ registry (mạnh hơn keyword)."""
    assert detect_legal_group("NĐ 168/2025 quy định gì") == "doanh_nghiep"
    assert detect_legal_group("Luật 48/2024") == "thue_gtgt"


def test_detect_legal_group_none_for_ood():
    assert detect_legal_group("lịch sử phát triển của Việt Nam") is None


@pytest.mark.parametrize(
    "query,intent",
    [
        ("Điều 100 NĐ 168/2025 còn hiệu lực không?", "hieu_luc"),
        ("Điều 5 Luật GTGT 48/2024 đã bãi bỏ chưa?", "hieu_luc"),
        ("Tại thời điểm 01/09/2025 điều 5 có hiệu lực không?", "time_travel"),
        ("So sánh Luật Doanh nghiệp 59/2020 với NĐ 168/2025", "so_sanh"),
        ("Điều 4 khoản 23 Luật Doanh nghiệp quy định gì?", "chinh_xac"),
        ("Hộ kinh doanh thay đổi nội dung đăng ký thì bao lâu?", "tra_cuu"),
        ("Mức vốn điều lệ tối thiểu là bao nhiêu?", "tra_cuu"),
    ],
)
def test_detect_intent(query, intent):
    assert detect_intent(query) == intent


def test_understanding_flags():
    u = understand("Điều 100 NĐ 168/2025 còn hiệu lực không?")
    assert u.needs_graph is True
    assert u.can_fast_track is False

    u2 = understand("Điều 4 khoản 23 Luật Doanh nghiệp 59/2020 quy định gì?")
    assert u2.can_fast_track is True
    assert u2.legal_group == "doanh_nghiep"


# ---------------------------------------------------------------------------
# QueryRouter (P4)
# ---------------------------------------------------------------------------
from QueryRouter import (  # noqa: E402
    QueryRouter,
    ROUTE_FAST_TRACK,
    ROUTE_VECTOR,
    anchor_node_ids,
    vb_key,
)


def test_vb_key_and_anchor_node_ids():
    assert vb_key("168/2025/NĐ-CP") == "168_2025_NĐ_CP"
    assert vb_key("59/2020/QH14") == "59_2020_QH14"

    from QueryUnderstanding import understand
    u = understand("Điều 4 khoản 23 điểm a của 59/2020/QH14")
    assert anchor_node_ids(u) == ["59_2020_QH14_D4_K23_Pa"]


def test_anchor_node_ids_multi_article_s6():
    """F3-lớp1 (S6): 'Điều 2 và Điều 4 của X' -> sinh CẢ D2 + D4.

    Lỗi cũ: nhánh `elif a:` chỉ dùng a[0] -> 'Điều 2 và Điều 4' mất D4,
    câu đa điều trả nửa answer. Sửa: loop mọi điều (chỉ điều, không kèm
    khoản/điểm)."""
    from QueryUnderstanding import understand
    u = understand("Điều 2 và Điều 4 của 108/2025/QH15")
    assert u.anchors.articles == ["2", "4"]
    assert anchor_node_ids(u) == ["108_2025_QH15_D2", "108_2025_QH15_D4"]


def test_anchor_node_ids_single_article_unchanged():
    """1 điều (không khoản/điểm) -> vẫn đúng 1 node, không bị thay đổi."""
    from QueryUnderstanding import understand
    u = understand("Điều 5 của 59/2020/QH14")
    assert anchor_node_ids(u) == ["59_2020_QH14_D5"]


def test_router_fast_track():
    """"Chinh xac" + confidence cao + có graph_lookup -> fast-track."""
    u = understand("Điều 4 khoản 23 của 59/2020/QH14")
    store = None  # không cần
    graph_lookup = lambda ids: [
        {"node_id": i, "text": "nội dung", "metadata": {"so_hieu": "59/2020/QH14"}}
        for i in ids
    ]
    router = QueryRouter(store=store, graph_lookup=graph_lookup, confidence_fn=lambda q, ds: 0.95)
    d = router.route("Điều 4 khoản 23 của 59/2020/QH14")
    assert d.route == ROUTE_FAST_TRACK
    assert "59_2020_QH14_D4_K23" in d.target_node_ids


def test_router_low_confidence_goes_vector():
    """"Chinh xac" nhưng confidence thấp -> đi nhánh vector."""
    router = QueryRouter(store=None, graph_lookup=lambda ids: [], confidence_fn=lambda q, ds: 0.4)
    d = router.route("Điều 4 khoản 23 của 59/2020/QH14")
    assert d.route == ROUTE_VECTOR


def test_router_hieu_luc_fast_track_graph():
    """Ý định hieu_luc + anchor -> fast-track (đồ thị, KHÔNG so cosine).

    P7 X1: câu QUAN HỆ (hỏi hiệu lực/sửa đổi) có cosine FAISS luôn
    thấp (~0.0) vì text nội dung không khớp ngữ nghĩa câu hỏi quan hệ.
    Áp ngưỡng cosine 0.70 thì fast-track không bao giờ bật -> rơi vào
    vector -> kéo nhầm văn bản khác vào evidence. Nên câu quan hệ đi
    thẳng nhánh đồ thị, bỏ qua bước so ngưỡng.
    """
    router = QueryRouter(store=None, graph_lookup=lambda ids: [], confidence_fn=lambda q, ds: 0.0)
    d = router.route("Điều 100 NĐ 168/2025 còn hiệu lực không?")
    assert d.route == ROUTE_FAST_TRACK
    assert d.target_node_ids == ["168_2025_NĐ_CP_D100"]


def test_router_relation_query_fast_track_without_cosine():
    """Câu quan hệ theo TỪ KHOÁ (sửa/bổ sung) -> fast-track, không so cosine.

    Ý định của câu này là `chinh_xac` (không phải hieu_luc) nhưng vẫn là
    câu QUAN HỆ -> phải đi đồ thị, không được áp ngưỡng cosine.
    """
    router = QueryRouter(store=None, graph_lookup=lambda ids: [], confidence_fn=lambda q, ds: 0.0)
    d = router.route("Điều 49 Luật Doanh nghiệp 2020 bị văn bản nào sửa đổi?")
    assert d.route == ROUTE_FAST_TRACK
    assert d.target_node_ids == ["59_2020_QH14_D49"]


def test_router_content_query_still_uses_cosine():
    """Câu NỘI DUNG (không phải quan hệ) -> vẫn so cosine để quyết định.

    Bảo đảm X1 không làm hỏng fast-track của câu nội dung: cosine thấp
    -> rơi về vector như trước.
    """
    router = QueryRouter(store=None, graph_lookup=lambda ids: [], confidence_fn=lambda q, ds: 0.4)
    d = router.route("Khoản 3 Điều 12 Nghị định 168/2025 quy định gì?")
    assert d.route == ROUTE_VECTOR

    router2 = QueryRouter(store=None, graph_lookup=lambda ids: [], confidence_fn=lambda q, ds: 0.9)
    d2 = router2.route("Khoản 3 Điều 12 Nghị định 168/2025 quy định gì?")
    assert d2.route == ROUTE_FAST_TRACK
