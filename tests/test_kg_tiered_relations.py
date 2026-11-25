"""Test P5 — cạnh quan hệ trỏ đúng TẦNG (điểm -> khoản -> điều).

Test trên assemble() trong bộ nhớ (không đụng Neo4j): chỉ cần data/processed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "graph",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from graph.kg.build import (  # noqa: E402
    _clause_targets,
    _resolve_target_node,
    assemble,
)
from graph.kg.config import EDGE_AMENDS, EDGE_GUIDES, EDGE_REPEALS  # noqa: E402


# ---------------------------------------------------------------------------
# _clause_targets — sinh node id theo tầng (thứ tự fallback sâu -> nông)
# ---------------------------------------------------------------------------
def test_clause_targets_khoan_diem():
    assert _clause_targets("D60", "khoản 2 điểm e") == [
        "D60_K2_Pe", "D60_K2", "D60",
    ]


def test_clause_targets_khoan():
    assert _clause_targets("D60", "khoản 5") == ["D60_K5", "D60"]


def test_clause_targets_bosung():
    assert _clause_targets("D4", "bổ sung 35") == ["D4_K35", "D4"]


def test_clause_targets_bare_number():
    """'14' trong '(khoản 5, 14, 16)' -> khoản 14 của điều."""
    assert _clause_targets("D4", "14") == ["D4_K14", "D4"]


def test_clause_targets_sua_diem_khoan():
    assert _clause_targets("D51", "sửa điểm c khoản 1") == [
        "D51_K1_Pc", "D51_K1", "D51",
    ]


def test_clause_targets_unknown_falls_back_to_article():
    """Token lạ -> không trỏ sâu, chỉ trỏ Điều (không bịa)."""
    assert _clause_targets("D4", "thay thế") == ["D4"]


def test_clause_targets_empty():
    assert _clause_targets("D4", "") == ["D4"]


# ---------------------------------------------------------------------------
# assemble() — cạnh quan hệ trỏ đúng tầng
# ---------------------------------------------------------------------------
def _assembled():
    return assemble()


def test_relation_edges_target_deep_tiers():
    """Cạnh quan hệ phải trỏ được tới Clause/Point khi CSV ghi rõ."""
    g = _assembled()
    tiers = {}
    for e in g.edges:
        node = g.nodes.get(e["to_id"])
        tiers[node.kind if node else "MISSING"] = (
            tiers.get(node.kind if node else "MISSING", 0) + 1
        )

    # Trước P5: 100% trỏ Article/Annex. Sau P5: có cả Clause và Point.
    assert tiers.get("clause", 0) > 0, "chưa có cạnh nào trỏ tới Khoản"
    assert tiers.get("point", 0) > 0, "chưa có cạnh nào trỏ tới Điểm"
    assert tiers.get("MISSING", 0) == 0, "có cạnh trỏ tới node không tồn tại"


def test_relation_edges_count_preserved():
    """Số cạnh quan hệ không đổi (264) — chỉ to_id đổi tầng."""
    g = _assembled()
    assert len(g.edges) == 264


def test_nodes_unchanged_by_p5():
    """P5 KHÔNG đổi số node — chỉ đổi tầng của node đích cạnh."""
    g = _assembled()
    assert len(g.nodes) == 5172
    assert len(g.structures) == 75


def test_no_missing_relations():
    g = _assembled()
    assert g.missing == []


def test_specific_known_edges():
    """Spot-check: quan hệ đã biết -> đúng tầng.

    P7: cạnh được NÂNG CẤP lên cấp khoản/điểm khi JSONL xác nhận
    (`amendment_operations`). Đích đã ở đúng tầng từ P5; NGUỒN thì
    nay có thể là cấp khoản thay vì cấp điều.
    """
    g = _assembled()
    pairs = {
        (e["relation_type"], e["from_id"], e["to_id"])
        for e in g.edges
    }
    # 03/2022/QH15 Điều 7 sửa 59/2020/QH14 Điều 60 khoản 2 điểm e
    assert ("AMENDS", "03_2022_QH15_D7", "59_2020_QH14_D60_K2_Pe") in pairs
    # 149/2025/QH15 bãi bỏ 48/2024/QH15 Điều 15 khoản 9 điểm c
    # (P7: nguồn nâng lên cấp khoản — JSONL xác nhận khoản 3 sửa
    #  khoản 9 điểm c)
    assert ("REPEALS", "149_2025_QH15_D1_K3", "48_2024_QH15_D15_K9_Pc") in pairs
    # 296/2026/NĐ-CP sửa 168/2025/NĐ-CP Điều 51 khoản 1 điểm c
    assert ("AMENDS", "296_2026_NĐ_CP_D10", "168_2025_NĐ_CP_D51_K1_Pc") in pairs


def test_relation_source_can_be_upgraded_to_clause():
    """P7: nguồn của cạnh được nâng lên cấp KHOẢN khi JSONL xác nhận."""
    g = _assembled()
    upgraded = [
        e for e in g.edges if e.get("detail_level") == "clause"
    ]
    assert upgraded, "không có cạnh nào được nâng lên cấp khoản"
    for e in upgraded:
        # Node nguồn phải tồn tại và đúng cấp khoản.
        assert e["from_id"] in g.nodes
        assert "_K" in e["from_id"]
        assert e.get("detail_operation_id")


def test_relation_source_keeps_article_when_unconfirmed():
    """P7: JSONL không xác nhận -> GIỮ cấp điều (không bịa, không mất quan hệ).

    Lưu ý: node NGUỒN cấp điều có thể **không tồn tại** trong KG — CSV
    có quan hệ từ Điều 1 của 141/2026/NĐ-CP nhưng `data/processed/` văn
    bản đó chỉ có Điều 2 (thiếu dữ liệu, không phải lỗi logic). Khi đó
    `seed()` bỏ cạnh (MATCH fail) → không trỏ hư không. Đây là hành vi
    ĐÚNG, có sẵn từ trước P7.
    """
    g = _assembled()
    for e in g.edges:
        if e.get("detail_level") == "article":
            assert "_K" not in e["from_id"]
            # Nếu node nguồn có tồn tại thì phải là node cấp ĐIỀU.
            if e["from_id"] in g.nodes:
                assert g.nodes[e["from_id"]].kind == "article"


def test_repeals_targets_are_repealed_nodes():
    """Node đích của REPEALS phải có invalid_at được gán."""
    g = _assembled()
    repealed = [
        e for e in g.edges if e["relation_type"] == EDGE_REPEALS
    ]
    assert repealed, "không có cạnh REPEALS nào"
    for e in repealed:
        node = g.nodes.get(e["to_id"])
        assert node is not None
        assert node.invalid_at == e["relation_effective_date"]


def test_resolve_target_node_prefers_existing():
    """_resolve_target_node chọn ứng viên sâu nhất TỒN TẠI."""
    g = _assembled()
    index = {
        ("X", "59/2020/QH14", "60"): [
            "59_2020_QH14_D60_K2_Pe",
            "59_2020_QH14_D60_K2",
            "59_2020_QH14_D60",
        ]
    }
    got = _resolve_target_node(
        g, index, "X", "59/2020/QH14", "60", "khoản 2 điểm e"
    )
    assert got == "59_2020_QH14_D60_K2_Pe"


def test_resolve_target_node_falls_back_when_deep_missing():
    """Ứng viên sâu không tồn tại -> lùi về tầng cha gần nhất."""
    g = _assembled()
    index = {
        ("Y", "59/2020/QH14", "60"): [
            "59_2020_QH14_D60_K99_Pz",  # không tồn tại
            "59_2020_QH14_D60_K2",     # tồn tại
            "59_2020_QH14_D60",
        ]
    }
    got = _resolve_target_node(
        g, index, "Y", "59/2020/QH14", "60", "khoản 99 điểm z"
    )
    assert got == "59_2020_QH14_D60_K2"
