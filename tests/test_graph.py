"""Test graph pháp lý (Giai đoạn 3).

Fast (không cần Neo4j): parser provision - ~12 case định dạng CSV thật.
Slow (@pytest.mark.slow, cần Neo4j chạy + driver): build idempotent,
amendments_of đúng văn bản theo CSV, provision_status (âm: điều chưa
bị sửa -> active), related_documents.

Chạy fast:  .venv/Scripts/python -m pytest tests/test_graph.py -q
Chạy slow:  .venv/Scripts/python -m pytest tests/test_graph.py --runslow -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "graph"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from ProvisionParser import (  # noqa: E402
    parse_provision_effective_dates,
    parse_provision_list,
    parse_provision_ref,
    provision_unparsed,
)


# ---------------------------------------------------------------------------
# Parser provision (fast, pure - không cần DB)
# Định dạng lấy TRỰC TIẾP từ DocumentRelations.csv
# ---------------------------------------------------------------------------


def test_simple_article():
    refs = parse_provision_list("Điều 7")
    assert len(refs) == 1
    assert refs[0].ok and refs[0].label == "7"
    assert refs[0].kind == "article"
    assert refs[0].clauses == []


def test_article_with_clause_range():
    """'Điều 7 (khoản 1-7)' -> label 7, clause token 'khoản 1-7'."""
    refs = parse_provision_list("Điều 7 (khoản 1-7)")
    assert refs[0].ok and refs[0].label == "7"
    assert refs[0].clauses == ["khoản 1-7"]


def test_article_multiple_clauses():
    """'Điều 4 (khoản 5, 14, 16, bổ sung 35)' -> 4 token khoản."""
    refs = parse_provision_list(
        "Điều 4 (khoản 5, 14, 16, bổ sung 35)"
    )
    assert refs[0].ok and refs[0].label == "4"
    assert refs[0].clauses == [
        "khoản 5", "14", "16", "bổ sung 35"
    ]


def test_article_clause_with_point():
    """'Điều 4 (khoản 2 điểm e)' giữ nguyên token (không bịa)."""
    refs = parse_provision_list("Điều 4 (khoản 2 điểm e)")
    assert refs[0].ok and refs[0].label == "4"
    assert "khoản 2 điểm e" in refs[0].clauses


def test_multi_article_semicolon():
    """'Điều 5; Điều 9 (khoản 1, 2)' -> 2 ref, 2 điều khác nhau."""
    refs = parse_provision_list("Điều 5; Điều 9 (khoản 1, 2)")
    labels = [r.label for r in refs]
    assert labels == ["5", "9"]
    assert all(r.ok for r in refs)


def test_semicolon_inside_parenthesis():
    """REL_144 (thật): 'Điều 4 (bổ sung khoản 3a; sửa điểm đ khoản 4;
    sửa khoản 14); Điều 9 (khoản 1 điểm b)' - ';' trong ngoặc KHÔNG
    phải phân cách provision."""
    refs = parse_provision_list(
        "Điều 4 (bổ sung khoản 3a; sửa điểm đ khoản 4; "
        "sửa khoản 14); Điều 9 (khoản 1 điểm b)"
    )
    labels = [r.label for r in refs]
    assert labels == ["4", "9"]
    assert provision_unparsed(refs) == []


def test_clauses_semicolon_split():
    """Trong ngoặc, ';' cũng tách token khoản (theo _split_clauses)."""
    refs = parse_provision_list("Điều 4 (bổ sung khoản 3a; sửa khoản 14)")
    assert refs[0].clauses == ["bổ sung khoản 3a", "sửa khoản 14"]


def test_annex_with_note():
    """'Phụ lục I, Phụ lục II (thay thế)' -> 2 ref annex."""
    refs = parse_provision_list("Phụ lục I, Phụ lục II (thay thế)")
    assert len(refs) == 2
    assert [r.label for r in refs] == ["Phụ lục I", "Phụ lục II"]
    assert all(r.ok for r in refs)


def test_unparseable_keeps_raw():
    """Text lạ -> ok=False, giữ raw, không bịa số (CLAUDE.md §6.4)."""
    refs = parse_provision_ref("???")
    assert refs[0].ok is False
    assert refs[0].raw == "???"
    assert provision_unparsed(refs) == ["???"]


def test_provision_unparsed_only_bad():
    refs = parse_provision_list("Điều 7 (khoản 1-7); ???")
    unp = provision_unparsed(refs)
    assert unp == ["???"]  # chỉ phần hỏng, điều 7 parse được


# ---------------------------------------------------------------------------
# Parser provision_effective_dates (phương án A, fast)
# ---------------------------------------------------------------------------


def test_eff_dates_empty():
    r = parse_provision_effective_dates("")
    assert r["article_dates"] == {}
    assert r["unparsed"] == []
    assert parse_provision_effective_dates(None)["unparsed"] == []


def test_eff_dates_whole_article():
    r = parse_provision_effective_dates("17:2026-01-01")
    assert r["article_dates"] == {"17": "2026-01-01"}


def test_eff_dates_clause_of_article():
    r = parse_provision_effective_dates("5.25:2026-01-01")
    assert r["clause_dates"] == {"5": {"25": "2026-01-01"}}


def test_eff_dates_mixed_like_registry():
    """Giá trị thật của 143/2025: cả điều, phụ lục, khoản riêng."""
    r = parse_provision_effective_dates(
        "7:2026-07-01; Phụ lục IV:2026-07-01; 50.3:2026-01-01"
    )
    assert r["article_dates"] == {"7": "2026-07-01"}
    assert r["annex_dates"] == {"Phụ lục IV": "2026-07-01"}
    assert r["clause_dates"] == {"50": {"3": "2026-01-01"}}
    assert r["unparsed"] == []


def test_eff_dates_bad_format_kept():
    """Sai format -> giữ raw vào unparsed, không bịa (§6.4)."""
    r = parse_provision_effective_dates("17:2026-01-01; abc:2026-01-01; 5:xx")
    assert r["article_dates"] == {"17": "2026-01-01"}
    assert "abc:2026-01-01" in r["unparsed"]
    assert "5:xx" in r["unparsed"]


def test_builder_plan_documents_21():
    """Registry thật phải cho ra 21 Document (không bịa/sót)."""
    from Neo4jBuilder import (
        plan_documents,
        read_registry,
    )

    registry = read_registry(PROJECT_ROOT)
    docs = plan_documents(registry)
    assert len(docs) == 22
    assert all(d["so_hieu"] for d in docs)
    # so_hieu unique
    assert len({d["so_hieu"] for d in docs}) == 22


def test_builder_plan_edges_all_verified_types():
    """Mọi quan hệ trong CSV thuộc {AMENDS, REPEALS, GUIDES} và
    parse OK -> plan_edges trả edges, 0 unparsed."""
    from Neo4jBuilder import (
        plan_edges,
        plan_provisions,
        read_relations,
        read_registry,
    )

    registry = read_registry(PROJECT_ROOT)
    relations = read_relations(PROJECT_ROOT)
    assert len(relations) == 41

    provisions, unparsed = plan_provisions(relations, registry)
    assert unparsed == [], (
        f"Có provision không parse được: {unparsed}"
    )
    edges = plan_edges(relations, registry)
    assert len(edges) > 0
    types = {e["relation_type"] for e in edges}
    assert types <= {"AMENDS", "REPEALS", "GUIDES"}
    # Mỗi edge có unique_key 2 đầu (không bịa)
    prov_keys = {p["unique_key"] for p in provisions}
    for e in edges:
        assert e["from_key"] in prov_keys
        assert e["to_key"] in prov_keys


# ---------------------------------------------------------------------------
# Slow: cần Neo4j + build (chạy --runslow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slow_build_idempotent_and_queries():
    """Build 2 lần -> count không đổi; query đúng theo CSV;
    provision_status âm (Điều 11 chưa bị sửa -> active)."""
    from Neo4jBuilder import Neo4jBuilder, read_relations, read_registry
    from Neo4jClient import Neo4jClient

    client = Neo4jClient()
    assert client.ping(), "Neo4j chưa chạy - không test được"
    builder = Neo4jBuilder(client)

    registry = read_registry(PROJECT_ROOT)
    relations = read_relations(PROJECT_ROOT)

    r1 = builder.build(registry, relations)
    r2 = builder.build(registry, relations)
    assert r1 == r2, "Build không idempotent"

    stats = client.document_graph_stats()
    assert stats.get("Document") == r1["documents"] == 22
    assert stats.get("Provision") == r1["provisions"]

    # Điều 4 Luật TNDN - phân loại theo LOẠI quan hệ (amendments_of
    # trả cả 3 loại): SỬA bởi (AMENDS/REPEALS) = 127/2025 (bổ sung
    # khoản 6), 141/2025 (khoản 11a,11b), 09/2026 (khoản 14a);
    # HƯỚNG DẪN bởi (GUIDES) = 320/2025 (NĐ-CP hướng dẫn thi hành).
    amend = client.amendments_of("67/2025/QH15", "4")
    amended_by = {
        r["by_document"] for r in amend
        if r["relation_type"] in ("AMENDS", "REPEALS")
    }
    guided_by = {
        r["by_document"] for r in amend
        if r["relation_type"] == "GUIDES"
    }
    assert amended_by == {
        "127/2025/QH15", "141/2025/QH15", "09/2026/QH16"
    }
    assert "141/2025/QH15" in amended_by  # sửa khoản 11a,11b
    assert "320/2025/NĐ-CP" in guided_by  # 320 = hướng dẫn, không phải sửa

    # Điều 11 Luật TNDN: không có AMENDS/REPEALS (chỉ bị 320/2025
    # GUIDES tại khoản 2 - CSV REL_320_..._D1_K1) -> status "active".
    # active = không bị sửa/bãi bỏ; GUIDES không làm provision mất
    # hiệu lực. (CLAUDE.md §6.4: trả đúng dữ liệu, không suy diễn.)
    status = client.provision_status("67/2025/QH15", "11")
    assert status["status"] == "active"
    assert status["is_repealed"] is False
    assert status["is_amended"] is False

    # 181/2025/NĐ-CP bị 359/2025 REPEALS (khoản 3 Điều 37, 39)
    # -> REPEALS có (điều 37, 39), kiểm tra theo CSV
    rep = client.amendments_of("181/2025/NĐ-CP", "37")
    rep_types = {row["relation_type"] for row in rep}
    assert "REPEALS" in rep_types

    # --- Time-travel (phương án A: điều + map khoản, hiệu lực 3 lớp) ---
    # 48/2024/QH15: provision_effective_dates = "17:2026-01-01; 5.25:2026-01-01"
    # -> khoản 25 Điều 5 có hiệu lực RIÊNG 01/01/2026 (muộn hơn ngày
    #    chung 01/07/2025); cả Điều 5 vẫn theo ngày chung.
    g_clause_before = client.governing_provision(
        "48/2024/QH15", "5", clause="25", as_of_date="2025-09-01"
    )
    assert g_clause_before["status"] == "chưa_hiệu_lực"
    assert g_clause_before["provision_in_effect_at_T"] is False
    assert g_clause_before["effective_date_used"] == "2026-01-01"
    assert g_clause_before["effective_date_source"] == "provision"

    # Cả Điều 5 (không hỏi khoản) -> ngày chung 01/07/2025, đã hiệu lực.
    g_article_before = client.governing_provision(
        "48/2024/QH15", "5", as_of_date="2025-09-01"
    )
    assert g_article_before["provision_in_effect_at_T"] is True
    assert g_article_before["status"] in ("gốc", "amended")

    # Sau 01/01/2026 khoản 25 đã hiệu lực.
    g_clause_after = client.governing_provision(
        "48/2024/QH15", "5", clause="25", as_of_date="2026-02-01"
    )
    assert g_clause_after["provision_in_effect_at_T"] is True

    # 09/2026/QH16: provision_effective_dates = "1:...;2:...;3:2026-01-01"
    # -> Điều 3 hiệu lực riêng 01/01/2026 (hồi tố so với ngày ban hành
    #    24/04/2026). T trước -> chưa hiệu lực.
    g_retro = client.governing_provision(
        "09/2026/QH16", "3", as_of_date="2025-12-31"
    )
    assert g_retro["status"] == "chưa_hiệu_lực"
    assert g_retro["effective_date_used"] == "2026-01-01"

    client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
