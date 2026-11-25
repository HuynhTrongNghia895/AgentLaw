"""Build KG legal structure, provisions, relations, and temporal fields.

Quy trình (verify từng bước §14.1):
    assemble()    -> {docs, nodes, edges} trong bộ nhớ (chưa chạm DB)
    seed()        -> MERGE toàn bộ vào db neo4j (idempotent, chạy lại không phồng)
    check_temporal-> 4 case GĐ3 + REPEALS bằng Cypher (số phải khớp backup)
    smoke() / seed_all()

Deterministic: 0 LLM, 0 training.

Cạnh quan hệ: dùng GĐ3's plan_edges (nguồn sự thật đã verify, 264 cạnh
article-level) — KHÔNG tự parse 41 dòng CSV (rủi ro nở sai 1168 cạnh bịa,
đã bắt trong dry-run).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import (DATA_PROCESSED, EDGE_AMENDS, EDGE_GUIDES, EDGE_HAS_ANNEX,
                     EDGE_HAS_ARTICLE, EDGE_HAS_CHAPTER, EDGE_HAS_CLAUSE,
                     EDGE_HAS_PART, EDGE_HAS_POINT, EDGE_HAS_SECTION,
                     EDGE_REPEALS, LABEL_ANNEX, LABEL_ARTICLE, LABEL_CHAPTER,
                     LABEL_CLAUSE, LABEL_DOCUMENT, LABEL_PART, LABEL_POINT,
                     LABEL_SECTION, NEO4J_DATABASE, NEO4J_URI,
                     REGISTRY_CSV, RELATIONS_CSV)
from .loader import (KgDocument, KgNode, KgStructureNode, annex_key, load_registry,
                     load_relations, load_structure_tree, load_tree, lookup_provision_date,
                     parse_provision_effective_dates, _vb_key)

# GĐ3 parser (nguồn sự thật cho cạnh) — vẫn trên đĩa, import trực tiếp
G3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G3))
try:
    from Neo4jBuilder import plan_edges  # noqa: E402
    from ProvisionParser import (  # noqa: E402
        parse_provision_list,
        provision_unparsed,
    )
except ImportError:
    plan_edges = None  # type: ignore[assignment]
    provision_unparsed = None  # type: ignore[assignment]
    parse_provision_list = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
@dataclass
class AssembledGraph:
    docs: dict[str, KgDocument] = field(default_factory=dict)
    nodes: dict[str, KgNode] = field(default_factory=dict)
    structures: dict[str, KgStructureNode] = field(default_factory=dict)
    # (parent_id, child_id, edge_type) — mỗi tầng con 1 loại cạnh riêng.
    containment_edges: set[tuple[str, str, str]] = field(default_factory=set)
    edges: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for n in self.nodes.values():
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        for n in self.structures.values():
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        return {"docs": len(self.docs), "nodes": len(self.nodes) + len(self.structures),
                "provision_nodes": len(self.nodes),
                "structure_nodes": len(self.structures),
                "containment_edges": len(self.containment_edges),
                "by_kind": by_kind, "relation_edges": len(self.edges),
                "missing": len(self.missing)}


def assemble(smoke_docs: list[str] | None = None) -> AssembledGraph:
    docs0, tree = load_tree(DATA_PROCESSED)
    structures0, article_parents = load_structure_tree(DATA_PROCESSED)
    registry = load_registry()
    g = AssembledGraph()
    # P7: index thông tin sửa đổi từ JSONL để nâng cạnh xuống cấp
    # khoản/điểm (CSV là nguồn chính, JSONL chỉ bổ sung mức chi tiết).
    amend_index = _load_amendment_index(DATA_PROCESSED)

    so_hieu_list = smoke_docs if smoke_docs is not None else sorted(docs0.keys())

    for sh in so_hieu_list:
        doc = docs0.get(sh)
        if doc is None:
            continue
        reg = registry.get(sh, {})
        ds = (reg.get("dataset") or "").split(";")
        if ds and ds != [""]:
            doc.nhom_pl = set(ds)
        doc.ten_file = doc.ten_file or reg.get("ten_file", "")
        doc.ban_hanh_boi = reg.get("ban_hanh_boi", "")
        doc.ngay_ban_hanh = reg.get("ngay_ban_hanh", "")
        doc.effective_date = reg.get("effective_date", "")
        doc.effective_status = reg.get("effective_status", "")
        doc.superseded_by = reg.get("superseded_by", "")
        doc.provision_effective_dates = parse_provision_effective_dates(
            reg.get("provision_effective_dates", ""))
        doc.applicable_from = reg.get("applicable_from", "")
        g.docs[sh] = doc

        for structure in structures0.get(sh, {}).values():
            g.structures[structure.node_id] = structure
            if structure.parent_id:
                g.containment_edges.add(
                    (structure.parent_id, structure.node_id, _edge_for(structure.kind))
                )

        for n in tree.get(sh, []):
            pef = lookup_provision_date(doc.provision_effective_dates,
                                        n.dieu, n.khoan, n.phu_luc)
            n.provision_effective_date = pef or ""
            if doc.applicable_from and n.kind == "clause" and "áp dụng từ" in (n.text or ""):
                n.applicable_from = doc.applicable_from
            g.nodes[n.node_id] = n
            if n.kind == "article":
                parent_id = article_parents.get(n.node_id, f"DOC_{sh}")
            elif n.kind in ("clause", "point"):
                parent_id = n.parent_id
            else:  # Annex
                parent_id = f"DOC_{sh}"
            if parent_id:
                g.containment_edges.add((parent_id, n.node_id, _edge_for(n.kind)))

    if smoke_docs is not None:
        # smoke: chỉ lấy quan hệ mà CẢ 2 đầu đều trong tập con
        keep = set(smoke_docs)
        relations = [r for r in load_relations()
                     if r["from_so_hieu"] in keep and r["to_so_hieu"] in keep]
    else:
        relations = load_relations()

    _add_relation_edges(g, relations, amend_index)

    # REPEALS: gán invalid_at xuống node đích (mốc 4 phần tử)
    for e in g.edges:
        if e["relation_type"] == EDGE_REPEALS:
            tgt = g.nodes.get(e["to_id"])
            if tgt is not None:
                tgt.invalid_at = e["relation_effective_date"]
    return g


def _add_relation_edges(
    g: AssembledGraph,
    relations: list[dict],
    amend_index: dict[tuple[str, str], list[dict]] | None = None,
) -> None:
    """Dùng GĐ3 plan_edges (264 cạnh, đã verify) + TRỎ ĐÚNG TẦNG (P5).

    Map key 'so_hieu|label' -> node_id của mình:
      '59/2020/QH14|49'          -> '59_2020_QH14_D49'
      '143/2025/QH15|Phụ lục IV' -> '143_2025_QH15_A4'

    P5: `to_provision` trong CSV có khi ghi rõ tầng ("Điều 60 (khoản 2 điểm
    e)"). Cạnh quan hệ PHẢI trỏ đúng tầng đó (điểm -> khoản -> điều) để
    agent "móc" đúng chỗ cần sửa, thay vì trỏ cả Điều. Node đích không
    tồn tại -> FALLBACK lên tầng cha gần nhất (không trỏ hư không, không
    bịa node).

    P7: NÂNG ĐẦU NGUỒN cạnh xuống cấp khoản/điểm. CSV (`from_provision`)
    chỉ ghi tới cấp ĐIỀU ("Điều 1 (khoản 1-28)"), nên cạnh là
    Article->Article, mất thông tin "khoản nào thực sự sửa". Thông tin
    đó có sẵn trong JSONL (`amendment_operations` của record cấp khoản).
    CSV quyết định CÓ quan hệ + phạm vi; JSONL chỉ quyết định MỨC CHI
    TIẾT. Không xác nhận được -> GIỮ cấp điều (không mất quan hệ).

    Mốc 4 (relation_effective_date) = GĐ3 edge['effective_date'].
    """
    if plan_edges is None:
        raise RuntimeError("Không import được GĐ3 plan_edges (đã xóa? xem "
                           "src/graph/Neo4jBuilder.py còn không)")
    registry_rows = [registry_row for registry_row in load_registry_rows()]
    g3_edges = plan_edges(relations, registry_rows)

    # Chuẩn hoá clause token theo tầng, dùng để trỏ đúng node đích.
    # key: (relation_id, to_so_hieu, to_label, clauses_text) -> target node
    target_index = _build_target_index(relations)

    for e in g3_edges:
        from_id = _g3_key_to_node_id(e["from_key"])
        to_id = _g3_key_to_node_id(e["to_key"])
        if from_id is None or to_id is None:
            g.missing.append(f"{e['relation_id']}: key không map được "
                            f"({e['from_key']} / {e['to_key']})")
            continue

        # P5: thử trỏ đúng tầng (khoản/điểm) trước, fallback về Điều.
        to_so, to_label = (e["to_key"].split("|", 1) + [""])[:2]
        resolved = _resolve_target_node(
            g, target_index, e["relation_id"], to_so, to_label,
            e.get("clauses") or "",
        )
        if resolved is not None:
            to_id = resolved
        elif to_id not in g.nodes:
            g.missing.append(f"{e['relation_id']}: đích thiếu {to_id}")
            continue

        # P7: nâng đầu NGUỒN xuống cấp khoản/điểm nếu JSONL xác nhận.
        from_id, from_detail = _resolve_source_node(
            g, amend_index, from_id, to_so, to_label,
            e.get("clauses") or "", e["relation_type"],
        )

        g.edges.append({
            "relation_id": e["relation_id"],
            "relation_type": e["relation_type"],
            "from_id": from_id, "to_id": to_id,
            "relation_effective_date": e.get("effective_date") or "",
            "clauses": e.get("clauses") or "",
            "source_reference": e.get("source_reference", ""),
            "effective_date_source": e.get("effective_date_source", ""),
            # P7: thông tin chi tiết từ JSONL (bổ sung, không thay CSV).
            "detail_level": from_detail.get("level", "article"),
            "detail_operation_id": from_detail.get("operation_id", ""),
            "detail_source_clause": from_detail.get("source_clause", ""),
        })


def _resolve_source_node(
    g: AssembledGraph,
    amend_index: dict[tuple[str, str], list[dict]] | None,
    from_id: str,
    to_so: str,
    to_label: str,
    clauses_text: str,
    relation_type: str,
) -> tuple[str, dict[str, Any]]:
    """Nâng node NGUỒN của cạnh xuống cấp khoản/điểm nếu JSONL xác nhận.

    from_id hiện là node cấp ĐIỀU (vd '76_2025_QH15_D1'). Ta tra JSONL
    của văn bản nguồn, đúng Điều đó, tìm operation khớp với ĐÍCH của
    quan hệ (văn bản + điều + khoản) -> node nguồn đúng tầng.

    Trả (node_id, detail). Nếu không xác nhận được -> giữ nguyên node
    cấp điều, detail.level='article' (KHÔNG mất quan hệ).
    """
    empty = {"level": "article", "operation_id": "", "source_clause": ""}
    if not amend_index or "_D" not in from_id:
        return from_id, empty
    try:
        vk, rest = from_id.split("_D", 1)
    except ValueError:
        return from_id, empty
    # rest = "1" hoặc "1_K24"
    parts = rest.split("_K", 1)
    dieu = parts[0]
    so_hieu = _so_by_vb_key().get(vk)
    if not so_hieu:
        return from_id, empty

    ops = amend_index.get((so_hieu, dieu), [])
    if not ops:
        return from_id, empty

    target_clause = _first_clause_token(clauses_text)
    best = None
    for op in ops:
        if op["target_document_number"] and to_so and (
            op["target_document_number"] != to_so
        ):
            continue
        if op["target_article"] != to_label:
            continue
        if target_clause and op["target_clause"]:
            if not _clause_match(target_clause, op["target_clause"]):
                continue
        best = op
        break
    if best is None:
        return from_id, empty

    # Dựng node nguồn đúng tầng: khoản -> điểm.
    art_id = f"{vk}_D{dieu}"
    src_khoan = best.get("source_clause") or ""
    src_diem = best.get("source_point") or ""
    candidates: list[str] = []
    if src_khoan and src_diem:
        candidates.append(f"{art_id}_K{src_khoan}_P{src_diem}")
    if src_khoan:
        candidates.append(f"{art_id}_K{src_khoan}")
    candidates.append(art_id)
    for nid in candidates:
        if nid in g.nodes:
            level = (
                "point" if "_P" in nid
                else ("clause" if "_K" in nid else "article")
            )
            return nid, {
                "level": level,
                "operation_id": best.get("operation_id", ""),
                "source_clause": src_khoan,
            }
    return from_id, empty


def _vb_key_back(vk: str) -> str | None:
    """'76_2025_QH15' -> '76/2025/QH15' (đảo của _vb_key).

    Ưu tiên tra registry (chuẩn xác tuyệt đối); fallback ghép chuỗi.
    """
    hit = _so_by_vb_key().get(vk)
    if hit:
        return hit
    parts = vk.split("_")
    if len(parts) < 3:
        return None
    num = parts[0]
    year = parts[1]
    tail = "_".join(parts[2:])
    return f"{num}/{year}/{tail}"


def _first_clause_token(clauses_text: str) -> str:
    """'khoản 1' -> '1'; 'khoản 2 điểm e' -> '2'."""
    text = (clauses_text or "").split(";")[0].strip()
    m = re.search(r"khoản\s+(\d+[a-z]?)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"^(\d+[a-z]?)$", text, re.IGNORECASE)
    return m.group(1) if m else ""


def _clause_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _load_amendment_index(
    processed_dir: Path,
) -> dict[tuple[str, str], list[dict]]:
    """Index operation sửa đổi từ JSONL: (so_hieu_nguon, dieu_nguon).

    Mỗi record cấp khoản/điểm của văn bản sửa có
    `amendment_operations` = [{action, target_document_number,
    target_article, target_clause, target_point, ...}] — tức chính là
    "khoản nào (trong văn bản nguồn) sửa điều/khoản/điểm nào (đích)".
    """
    import json

    index: dict[tuple[str, str], list[dict]] = {}
    if not processed_dir.exists():
        return index
    for f in sorted(processed_dir.glob("*/*.jsonl")):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict):
                    continue
                ops = r.get("amendment_operations")
                if not isinstance(ops, list) or not ops:
                    continue
                so = str(r.get("document_number") or "").strip()
                dieu = str(r.get("article_number") or "").strip()
                if not so or not dieu:
                    continue
                khoan = str(r.get("clause_number") or "").strip()
                diem = str(r.get("clause_points") or "").strip()
                bucket = index.setdefault((so, dieu), [])
                for op in ops:
                    if not isinstance(op, dict):
                        continue
                    t_art = str(op.get("target_article") or "").strip()
                    if not t_art:
                        continue
                    bucket.append({
                        "action": str(op.get("action") or "").strip(),
                        "target_document_number": str(
                            op.get("target_document_number") or ""
                        ).strip(),
                        "target_article": t_art,
                        "target_clause": str(
                            op.get("target_clause") or ""
                        ).strip(),
                        "target_point": str(
                            op.get("target_point") or ""
                        ).strip(),
                        # TẦNG NGUỒN: khoản (và điểm) của văn bản sửa.
                        "source_clause": khoan,
                        "source_point": "",
                        "operation_id": str(
                            op.get("operation_id") or ""
                        ).strip(),
                    })
    return index


def _g3_key_to_node_id(key: str) -> str | None:
    """'<so_hieu>|<label>' -> node_id. Label: số điều (49) hoặc 'Phụ lục IV'."""
    if "|" not in key:
        return None
    sh, label = key.split("|", 1)
    label = label.strip()
    if label.isdigit():
        return f"{_vb_key(sh)}_D{label}"
    if label.lower().startswith("phụ lục"):
        return annex_key(sh, label)
    return None


# ---------------------------------------------------------------------------
# P5: trỏ cạnh quan hệ đúng TẦNG (điểm -> khoản -> điều)
# ---------------------------------------------------------------------------
# Token clause trong CSV: 'khoản 5', 'khoản 2 điểm e', 'bổ sung 35',
# 'sửa điểm đ khoản 4', '14', 'thay thế' (ghi chú phụ lục).
_CLAUSE_TOKEN_RE = re.compile(
    r"^(?:sửa\s+|bổ\s+sung\s+)?khoản\s+(\d+[a-z]?)$", re.IGNORECASE
)
# "bổ sung 35" (không chữ khoản) -> bổ sung khoản 35
_BOSUNG_BARE_RE = re.compile(r"^(?:bổ\s+sung|sửa)\s+(\d+[a-z]?)$", re.IGNORECASE)
_KHOAN_DIEM_RE = re.compile(
    r"^(?:sửa\s+|bổ\s+sung\s+)?khoản\s+(\d+[a-z]?)\s+điểm\s+([a-zđ])$",
    re.IGNORECASE,
)
_DIEM_KHOAN_RE = re.compile(
    r"^(?:sửa\s+|bổ\s+sung\s+)?điểm\s+([a-zđ])\s+khoản\s+(\d+[a-z]?)$",
    re.IGNORECASE,
)
_DIEM_TOKEN_RE = re.compile(
    r"^(?:sửa\s+|bổ\s+sung\s+)?điểm\s+([a-zđ])$", re.IGNORECASE
)
_BARE_NUM_RE = re.compile(r"^(\d+[a-z]?)$", re.IGNORECASE)


def _clause_targets(article_id: str, clause_text: str) -> list[str]:
    """Sinh danh sách node id ứng viên theo tầng, từ sâu -> nông.

    'khoản 2 điểm e' -> ['D60_K2_Pe', 'D60_K2', 'D60']
    'khoản 5'       -> ['D60_K5', 'D60']
    'bổ sung 35'    -> ['D60_K35', 'D60']
    'điểm đ khoản 4'-> ['D60_K4_Pđ', 'D60_K4', 'D60']

    Thứ tự = thứ tự fallback: dùng node đầu tiên TỒN TẠI.
    """
    text = (clause_text or "").strip()
    if not text:
        return [article_id]

    out: list[str] = []

    m = _KHOAN_DIEM_RE.match(text)
    if m:
        k, p = m.group(1), m.group(2).lower()
        out += [f"{article_id}_K{k}_P{p}", f"{article_id}_K{k}"]
        return out + [article_id]

    m = _DIEM_KHOAN_RE.match(text)
    if m:
        p, k = m.group(1).lower(), m.group(2)
        out += [f"{article_id}_K{k}_P{p}", f"{article_id}_K{k}"]
        return out + [article_id]

    m = _CLAUSE_TOKEN_RE.match(text)
    if m:
        k = m.group(1)
        return [f"{article_id}_K{k}", article_id]

    m = _BOSUNG_BARE_RE.match(text)
    if m:
        k = m.group(1)
        return [f"{article_id}_K{k}", article_id]

    m = _BARE_NUM_RE.match(text)
    if m:
        # '14' -> khoản 14 của điều đó
        return [f"{article_id}_K{m.group(1)}", article_id]

    # 'thay thế' (ghi chú phụ lục) hoặc token lạ -> không trỏ sâu
    return [article_id]


def _build_target_index(
    relations: list[dict],
) -> dict[tuple[str, str, str], list[str]]:
    """Index các node đích ứng viên theo (relation_id, to_so_hieu, label).

    Mỗi entry = danh sách node id theo thứ tự ưu tiên (sâu -> nông).
    """
    index: dict[tuple[str, str, str], list[str]] = {}
    for rel in relations:
        rid = (rel.get("relation_id") or "").strip()
        to_so = (rel.get("to_so_hieu") or "").strip()
        for ref in parse_provision_list(rel.get("to_provision")):
            if not ref.ok:
                continue
            if ref.kind == "annex":
                node_id = annex_key(to_so, ref.label)
                index[(rid, to_so, ref.label)] = [node_id]
                continue
            article_id = f"{_vb_key(to_so)}_D{ref.label}"
            cands: list[str] = []
            for c in ref.clauses:
                for nid in _clause_targets(article_id, c):
                    if nid not in cands:
                        cands.append(nid)
            if article_id not in cands:
                cands.append(article_id)
            index[(rid, to_so, ref.label)] = cands
    return index


def _resolve_target_node(
    g: AssembledGraph,
    target_index: dict[tuple[str, str, str], list[str]],
    relation_id: str,
    to_so: str,
    to_label: str,
    clauses_text: str,
) -> str | None:
    """Chọn node đích theo tầng cụ thể nhất TỒN TẠI trong KG.

    Trả None nếu không có ứng viên nào tồn tại (caller fallback về Điều
    hoặc bỏ + report).
    """
    cands = target_index.get((relation_id, to_so, to_label))
    if not cands:
        # Không có trong index (vd relation không có trong CSV gốc) ->
        # thử suy ra từ chuỗi clauses gộp.
        article_id = f"{_vb_key(to_so)}_D{to_label}"
        cands = []
        for c in (clauses_text or "").split(";"):
            c = c.strip()
            if not c:
                continue
            for nid in _clause_targets(article_id, c):
                if nid not in cands:
                    cands.append(nid)
        if article_id not in cands:
            cands.append(article_id)

    for nid in cands:
        if nid in g.nodes:
            return nid
    return None


def load_registry_rows() -> list[dict]:
    """GĐ3 plan_edges cần DANH SÁCH registry (không phải dict) — đọc lại."""
    import csv
    with open(REGISTRY_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# P7: bảng tra ngược vb_key -> so_hieu (từ registry, chuẩn xác tuyệt đối).
_SO_BY_VB_KEY: dict[str, str] = {}


def _so_by_vb_key() -> dict[str, str]:
    """Lấy bảng _SO_BY_VB_KEY (lazy)."""
    global _SO_BY_VB_KEY
    if not _SO_BY_VB_KEY:
        for row in load_registry_rows():
            sh = (row.get("so_hieu") or "").strip()
            if sh:
                _SO_BY_VB_KEY[_vb_key(sh)] = sh
    return _SO_BY_VB_KEY


# ---------------------------------------------------------------------------
# Seed Cypher (idempotent)
# ---------------------------------------------------------------------------
_LABEL = {"article": LABEL_ARTICLE, "clause": LABEL_CLAUSE,
          "point": LABEL_POINT, "annex": LABEL_ANNEX}
_STRUCT_LABEL = {"part": LABEL_PART, "chapter": LABEL_CHAPTER,
                 "section": LABEL_SECTION}
_STRUCT_NAME = {"part": "Phần", "chapter": "Chương", "section": "Mục"}

# Loại cạnh cha -> con, theo TẦNG của node con (không dùng CONTAINS chung).
_EDGE_FOR_KIND = {
    "part": EDGE_HAS_PART,
    "chapter": EDGE_HAS_CHAPTER,
    "section": EDGE_HAS_SECTION,
    "article": EDGE_HAS_ARTICLE,
    "clause": EDGE_HAS_CLAUSE,
    "point": EDGE_HAS_POINT,
    "annex": EDGE_HAS_ANNEX,
}


def _edge_for(kind: str) -> str:
    return _EDGE_FOR_KIND.get(kind, EDGE_HAS_ARTICLE)


def _driver():
    import os
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", NEO4J_URI),
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "")),
        database=NEO4J_DATABASE,
    )


def seed(g: AssembledGraph, wipe: bool = True) -> dict:
    """MERGE cây + cạnh. Xóa KG cũ trước (cờ kind4tier) để chạy lại không phồng."""
    d = _driver()
    stats = {"docs": 0, "nodes": 0, "structure_nodes": 0,
             "parent_edges": 0, "containment_edges": 0, "relation_edges": 0}
    try:
        with d.session() as s:
            if wipe:
                s.run("MATCH (n {kind4tier: true}) DETACH DELETE n").consume()
            for _lab in (LABEL_DOCUMENT, LABEL_ARTICLE, LABEL_CLAUSE,
                         LABEL_POINT, LABEL_ANNEX, LABEL_PART,
                         LABEL_CHAPTER, LABEL_SECTION):
                s.run(f"CREATE CONSTRAINT node_id4tier_{_lab} IF NOT EXISTS "
                      f"FOR (n:{_lab}) REQUIRE n.id IS UNIQUE").consume()

            for sh, doc in g.docs.items():
                s.run("""
                    MERGE (d:Document {id: $id})
                    SET d.kind4tier = true,
                        d.so_hieu = $so_hieu,
                        d.nhom_pl = $nhom,
                        d.ten_file = $ten_file,
                        d.ban_hanh_boi = $bh,
                        d.ngay_ban_hanh = $nbh,
                        d.effective_date = $eff,
                        d.effective_status = $st,
                        d.superseded_by = $sup,
                        d.applicable_from = $app,
                        d.uuid = $uuid,
                        d.name = $name
                """, id=f"DOC_{sh}", so_hieu=sh, nhom=sorted(doc.nhom_pl),
                     ten_file=doc.ten_file, bh=doc.ban_hanh_boi,
                     nbh=doc.ngay_ban_hanh, eff=doc.effective_date,
                     st=doc.effective_status, sup=doc.superseded_by or None,
                     app=doc.applicable_from or None,
                     uuid=_uuid(f"DOC_{sh}"),
                     name=sh)  # Graphiti: tìm kiếm theo số hiệu văn bản
                stats["docs"] += 1

            for n in g.structures.values():
                label = _STRUCT_LABEL[n.kind]
                s.run(f"""
                    MERGE (n:{label} {{id: $id}})
                    SET n.kind4tier = true,
                        n.so_hieu = $sh,
                        n.number = $number,
                        n.title = $title,
                        n.uuid = $uuid,
                        n.name = $name
                """, id=n.node_id, sh=n.so_hieu,
                     number=n.number or None, title=n.title or None,
                     uuid=_uuid(n.node_id), name=_structure_name(n))
                stats["nodes"] += 1
                stats["structure_nodes"] += 1

            for kind in ("article", "annex", "clause", "point"):
                has_title = kind in ("article", "annex")  # khoản/điểm KHÔNG có tiêu đề
                for n in [x for x in g.nodes.values() if x.kind == kind]:
                    eff = g.docs[n.so_hieu].effective_date
                    s.run(f"""
                        MERGE (n:{_LABEL[kind]} {{id: $id}})
                        SET n.kind4tier = true,
                            n.so_hieu = $sh,
                            n.dieu = $dieu,
                            n.khoan = $khoan,
                            n.diem = $diem,
                            n.phu_luc = $plc,
                            n.text = $text,
                            n.title = $title,
                            n.effective_date = $eff,
                            n.provision_effective_date = $pef,
                            n.applicable_from = $app,
                            n.invalid_at = $inv,
                            n.uuid = $uuid,
                            n.name = $name
                    """, id=n.node_id, sh=n.so_hieu,
                         dieu=(n.dieu or None), khoan=(n.khoan or None),
                         diem=(n.diem or None), plc=(n.phu_luc or None),
                         text=n.text,
                         title=(n.title if has_title else None),
                         eff=eff,
                         pef=n.provision_effective_date or None,
                         app=n.applicable_from or None,
                         inv=n.invalid_at or None,
                         uuid=_uuid(n.node_id),
                         name=_node_name(n))
                    stats["nodes"] += 1

            # Containment edges: cha -> con, mỗi tầng 1 loại cạnh riêng.
            for parent_id, child_id, edge_type in sorted(g.containment_edges):
                s.run(f"""
                    MATCH (p {{id: $pid}}), (c {{id: $cid}})
                    MERGE (p)-[r:{edge_type}]->(c)
                    SET r.kind4tier = true
                """, pid=parent_id, cid=child_id)
                stats["containment_edges"] += 1
                stats["parent_edges"] += 1

            for e in g.edges:
                s.run(f"""
                    MATCH (a {{id: $aid}}), (b {{id: $bid}})
                    MERGE (a)-[r:{e['relation_type']} {{edge_key: $key}}]->(b)
                    SET r.kind4tier = true,
                        r.relation_id = $rid,
                        r.relation_effective_date = $red,
                        r.clauses = $cl,
                        r.source_reference = $ref,
                        r.effective_date_source = $src,
                        r.detail_level = $dlvl,
                        r.detail_operation_id = $dop,
                        r.detail_source_clause = $dkh
                """, aid=e["from_id"], bid=e["to_id"],
                     key=f"{e['from_id']}->{e['to_id']}:{e['relation_type']}",
                     rid=e["relation_id"], red=e["relation_effective_date"],
                     cl=e.get("clauses", ""), ref=e.get("source_reference", ""),
                     src=e.get("effective_date_source", ""),
                     dlvl=e.get("detail_level", "article"),
                     dop=e.get("detail_operation_id", ""),
                     dkh=e.get("detail_source_clause", ""))
                stats["relation_edges"] += 1
    finally:
        d.close()
    return stats


def _uuid(node_id: str) -> str:
    import hashlib
    return hashlib.md5(node_id.encode("utf-8")).hexdigest()


def _structure_name(n: KgStructureNode) -> str:
    label = _STRUCT_NAME[n.kind]
    identifier = f" {n.number}" if n.number else ""
    title = f". {n.title}" if n.title else ""
    return f"{label}{identifier}{title} ({n.so_hieu})"


def _node_name(n) -> str:
    """'Chìa khóa tìm kiếm' cho Graphiti, đọc như cách người hỏi gọi:
       Điều: 'Điều 49. Quyền của thành viên... (59/2020/QH14)' (dòng in đậm + số hiệu)
       Khoản: 'Điều 49 Khoản 1 (59/2020/QH14)'
       Điểm: 'Điều 49 Khoản 1 Điểm a (59/2020/QH14)'
       Phụ lục: 'Phụ lục I (144/2026/NĐ-CP)' """
    if n.kind == "article":
        # text điều có khoản = dòng in đậm; điều đứng riêng = nội dung đầy đủ
        # cắt ngắn khi quá dài (max 80 ký tự, tránh cắt giữa câu)
        core = n.text.strip()
        if len(core) > 80:
            core = core[:80].rsplit(" ", 1)[0] + "..."
        return f"{core} ({n.so_hieu})"
    if n.kind == "clause":
        return f"Điều {n.dieu} Khoản {n.khoan} ({n.so_hieu})"
    if n.kind == "point":
        return f"Điều {n.dieu} Khoản {n.khoan} Điểm {n.diem} ({n.so_hieu})"
    return f"Phụ lục {n.phu_luc} ({n.so_hieu})"


# ---------------------------------------------------------------------------
# Kiểm temporal: 4 case GĐ3 + REPEALS
# ---------------------------------------------------------------------------
def check_temporal() -> dict:
    """4 case GĐ3 (ảnh time-travel) + REPEALS 149/2025. Số phải khớp backup."""
    d = _driver()
    out: dict[str, dict] = {}
    try:
        with d.session() as s:
            def active(node_id: str, asof: str) -> bool:
                r = s.run("""
                    MATCH (n {id: $id})
                    WITH n, coalesce(n.provision_effective_date, n.effective_date) AS eff
                    WHERE eff <= $asof
                      AND (n.invalid_at IS NULL OR n.invalid_at > $asof)
                    RETURN n.id AS found
                """, id=node_id, asof=asof).single()
                return bool(r and r["found"])

            cases = [
                ("09/2026 D3 @31/12/2025", "09_2026_QH16_D3", "2025-12-31", False),
                ("09/2026 D3 @01/02/2026", "09_2026_QH16_D3", "2026-02-01", True),
                ("48/2024 D5K25 @01/09/2025", "48_2024_QH15_D5_K25", "2025-09-01", False),
                ("48/2024 D5 @01/09/2025", "48_2024_QH15_D5", "2025-09-01", True),
            ]
            for label, nid, asof, exp in cases:
                got = active(nid, asof)
                out[label] = {"got": got, "expected": exp, "pass": got == exp}

            # Cạnh REPEALS nằm ở tầng Điều (149/2025 D1 -> 48/2024 D12, D15),
            # chi tiết "khoản 3 / khoản 9 điểm c" nằm trong property `clauses`.
            r = s.run("""
                MATCH (a {id: $a})-[r:REPEALS]->(b)
                RETURN b.id AS bid, r.relation_effective_date AS red, r.clauses AS cl
            """, a="149_2025_QH15_D1").data()
            out["REPEALS D1(149/2025)"] = {
                "repealed": [x["bid"] for x in r],
                "clauses": [x["cl"] for x in r],
                "edge_eff": [x["red"] for x in r]}
    finally:
        d.close()
    return out


# ---------------------------------------------------------------------------
SMOKE_DOCS = ["09/2026/QH16", "149/2025/QH15", "48/2024/QH15"]


def smoke() -> dict:
    g = assemble(SMOKE_DOCS)
    st = seed(g, wipe=True)
    return {"stats": g.stats(), "seed": st, "temporal": check_temporal(),
            "missing": g.missing}


def seed_all() -> dict:
    g = assemble(None)
    st = seed(g, wipe=True)
    return {"stats": g.stats(), "seed": st, "temporal": check_temporal(),
            "missing": g.missing}
