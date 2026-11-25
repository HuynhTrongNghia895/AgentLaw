"""Loader provision nodes and optional Part/Chapter/Section metadata.

Nguồn (chỉ-đọc, đã verify GĐ1–GĐ3):
- data/chunked/<dataset>/<vb>.jsonl : text + cấu trúc (record_type: article/clause/annex)
- data/metadata/DocumentRegistry.csv  : 22 vb, thuộc tính Document + ngày hiệu lực
- data/metadata/DocumentRelations.csv : 41 quan hệ verified (cạnh do build.py mở
  rộng qua GĐ3 ProvisionParser/plan_edges — nguồn sự thật, KHÔNG parse lại)

Quy tắc đã verify trên đĩa (30/10–31/10):
- Sub-chunk (is_chunk=true, 320 records) KHÔNG vào graph: gộp text C1+C2+...
  theo chunk_index về 1 node cha (parent_id). content_blocks lặp y nguyên trên
  mọi sub-chunk -> lấy từ sub-chunk đầu (gối=0, verify 320/2025 D3K3).
- 09/2026 xuất hiện 2 file (thue_gtgt, thue_tndn): dedup theo (so_hieu, kind, key)
  -> 1 node, nhom_pl gom 2 dataset.
- ĐIỀU CÒN THỤ ĐỘNG: ~500 điều chỉ hiện trong parent_id của khoản (record khoản
  D5_K2 -> Điều 5). Node Article phải được sinh cho MỌI (vb, điều) xuất hiện ở
  bài (trên 566 điều khác biệt trên đĩa, đã kiểm tra). Text của Article "ẩn" =
  dòng tiêu đề "Điều X. ..." lấy từ record khoản đầu tiên (không bịa nội dung).
- Điểm a) b) c): parse từ content_blocks của KHOÁN, chỉ khi block đầu khớp
  '^([a-z])\\)' (khoản có điểm); số thứ tự = vị trí, chữ = ký hiệu (P_c).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_CHUNKED, DATA_PROCESSED, REGISTRY_CSV, RELATIONS_CSV

RE_POINT_START = re.compile(r"^([a-ỹđi])\)\s")  # "a) ", "b) ", ... "i) ", "đ) "
RE_ARTICLE_HDR = re.compile(r"^Điều\s+\d+\.[^\n]*\n")  # dòng tiêu đề "Điều N. ..."
RE_KHOAN_PREFIX = re.compile(r"^Khoản\s+\S+\.\s*")      # tiền tố "Khoản K. "

# ---------------------------------------------------------------------------
# Quy tắc lưu text (spec 30/10): tách "tiêu đề in đậm" khỏi "thân"
#   - Article: title = dòng in đậm (article_title, KHÔNG tiền tố "Điều N."),
#              text  = đoạn thân GIỮA tiêu đề và Khoản 1 (nếu có); điều có
#              khoản (implicit) KHÔNG có đoạn thân trong data -> text = "".
#   - Clause : title = "" (khoản không có tiêu đề in đậm),
#              text  = đoạn thân TRƯỚC điểm a) đầu tiên; khoản không có
#              điểm thì text = toàn bộ thân.
#   - Point  : text = nội dung điểm, BỎ ký hiệu "a) " (lấy từ content_blocks).
# Không bịa: data không có gì thì để rỗng.
# ---------------------------------------------------------------------------


def _strip_marker(block: str) -> str:
    """Bỏ ký hiệu điểm ("a) ", "đ) ") khỏi 1 block; giữ nguyên nội dung."""
    b = block.strip()
    m = RE_POINT_START.match(b)
    return b[m.end():].strip() if m else b


# ---------------------------------------------------------------------------
# Kiểu dữ liệu trong bộ nhớ (seed.py mới MERGE vào Neo4j)
# ---------------------------------------------------------------------------
@dataclass
class KgDocument:
    so_hieu: str
    nhom_pl: set[str] = field(default_factory=set)
    ten_file: str = ""
    ban_hanh_boi: str = ""
    ngay_ban_hanh: str = ""
    effective_date: str = ""
    effective_status: str = ""
    superseded_by: str = ""
    provision_effective_dates: dict[str, str] = field(default_factory=dict)
    # {"17": "...", "5.25": "...", "Phụ lục IV": "..."}
    applicable_from: str = ""
    ghi_chu: str = ""


@dataclass
class KgNode:
    """1 node Điều/Khoản/Điểm/Phụ lục.

    Quy tắc text (spec đã duyệt):
      - Article: title = tiêu đề in đậm; text = chữ TRƯỚC khoản (điều có
        khoản) hoặc toàn bộ thân (điều đứng riêng). Bỏ tiền tố "Điều N.".
      - Clause : title = ""; text = chữ từ sau "Khoản N." đến trước điểm a)
        (bao gồm trích dẫn nằm trong ngoặc kép); khoản không có điểm thì
        text = toàn bộ thân khoản.
      - Point  : title = ""; text = nội dung điểm, BỎ ký hiệu "a)".
    """

    node_id: str          # "59_2020_QH14_D49" / "..._K1" / "..._K1_Pc" / "..._A1"
    so_hieu: str
    kind: str             # "article" | "clause" | "point" | "annex"
    dieu: str = ""
    khoan: str = ""
    diem: str = ""        # CHỮ điểm ("a", "b", ...), rỗng nếu không phải điểm
    diem_index: int = 0   # vị trí trong khoản (1-based)
    phu_luc: str = ""     # "I".."VI" hoặc số
    title: str = ""
    text: str = ""
    implicit: bool = False  # True: node Điều sinh từ parent_id (không có record riêng)
    provision_effective_date: str = ""
    applicable_from: str = ""
    invalid_at: str = ""
    parent_id: str = ""   # article:"", clause->id điều, point->id khoản, annex->""


@dataclass
class KgStructureNode:
    """Phần/Chương/Mục. Chỉ giữ tiêu đề và định danh; không có text/order."""

    node_id: str
    so_hieu: str
    kind: str             # "part" | "chapter" | "section"
    number: str
    title: str
    parent_id: str


def _heading_parts(value: object) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    number = str(value.get("number") or "").strip()
    title = str(value.get("title") or "").strip()
    if not number and not title:
        return None
    return number, title


def _structure_token(number: str, title: str) -> str:
    raw = number or title
    token = re.sub(r"[^\w]+", "_", raw, flags=re.UNICODE).strip("_")
    return token or "UNNUMBERED"


def load_structure_tree(
    chunked_dir: Path,
) -> tuple[dict[str, dict[str, KgStructureNode]], dict[str, str]]:
    """Read optional part/chapter/section metadata without changing provision nodes.

    Returns structure nodes by document and the nearest structural parent id
    for each Article. Missing levels are skipped in the resulting hierarchy.
    """
    structures: dict[str, dict[str, KgStructureNode]] = {}
    article_parents: dict[str, str] = {}

    def add_structure(
        so_hieu: str,
        node_id: str,
        kind: str,
        number: str,
        title: str,
        parent_id: str,
    ) -> None:
        doc_structures = structures.setdefault(so_hieu, {})
        existing = doc_structures.get(node_id)
        if existing is None:
            doc_structures[node_id] = KgStructureNode(
                node_id=node_id, so_hieu=so_hieu, kind=kind,
                number=number, title=title, parent_id=parent_id,
            )
        elif title and (not existing.title or len(title) > len(existing.title)):
            existing.title = title

    for file_path in sorted(chunked_dir.glob("*/*.jsonl")):
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            so_hieu = str(record.get("document_number") or "").strip()
            article_number = str(record.get("article_number") or "").strip()
            if not so_hieu or not article_number:
                continue

            doc_id = f"DOC_{so_hieu}"
            vb_key = _vb_key(so_hieu)
            parent_id = doc_id

            part = _heading_parts(record.get("part"))
            if part:
                number, title = part
                part_id = f"{vb_key}_P{_structure_token(number, title)}"
                add_structure(so_hieu, part_id, "part", number, title, doc_id)
                parent_id = part_id

            chapter = _heading_parts(record.get("chapter"))
            if chapter:
                number, title = chapter
                prefix = f"{part_id}_" if part else f"{vb_key}_"
                chapter_id = f"{prefix}C{_structure_token(number, title)}"
                add_structure(so_hieu, chapter_id, "chapter", number, title, parent_id)
                parent_id = chapter_id

            section = _heading_parts(record.get("section"))
            if section:
                number, title = section
                section_prefix = parent_id if parent_id != doc_id else vb_key
                section_id = f"{section_prefix}_S{_structure_token(number, title)}"
                add_structure(so_hieu, section_id, "section", number, title, parent_id)
                parent_id = section_id

            article_id = f"{vb_key}_D{article_number}"
            article_parents[article_id] = parent_id

    return structures, article_parents


# ---------------------------------------------------------------------------
# provision_effective_dates (registry): "17:2026-01-01; 5.25:2026-01-01"
# ---------------------------------------------------------------------------
def parse_provision_effective_dates(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split(";"):
        part = part.strip()
        if part and ":" in part:
            key, val = part.rsplit(":", 1)
            out[key.strip()] = val.strip()
    return out


def lookup_provision_date(dates: dict[str, str], dieu: str, khoan: str,
                          phu_luc: str = "") -> str | None:
    """Theo độ cụ thể: 'dieu.khoan' -> 'dieu' -> 'Phụ lục X'."""
    if dieu and khoan:
        v = dates.get(f"{dieu}.{khoan}")
        if v:
            return v
    if dieu and dieu in dates:
        return dates[dieu]
    if phu_luc:
        for key, v in dates.items():
            if key.lower().startswith("phụ lục") and key.split()[-1].lower() == phu_luc.lower():
                return v
    return None


# ---------------------------------------------------------------------------
# JSONL -> cây
# ---------------------------------------------------------------------------
def _vb_key(so_hieu: str) -> str:
    """'59/2020/QH14' -> '59_2020_QH14' (khớp định danh JSONL; 'Phụ lục I' -> A1...)."""
    return so_hieu.replace("/", "_").replace("NĐ-CP", "NĐ_CP")


def _merge_group(group: list[dict]) -> dict:
    """Gộp records cùng parent_id (tương thích dữ liệu chunk cũ).

    Dữ liệu MỚI đã bỏ bước chunk: mỗi điều/khoản = 1 record, nên nhánh
    "is_chunk" chỉ còn để đọc được data cũ (không dùng nữa).
    """
    subs = [r for r in group if r.get("is_chunk")]
    if subs:
        by_idx = {r["chunk_index"]: r for r in subs}
        order = sorted(by_idx)
        base = by_idx[order[0]]
        text = "\n".join(by_idx[i]["text"] for i in order)
        blocks = list(base.get("content_blocks") or [])
    else:
        base = group[0]
        text = base.get("text") or ""
        blocks = list(base.get("content_blocks") or [])
    return {
        "text": text,
        "blocks": blocks,
        "title": (base.get("clause_title") or base.get("article_title")
                  or base.get("annex_title") or "").strip(),
        "article_title": (base.get("article_title") or "").strip(),
        "article_body": (base.get("article_body") or "").strip(),
        "clause_body": (base.get("clause_body") or "").strip(),
        "clause_points": list(base.get("clause_points") or []),
        "kind": base.get("record_type", ""),
        "doc_no": base.get("document_number", ""),
        "source": base.get("source_filename", ""),
    }


def load_tree(
    processed_dir: Path,
) -> tuple[dict[str, KgDocument], dict[str, list[KgNode]]]:
    """Trả {so_hieu: KgDocument} + {so_hieu: [KgNode...]} (4 tầng đầy đủ).

    Nguồn: data/processed/<dataset>/*.jsonl — output trực tiếp của parse
    DOCX (đã bỏ bước chunk nên không có sub-chunk, không cần gộp).
    Mỗi record có article_number/clause_number/annex_number rõ ràng.
    """
    docs: dict[str, KgDocument] = {}
    by_so_hieu: dict[str, list[KgNode]] = {}
    seen_a: set[str] = set()
    seen_k: set[tuple[str, str, str]] = set()

    for f in sorted(processed_dir.glob("*/*.jsonl")):
        dataset = f.parent.name
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            so_hieu = str(r.get("document_number") or "").strip()
            if not so_hieu:
                continue

            doc = docs.setdefault(so_hieu, KgDocument(so_hieu=so_hieu))
            doc.nhom_pl.add(dataset)
            doc.ten_file = doc.ten_file or str(
                r.get("source_filename") or "")
            nodes = by_so_hieu.setdefault(so_hieu, [])
            vk = _vb_key(so_hieu)

            kind = str(r.get("record_type") or "")
            dieu = str(r.get("article_number") or "").strip()
            khoan = str(r.get("clause_number") or "").strip()
            art_title = str(r.get("article_title") or "").strip()
            art_body = str(r.get("article_body") or "").strip()

            if kind == "article" and dieu:
                aid = f"{vk}_D{dieu}"
                if aid in seen_a:
                    continue
                seen_a.add(aid)
                # Điều ĐỨNG RIÊNG: text = toàn bộ thân điều (đã bỏ tiền tố
                # "Điều N." ở tầng parse).
                nodes.append(KgNode(
                    node_id=aid, so_hieu=so_hieu, kind="article", dieu=dieu,
                    title=art_title or f"Điều {dieu}",
                    text=str(r.get("text") or "").strip(),
                    implicit=False,
                ))
                continue

            if kind == "clause" and dieu and khoan:
                # Điều CÓ khoản: title = tiêu đề in đậm; text = chữ TRƯỚC
                # khoản (article_body). Không có thì rỗng — không bịa.
                aid = f"{vk}_D{dieu}"
                if aid not in seen_a:
                    seen_a.add(aid)
                    nodes.append(KgNode(
                        node_id=aid, so_hieu=so_hieu, kind="article", dieu=dieu,
                        title=art_title or f"Điều {dieu}",
                        text=art_body,
                        implicit=True,
                    ))

                kkey = (so_hieu, dieu, khoan)
                if kkey in seen_k:
                    continue
                seen_k.add(kkey)
                kid = f"{vk}_D{dieu}_K{khoan}"
                # text khoản = clause_body (phần dẫn trước điểm a) + trích
                # dẫn trong ngoặc kép). Fallback text cũ nếu thiếu field mới.
                ktext = str(r.get("clause_body") or "").strip() or str(
                    r.get("text") or "").strip()
                nodes.append(KgNode(
                    node_id=kid, so_hieu=so_hieu, kind="clause", dieu=dieu,
                    khoan=khoan, title="", text=ktext,
                    parent_id=aid,
                ))

                points = r.get("clause_points") or []
                if points:
                    for i, p in enumerate(points, start=1):
                        letter = str(p.get("marker") or i).strip()
                        ptext = str(p.get("text") or "").strip()
                        if not ptext:
                            continue
                        nodes.append(KgNode(
                            node_id=f"{kid}_P{letter}", so_hieu=so_hieu,
                            kind="point", dieu=dieu, khoan=khoan,
                            diem=letter, diem_index=i, text=ptext,
                            parent_id=kid,
                        ))
                else:
                    # Fallback data cũ: suy điểm từ content_blocks.
                    blocks = r.get("content_blocks") or []
                    if blocks and RE_POINT_START.match(
                        str(blocks[0]).strip()
                    ):
                        for i, b in enumerate(blocks, start=1):
                            bm = RE_POINT_START.match(str(b).strip())
                            letter = bm.group(1) if bm else str(i)
                            nodes.append(KgNode(
                                node_id=f"{kid}_P{letter}", so_hieu=so_hieu,
                                kind="point", dieu=dieu, khoan=khoan,
                                diem=letter, diem_index=i,
                                text=_strip_marker(str(b)), parent_id=kid,
                            ))
                continue

            if kind == "annex":
                plu = str(r.get("annex_number") or "").strip()
                if not plu:
                    continue
                akey = f"{so_hieu}|A{plu}"
                if akey in seen_k:
                    continue
                seen_k.add(akey)
                nodes.append(KgNode(
                    node_id=f"{vk}_A{plu}", so_hieu=so_hieu, kind="annex",
                    phu_luc=plu,
                    title=str(r.get("annex_title") or "").strip(),
                    text=str(r.get("text") or "").strip(),
                    parent_id="",
                ))
    return docs, by_so_hieu


# ---------------------------------------------------------------------------
# 2 CSV (nguồn sự thật, đã verified 100% ở GĐ1–GĐ3)
# ---------------------------------------------------------------------------
def load_registry() -> dict[str, dict]:
    with open(REGISTRY_CSV, encoding="utf-8") as f:
        return {r["so_hieu"].strip(): r for r in csv.DictReader(f)}


def load_relations() -> list[dict]:
    with open(RELATIONS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("verification_status", "").strip() == "verified"]


ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6"}


def annex_key(so_hieu: str, label: str) -> str:
    """'Phụ lục IV' (09/2026 dạng 143) -> '<vk>_A4'; 'A3' -> '<vk>_A3'."""
    m = re.match(r"^(?:Phụ lục\s+)?([A-Za-z0-9]+)$", label)
    v = m.group(1) if m else label
    v = ROMAN.get(v.upper(), v)
    return f"{_vb_key(so_hieu)}_A{v}"
