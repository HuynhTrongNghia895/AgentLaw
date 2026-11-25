"""Ontology KG — schema node/cạnh + 5 mốc hiệu lực.

Dùng chung 2 mục đích:
1. Schema nguồn để seed Cypher deterministic (loader + seed).
2. entity_types / edge_types Pydantic cho Graphiti add_episode
   (khi add văn bản luật MỚI — con đường LLM, chưa trigger hiện tại).

Tên trường giữ convention GĐ3 (đặt tên cũ, dễ nhớ), KHÔNG theo tên
Graphiti (Graphiti chỉ là hòm + retrieval ở GĐ4).
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Định nghĩa 5 mốc hiệu lực (để trên node/cạnh, KHÔNG phải Graphiti ép)
#   1. effective_date              : ngày hiệu lực CHUNG của văn bản (registry),
#                                    kế thừa xuống MỌI node con.
#   2. provision_effective_date   : ngày hiệu lực RIÊNG của điều/khoản (registry
#                                    cột provision_effective_dates, 9 item/4 vb),
#                                    null nếu không có. Priority > mốc 1.
#   3. applicable_from            : "áp dụng từ kỳ nào" — string nguyên văn,
#                                    KHÔNG parse ngày (2 khoản: 67/2025 D19K1,
#                                    320/2025 D24K1).
#   4. relation_effective_date    : hiệu lực CON QUAN HỆ trên cạnh
#                                    AMENDS/REPEALS/GUIDES = coalesce(provision
#                                    effect date, effective_date) của KHOẢN SỬA.
#   5. as_of_date                 : mốc cảnh báo "dữ liệu tính đến 29/10/2026",
#                                    1 config constant (config.py), KHÔNG lưu node.
# ---------------------------------------------------------------------------


def make_uuid(node_id: str) -> str:
    """UUID5 deterministic từ id node — để Graphiti dedup + truy cập node."""
    return str(hashlib.md5(node_id.encode("utf-8")).hexdigest())


class StructuralHeading(BaseModel):
    """Common stored fields for optional Part/Chapter/Section nodes."""

    id: str = Field(description="ID duy nhất, có số hiệu văn bản và đường dẫn cấu trúc")
    so_hieu: str = Field(description="số hiệu văn bản")
    number: str = Field(default="", description="số Phần/Chương/Mục như trong nguồn")
    title: str = Field(default="", description="tiêu đề cấu trúc")
    uuid: str = ""
    name: str = ""
    kind4tier: bool = True


class Part(StructuralHeading):
    """Phần (chỉ có ở văn bản có phân Phần)."""


class Chapter(StructuralHeading):
    """Chương (chỉ có ở văn bản có Chương)."""


class Section(StructuralHeading):
    """Mục (chỉ có ở văn bản có Mục)."""


class Document(BaseModel):
    """Node tầng 1 (văn bản). 1 văn bản = 1 node, thuộc 1 hoặc nhiều dataset."""

    id: str = Field(description="id node, = so_hieu (số hiệu văn bản)")
    so_hieu: str = Field(description="số hiệu văn bản, ví dụ 59/2020/QH14")
    document_title: str = Field(default="", description="tên/tiêu đề văn bản")
    nhom_pl: list[str] = Field(default_factory=list, description="4 nhóm: doanh_nghiep/thue_gtgt/thue_tndn/quan_ly_thue")
    ten_file: str = Field(default="", description="tên file DOCX gốc")
    ban_hanh_boi: str = Field(default="", description="cơ quan ban hành (QH/CP...)")
    ngay_ban_hanh: str | None = Field(default=None, description="ngày ban hành")
    effective_date: str | None = Field(default=None, description="MỐC 1: ngày hiệu lực chung văn bản")
    effective_status: str = Field(default="", description="con_hieu_luc/bi_thay_the_mot_phan/...")
    superseded_by: str = Field(default="", description="văn bản thay thế (nếu có)")
    uuid: str = ""  # Graphiti compat
    name: str = ""  # Graphiti compat: "Luật 59/2020/QH14"
    summary: str = ""  # Graphiti compat


class Article(BaseModel):
    """Node tầng 2 (điều). Mỗi điều = 1 node, text nguyên vẹn."""

    id: str = Field(description="id node, ví dụ 59_2020_QH14_D49")
    so_hieu: str = Field(description="số hiệu văn bản cha")
    dieu: str = Field(description="số điều, ví dụ 49")
    title: str = Field(default="", description="tiêu đề điều")
    text: str = Field(default="", description="nội dung điều (đã gộp sub-chunk)")
    effective_date: str | None = Field(default=None, description="MỐC 1: kế thừa từ văn bản cha")
    provision_effective_date: str | None = Field(default=None, description="MỐC 2: ngày riêng (null nếu không có)")
    uuid: str = ""
    name: str = ""
    summary: str = ""


class Clause(BaseModel):
    """Node tầng 3 (khoản). Mỗi khoản = 1 node, text nguyên vẹn (gộp sub-chunk)."""

    id: str = Field(description="id node, ví dụ 59_2020_QH14_D49_K1")
    so_hieu: str = Field(description="số hiệu văn bản")
    dieu: str = Field(description="số điều cha")
    khoan: str = Field(description="số khoản, ví dụ 1, 5a, 11a")
    title: str = Field(default="", description="tiêu đề khoản")
    text: str = Field(default="", description="nội dung khoản (đã gộp sub-chunk)")
    effective_date: str | None = Field(default=None, description="MỐC 1: kế thừa từ văn bản")
    provision_effective_date: str | None = Field(default=None, description="MỐC 2: ngày riêng")
    applicable_from: str | None = Field(default=None, description="MỐC 3: áp dụng từ kỳ nào (string)")
    invalid_at: str | None = Field(default=None, description="ngày hết hiệu lực (REPEALS), null nếu còn")
    uuid: str = ""
    name: str = ""
    summary: str = ""


class Point(BaseModel):
    """Node tầng 4 (điểm a) b) c) ...). Parse từ content_blocks của khoản."""

    id: str = Field(description="id node, ví dụ 59_2020_QH14_D49_K1_P1")
    so_hieu: str = Field(description="số hiệu văn bản")
    dieu: str = Field(description="số điều")
    khoan: str = Field(description="số khoản cha")
    diem: str = Field(description="chỉ số điểm (1-based trong khoản), ví dụ 1, 2, 3")
    text: str = Field(default="", description="nội dung điểm")
    effective_date: str | None = Field(default=None, description="MỐC 1: kế thừa")
    provision_effective_date: str | None = Field(default=None, description="MỐC 2: kế thừa từ khoản/điều")
    uuid: str = ""
    name: str = ""
    summary: str = ""


class Annex(BaseModel):
    """Phụ lục (55 records, record_type='annex'). Treo dưới Document."""

    id: str = Field(description="id node, ví dụ 144_2026_ND_CP_A1")
    so_hieu: str = Field(description="số hiệu văn bản")
    phu_luc: str = Field(description="số phụ lục, ví dụ 1, IV")
    title: str = Field(default="", description="tên phụ lục")
    text: str = Field(default="", description="nội dung phụ lục")
    effective_date: str | None = Field(default=None, description="MỐC 1: kế thừa")
    uuid: str = ""
    name: str = ""
    summary: str = ""


class RelationEdge(BaseModel):
    """Cạnh AMENDS/REPEALS/GUIDES provision-level (mở rộng từ 41 quan hệ CSV)."""

    relation_id: str = Field(description="mã REL_xxx (truy về DocumentRelations.csv)")
    relation_type: str = Field(description="AMENDS/REPEALS/GUIDES")
    from_id: str = Field(description="id node nguồn (khoản/điều SỬA)")
    to_id: str = Field(description="id node đích (khoản/điều ĐIỀU BỊ SỬA)")
    relation_effective_date: str | None = Field(
        default=None,
        description="MỐC 4: hiệu lực con quan hệ = coalesce(provision_effective_date, "
        "effective_date) của khoản sửa. Query: T >= ngày này → quan hệ đã có hiệu lực.",
    )
    source_reference: str = Field(default="", description="nguồn chính văn (cột CSV)")
