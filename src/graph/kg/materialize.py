"""Materialize — ghép cây node thành EVIDENCE đưa cho LLM.

Tách khỏi `vector_store` (P1, 04/11) vì 2 mục đích khác nhau:

  - **Text để TÌM** (embed): node + nội dung con gộp lại, để query khớp
    được cả điều lẫn khoản lẫn điểm.
  - **Evidence để TRẢ LỜI**: giữ ranh giới khoản/điểm, để LLM biết
    "a) Tên hộ kinh doanh" là ĐIỂM a của KHOẢN 1 của ĐIỀU 100, không
    phải một mệnh đề lơ lửng. LLM không được bịa ranh giới.

Nguyên tắc: **không bao giờ đưa 1 node "trần" cho LLM** — luôn kèm
ngữ cảnh dọc (điều → khoản → điểm) + citation + tình trạng hiệu lực.

Không merge/rewrite văn bản pháp luật (nguyên tắc §2.4 PROJECT_CONTEXT):
mọi khẳng định trạng thái đều kèm nguồn + caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .loader import KgNode


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------
def node_citation(node: KgNode) -> str:
    """Citation chuẩn hoá: 'Điều 100 Khoản 1 Điểm a (168/2025/NĐ-CP)'."""
    if node.kind == "annex":
        label = f"Phụ lục {node.phu_luc}"
    elif node.kind == "article":
        label = f"Điều {node.dieu}"
    elif node.kind == "clause":
        label = f"Điều {node.dieu} Khoản {node.khoan}"
    else:  # point
        label = f"Điều {node.dieu} Khoản {node.khoan} Điểm {node.diem}"
    return f"{label} ({node.so_hieu})"


def node_title_line(node: KgNode) -> str:
    """Dòng tiêu đề của node (Điều/Khoản) để dựng lại văn bản."""
    if node.kind == "annex":
        head = f"Phụ lục {node.phu_luc}"
    elif node.kind == "article":
        head = f"Điều {node.dieu}"
    elif node.kind == "clause":
        head = f"Khoản {node.khoan}"
    else:
        head = f"Điểm {node.diem}"
    if node.title:
        return f"{head}. {node.title}"
    return head


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    """1 mảnh bằng chứng đưa cho LLM."""

    node_id: str
    citation: str
    text: str                 # văn bản đã dựng lại (giữ cấu trúc)
    kind: str = ""
    node_type: str = ""       # "article" | "clause" | "point" | "annex"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "citation": self.citation,
            "text": self.text,
            "kind": self.kind,
            "node_type": self.node_type,
            "metadata": self.metadata,
        }


def _effective_note(node: KgNode, doc_meta: dict[str, Any] | None) -> str:
    """Ghi chú tình trạng hiệu lực (có nguồn, không suy tuyệt đối)."""
    bits: list[str] = []
    doc = doc_meta or {}
    doc_date = str(doc.get("effective_date") or "")
    if doc_date:
        bits.append(f"Hiệu lực văn bản từ {doc_date}")
    if node.provision_effective_date:
        bits.append(
            f"Hiệu lực riêng từ {node.provision_effective_date}"
        )
    if node.applicable_from:
        bits.append(f"Áp dụng từ {node.applicable_from}")
    if node.invalid_at:
        bits.append(f"Bị bãi bỏ từ {node.invalid_at}")
    return " | ".join(bits)


def materialize(
    node: KgNode,
    children: dict[str, list[KgNode]] | None = None,
    doc_meta: dict[str, Any] | None = None,
    include_status: bool = True,
) -> Evidence:
    """Dựng 1 node + toàn bộ cây con thành văn bản có cấu trúc.

    Args:
        node: node gốc (Điều / Khoản / Điểm / Phụ lục).
        children: {parent_id: [node con...]} — adjacency map của cây KG.
        doc_meta: metadata văn bản (effective_date, ...) để ghi chú hiệu lực.
        include_status: có gắn ghi chú hiệu lực không.

    Returns:
        Evidence với `text` giữ ranh giới khoản/điểm rõ ràng.
    """
    child_map = children or {}
    lines: list[str] = []
    visited = {node.node_id}

    def subtree_has_text(parent: KgNode) -> bool:
        if parent.text and parent.text.strip():
            return True
        return any(subtree_has_text(child) for child in child_map.get(parent.node_id, []))

    def append_subtree(parent: KgNode) -> None:
        for child in child_map.get(parent.node_id, []):
            if child.node_id in visited or not subtree_has_text(child):
                continue
            visited.add(child.node_id)
            if child.kind == "clause":
                lines.append(f"Khoản {child.khoan}.")
                if child.text and child.text.strip():
                    lines.append(child.text.strip())
            elif child.kind == "point":
                marker = child.diem or ""
                prefix = f"{marker})" if marker else "Điểm."
                if child.text and child.text.strip():
                    lines.append(f"{prefix} {child.text.strip()}")
                else:
                    lines.append(prefix)
            else:
                if child.text and child.text.strip():
                    lines.append(child.text.strip())
            append_subtree(child)

    if node.kind == "article":
        lines.append(node_title_line(node))
        if node.text and node.text.strip():
            lines.append(node.text.strip())
        append_subtree(node)
    elif node.kind == "clause":
        lines.append(f"Khoản {node.khoan}.")
        if node.text and node.text.strip():
            lines.append(node.text.strip())
        append_subtree(node)
    elif node.kind == "point":
        marker = f"Điểm {node.diem}." if node.diem else "Điểm."
        lines.append(f"{marker} {node.text.strip()}" if node.text and node.text.strip() else marker)
        append_subtree(node)
    else:  # annex
        lines.append(node_title_line(node))
        if node.text and node.text.strip():
            lines.append(node.text.strip())
        append_subtree(node)

    body = "\n".join(line for line in lines if line).strip()

    meta: dict[str, Any] = {
        "dieu": node.dieu,
        "khoan": node.khoan,
        "diem": node.diem,
        "phu_luc": node.phu_luc,
        "so_hieu": node.so_hieu,
    }
    if include_status:
        note = _effective_note(node, doc_meta)
        if note:
            meta["effective_note"] = note
            body = f"{body}\n({note})" if body else f"({note})"

    return Evidence(
        node_id=node.node_id,
        citation=node_citation(node),
        text=body,
        kind=node.kind,
        node_type=node.kind,
        metadata=meta,
    )


def materialize_many(
    nodes: Iterable[KgNode],
    children: dict[str, list[KgNode]] | None = None,
    doc_meta: dict[str, Any] | None = None,
    include_status: bool = True,
) -> list[Evidence]:
    return [
        materialize(n, children=children, doc_meta=doc_meta,
                    include_status=include_status)
        for n in nodes
    ]


def format_evidence_block(evidences: Iterable[Evidence]) -> str:
    """Ghép evidence thành 1 khối text cho LLM (có nhãn nguồn rõ ràng)."""
    blocks: list[str] = []
    for ev in evidences:
        blocks.append(f"[{ev.citation}]\n{ev.text}")
    return "\n\n".join(blocks)


__all__ = [
    "Evidence",
    "format_evidence_block",
    "materialize",
    "materialize_many",
    "node_citation",
    "node_title_line",
]
