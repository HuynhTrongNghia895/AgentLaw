"""BuildPrompt — LỚP DỰNG PROMPT của hệ thống (X7).

TỪNG CHẾT, nay sống lại
----------------------
File này từng import `SearchVectorDb` — module đã bị xoá từ P1 (Vector DB
chuyển sang `data/processed/` + FAISS + KG 4 tầng). Không ai import nó →
**chết**. Nay viết lại thành **lớp dựng prompt chuẩn**, dùng chung cho
Legal Agent.

VÌ SAO CẦN MỘT CHỖ DUY NHẤT
----------------------------
Trước đây prompt được dựng rải rác ở 2 nơi:
  - `src/graph/kg/materialize.py::format_evidence_block` — ghép evidence
  - `src/agents/prompts.py::build_legal_prompt` — dàn khung prompt

`LegalRetrieval` phải tự gọi cả 2. Dễ sót: gọi 1 trong 2 → mất block
hiệu lực hoặc mất citation. Nay `BuildPrompt.build_legal_prompt()` là
**call site duy nhất** — nhận thẳng list `Evidence`, tự ghép đủ.

NGUYÊN TẮC (CLAUDE.md §6.2, §6.3, §8 — không nới lỏng)
-------------------------------------------------------
  - Văn bản pháp luật trong ngữ cảnh là **DỮ LIỆU KHÔNG TIN CẬY** → LLM
    được nhắc không làm theo chỉ dẫn bên trong.
  - **Không merge/rewrite** văn bản pháp luật (giữ ranh giới điều/khoản/
    điểm) → LLM không bịa. (nguyên tắc §2.4 của PROJECT_CONTEXT)
  - Luôn kèm **citation** + **tình trạng hiệu lực** nếu có.
  - Không đủ căn cứ → thông báo "không đủ căn cứ", **không bịa**.

Ghi chú: `agents/prompts.py::build_legal_prompt` **giữ nguyên** làm nguồn
sự thật của *nội dung prompt* — module này chỉ đảm nhiệm phần "evidence →
context block" + làm call site duy nhất. Không nhân bản logic dàn khung.
"""

from __future__ import annotations

from typing import Any, Iterable

# Nguồn sự thật của nội dung prompt (system prompt + dàn khung section).
from agents.prompts import (  # noqa: E402
    NO_EVIDENCE_MESSAGE,
    build_legal_prompt as _build_legal_prompt_text,
)

# `Evidence` + ghép block. Import có thể lỗi nếu thiếu dependency đồ thị
# → fallback cục bộ (3 dòng, không cần DB).
try:  # pragma: no cover - phụ thuộc môi trường import
    from graph.kg.materialize import Evidence, format_evidence_block
except ImportError:  # pragma: no cover
    Evidence = Any  # type: ignore[misc,assignment]

    def format_evidence_block(evidences: Iterable[Any]) -> str:  # type: ignore[misc]
        """Fallback khi không import được `graph.kg.materialize`."""
        blocks: list[str] = []
        for ev in evidences or []:
            citation = getattr(ev, "citation", "") or ""
            text = getattr(ev, "text", "") or ""
            blocks.append(f"[{citation}]\n{text}")
        return "\n\n".join(blocks)


# Hằng cũ (từ thời SearchVectorDb) — giữ lại để không phá code đọc nó.
NO_CONTEXT_MESSAGE = (
    "Không tìm thấy đủ căn cứ pháp lý "
    "trong dữ liệu hiện có."
)


def build_legal_prompt(
    question: str,
    evidences: Iterable[Any] | None = None,
    status_block: str = "",
    graphiti_block: str = "",
) -> str:
    """Dựng prompt cho Legal Agent — **call site duy nhất**.

    Tham số:
        question     — câu hỏi người dùng.
        evidences    — list `Evidence` (dataclass ở `graph.kg.materialize`):
                       `node_id`, `citation`, `text`, `kind`, `metadata`.
                       Rỗng/None → prompt dùng thông báo "không đủ căn cứ".
        status_block — tình trạng hiệu lực lấy từ đồ thị (tùy chọn).
        graphiti_block — quan hệ sửa đổi / tri thức mới từ Graphiti (P6);
                        là dữ liệu **không tin cậy** → được đặt ở section
                        riêng kèm caveat.

    Trả về prompt hoàn chỉnh. **Không raise** khi thiếu evidence — trả
    prompt có thông báo không đủ căn cứ (quyết định #12: không bịa).
    """
    context = format_evidence_block(evidences or [])
    return _build_legal_prompt_text(
        question, context, status_block, graphiti_block
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def build_reference(
    metadata: dict[str, Any],
) -> str:
    """Chuỗi trích dẫn từ metadata dạng dict (bổ sung cho evidence dict).

    Ưu tiên `citation` nếu có; nếu không thì dựng từ
    article_number/clause_number/document_number.
    """
    citation = normalize_text(metadata.get("citation"))
    if citation:
        return citation

    parts: list[str] = []
    article_number = normalize_text(metadata.get("article_number"))
    clause_number = normalize_text(metadata.get("clause_number"))
    document_number = normalize_text(metadata.get("document_number"))

    if article_number:
        parts.append(f"Điều {article_number}")
    if clause_number:
        parts.append(f"Khoản {clause_number}")
    if document_number:
        parts.append(f"văn bản {document_number}")

    if not parts:
        return "Không xác định căn cứ"

    return ", ".join(parts)


def build_context(
    results: list[dict[str, Any]],
) -> str:
    """Ghép kết quả dạng dict thành block ngữ cảnh (tương thích cũ).

    Bỏ qua kết quả thiếu `storage_id` hoặc `text` rỗng, và loại trùng
    `storage_id` (giữ lần đầu).
    """
    cleaned_results: list[dict[str, Any]] = []
    seen_storage_ids: set[str] = set()

    for result in results or []:
        storage_id = normalize_text(result.get("storage_id"))
        text = normalize_text(result.get("text"))
        if not storage_id or not text:
            continue
        if storage_id in seen_storage_ids:
            continue
        seen_storage_ids.add(storage_id)
        cleaned_results.append(dict(result))

    if not cleaned_results:
        return NO_CONTEXT_MESSAGE

    context_parts: list[str] = []
    for rank, result in enumerate(cleaned_results, start=1):
        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        article_title = normalize_text(metadata.get("article_title"))
        reference = build_reference(metadata)
        context_lines = [
            f"[NGUỒN {rank}]",
            f"Căn cứ: {reference}",
        ]
        if article_title:
            context_lines.append(f"Tên điều: {article_title}")
        context_lines.extend([
            "Nội dung:",
            normalize_text(result.get("text")),
        ])
        context_parts.append("\n".join(context_lines))

    return "\n\n".join(context_parts)


__all__ = [
    "NO_CONTEXT_MESSAGE",
    "NO_EVIDENCE_MESSAGE",
    "build_context",
    "build_legal_prompt",
    "build_reference",
    "format_evidence_block",
    "normalize_text",
]
