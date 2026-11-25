"""QueryRefiner (P7 X3) — hỏi lại user khi câu hỏi chưa đủ thông tin.

Bối cảnh (ý thầy): "Khi user hỏi vào, mình cho nó hỏi kĩ lại, xong khi
nó có đủ thông tin nó sẽ tạo một câu hỏi lại cho hệ thống để hiểu rõ,
rồi đưa câu trả lời."

Thiết kế HYBRID (rule-based trước, LLM chỉ khi thật sự cần):

    Câu hỏi vào
       ├── GREETING  (rule) -> chào + hỏi cần tra gì -> DỪNG, KHÔNG search
       ├── CLEARED   (rule) -> đã có anchor đủ -> chuẩn hóa -> thả vào
       │                          pipeline, KHÔNG gọi LLM (giữ fast-track
       │                          nhanh, không tốn token)
       └── VAGUE     (LLM)  -> hỏi lại KÈM LỰA CHỌN đóng; sau khi user
                              chọn -> viết lại thành câu chuẩn

Vì sao rule trước:
  - "Chào bạn" / "Xin chào" là câu hỏi rác -> search là lãng phí.
  - Câu đã có số hiệu + điều là câu ĐÃ đủ thông tin -> hỏi lại là làm
    khó chịu user + làm mất lợi thế fast-track (0 embedding).
  - Chỉ câu MƠ HỒ mới cần LLM suy nghĩ xem còn thiếu gì.

LLM ở đây là Qwen chạy LOCAL (Ollama) -> không tốn tiền, chỉ tốn thời
gian. Nguyên tắc: KHÔNG bịa số hiệu văn bản không có trong input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from .QueryUnderstanding import (
        QueryUnderstanding,
        extract_anchors,
        understand,
    )
except ImportError:
    from QueryUnderstanding import (  # type: ignore
        QueryUnderstanding,
        extract_anchors,
        understand,
)


# ---------------------------------------------------------------------------
# Kết quả refine
# ---------------------------------------------------------------------------

KIND_GREETING = "greeting"
KIND_CLEARED = "cleared"
KIND_VAGUE = "vague"

# Trần mềm số lượt hỏi lại (quyết định #7)
MAX_CLARIFY_ROUNDS = 3


@dataclass
class RefineResult:
    """Kết quả của 1 lượt refine."""

    kind: str                      # greeting | cleared | vague
    question: str = ""             # câu đã chuẩn hóa (dùng cho pipeline)
    answer: str = ""               # trả lời trực tiếp (greeting) - không search
    follow_up: str = ""            # câu hỏi lại cho user
    choices: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    legal_group: str | None = None
    intent: str | None = None
    confidence: float = 0.0
    source: str = "rule"           # rule | llm
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_greeting(self) -> bool:
        return self.kind == KIND_GREETING

    @property
    def is_cleared(self) -> bool:
        return self.kind == KIND_CLEARED

    @property
    def needs_clarification(self) -> bool:
        """Có cần hỏi lại user không (dừng pipeline)?"""
        return self.kind == KIND_GREETING or (
            self.kind == KIND_VAGUE and bool(self.follow_up)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "question": self.question,
            "answer": self.answer,
            "follow_up": self.follow_up,
            "choices": self.choices,
            "missing": self.missing,
            "legal_group": self.legal_group,
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Rule-based (không tốn LLM)
# ---------------------------------------------------------------------------

# Câu chào: chỉ khớp khi câu NGẮN và toàn là lời chào -> không nhầm với
# câu hỏi thật có chứa từ "chào" (vd "chào mừng doanh nghiệp mới").
GREETING_PATTERNS = (
    r"^\s*(xin\s+)?chào\b(\s+(bạn|anh|chị|em|ad))?[\s!.?]*$",
    r"^\s*hello\b(\s+(there|you))?[\s!.?]*$",
    r"^\s*hi\b[\s!.?]*$",
    r"^\s*(cảm ơn|thank you|thanks)\b[\s!.?]*$",
    r"^\s*tôi cần tra cứu\b",
    r"^\s*(giúp tôi tra cứu|tra cứu giúp tôi)\b",
)

GREETING_REPLY = (
    "Xin chào! Tôi có thể tra cứu pháp luật Việt Nam cho bạn. "
    "Bạn muốn tra vấn vấn đề gì?"
)


def is_greeting(question: str) -> bool:
    """Câu này có phải lời chào / câu mở đầu không?"""
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(re.match(p, q) for p in GREETING_PATTERNS)


def has_sufficient_anchor(u: QueryUnderstanding) -> bool:
    """Câu hỏi đã đủ tọa độ để tra KG tất định chưa?

    Đủ = có số hiệu văn bản VÀ có điều (hoặc khoản/điểm).
    """
    a = u.anchors
    return bool(a.doc_numbers) and bool(
        a.articles or a.clauses or a.points
    )


# ---------------------------------------------------------------------------
# Prompt cho LLM (chỉ dùng cho câu mơ hồ)
# ---------------------------------------------------------------------------

REFINER_SYSTEM_PROMPT = """\
Bạn là trợ lý pháp lý Việt Nam. Nhiệm vụ: đánh giá câu hỏi của người dùng
về luật và, nếu câu hỏi còn MƠ HỒ, soạn câu hỏi làm rõ KÈM CÁC LỰA CHỌN
đóng để người dùng chỉ cần chọn số thứ tự.

NGUYÊN TẮC TUYỆT ĐỐI:
1. TUYỆT ĐỐI KHÔNG bịa số hiệu văn bản. Chỉ được dùng số hiệu xuất hiện
   trong danh sách VĂN BẢN CÓ SẴN được cung cấp. Nếu không chắc, để
   doc_numbers rỗng.
2. Trả lời DUY NHẤT bằng 1 JSON hợp lệ, không thêm chữ nào khác.
3. Lựa chọn phải CỤ THỂ và HỮU HẠN (3-5 lựa chọn). Khi câu hỏi thiếu
   thông tin (chưa rõ văn bản nào), mỗi lựa chọn nêu dạng:
   "số hiệu văn bản – phạm vi/mô tả ngắn" (dùng mô tả trong ngoặc của
   danh sách VĂN BẢN CÓ SẴN) để người dùng KHÔNG rành luật vẫn chọn
   được. Không để lựa chọn chỉ là số hiệu trần, không hỏi chung chung
   kiểu "bạn muốn hỏi gì".
4. Nếu câu hỏi đã đủ rõ (có số hiệu/điều, hoặc đã nêu rõ loại thuế và
   nội dung) -> kind = "cleared", viết lại câu hỏi đầy đủ, không hỏi lại.
5. Không hỏi chọn văn bản chỉ vì người dùng chưa nêu số hiệu nếu chủ đề và
   việc cần tra đã rõ; chuyển câu hỏi để retrieval tìm căn cứ phù hợp.
6. Nếu câu hỏi ngoài phạm vi pháp luật nhưng đã rõ (ví dụ hỏi thời tiết),
   không hỏi làm rõ nội dung đó; giữ nguyên câu hỏi để retrieval phân loại OOD.
7. Nếu thực sự thiếu dữ kiện, liệt kê tất cả dữ kiện thiết yếu còn thiếu
   trong cùng một câu hỏi lại, không hỏi từng dữ kiện đã có.

JSON cần đúng các trường sau:
{
  "kind": "cleared" | "vague",
  "question": "câu hỏi đã chuẩn hóa, đầy đủ, dùng để tra cứu",
  "legal_group": "doanh_nghiep" | "thue_gtgt" | "thue_tndn" | "quan_ly_thue" | null,
  "intent": "tra_cuu" | "hieu_luc" | "time_travel" | "so_sanh" | "chinh_xac",
  "missing": ["danh sách thông tin còn thiếu"],
  "choices": ["lựa chọn 1", "lựa chọn 2", "lựa chọn 3"],
  "clarifying_question": "câu hỏi làm rõ (chỉ khi kind = vague)",
  "confidence": 0.0
}
"""


def build_refiner_prompt(
    question: str,
    allowed_documents: list[str],
    user_answer: str = "",
    conversation_context: str = "",
) -> str:
    """Dựng prompt cho LLM refine.

    `conversation_context`: lược sử các lượt hội thoại trước (agent
    STATELESS — không tự nhớ; người gọi truyền lại ở mỗi lượt). Rỗng
    -> prompt y hệt trước đây (đường 1 lượt không đổi).
    """
    docs = "\n".join(f"- {d}" for d in allowed_documents) or "- (không có)"
    parts = [
        REFINER_SYSTEM_PROMPT,
        "",
        "=== VĂN BẢN CÓ SẴN (số hiệu – phạm vi) ===",
        docs,
        "",
        "=== CÂU HỎI CỦA NGƯỜI DÙNG ===",
        question.strip(),
    ]
    if user_answer:
        parts += [
            "",
            "=== NGƯỜI DÙNG ĐÃ TRẢ LỜI ===",
            user_answer.strip(),
            "",
            "Dựa trên câu trả lời trên, hãy viết lại câu hỏi đầy đủ và "
            "chọn kind phù hợp.",
        ]
    if conversation_context:
        parts += [
            "",
            "=== LƯỢC SỬ HỘI THOẠI TRƯỚC ĐÓ ===",
            conversation_context.strip(),
            "",
            "Dùng lược sử để KHÔNG hỏi lại thông tin người dùng đã cung "
            "cấp và giữ đúng ngữ cảnh các lượt trước.",
        ]
    parts += ["", "=== JSON ==="]
    return "\n".join(parts)


def parse_refiner_json(raw: str) -> dict[str, Any] | None:
    """Parse JSON từ output LLM (chịu được ```json fence + text thừa)."""
    if not raw:
        return None
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", raw,
                  flags=re.IGNORECASE | re.DOTALL).strip()
    # Bóc fence ```json ... ```
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class QueryRefiner:
    """Hỏi lại + chuẩn hóa câu hỏi trước khi vào pipeline.

    Args:
        llm: BaseLLM (Qwen local). None -> chỉ chạy rule-based (không hỏi
            lại được câu mơ hồ, vẫn trả lời bằng pipeline).
        allowed_documents: số hiệu văn bản được phép nhắc (chống bịa).
    """

    def __init__(
        self,
        llm: Any | None = None,
        allowed_documents: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.allowed_documents = allowed_documents or []

    # -- Rule ------------------------------------------------------------

    def refine(
        self,
        question: str,
        user_answer: str = "",
        conversation_context: str = "",
    ) -> RefineResult:
        """Refine 1 câu hỏi.

        `user_answer` = câu user đã chọn/trả lời.
        `conversation_context` = lược sử hội thoại các lượt trước (agent
        STATELESS: chính người gọi truyền lại). Rỗng -> hành vi cũ y nguyên.
        """
        q = (question or "").strip()

        # 1) Lời chào / câu mở đầu -> dừng, không search.
        if not user_answer and is_greeting(q):
            return RefineResult(
                kind=KIND_GREETING,
                answer=GREETING_REPLY,
                follow_up="Bạn muốn tra vấn vấn đề pháp lý nào?",
                source="rule",
            )

        # 2) Câu đã đủ thông tin -> chuẩn hóa, KHÔNG gọi LLM.
        #    Có lược sử hội thoại -> BỎ qua shortcut (cho LLM xem ngữ
        #    cảnh trước, tránh hỏi lại cái user đã cung cấp ở lượt trước).
        if not user_answer and not conversation_context:
            u = understand(q)
            if has_sufficient_anchor(u):
                return RefineResult(
                    kind=KIND_CLEARED,
                    question=q,
                    legal_group=u.legal_group,
                    intent=u.intent,
                    confidence=1.0,
                    source="rule",
                )

        # 3) Cần LLM để biết còn thiếu gì (câu mơ hồ / có ngữ cảnh).
        return self._refine_with_llm(q, user_answer, conversation_context)

    # -- F4a/F4b: guard menu hỏi-lại (không bịa vb, buộc hỏi khi thiếu vb) --

    def _valid_doc_numbers(self) -> list[str]:
        """F4b: số hiệu vb CÓ THẬT, trích từ danh sách vb có sẵn (registry).

        Dùng để rã soát menu: chỉ giữ lựa chọn có chứa số hiệu này. Danh
        sách rỗng (chưa nối registry) -> trả [] -> không rã soát (degrade
        an toàn, đường cũ không đổi).
        """
        out: list[str] = []
        for doc in self.allowed_documents:
            m = re.match(
                r"\d{1,3}/\d{4}/(?:QH\d{1,2}|NĐ-?CP)", str(doc).strip(), re.I
            )
            if m:
                out.append(m.group(0))
        return out

    def _fallback_doc_choices(self, question: str) -> list[str]:
        """F4a: menu fallback khi câu thiếu số hiệu vb (buộc HỎI, không bịa).

        Chọn tối đa 5 vb từ `allowed_documents` (vb CÓ SẴN — bao giờ cũng
        thật), ưu tiên vb có tên/dạng gần nghĩa với câu hỏi (keyword
        overlap), tie-break theo thứ tự danh sách. Danh sách rỗng -> []
        (giữ degrade, không crash, không bịa).
        """
        docs = [str(d).strip() for d in self.allowed_documents if str(d).strip()]
        if not docs:
            return []
        toks = set(re.findall(r"[a-zà-ỹ\w]{3,}", (question or "").lower()))
        scored: list[tuple[int, int, str]] = []
        for i, doc in enumerate(docs):
            low = doc.lower()
            overlap = sum(1 for t in toks if t in low)
            scored.append((-overlap, i, doc))
        scored.sort()
        return [s[2] for s in scored[:5]]

    # -- LLM -------------------------------------------------------------

    def _refine_with_llm(
        self,
        question: str,
        user_answer: str,
        conversation_context: str = "",
    ) -> RefineResult:
        if self.llm is None:
            # Không có LLM -> trả về câu gốc, để pipeline tự xử lý
            # (không hỏi lại, không chặn user).
            u = understand(question)
            return RefineResult(
                kind=KIND_CLEARED,
                question=question,
                legal_group=u.legal_group,
                intent=u.intent,
                confidence=0.5,
                source="rule",
            )

        prompt = build_refiner_prompt(
            question,
            self.allowed_documents,
            user_answer,
            conversation_context=conversation_context,
        )
        try:
            _sp = getattr(self.llm, "set_purpose", None)
            if _sp:
                _sp("refiner")
            raw = self.llm.complete(prompt)
        except Exception:
            u = understand(question)
            return RefineResult(
                kind=KIND_CLEARED,
                question=question,
                legal_group=u.legal_group,
                intent=u.intent,
                confidence=0.5,
                source="rule",
            )

        data = parse_refiner_json(raw)
        if not data:
            u = understand(question)
            return RefineResult(
                kind=KIND_CLEARED,
                question=question,
                legal_group=u.legal_group,
                intent=u.intent,
                confidence=0.5,
                source="rule",
            )

        kind = str(data.get("kind") or KIND_VAGUE).strip()
        if kind not in (KIND_CLEARED, KIND_VAGUE):
            kind = KIND_VAGUE

        choices = [
            str(c).strip()
            for c in (data.get("choices") or [])
            if str(c).strip()
        ][:5]
        missing = [
            str(m).strip()
            for m in (data.get("missing") or [])
            if str(m).strip()
        ]

        q_new = str(data.get("question") or "").strip() or question
        follow_up = str(data.get("clarifying_question") or "").strip()

        # F4b: rã soát MENU — giữ chỉ lựa chọn MỌI số hiệu vb trong đó đều
        # THẬT (nằm trong danh sách vb có sẵn). LLM hallucinate "48/2025/QH15"
        # (registry chỉ có 48/2024) -> bị loại. Trích số hiệu bằng regex (khớp
        # đúng cách grader chấm), KHÔNG dùng substring (tránh false khi số
        # bịa chứa số thật làm tiền tố). Lựa chọn mô tả thuần (không có số
        # hiệu vb) -> giữ nguyên. Danh sách vb có sẵn rỗng -> không rã soát
        # (đường degrade an toàn, không crash, không cắt oan).
        valid_docs = self._valid_doc_numbers()
        if valid_docs:
            # So khớp KHÔNG phân biệt hoa thường (LLM có thể viết qh15).
            valid_set = {v.upper() for v in valid_docs}
            choices = [
                c for c in choices
                if not any(
                    m.upper() not in valid_set
                    for m in re.findall(
                        r"\d{1,4}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)?",
                        c,
                        re.I,
                    )
                )
            ]

        # LLM nói "cleared" nhưng câu vẫn mơ hồ (không có lựa chọn và
        # không có câu hỏi làm rõ) -> tin rule: nếu câu đã đủ anchor thì
        # cleared, ngược lại vague.
        #
        # CHỈ áp ở lượt ĐẦU. Nếu user đã trả lời (user_answer) thì
        # KHÔNG hỏi lại nữa (trần 3 lượt) — tin LLM, câu đã đủ ý.
        if kind == KIND_CLEARED and not choices and not user_answer:
            u = understand(q_new)
            if not has_sufficient_anchor(u):
                kind = KIND_VAGUE
                # F4a: câu có Điều/Khoản NHƯNG thiếu số hiệu vb (vd S3
                # "Điều 2...người nộp thuế" — "người nộp thuế" có trong NHIỀU
                # văn bản) -> trả thẳng sẽ SAI vb. Buộc HỎI + cấp menu fallback
                # (các vb có sẵn, không bịa) để user chọn đúng.
                if (
                    not u.anchors.doc_numbers
                    and (u.anchors.articles or u.anchors.clauses)
                    and not choices
                ):
                    choices = self._fallback_doc_choices(question)
                    if choices and not follow_up:
                        arts = u.anchors.articles or []
                        label = (
                            " và ".join("Điều " + a for a in arts[:2])
                            if arts
                            else "vấn đề"
                        )
                        follow_up = f"Bạn muốn hỏi {label} của văn bản nào?"

        if kind == KIND_VAGUE and not follow_up and choices:
            follow_up = (
                "Bạn muốn tra phần nào?\n"
                + "\n".join(f"{i + 1}) {c}" for i, c in enumerate(choices))
            )

        return RefineResult(
            kind=kind,
            question=q_new,
            follow_up=follow_up,
            choices=choices,
            missing=missing,
            legal_group=(
                str(data["legal_group"]).strip()
                if data.get("legal_group")
                else None
            ),
            intent=(
                str(data["intent"]).strip()
                if data.get("intent")
                else None
            ),
            confidence=float(data.get("confidence") or 0.0),
            source="llm",
            raw=data,
        )


__all__ = [
    "GREETING_REPLY",
    "KIND_CLEARED",
    "KIND_GREETING",
    "KIND_VAGUE",
    "MAX_CLARIFY_ROUNDS",
    "QueryRefiner",
    "RefineResult",
    "build_refiner_prompt",
    "has_sufficient_anchor",
    "is_greeting",
    "parse_refiner_json",
]
