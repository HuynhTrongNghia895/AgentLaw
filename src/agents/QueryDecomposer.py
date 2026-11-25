"""QueryDecomposer (P7) — xử lý câu hỏi NHIỀU Ý bằng gap-filling.

Triết lý (không "cắt N ý" cố định — con số trần chỉ là NGÂN SÁCH CHI
PHÍ, không phải số ý):

    Câu hỏi (nhiều ý bất kỳ: 2, 3, 4, 5...)
       │
       ├─ [1] LỌC RẺ (rule, không LLM): có "tín hiệu đa ý" không?
       │      Không -> coi là câu đơn, no-op (đa số câu, 0 chi phí).
       │
       ├─ [2] Qwen kiểm tra ĐỘ PHỦ: câu gồm các ý A,B,C... — ý nào
       │      CHƯA có căn cứ trong evidence hiện có? (dựa trên DỮ LIỆU,
       │      Qwen tự đếm, không cần trần "số ý").
       │
       └─ [3] LẤP: với MỖI ý thiếu -> chạy pipeline lại -> gộp evidence.
              Lặp [2]-[3] trong NGÂN SÁCH (max vòng / trần evidence /
              thời gian). Hết ý thiếu hoặc hết ngân sách -> dừng.

Điểm khác với "tách N ý trước rồi map-reduce":
  - Không đoán N trước. N do Qwen đọc câu + evidence mà đếm.
  - Câu đơn: bước lọc chặn trước, không tốn LLM, số không đổi.
  - Ý trùng / ý thừa: Qwen nhận diện, không xử lý 2 lần.
  - 1 ý ngoài phạm vi bị OOD chặn RIÊNG, không kéo thụt điểm cả câu.

An toàn: không LLM (test NoLLM trả "") -> parse coverage lỗi -> coi
"hết ý thiếu" -> dừng ngay. Không crash.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from retrieval.QueryUnderstanding import extract_anchors, parse_temporal_constraint

# Ngắn gọn tên import để dùng trong prompt
try:
    from retrieval.QueryRefiner import parse_refiner_json
except ImportError:
    from QueryRefiner import parse_refiner_json  # type: ignore


# ---------------------------------------------------------------------------
# Ngưỡng / trần NGÂN SÁCH (chi phí, KHÔNG phải "số ý tối đa")
# ---------------------------------------------------------------------------
MAX_ROUNDS = 3          # số vòng lấp-trong tối đa
MAX_EVIDENCE = 30      # trần evidence gộp (trần token cho prompt)
# Số "tín hiệu đa ý" cần để bước lọc rẻ cho Qwen kiểm tra (tiết kiệm).
MIN_MULTI_SIGNALS = 2

# P-B (B1 + điều chỉnh #3): intent_verdict (KẾT LUẬN CUỐI) — riêng rẽ với
# trục execution (exec_state: OK/EMPTY/FAILED, trung gian). KHÔNG đánh đồng:
# - COVERED      : required node (hoặc con) đã đủ trong pool (BỀN, không bị
#                  lỗi bước phụ xóa — điều chỉnh #3).
# - OOD          : chủ đề ngoài phạm vi DB. Chỉ LegalRetreival (OOD gate F5b)
#                  mới được gán OOD; decomposer KHÔNG tự gán (B1) — decomposer
#                  chỉ đánh dấu ứng viên (pending_gate).
# - INSUFFICIENT : intent hợp lệ nhưng chưa đủ căn cứ để trả.
# - FAILED       : lỗi kỹ thuật THỰC (exception) khi fill intent -> không thể
#                  kết luận. Không đánh đồng "không tìm thấy" (đó là EMPTY).
VERDICT_COVERED = "COVERED"
VERDICT_OOD = "OOD"
VERDICT_INSUFFICIENT = "INSUFFICIENT"
VERDICT_FAILED = "FAILED"


# Tín hiệu "nhiều ý" — dùng để LỌC RẺ (không quyết N).
_MULTI_INTENT_PATTERNS = [
    # ≥2 từ nối / "cũng / cả ... và"
    r"(và|cũng|cùng|thêm|ngoài ra|đồng thời|cùng với|cũng như)\s",
    r"(\bA\b|\bB\b|\bC\b)\s*(và|/|,)",
    # ≥2 mệnh đề hỏi (câu dài + nhiều động từ hỏi / số điều)
    r"điều\s+\d+",
    r"(quy định|bao nhiêu|thế nào|khi nào|ai|đăng ký|nộp|phạt|mức|thời hạn|hiệu lực)",
    # Dấu tách ý rõ: "1)... 2)..." / "trước hết... sau đó..."
    r"\d\)\s*.\{0,40}\d\)",
]

# Prompt kiểm tra độ phủ (cho Qwen local).
_COVERAGE_SYSTEM = """\
Bạn là trợ lý pháp lý Việt Nam. Đọc CÂU HỎI của người dùng và DANH SÁCH
CĂN CỨ đã tìm được. Nhiệm vụ: tách câu hỏi thành các "ý con" (mỗi ý là
một thứ cần trả lời riêng), rồi chỉ ra ý nào CHƯA có căn cứ khớp.

Trả về DUY NHẤT 1 JSON hợp lệ:
{
  "intents": ["ý con 1 (ngắn)", "ý con 2", ...],
  "uncovered": ["ý nào còn thiếu căn cứ", ...],
  "clarify": null,
  "confidence": 0.0
}

NGUYÊN TẮC:
- "uncovered" RỖNG nếu mọi ý con đều đã có căn cứ.
- Chỉ liệt kê ý CON RIÊNG BIỆT; ý trùng/ăn nhau thì hợp nhất, không
  lặp lại. Không bịa ý không có trong câu hỏi.
- "clarify": nếu có ý con KHÔNG tra đúng được vì thiếu thông tin (vd
  "điều 4 của văn bản nào?"), cho MỘT câu hỏi gộp duy nhất để hỏi lại
  người dùng (liệt kê các ý thiếu trong 1 câu). Nếu mọi ý đủ thông tin
  để tra, để "clarify" = null.
"""

# Prompt "vẽ lại ý con" (rewriter) — viết lại ý con ngắn thành câu tra
# đủ ngữ cảnh để tìm đúng điều/khoản. CHỈ làm rõ hơn, KHÔNG bịa nội dung.
_REWRITER_SYSTEM = """\
Bạn là trợ lý pháp lý Việt Nam. Bạn nhận MỘT "ý con" trong câu hỏi pháp
lý và BỐI CẢNH (câu hỏi gốc + các căn cứ đã tìm được). Nhiệm vụ: viết lại
"ý con" thành một CÂU TRA CỨU PHÁP LÝ rõ ràng hơn, đủ ngữ cảnh để tìm
đúng điều/khoản.

NGUYÊN TẮC (bắt buộc):
- CHỈ làm RÕ HƠN. TUYỆT ĐỐI KHÔNG bịa: không thêm quy định, không thêm
  số hiệu văn bản, không đoán loại văn bản.
- Số hiệu / loại văn bản chỉ được dùng nếu NÓ CÓ ĐÚNG trong bối cảnh,
  và phải viết ĐÚNG NGUYÊN VĂN như trong bối cảnh (vd bối cảnh ghi
  "108/2025/QH15" thì viết "108/2025/QH15", không tự đổi thành
  "Thông tư 15/2025/TT-BTC" hay bỏ "QH15").
- Nếu bối cảnh KHÔNG nêu rõ văn bản -> viết câu cho rõ hơn theo cách
  diễn đạt tự nhiên, KHÔNG chèn số hiệu.
- Giữ nguyên ý nghĩa của ý con, chỉ bổ sung từ khoá giúp tìm đúng.
- Trả DUY NHẤT 1 câu tra (câu văn, không giải thích, không JSON).
"""


def build_rewriter_prompt(sub_query: str, context: str) -> str:
    """Dựng prompt "vẽ lại ý con" -> câu tra đủ ngữ cảnh."""
    return "\n".join(
        [
            _REWRITER_SYSTEM,
            "",
            "=== BỐI CẢNH ===",
            (context or "").strip() or "(không có)",
            "",
            "=== Ý CON CẦN VIẾT LẠI ===",
            (sub_query or "").strip(),
            "",
            "=== Câu tra mới ===",
        ]
    )


def default_sub_query_rewriter(llm: Any) -> Callable[[str, str], str]:
    """Trả rewriter `(ý con, bối cảnh) -> câu tra` dùng LLM.

    LLM None / lỗi / trả rỗng -> trả lại Ý CON GỐC (fallback an toàn,
    không bịa, không crash) — đúng nguyên tắc "không làm hỏng đường cũ".
    """

    def _rewrite(sub: str, context: str) -> str:
        # KHÔNG viết lại nếu không có LLM THẬT. `llm.name == "none"` là
        # NoLLM trong test (65 câu + OOD chạy không LLM) -> trả chuỗi
        # cố định; nếu dùng làm sub-query sẽ bậy. Giữ ý con NGUYÊN.
        if llm is None or getattr(llm, "name", "") in ("none", ""):
            return sub
        try:
            # Tracer-only, dispatch an toàn (stub trần không có method
            # này -> bỏ qua; không ảnh hưởng pipeline).
            _sp = getattr(llm, "set_purpose", None)
            if _sp:
                _sp("sub_query_rewrite")
            out = llm.complete(build_rewriter_prompt(sub, context))
        except Exception:
            return sub
        out = (out or "").strip().strip('"')
        # Lấy dòng đầu, bỏ thừa; rỗng -> giữ ý gốc.
        out = out.splitlines()[0].strip() if out else ""
        return out or sub

    return _rewrite


def is_multi_intent(question: str) -> bool:
    """[1] Lọc RẺ: câu này có "tín hiệu đa ý" đáng cho Qwen kiểm tra không.

    Chỉ dùng để TIẾT KIỆM (tránh gọi Qwen cho câu đơn rõ ràng); KHÔNG
    quyết định số ý. Gate CHỈ quyết "có gọi LLM phân tích không";
    kết luận cuối cùng đơn-ý/đa-ý thuộc Planner/LLM.

    Tín hiệu:
    - >=2 số hiệu vb KHÁC NHAU: mạnh (có thể 2 ý, cũng có thể 1 ý quan
      hệ sửa đổi giữa 2 vb) -> gọi LLM, LLM phán quyết.
    - >=2 số Điều: mạnh (user hỏi 2 provision riêng) -> gọi LLM.
    - Nhiều động từ / từ khóa chủ đề KHÁC NHAU: chỉ BỔ TRỢ, không tự
      kết luận đa ý.
    - Từ nối + vế sau là mệnh đề hỏi độc lập ("...không?" "...là ai"):
      BỔ TRỢ -> gọi LLM.

    Sai lệch chấp nhận được: false-positive chỉ tốn 1 Qwen rồi dừng;
    false-negative thì trả lời như câu đơn (không mất đúng).
    """
    q = (question or "").strip()
    hits = 0
    n_joins = len(re.findall(r"(?:\bvà\b|\bcũng\b|\bcùng\b|\bthêm\b|ngoài ra|đồng thời)", q, re.I))
    if n_joins >= 2:
        hits += 1
    n_articles = len(re.findall(r"điều\s+\d+", q, re.I))
    if n_articles >= 2:
        # ≥2 số điều là TÍN HIỆU MẠNH (user hỏi 2 provision riêng) ->
        # gọi LLM (LLM vẫn phán quyết số ý, vd 2 vb quan hệ = 1 ý).
        hits += 2
    # Strong separator patterns: a semicolon followed by another question,
    # or an explicit comparison of two legal time points. The LLM still
    # decides whether these are separate intents.
    if re.search(r";\s*(?:nếu|đồng thời|và|sau đó|trong trường hợp)\b", q, re.I):
        hits += 2
    if re.search(
        r"trước\s+(?:ngày\s+)?\d{1,2}/\d{1,2}/\d{4}.*?\b(?:và|;).*?"
        r"\b(?:từ|kể từ|sau)\s+(?:ngày\s+)?\d{1,2}/\d{1,2}/\d{4}",
        q, re.I | re.S,
    ):
        hits += 2
    # >=2 số hiệu văn bản KHÁC NHAU -> TÍN HIỆU MẠNH: có thể 2 ý, cũng
    # có thể 1 ý quan hệ (vb A sửa vb B) -> gọi LLM phán quyết, gate
    # không tự kết luận. (?<!/) tránh đếm "mm/yyyy" trong ngày dd/mm/yyyy.
    n_docs = len({
        m.group(1).upper()
        for m in re.finditer(
            r"\b(?<!/)(\d{1,3}/\d{4})(?:/(?:QH\d{1,2}|NĐ-?CP))?", q, re.I
        )
    })
    if n_docs >= 2:
        hits += 2
    # Nhiều động từ hỏi / chuyên từ khác nhau (CHỦ ĐỀ: thuế, giảm, đối
    # tượng, trách nhiệm...) -> chỉ BỔ TRỢ.
    verbs = re.findall(
        r"(quy định|bao nhiêu|thế nào|khi nào|đăng ký|nộp|phạt|hiệu lực|mức thuế|mức|điều kiện|giảm|thuế|đối tượng|trách nhiệm)",
        q, re.I,
    )
    if len(set(v.lower() for v in verbs)) >= 2:
        hits += 1
    # Câu có dấu phân tách ý rõ: "và ... bao nhiêu" / "và ... thế nào"
    # (1 từ nối + nhiều chủ đề) -> đủ nghi đa ý (vẫn chỉ gọi LLM).
    if n_joins >= 1 and len(set(v.lower() for v in verbs)) >= 2:
        hits += 1
    # Từ nối + vế SAU là mệnh đề hỏi độc lập ("...không?" / "...là ai")
    # -> BỔ TRỢ (vd "A là ai, và B có được giảm không?").
    last_join = q.rfind("và")
    if last_join != -1:
        tail = q[last_join + 2:]
        if re.search(r"\b(không|ai|se|bao nhiêu|thế nào|khi nào)\b", tail, re.I):
            hits += 1
    # Dấu tách ý rõ: "1)... 2)..." / "trước hết ... sau đó ..."
    if re.search(r"\d\)\s*.\{0,40}\d\)", q, re.S):
        hits += 1
    if hits >= MIN_MULTI_SIGNALS:
        return True
    return False


def format_intent_clarifications(intents: list[dict[str, Any]]) -> str:
    """Present independent per-intent questions together in one user turn."""
    pending = [
        item for item in intents
        if item.get("needs_clarification") and str(item.get("clarification") or "").strip()
    ]
    if len(pending) == 1:
        return str(pending[0]["clarification"]).strip()
    return "\n".join(
        f"{index}) {str(item['clarification']).strip()}"
        for index, item in enumerate(pending, start=1)
    )


def parse_intent_clarification_answers(
    pending_intent_ids: list[str], user_answer: str
) -> dict[str, str]:
    """Map numbered one-turn answers to pending intents without guessing."""
    ids = [str(item) for item in pending_intent_ids if str(item)]
    answer = str(user_answer or "").strip()
    if not ids or not answer:
        return {}
    if len(ids) == 1:
        return {ids[0]: answer}

    parsed: dict[int, list[str]] = {}
    current: int | None = None
    for line in answer.splitlines():
        match = re.match(r"^\s*(\d+)[.)]\s*(.*)$", line)
        if match:
            current = int(match.group(1))
            parsed.setdefault(current, []).append(match.group(2).strip())
        elif current is not None and line.strip():
            parsed[current].append(line.strip())
    return {
        ids[number - 1]: text
        for number, parts in parsed.items()
        if 1 <= number <= len(ids) and (text := "\n".join(parts).strip())
    }


def coverage_prompt(question: str, evidence_lines: list[str]) -> str:
    """[2] Dựng prompt kiểm tra độ phủ."""
    ev = "\n".join(f"- {ln}" for ln in evidence_lines) or "- (chưa có)"
    return "\n".join(
        [
            _COVERAGE_SYSTEM,
            "",
            "=== CÂU HỎI ===",
            question.strip(),
            "",
            "=== CĂN CỨ ĐÃ TÌM ĐƯỢC ===",
            ev,
            "",
            "=== JSON ===",
        ]
    )


def _parse_coverage(raw: str) -> dict[str, Any] | None:
    return parse_refiner_json(raw)


class QueryDecomposer:
    """Gap-filling cho câu nhiều ý.

    `runner(sub_question)` -> list[evidence] (bắt buộc: cách chạy
    pipeline cho 1 sub-question). `llm` -> Qwen local kiểm tra độ phủ.
    """

    def __init__(
        self,
        llm: Any | None = None,
        max_rounds: int = MAX_ROUNDS,
        clarify_question: str = "",
        sub_query_rewriter: Callable[[str, str], str] | None = None,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.clarify_question = clarify_question
        self.sub_query_rewriter = sub_query_rewriter

    @staticmethod
    def _single_intent_scope_hint(text: str) -> dict[str, Any] | None:
        """Record a clarification hint for an intent missing document or essential scope."""
        question = (text or "").strip()
        if not question:
            return None
        has_provision = bool(re.search(
            r"(?i)\b(?:điểm|khoản|điều)\s*[0-9]", question
        ))
        has_document = bool(extract_anchors(question).doc_numbers)
        if has_provision and not has_document:
            return {
                "needs_clarification": True,
                "ambiguity_type": "document",
                "missing_scope": ["document"],
                "clarification": (
                    "Bạn muốn tra Điều/Khoản/Điểm này của văn bản nào? "
                    "Vui lòng cho tên hoặc số hiệu văn bản."
                ),
            }
        vague = bool(re.search(
            r"(?i)^(?:cho tôi kết luận pháp lý cụ thể|hãy cho tôi kết luận pháp lý|"
            r"kết luận pháp lý)\b",
            question,
        ))
        if vague:
            return {
                "needs_clarification": True,
                "ambiguity_type": "scope",
                "missing_scope": ["scope"],
                "clarification": (
                    "Vui lòng nêu vấn đề pháp lý cụ thể, văn bản/điều khoản "
                    "liên quan và dữ kiện vụ việc nếu có."
                ),
            }

        # Open tax queries lacking both document anchor and tax/taxpayer scope
        broad_tax_query = bool(re.search(
            r"(?i)\b(?:nghĩa vụ\s+kê\s+khai\s+thuế|kê\s+khai\s+thuế|nộp\s+thuế|hoàn\s+thuế|"
            r"miễn\s+thuế|giảm\s+thuế|thuế\s+suất|đóng\s+góp\s+từ\s+thiện)\b",
            question,
        ))
        if broad_tax_query and not has_document:
            missing: list[str] = []
            if not re.search(
                r"(?i)(?:thuế\s+(?:TNCN|TNDN|GTGT|môn\s+bài|tiêu\s+thụ\s+đặc\s+biệt|"
                r"xuất\s+nhập\s+khẩu|thu\s+nhập\s+cá\s+nhân|thu\s+nhập\s+doanh\s+nghiệp|"
                r"giá\s+trị\s+gia\s+tăng)|GTGT|TNDN|TNCN)",
                question,
            ):
                missing.append("tax_type")
            if not re.search(
                r"(?i)(?:cá\s+nhân|doanh\s+nghiệp|công\s+ty|hộ\s+kinh\s+doanh|tổ\s+chức|người\s+nộp\s+thuế)",
                question,
            ):
                missing.append("taxpayer")
            if re.search(r"(?i)(?:đóng\s+góp\s+từ\s+thiện|khoản\s+từ\s+thiện)", question) and not re.search(
                r"(?i)(?:ủng\s+hộ|quyên\s+góp|từ\s+thiện\s+(?:cho|tại|đến)\s+\w+)", question
            ):
                missing.append("donation_type")
            if missing:
                labels = {
                    "tax_type": "sắc thuế (ví dụ: thuế GTGT, TNDN, TNCN)",
                    "taxpayer": "đối tượng nộp thuế (doanh nghiệp, hộ kinh doanh hay cá nhân)",
                    "donation_type": "loại khoản đóng góp",
                }
                return {
                    "needs_clarification": True,
                    "ambiguity_type": "scope",
                    "missing_scope": missing,
                    "clarification": "Vui lòng nêu rõ "
                        + " và ".join(labels[item] for item in missing) + ".",
                }

        # Open enterprise registration queries lacking both document anchor and enterprise type
        broad_enterprise_query = bool(re.search(
            r"(?i)\b(?:điều\s+kiện\s+đăng\s+ký\s+doanh\s+nghiệp|thủ\s+tục\s+đăng\s+ký\s+doanh\s+nghiệp|"
            r"thủ\s+tục\s+thành\s+lập\s+doanh\s+nghiệp|hồ\s+sơ\s+đăng\s+ký\s+doanh\s+nghiệp)\b",
            question,
        ))
        if broad_enterprise_query and not has_document:
            has_subject = bool(re.search(
                r"(?i)(?:TNHH|trách\s+nhiệm\s+hữu\s+hạn|cổ\s+phần|tư\s+nhân|hợp\s+danh|hộ\s+kinh\s+doanh|chi\s+nhánh|văn\s+phòng\s+đại\s+diện)",
                question,
            ))
            if not has_subject:
                return {
                    "needs_clarification": True,
                    "ambiguity_type": "scope",
                    "missing_scope": ["subject", "document"],
                    "clarification": (
                        "Bạn muốn tra cứu cho loại hình doanh nghiệp nào "
                        "(công ty TNHH, công ty cổ phần, doanh nghiệp tư nhân...) "
                        "hoặc theo văn bản cụ thể nào?"
                    ),
                }
        return None
        """Record a clarification hint for an intent missing document or essential scope."""
        question = (text or "").strip()
        if not question:
            return None
        has_provision = bool(re.search(
            r"(?i)\b(?:điểm|khoản|điều)\s*[0-9]", question
        ))
        has_document = bool(extract_anchors(question).doc_numbers)
        if has_provision and not has_document:
            return {
                "needs_clarification": True,
                "ambiguity_type": "document",
                "missing_scope": ["document"],
                "clarification": (
                    "Bạn muốn tra Điều/Khoản/Điểm này của văn bản nào? "
                    "Vui lòng cho tên hoặc số hiệu văn bản."
                ),
            }
        vague = bool(re.search(
            r"(?i)^(?:cho tôi kết luận pháp lý cụ thể|hãy cho tôi kết luận pháp lý|"
            r"kết luận pháp lý)\b",
            question,
        ))
        if vague:
            return {
                "needs_clarification": True,
                "ambiguity_type": "scope",
                "missing_scope": ["scope"],
                "clarification": (
                    "Vui lòng nêu vấn đề pháp lý cụ thể, văn bản/điều khoản "
                    "liên quan và dữ kiện vụ việc nếu có."
                ),
            }

        # Open tax obligation / declaration / rate / exemption queries lacking both doc anchor and tax/taxpayer scope
        broad_tax_query = bool(re.search(
            r"(?i)\b(?:nghĩa vụ\s+kê\s+khai\s+thuế|kê\s+khai\s+thuế|nộp\s+thuế|hoàn\s+thuế|"
            r"miễn\s+thuế|giảm\s+thuế|thuế\s+suất|đóng\s+góp\s+từ\s+thiện)\b",
            question,
        ))
        if broad_tax_query and not has_document:
            missing: list[str] = []
            if not re.search(
                r"(?i)(?:thuế\s+(?:TNCN|TNDN|GTGT|môn\s+bài|tiêu\s+thụ\s+đặc\s+biệt|"
                r"xuất\s+nhập\s+khẩu|thu\s+nhập\s+cá\s+nhân|thu\s+nhập\s+doanh\s+nghiệp|"
                r"giá\s+trị\s+gia\s+tăng)|GTGT|TNDN|TNCN)",
                question,
            ):
                missing.append("tax_type")
            if not re.search(
                r"(?i)(?:cá\s+nhân|doanh\s+nghiệp|công\s+ty|hộ\s+kinh\s+doanh|tổ\s+chức|người\s+nộp\s+thuế)",
                question,
            ):
                missing.append("taxpayer")
            if re.search(r"(?i)(?:đóng\s+góp\s+từ\s+thiện|khoản\s+từ\s+thiện)", question) and not re.search(
                r"(?i)(?:ủng\s+hộ|quyên\s+góp|từ\s+thiện\s+(?:cho|tại|đến)\s+\w+)", question
            ):
                missing.append("donation_type")
            if missing:
                labels = {
                    "tax_type": "sắc thuế (ví dụ: thuế GTGT, TNDN, TNCN)",
                    "taxpayer": "đối tượng nộp thuế (doanh nghiệp, hộ kinh doanh hay cá nhân)",
                    "donation_type": "loại khoản đóng góp",
                }
                return {
                    "needs_clarification": True,
                    "ambiguity_type": "scope",
                    "missing_scope": missing,
                    "clarification": "Vui lòng nêu rõ "
                        + " và ".join(labels[item] for item in missing) + ".",
                }

        # Open enterprise registration / establishment queries lacking both doc anchor and enterprise type
        broad_enterprise_query = bool(re.search(
            r"(?i)\b(?:điều\s+kiện\s+đăng\s+ký\s+doanh\s+nghiệp|thủ\s+tục\s+đăng\s+ký\s+doanh\s+nghiệp|"
            r"thủ\s+tục\s+thành\s+lập\s+doanh\s+nghiệp|hồ\s+sơ\s+đăng\s+ký\s+doanh\s+nghiệp)\b",
            question,
        ))
        if broad_enterprise_query and not has_document:
            has_subject = bool(re.search(
                r"(?i)(?:TNHH|trách\s+nhiệm\s+hữu\s+hạn|cổ\s+phần|tư\s+nhân|hợp\s+danh|hộ\s+kinh\s+doanh|chi\s+nhánh|văn\s+phòng\s+đại\s+diện)",
                question,
            ))
            if not has_subject:
                return {
                    "needs_clarification": True,
                    "ambiguity_type": "scope",
                    "missing_scope": ["subject", "document"],
                    "clarification": (
                        "Bạn muốn tra cứu cho loại hình doanh nghiệp nào "
                        "(công ty TNHH, công ty cổ phần, doanh nghiệp tư nhân...) "
                        "hoặc theo văn bản cụ thể nào?"
                    ),
                }
        return None
    def rewrite_clarified_intent(
        self,
        *,
        original_question: str,
        original_sub_query: str,
        clarification: str,
        user_answer: str,
    ) -> tuple[str, dict[str, Any]]:
        """Create one auditable standalone query for a clarified intent."""
        llm = self.llm
        if llm is None or getattr(llm, "name", "") in ("none", ""):
            return "", {"status": "ERROR", "reason": "rewrite_llm_unavailable"}

        source = "\n".join(part for part in (
            original_sub_query.strip(),
            f"Câu hỏi làm rõ: {clarification.strip()}" if clarification.strip() else "",
            f"Thông tin người dùng: {user_answer.strip()}" if user_answer.strip() else "",
        ) if part)
        context = self._context(original_question, [])
        context += "\n\nÝ cần xử lý độc lập: " + source
        try:
            setter = getattr(llm, "set_purpose", None)
            if setter:
                setter("clarified_intent_rewrite")
            rewritten = str(llm.complete(build_rewriter_prompt(source, context)) or "")
            rewritten = rewritten.strip().strip('"').splitlines()[0].strip()
        except Exception as exc:
            return "", {"status": "ERROR", "reason": "rewrite_llm_error"}
        if not rewritten:
            return "", {"status": "ERROR", "reason": "rewrite_empty"}

        original_anchors = extract_anchors(original_sub_query)
        allowed_anchors = extract_anchors(source)
        rewritten_anchors = extract_anchors(rewritten)
        fields = ("doc_numbers", "articles", "clauses", "points", "dates")
        has_original_anchor = any(getattr(original_anchors, field) for field in fields)
        missing = {
            field: [value for value in getattr(original_anchors, field)
                    if value not in getattr(rewritten_anchors, field)]
            for field in fields
        }
        added = {
            field: [value for value in getattr(rewritten_anchors, field)
                    if value not in getattr(allowed_anchors, field)]
            for field in fields
        }
        missing = {key: value for key, value in missing.items() if value}
        added = {key: value for key, value in added.items() if value}
        if not has_original_anchor:
            missing = {}
        audit = {
            "status": "REJECTED" if missing or added else "ANCHORS_VALIDATED",
            "original_sub_query": original_sub_query,
            "clarification": clarification,
            "user_answer": user_answer,
            "rewritten_query": rewritten,
            "missing_anchors": missing,
            "unsupported_anchors": added,
            "semantic_review": "not_proven_automatically",
        }
        return ("" if missing or added else rewritten), audit

    # -- API chính -------------------------------------------------------

    def should_check(self, question: str) -> bool:
        """Có đáng gọi Qwen kiểm tra độ phủ không (lọc rẻ)?"""
        if self.llm is None:
            return False
        return is_multi_intent(question)

    def _rule_based_multi_intent_plan(self, question: str) -> dict[str, Any] | None:
        """Fallback chỉ dùng cho các split không mơ hồ.

        Áp dụng khi LLM planner lỗi nhưng câu có ít nhất hai chuẩn văn bản
        khác nhau, một vế rõ ràng ngoài phạm vi pháp luật kèm một vế pháp lý
        có anchor, hoặc một vế có anchor kèm một vế thiếu phạm vi cần làm rõ.
        Không dùng heuristic này thay cho planner trên câu thiếu dữ kiện
        hoặc câu nội dung pháp lý không rõ.
        """
        temporal_matches = list(re.finditer(
            r"(?i)(?:trước|từ|kể từ|sau|tại|vào)\s+(?:ngày\s+)?"
            r"\d{1,2}/\d{1,2}/\d{4}",
            question,
        ))
        if len(temporal_matches) >= 2 and re.search(r"(?i)\bso\s+sánh\b", question):
            intents: list[dict[str, Any]] = []
            article_match = re.search(r"(?i)Điều\s+\d+[a-z]?", question)
            doc_match = re.search(
                r"\d{1,3}/\d{4}/(?:NĐ-CP|NĐ|QH\d+|TT-[A-Z0-9-]+)",
                question,
                re.I,
            )
            doc_label_match = re.search(r"(?i)(Nghị định|Luật|Thông tư)\s+", question)
            anchor_prefix = " ".join(
                part for part in (
                    article_match.group(0).strip() if article_match else "",
                    f"{doc_label_match.group(1)} {doc_match.group(0)}".strip()
                    if doc_label_match and doc_match else "",
                ) if part
            )
            if not anchor_prefix and doc_match:
                anchor_prefix = doc_match.group(0)
            for index, match in enumerate(temporal_matches[:2], start=1):
                sub = (
                    f"{anchor_prefix} {match.group(0)} "
                    f"{'quy định những khoản nào' if index == 1 else 'được bổ sung những khoản nào'}"
                ).strip()
                constraint = parse_temporal_constraint(sub)
                intents.append({
                    "intent_id": f"intent-{index}",
                    "intent_index": index - 1,
                    "sub_query": sub,
                    "original_sub_query": sub,
                    "original_sub_question": sub,
                    "source_clause": sub,
                    "is_out_of_domain": False,
                    "needs_clarification": False,
                    "clarification": "",
                    "ambiguity_type": "",
                    "missing_scope": [],
                    "temporal_expression": match.group(0),
                    "temporal_constraint": constraint,
                    "temporal_operator": constraint.get("operator") if constraint else None,
                    "raw_date": constraint.get("raw_date") if constraint else None,
                    "normalized_target_date": constraint.get("normalized_target_date") if constraint else None,
                    "temporal_constraint_source": constraint.get("temporal_constraint_source") if constraint else None,
                })
            return {"planned": True, "status": "planned", "clarification_round": 0, "intents": intents}

        parts = [
            part.strip(" ,;:-")
            for part in re.split(
                r"\b(?:và|cũng như)\b|;",
                question,
                flags=re.I,
            )
            if part.strip(" ,;:-")
        ]
        if len(parts) < 2:
            return None

        def out_of_domain(text: str) -> bool:
            return bool(re.search(
                r"(?i)\b(?:thời tiết|bản tin thời tiết|dự báo)\b"
                r"|ngoài phạm vi pháp luật|out of domain",
                text,
            ))

        doc_sets = [set(extract_anchors(part).doc_numbers) for part in parts]
        distinct_docs = set().union(*doc_sets)
        ood_count = sum(1 for part in parts if out_of_domain(part))
        scope_hints = [
            None if out_of_domain(part) else self._single_intent_scope_hint(part)
            for part in parts
        ]
        has_semicolon_split = bool(re.search(r";", question))
        complete_semicolon_parts = has_semicolon_split and len(parts) == 2 and all(
            re.search(r"(?:gì|nào|thế nào|\?)", part, re.I)
            for part in parts
        )
        if len(distinct_docs) < 2 and not (
            (ood_count == 1 and any(doc_sets))
            or complete_semicolon_parts
            or (any(doc_sets) and any(scope_hints))
        ):
            return None

        intents: list[dict[str, Any]] = []
        for index, part in enumerate(parts[:12], start=1):
            is_ood = out_of_domain(part)
            hint = scope_hints[index - 1] if not is_ood else None
            intents.append({
                "intent_id": f"intent-{index}",
                "intent_index": index - 1,
                "sub_query": part,
                "original_sub_query": part,
                "original_sub_question": part,
                "source_clause": part,
                "is_out_of_domain": is_ood,
                "needs_clarification": bool(hint),
                "clarification": str((hint or {}).get("clarification") or ""),
                "ambiguity_type": str((hint or {}).get("ambiguity_type") or ""),
                "missing_scope": list((hint or {}).get("missing_scope") or []),
                "temporal_expression": "",
                "temporal_constraint": parse_temporal_constraint(part),
            })
        return {"planned": True, "status": "planned", "clarification_round": 0, "intents": intents}
    @staticmethod
    def _is_actionable_subquestion(part: str) -> bool:
        """A split fragment usable as an independent retrieval target."""
        text = (part or "").strip()
        if len(text) < 8:
            return False
        if re.search(r"(?i)(?:gì|nào|thế nào|như thế nào|\?|？)", text):
            return True
        if re.search(r"(?i)\b(?:điểm|khoản|điều)\s*\d+|\d{1,3}/\d{4}/", text):
            return True
        return False

    def _permissive_split_plan(self, question: str) -> dict[str, Any] | None:
        """Planner-error fallback (Fix E): split a multi-intent question into
        clear sub-questions on conjunctions/semicolons so one broken intent
        does not fail the whole question. Clear intents go to retrieval;
        only intents actually missing data get a clarification hint.

        Used ONLY when the LLM planner errored/failed to parse (the strict
        rule plan already ran and returned None). Not a replacement for a
        successful planner.
        """
        parts = [
            part.strip(" ,;:-")
            for part in re.split(r"\b(?:và|cũng như)\b|;", question, flags=re.I)
            if part.strip(" ,;:-")
        ]
        kept = [part for part in parts if self._is_actionable_subquestion(part)]
        if len(kept) < 2:
            # Secondary try: comma-separated short clauses (e.g. "nghĩa vụ thuế,
            # thời hạn, mức phạt ...?").
            comma_parts = [
                part.strip(" ,;:-")
                for part in re.split(r",", question)
                if part.strip(" ,;:-")
            ]
            kept = [p for p in comma_parts if len(p) >= 8]
            if len(kept) < 2:
                return None
        intents: list[dict[str, Any]] = []
        for index, part in enumerate(kept[:4], start=1):
            is_ood = bool(re.search(
                r"(?i)\b(?:thời tiết|bản tin thời tiết|dự báo)\b"
                r"|ngoài phạm vi pháp luật|out of domain",
                part,
            ))
            intent: dict[str, Any] = {
                "intent_id": f"intent-{index}",
                "intent_index": index - 1,
                "sub_query": part,
                "original_sub_query": part,
                "original_sub_question": part,
                "source_clause": part,
                "is_out_of_domain": is_ood,
                "needs_clarification": False,
                "clarification": "",
                "temporal_expression": "",
                "temporal_constraint": parse_temporal_constraint(part),
            }
            # Only the genuinely missing intents are flagged for clarification.
            hint = self._single_intent_scope_hint(part)
            if hint and not is_ood:
                intent.update(hint)
            intents.append(intent)
        return {
            "planned": True,
            "status": "planned",
            "intents": intents,
            "planner_fallback": "permissive_split",
            "planner_diagnostic": {"fallback": "permissive_split"},
        }

    def plan(self, question: str) -> dict[str, Any]:
        """Create an intent plan before retrieval; never invent missing anchors."""
        question = (question or "").strip()
        fallback = [{
            "intent_id": "intent-1",
            "sub_query": question,
            "original_sub_query": question,
            "is_out_of_domain": False,
            "needs_clarification": False,
            "clarification": "",
            "temporal_expression": "",
        }]
        # Scope hints (document/scope/temporal) are recorded on the intent so the
        # clarification wizard can target the missing field. They do NOT force a
        # clarification here: the context resolver runs before planning in
        # Orchestrator.run, so a follow-up that inherits its document from the
        # previous turn reaches this gate already carrying a doc number and is
        # left to semantic retrieval. For multi-intent questions the gate is
        # deferred to the per-intent loop, which evaluates each sub-question.
        hint = self._single_intent_scope_hint(question)
        if hint and not is_multi_intent(question):
            fallback[0].update(hint)
            return {"planned": True, "status": "single_intent", "intents": fallback}
        if not self.should_check(question):
            return {"planned": False, "status": "single_intent", "intents": fallback}

        rule_plan = self._rule_based_multi_intent_plan(question)
        if rule_plan is not None:
            return rule_plan

        prompt = """Tách câu hỏi pháp lý thành từng yêu cầu độc lập TRƯỚC KHI TRA CỨU.
Chỉ tách nội dung người dùng đã nêu; không thêm số hiệu văn bản, điều khoản,
giả định sự kiện hoặc câu trả lời. Mỗi ý phải có sub_query riêng. Nếu một ý
thiếu thông tin thiết yếu để tra đúng, đánh dấu cần làm rõ ý đó riêng; các ý
đủ thông tin vẫn được tra cứu độc lập. Nếu cần làm rõ, nêu gộp trong một câu
hỏi lại TẤT CẢ dữ kiện thiết yếu còn thiếu cho chính ý đó; không hỏi lại dữ
kiện người dùng đã cung cấp và không hỏi chung chung khi có thể nêu rõ từng mục.
Mỗi intent phải có source_clause là đoạn nguyên văn ngắn nhất trong câu hỏi
gốc tương ứng với ý đó. Khi hai ý cùng chủ thể/thủ tục, hãy giữ chủ thể đó
trong từng sub_query để mỗi câu con tự đứng độc lập. Không hỏi số hiệu văn bản
chỉ vì người dùng không nêu số hiệu nếu chủ thể và thủ tục đã rõ.
Không chép số hiệu văn bản từ source_clause của intent khác sang intent này,
trừ khi câu gốc nêu rõ số hiệu đó áp dụng chung cho cả hai ý.
Ý ngoài phạm vi pháp luật nhưng đã rõ nội dung không phải là ý thiếu dữ kiện:
đánh dấu không cần làm rõ để bước retrieval phân loại OOD riêng. Khi người dùng
so sánh cùng một quy định ở hai mốc ngày, tạo intent riêng cho mỗi mốc và giữ
nguyên ngày/mốc áp dụng tương ứng trong từng câu con.
Với câu hỏi ưu đãi/miễn/giảm thuế từ khoản đóng góp từ thiện của cá nhân, nếu
ý đó chưa nêu sắc thuế, đối tượng nộp thuế, hoặc loại khoản đóng góp thì hỏi
làm rõ riêng ý đó và hỏi đủ các dữ kiện còn thiếu trong một câu.
Trả về duy nhất JSON:
For clear out-of-domain intents, set is_out_of_domain=true and leave clarification empty.
For two-date comparisons, create separate intents with their own temporal_expression.
Copy each time phrase exactly as written; do not reuse one intent's time constraint for another.
{"intents":[{"source_clause":"...","sub_query":"...","is_out_of_domain":false,"needs_clarification":false,"clarification":"","temporal_expression":""}]}

CÂU HỎI:
""" + question
        raw_output = ""
        try:
            setter = getattr(self.llm, "set_purpose", None)
            if setter:
                setter("decomposition_plan")
            raw_output = str(self.llm.complete(prompt) or "")
        except Exception as exc:
            if rule_plan is not None:
                return rule_plan
            permissive = self._permissive_split_plan(question)
            if permissive is not None:
                permissive["planner_diagnostic"] = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:240],
                    "fallback": "permissive_split",
                }
                return permissive
            return {
                "planned": True, "status": "llm_error", "intents": fallback,
                "planner_diagnostic": {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:240],
                },
            }
        safe_output = re.sub(
            r"<think\b[^>]*>.*?</think\s*>", "", raw_output,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()[:500]
        data = _parse_coverage(raw_output)
        if data is None:
            if rule_plan is not None:
                return rule_plan
            return {
                "planned": True, "status": "parse_error", "intents": fallback,
                "planner_diagnostic": {"raw_output": safe_output},
            }
        raw_intents = data.get("intents") if isinstance(data, dict) else None
        if not isinstance(raw_intents, list) or not raw_intents:
            if rule_plan is not None:
                return rule_plan
            return {
                "planned": True, "status": "invalid_plan", "intents": fallback,
                "planner_diagnostic": {"raw_output": safe_output},
            }
        temporal_matches = list(re.finditer(
            r"(?i)(?:\u0111\u1ebfn\s+tr\u01b0\u1edbc|\u0111\u1ebfn\s+(?:h\u1ebft\s+)?|"
            r"k\u1ec3\s+t\u1eeb|t\u1eeb|tr\u01b0\u1edbc|sau|t\u1ea1i|v\u00e0o)\s+"
            r"(?:ng\u00e0y\s+)?\d{1,2}/\d{1,2}/\d{4}",
            question,
        ))
        is_time_comparison = (
            len(temporal_matches) >= 2
            and bool(re.search(r"(?i)so\s+s\u00e1nh|\bv\u00e0\b", question))
        )
        if is_time_comparison:
            planned_dates = {
                (parse_temporal_constraint(
                    str(item.get("sub_query") or item.get("query") or "")
                ) or {}).get("normalized_target_date")
                for item in raw_intents if isinstance(item, dict)
            }
            if len(raw_intents) < 2 or len({value for value in planned_dates if value}) < 2:
                common = question[:temporal_matches[0].start()].strip(" ,;:-")
                raw_intents = [{
                    "source_clause": question,
                    "sub_query": f"{common} {match.group(0)}".strip(),
                    "temporal_expression": match.group(0),
                    "needs_clarification": False,
                } for match in temporal_matches[:2]]

        intents: list[dict[str, Any]] = []
        seen: set[str] = set()
        for intent_number, raw in enumerate(raw_intents[:12], start=1):
            if isinstance(raw, str):
                raw = {"sub_query": raw}
            if not isinstance(raw, dict):
                continue
            sub = str(raw.get("sub_query") or raw.get("query") or "").strip()
            if not sub or sub in seen:
                continue
            seen.add(sub)
            source_clause = str(raw.get("source_clause") or "").strip()
            if source_clause and source_clause.casefold() not in question.casefold():
                source_clause = ""
            if not source_clause:
                source_clause = question
            source_anchors = set(extract_anchors(source_clause).doc_numbers)
            elsewhere_anchors = set().union(*(
                set(extract_anchors(str(other.get("source_clause") or "")).doc_numbers)
                for other in raw_intents if isinstance(other, dict) and other is not raw
            )) if len(raw_intents) > 1 else set()
            sub_anchors = set(extract_anchors(sub).doc_numbers)
            for doc_number in sub_anchors - source_anchors:
                if doc_number not in elsewhere_anchors:
                    continue
                sub = re.sub(
                    rf"(?i)(?:luật|nghị định|thông tư|văn bản số)\s*{re.escape(doc_number)}|{re.escape(doc_number)}",
                    "", sub,
                )
                sub = re.sub(r"\s{2,}", " ", sub).strip(" ,;:.-")
            clarification = str(raw.get("clarification") or "").strip()
            ambiguity_type = str(raw.get("ambiguity_type") or "").strip()
            missing_scope_raw = raw.get("missing_scope")
            if isinstance(missing_scope_raw, (list, tuple)):
                missing_scope = [str(value).strip() for value in missing_scope_raw if str(value).strip()]
            elif missing_scope_raw:
                missing_scope = [str(missing_scope_raw).strip()]
            else:
                missing_scope = []
            ood_text = bool(re.search(
                r"(?i)ngo\u00e0i\s+ph\u1ea1m\s+vi|ngo\u00e0i\s+l\u0129nh\s+v\u1ef1c|"
                r"kh\u00f4ng\s+thu\u1ed9c\s+ph\u1ea1m\s+vi|kh\u00f4ng\s+ph\u1ea3i\s+c\u00e2u\s+h\u1ecfi\s+ph\u00e1p\s+lu\u1eadt|\bOOD\b|out\s+of\s+domain",
                clarification,
            ))
            raw_ood = raw.get("is_out_of_domain")
            is_out_of_domain = (
                raw_ood is True or str(raw_ood).strip().lower() in {"true", "yes"}
                or ood_text
                or bool(re.search(
                    r"(?i)\b(?:thời tiết|bản tin thời tiết|dự báo)\b",
                    sub,
                ))
            )
            needs = bool(raw.get("needs_clarification") or clarification)
            if is_out_of_domain:
                needs = False
                clarification = ""
            temporal_expression = str(raw.get("temporal_expression") or "").strip()
            normalized_time = parse_temporal_constraint(
                sub if not temporal_expression else f"{sub} {temporal_expression}"
            )
            tax_donation = bool(re.search(
                r"(?i)(?:đóng góp từ thiện|khoản từ thiện).*(?:giảm thuế|ưu đãi thuế|miễn thuế)",
                sub,
            ))
            if tax_donation:
                required = {
                    "tax_type": bool(re.search(r"(?i)thuế\s+(?:TNCN|TNDN|GTGT|thu nhập cá nhân|thu nhập doanh nghiệp)", source_clause)),
                    "taxpayer": bool(re.search(r"(?i)cá nhân|doanh nghiệp|hộ kinh doanh|người nộp thuế", source_clause)),
                    "donation_type": bool(re.search(r"(?i)ủng hộ|quyên góp|từ thiện\s+(?:cho|tại|đến)\s+\w+", source_clause)),
                }
                missing = [key for key, present in required.items() if not present]
                if missing:
                    needs = True
                    questions = {
                        "tax_type": "sắc thuế đang hỏi",
                        "taxpayer": "đối tượng nộp thuế",
                        "donation_type": "loại khoản đóng góp từ thiện",
                    }
                    clarification = "Vui lòng cho biết " + ", ".join(questions[k] for k in missing) + "."
            if needs and not clarification:
                clarification = f"Bạn có thể làm rõ ý này: {sub}?"
            scope_hint = self._single_intent_scope_hint(sub)
            if scope_hint and not is_out_of_domain and not is_time_comparison:
                if not needs:
                    needs = True
                    clarification = scope_hint["clarification"]
                if not ambiguity_type:
                    ambiguity_type = str(scope_hint.get("ambiguity_type") or "")
                if not missing_scope:
                    missing_scope = list(scope_hint.get("missing_scope") or [])
            temporal_constraint = normalized_time
            intents.append({
                "intent_id": str(raw.get("intent_id") or f"intent-{intent_number}"),
                "intent_index": len(intents),
                "sub_query": sub,
                "original_sub_query": sub,
                "original_sub_question": sub,
                "source_clause": source_clause,
                "is_out_of_domain": is_out_of_domain,
                "needs_clarification": needs,
                "clarification": clarification if needs else "",
                "ambiguity_type": ambiguity_type,
                "missing_scope": missing_scope,
                "temporal_expression": temporal_expression or str(
                    (temporal_constraint or {}).get("temporal_expression") or ""
                ),
                "temporal_constraint": temporal_constraint,
                "temporal_operator": temporal_constraint.get("operator") if temporal_constraint else None,
                "raw_date": temporal_constraint.get("raw_date") if temporal_constraint else None,
                "normalized_target_date": temporal_constraint.get("normalized_target_date") if temporal_constraint else None,
                "temporal_constraint_source": temporal_constraint.get("temporal_constraint_source") if temporal_constraint else None,
            })
        if not intents:
            if rule_plan is not None:
                return rule_plan
            return {"planned": True, "status": "invalid_plan", "intents": fallback,
                    "planner_diagnostic": {"raw_output": safe_output}}
        return {"planned": True, "status": "planned", "clarification_round": 0, "intents": intents}

    def fill(
        self,
        question: str,
        evidence: list[Any],
        runner: Callable[[str], list[Any]],
        evidence_cap: int = MAX_EVIDENCE,
    ) -> tuple[list[Any], dict[str, Any]]:
        """[3] Lấp các ý con CHƯA có căn cứ trong `evidence`.

        Trả (evidence_gộp, meta). Không đổi gì (no-op) nếu:
          - không LLM, hay câu không đa ý (should_check False).
          - Qwen nói không còn ý thiếu.
          - hết ngân sách (vòng / trần evidence).

        F1 (chấm theo NODE, không theo chữ): ý con có "số hiệu vb +
        Điều/Khoản" -> tính node CẦN TÌM (tất định); chỉ "covered" khi
        node đó (hoặc node con) CÓ THẬT trong pool. Không để LLM khai
        "covered" vì nhìn thấy chữ "Điều N" trong provision khác.

        F5b (mỗi ý = 1 câu đơn): meta["per_intent"] cho mỗi ý con gồm
        `sub_query` (đã rewriter) + `node_ids` (seed pool + evidence ý
        đó tự sinh) — để `_final_rerank` xếp hạng RIÊNG theo câu con,
        không để 1 ý chiếm hết top-5 của ý kia.

        meta có thể chứa thêm:
          - "clarify": CÂU HỎI GỘP (nếu LLM thấy có ý con thiếu thông tin
            cần hỏi lại người dùng). Hệ thống thật gắn vào follow_up;
            test 1 lượt (không user) thì chỉ ghi nhận.
          - "rewritten": {ý con gốc: câu tra đã viết lại} — để quan sát.
        """
        if not self.should_check(question):
            seed = [
                str(getattr(e, "node_id", None) or getattr(e, "id", None))
                for e in evidence
            ]
            return evidence, {
                "filled": 0,
                "rounds": 0,
                "per_intent": [
                    {"sub_query": question, "node_ids": [s for s in seed if s]}
                ],
            }

        merged = list(evidence)
        seen = {getattr(e, "node_id", None) or getattr(e, "id", None) for e in merged}
        total_filled = 0
        meta: dict[str, Any] = {"filled": 0, "rounds": 0}
        rewritten: dict[str, str] = {}
        # F1 + F5b: theo dõi provenance "ý nào sinh evidence nào" (để _final
        # rerank theo từng ý) + "ý nào đã covered theo NODE".
        seed_ids = [str(nid) for nid in seen if nid is not None]
        own_ids: dict[str, set[str]] = {}
        intents: list[str] = []  # cập nhật mỗi vòng (dù parse lỗi -> [] cũ)

        # P-B (B1, micro-mốc #1): early-stop PER-INTENT theo "no-progress".
        # - no_progress_streak[X]: số vòng LIÊN TIẾP X uncovered MÀ X không
        #   kéo node MỚI (added==0). Đạt >=2 -> stop X (micro-mốc #1: per-intent).
        # - skipped_calls[X]: số rewriter call ĐÃ TIẾT KIỆM khi bỏ qua fill X
        #   ở các vòng sau (ghi trace, chứng minh giá trị early-stop).
        # - Intent đã stop KHÔNG làm gián đoạn fill của intent khác (độc lập).
        # - T7: X nhận node MỚI (added>0) -> streak RESET (thông tin mới có thể
        #   đổi kết quả) -> X được fill lại.
        no_progress_streak: dict[str, int] = {}
        skipped_calls: dict[str, int] = {}
        intent_exec: dict[str, str] = {}   # execution: OK/EMPTY/FAILED (trục 1)
        stopped: dict[str, dict[str, Any]] = {}  # X -> {round, exec, calls_saved, pending}
        pending: dict[str, str] = {}        # X -> "ood_gate" | "insufficient" | "failed"

        for round_idx in range(1, self.max_rounds + 1):
            meta["rounds"] = round_idx
            lines = [self._line(e) for e in merged]
            data = self._coverage(question, lines)
            if data is None:
                # Không parse được -> an toàn: dừng (không bịa).
                meta["stop_reason"] = "coverage_unparsed"
                break

            # Câu hỏi lại GỘP (nếu LLM thấy có ý thiếu thông tin). Ghi
            # nhận để hệ thống thật gắn vào follow_up; test 1 lượt chỉ
            # lưu, không ai trả lời (rewriter lo phần "viết lại").
            clarify = data.get("clarify")
            if clarify:
                meta["clarify"] = str(clarify).strip()

            llm_uncovered = [
                str(x).strip() for x in (data.get("uncovered") or []) if str(x).strip()
            ]
            intents = [
                str(x).strip() for x in (data.get("intents") or []) if str(x).strip()
            ]
            meta.setdefault("intents", intents)

            # F1: CHẤM THEO NODE, không theo chữ. Ý con có "số hiệu vb +
            # Điều/Khoản" -> tính node cần tìm (tất định); chỉ "covered"
            # khi node đó (hoặc khoản/điểm con của nó) CÓ THẬT trong pool.
            # (Bug cũ: LLM nhìn chữ "Điều 4" trong text của Điều 9 -> khai
            # covered -> rewriter không chạy.) Ý không có anchor -> tin LLM.
            found = {nid for nid in seen if nid is not None}
            uncovered: list[str] = []
            for it in intents:
                req = self._intent_required_nodes(it)
                if req:
                    if not any(
                        self._node_in_pool(r, f) for r in req for f in found
                    ):
                        uncovered.append(it)
                else:
                    if it in llm_uncovered:
                        uncovered.append(it)
            # Ý LLM liệt kê thiếu nhưng không nằm trong "intents" -> giữ.
            for x in llm_uncovered:
                if x not in intents and x not in uncovered:
                    uncovered.append(x)

            if not uncovered:
                meta["stop_reason"] = "covered"
                break
            # Lấp: chạy pipeline cho MỖI ý thiếu, gộp evidence mới.
            # P-B: BỎ QUA intent đã early-stop (no-progress) để tiết kiệm
            # LLM call; intent khác vẫn fill bình thường (độc lập).
            context = self._context(question, lines)
            added_this_round: dict[str, int] = {}
            for sub in uncovered:
                if len(merged) >= evidence_cap:
                    meta["stop_reason"] = "budget_evidence"
                    break
                if sub in stopped:
                    # T7 no-retry: intent đã stop, chưa có thông tin mới
                    # (vẫn no-progress) -> bỏ qua fill, ĐẾM LLM call tiết
                    # kiệm (1 rewriter call/vòng). Không chạy rewriter/runner.
                    skipped_calls[sub] = skipped_calls.get(sub, 0) + 1
                    continue
                # "Vẽ lại ý con": viết lại sub thành câu tra đủ ngữ cảnh
                # (bằng LLM, dùng câu cha + evidence đã tìm). Không LLM
                # / lỗi -> giữ sub nguyên (fallback an toàn, không bịa).
                sub_q = sub
                if self.sub_query_rewriter is not None:
                    out = self.sub_query_rewriter(sub, context)
                    if out and out != sub:
                        sub_q = out
                        rewritten[sub] = out
                # P-B: catch lỗi THỰC khi fill intent NÀY (runner truyền vào
                # có thể raise; runner pipeline tự swallow). -> execution
                # =FAILED cho intent này, KHÔNG lan sang intent khác, không
                # xóa COVERED của intent đã có (điều chỉnh #3).
                try:
                    new_ev = runner(sub_q)
                except Exception:
                    intent_exec[sub] = "FAILED"
                    new_ev = []
                bag = own_ids.setdefault(sub, set())
                added = 0
                for ev in new_ev:
                    key = getattr(ev, "node_id", None) or getattr(ev, "id", None)
                    if key is not None and key in seen:
                        continue
                    seen.add(key)
                    if key is not None:
                        bag.add(str(key))
                    merged.append(ev)
                    added += 1
                if new_ev and intent_exec.get(sub) != "FAILED":
                    intent_exec[sub] = "OK"
                elif not new_ev and intent_exec.get(sub) != "FAILED":
                    intent_exec[sub] = "EMPTY"
                added_this_round[sub] = added
                total_filled += added
                meta["filled"] = total_filled
                if len(merged) >= evidence_cap:
                    meta["stop_reason"] = "budget_evidence"
                    break
            # P-B: cập nhật streak no-progress (CHỈ intent chưa stopped).
            # added==0 (không fill vì stop, hay fill về rỗng/trùng) mà intent
            # vẫn uncovered 2 vòng liên tiếp -> STOP intent NÀY.
            for sub in uncovered:
                if sub in stopped:
                    continue
                a = added_this_round.get(sub, 0)
                if a == 0:
                    no_progress_streak[sub] = no_progress_streak.get(sub, 0) + 1
                else:
                    no_progress_streak[sub] = 0  # T7: node mới -> reset
                if no_progress_streak[sub] >= 2:
                    stopped[sub] = {"round": round_idx, "exec": intent_exec.get(sub, "EMPTY"), "calls_saved": 0}
                    # Triage dự tính (decomposer KHÔNG tự gán OOD — B1;
                    # LegalRetrieval sẽ finalize qua OOD gate F5b / KG).
                    pending[sub] = (
                        "failed"
                        if intent_exec.get(sub) == "FAILED"
                        else ("insufficient" if self._intent_required_nodes(sub) else "ood_gate")
                    )
            if len(merged) >= evidence_cap:
                break
        else:
            meta["stop_reason"] = "max_rounds"

        if rewritten:
            meta["rewritten"] = rewritten

        # P-B: tổng hợp "intent stopped" + số LLM call tiết kiệm + trạng
        # thái dự tính (LegalRetrieval sẽ finalize verdict + ghi trace).
        for sub, s in stopped.items():
            s["calls_saved"] = skipped_calls.get(sub, 0)
            s["pending"] = pending.get(sub, "insufficient")
        if stopped:
            meta["stopped"] = stopped

        # F5b: cấu trúc "mỗi ý = 1 câu đơn" cho rerank theo từng ý.
        # node_ids = seed (pool chung) + evidence ý đó tự sinh (provenance).
        # P-B: thêm status/execution/pending (dự tính) — LegalRetrieval
        # finalize verdict (OOD gate F5b / KG). Không đổi 2 trường cũ.
        meta["per_intent"] = [
            {
                "sub_query": rewritten.get(it, it),
                # Shared initial recall is attributed only when it contains
                # the intent's explicit provision anchor. Unanchored intents
                # must be retrieved separately to avoid evidence leakage.
                "node_ids": sorted({
                    nid for nid in seed_ids
                    if any(
                        self._node_in_pool(required, nid)
                        for required in self._intent_required_nodes(it)
                    )
                } | own_ids.get(it, set())),
                "status": "stopped"
                if it in stopped
                else ("covered" if meta.get("stop_reason") == "covered" else "open"),
                "execution": intent_exec.get(
                    it, "EMPTY" if it in stopped else "OK"
                ),
                "pending": pending.get(it),
            }
            for it in (intents or [question])
        ]

        return merged, meta

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _node_in_pool(required: str, found: str) -> bool:
        """Node `found` có THỎA yêu cầu `required` không (bidirectional cha/con).

        "108_2025_QH15_D4" thỏa khi pool có "108_2025_QH15_D4" (đúng node),
        hoặc "108_2025_QH15_D4_K8" (node con = nội dung Điều đó).
        """
        return (
            required == found
            or found.startswith(required + "_")
            or required.startswith(found + "_")
        )

    def _intent_required_nodes(self, intent_text: str) -> list[str]:
        """Node CẦN TÌM cho 1 ý con (tất định, không LLM).

        Ý con có "số hiệu vb + Điều/Khoản" -> node id chính xác. Không có
        -> [] (caller fallback: tin LLM về độ phủ).
        """
        try:
            from retrieval.QueryUnderstanding import understand
            from retrieval.QueryRouter import anchor_node_ids

            return list(anchor_node_ids(understand(intent_text)))
        except Exception:
            return []

    def _line(self, evidence: Any) -> str:
        if isinstance(evidence, dict):
            return (
                f"{evidence.get('citation') or evidence.get('node_id') or ''}"
                f" | {(evidence.get('text') or '')[:120]}"
            )
        return (
            f"{getattr(evidence, 'citation', '') or getattr(evidence, 'node_id', '')}"
            f" | {(getattr(evidence, 'text', '') or '')[:120]}"
        )

    def _coverage(self, question: str, lines: list[str]) -> dict[str, Any] | None:
        prompt = coverage_prompt(question, lines)
        try:
            _sp = getattr(self.llm, "set_purpose", None)
            if _sp:
                _sp("coverage_check")
            raw = self.llm.complete(prompt)
        except Exception:
            return None
        return _parse_coverage(raw)

    @staticmethod
    def _context(question: str, lines: list[str]) -> str:
        """Bối cảnh cho rewriter: câu hỏi gốc + các căn cứ đã tìm (dài
        hơn 1 dòng evidence) — để viết lại ý con mà KHÔNG bịa."""
        parts = [f"CÂU HỎI GỐC: {question.strip()}"]
        if lines:
            parts.append("CĂN CỨ ĐÃ TÌM ĐƯỢC:\n" + "\n".join(lines))
        return "\n\n".join(parts)


__all__ = [
    "MAX_EVIDENCE",
    "MAX_ROUNDS",
    "QueryDecomposer",
    "VERDICT_COVERED",
    "VERDICT_FAILED",
    "VERDICT_INSUFFICIENT",
    "VERDICT_OOD",
    "build_rewriter_prompt",
    "coverage_prompt",
    "default_sub_query_rewriter",
    "is_multi_intent",
]
