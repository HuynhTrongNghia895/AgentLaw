"""AnchorVerifier (PC-1, §4.4/§14.1) — verify anchor THEO CẤP, rule-first.

Chuỗi verify (KHÔNG LLM, chỉ dữ liệu cấu trúc):

    registry (vb tồn tại? + hiệu lực)
      -> node đúng cấp user chỉ định (điều/khoản/điểm, trên ĐÚNG văn bản)
      -> cấu trúc + text của cha (fallback article_body, §4.2/§4.3)
      -> route: EXACT / ANCHOR_SEMANTIC / SEMANTIC / ANCHOR_UNRESOLVED

Nguyên tắc đã chốt (§2.2/§4.4/§14.1):
  - Semantic search KHÔNG BAO GIỜ thay anchor không hợp lệ.
  - Đúng văn bản (R5 anti-trùng-số): "Điều 4" của 48/2024 chỉ verify trên
    node 48/2024, không bao giờ map sang vb khác.
  - Đúng cấp (R4): user hỏi K9 mà KG không có K9 -> UNRESOLVED, không
    trả K khác thay thế. Ngoại lệ §4.3: điều CHƯA có node khoản nào +
    text thân điều không rỗng -> promote điều (article_body fallback).
  - Provision thiếu văn bản (S3) -> không tự map vb -> UNRESOLVED +
    follow-up hỏi vb.

Phân loại EXACT vs ANCHOR_SEMANTIC (chỉ khi anchor hợp lệ, §14.1):
  - Intent đồ thị (hieu_luc/time_travel/so_sanh) -> EXACT label (caller
    giữ nhánh relation/fast-track — behavior S8/R7 không đổi), 0 LLM.
  - Rule chắc (0 LLM): intent ``chinh_xac`` (nêu provision đích danh)
    HOẶC câu khớp mẫu EXACT ("quy định gì về...", "trích điều...").
  - Chỉ có vb (đã verify), không chỉ định điều/khoản -> ANCHOR_SEMANTIC
    (vb làm filter, semantic bổ sung trong đúng vb), 0 LLM.
  - Intent mở còn lại -> 1 LLM call ngắn (purpose ``route_classify``,
    trả 1 JSON). LLM None/NoLLM/lỗi -> ANCHOR_SEMANTIC (an toàn: anchor
    vẫn pin, semantic chỉ bổ sung — không bịa EXACT).

**LLM call accounting (§14.1):** result.llm_calls_added ∈ {0, 1}; trace
`route_decision.data.llm_calls_added` phải đọc được đúng con số
(0 = rule/reuse understand+refiner, 1 = gọi thêm route_classify).

Giao diện `AnchorVerifySource` (production = `KgAccess` + `registry()`;
test = fake):
  - `registry() -> {so_hieu: row}` — DocumentRegistry (vb tồn tại?).
  - `node(node_id) -> KgNode | None` — cây 4 tầng.
  - `children(node_id) -> [KgNode]` — con trực tiếp.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Route labels 5-route (§2.1)
# ---------------------------------------------------------------------------
ROUTE_EXACT = "exact"
ROUTE_ANCHOR_SEMANTIC = "anchor_semantic"
ROUTE_SEMANTIC = "semantic"
ROUTE_FULL = "full"
ROUTE_ANCHOR_UNRESOLVED = "anchor_unresolved"

# Rule CHẮC CHỐN "hỏi đúng provision" (0 LLM) — R1 (S1):
# "Khoản 8 Điều 4 ... quy định gì về kỳ tính thuế?" = hỏi ĐÚNG nội dung
# provision được nêu đích danh (scope nằm TRONG provision). KHÔNG dùng
# pattern "quy định gì" chung: "quy định NHỮNG ... gì?" (R2) là hỏi MỞ
# RỘNG (nội dung rải nhiều điều) -> rule không quyết -> LLM route_classify
# (hoặc ANCHOR_SEMANTIC an toàn khi NoLLM). Cẩn thận: "nội dung điều"
# là EXACT (hỏi thân provision), "quy định những quyền gì" KHÔNG khớp
# (từ "những" ngăn pattern "quy định gì").
_EXACT_PATTERNS: tuple[str, ...] = (
    r"quy định gì (?:về|gì|là gì)?\b",
    r"quy định như thế nào",
    r"quy định ra sao",
    r"nội dung (?:của )?(?:điều|khoản|điểm)\b",
    r"trích (?:điều|khoản)",
    r"căn cứ (?:điều|khoản)\b",
)
_EXACT_RE = [re.compile(p, re.IGNORECASE) for p in _EXACT_PATTERNS]

# Tín hiệu HỎI MỞ RỘNG (R2): "quy định NHỮNG/NHƯNG ... gì" — nội dung
# rải nhiều điều (quyền/nghĩa vụ...), KHÔNG phải hỏi đúng 1 provision.
# Khi có tín hiệu này: rule EXACT (kể cả intent chinh_xac) KHÔNG được
# quyết định -> phải LLM route_classify (hoặc ANCHOR_SEMANTIC an toàn
# khi NoLLM). Đúng §14.1: "chỉ câu anchor hợp lệ + hỏi mở rộng nội
# dung mới cần LLM".
_EXPANSION_RE = re.compile(
    r"(?:quy định|được|có|còn|bao gồm)\s+(?:những|các)\b", re.IGNORECASE
)

# Intent đồ thị (caller giữ nhánh relation — không phân loại EXACT/SEM).
_GRAPH_INTENTS = ("hieu_luc", "time_travel", "so_sanh")

_MISSING_LABELS = {
    "vb_khong_ton_tai": "văn bản {vb} không có trong dữ liệu (registry)",
    "dieu_khong_ton_tai": "điều chỉ định không tồn tại trong văn bản {vb}",
    "khoan_khong_ton_tai": (
        "khoản chỉ định không tồn tại trong văn bản {vb}"
    ),
    "diem_khong_ton_tai": (
        "điểm chỉ định không tồn tại trong văn bản {vb}"
    ),
    "thieu_van_ban": "chưa rõ điều/khoản thuộc văn bản nào",
}


# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------
@dataclass
class AnchorCheck:
    """Kết quả verify 1 (văn bản + điều/khoản/điểm)."""

    ok: bool
    vb: str = ""                 # số hiệu vb (user nêu / đã chuẩn hoá)
    level: str = ""              # "article" | "clause" | "point" |
                                 # "article_body(...)" (fallback §4.3)
    node_id: str = ""            # node trả về (khi ok, kể cả fallback)
    fallback: bool = False       # True = promote cha (article_body)
    missing: str = ""            # key trong _MISSING_LABELS (khi không ok)
    vb_in_registry: bool = False


@dataclass
class AnchorVerifyResult:
    """Kết quả verify-by-level cho 1 câu / 1 intent con (gọi RIÊNG
    từng intent — requirement #2: intent UNRESOLVED không kéo thụt
    intent khác)."""

    route: str
    checks: list[AnchorCheck] = field(default_factory=list)
    llm_calls_added: int = 0     # {0, 1} — accounting §14.1
    follow_up: str = ""
    reason: str = ""

    @property
    def unresolved(self) -> bool:
        return self.route == ROUTE_ANCHOR_UNRESOLVED

    def pin_node_ids(self) -> list[str]:
        """Node đã verify (kể cả fallback §4.3 — cha promote) phải PIN
        trong evidence: semantic/rerank không được đẩy provision được
        hỏi (hoặc phần gần nhất hợp lệ của nó) đi mất. Node CHẶNG
        verify được (checks không ok) không bao giờ xuất hiện —
        không bịa."""
        out: list[str] = []
        for c in self.checks:
            if c.ok and c.node_id and c.node_id not in out:
                out.append(c.node_id)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vb_key(so_hieu: str) -> str:
    """'59/2020/QH14' -> '59_2020_QH14' (khớp loader._vb_key)."""
    return so_hieu.replace("/", "_").replace("NĐ-CP", "NĐ_CP")


# Số hiệu vb user VIẾT RÕ trong câu ("NN/YYYY" / "NN/YYYY/QH.."...) —
# khác vb SUY RA từ keyword/alias trong QueryUnderstanding. A1/P7: nguồn
# keyword là "nguồn đoán", không được dùng làm anchor (R6: "thuế GTGT
# hiện hành" KHÔNG có số hiệu -> SEMANTIC, không anchor_semantic).
# (?<!/) : không nhận "mm/yyyy" trong ngày dd/mm/yyyy làm số hiệu vb.
_RE_DOC_EXPLICIT = re.compile(r"\b(?<!/)(\d{1,3})/(\d{4})\b")


def _explicit_doc_numbers(
    query: str, registry: dict[str, dict[str, Any]]
) -> list[str]:
    """Số hiệu vb user nêu RÕ trong câu hỏi.

    Chuẩn hoá qua registry (tiền tố 'số/năm'). VB KHÔNG có trong
    registry (vd R3 '999/2030/QH16') VẪN được giữ (dạng '999/2030')
    để caller đánh dấu UNRESOLVED — không tự drop vb chết. Trả [] nếu
    câu không có số hiệu (vd R6/R8)."""
    out: list[str] = []
    for num, year in _RE_DOC_EXPLICIT.findall(query or ""):
        so = _vb_in_registry(f"{num}/{year}", registry)
        so = so or f"{num}/{year}"  # giữ vb không có trong registry
        if so not in out:
            out.append(so)
    return out


def _vb_in_registry(
    so_hieu: str, registry: dict[str, dict[str, Any]]
) -> str | None:
    """Tra registry: chính xác trước, rồi tiền tố 'số/năm'
    (user viết '168/2025' -> '168/2025/NĐ-CP'). Không có -> None."""
    if so_hieu in registry:
        return so_hieu
    prefix = so_hieu.split("/")[:2]
    for key in registry:
        if str(key).split("/")[:2] == prefix:
            return str(key)
    return None


def _anchor_levels(
    articles: list[str], clauses: list[str], points: list[str]
) -> list[dict[str, str]]:
    """Tổ hợp (điều, khoản, điểm) user chỉ định (quy tắc như
    ``QueryRouter.anchor_node_ids`` — không cross-product):

    - điều + khoản + điểm -> 1 point (a[0]/k[0]/p[0]).
    - điều + khoản       -> 1 clause.
    - chỉ điều           -> MỖI điều 1 article (F3-lớp1: "Điều 2 và
      Điều 4" -> D2 + D4, verify riêng từng điều trên đúng văn bản).
    """
    out: list[dict[str, str]] = []
    if articles and clauses and points:
        out.append({
            "level": "point", "dieu": articles[0],
            "khoan": clauses[0], "diem": points[0],
        })
    elif articles and clauses:
        out.append(
            {"level": "clause", "dieu": articles[0], "khoan": clauses[0]}
        )
    elif articles:
        out.extend({"level": "article", "dieu": a} for a in articles)
    return out


def _check_one(
    vb: str,
    levels: list[dict[str, str]],
    registry: dict[str, dict[str, Any]],
    source: Any,
) -> AnchorCheck:
    """Verify 1 (văn bản + levels) trên ĐÚNG văn bản đó (R5).

    Registry không có vb -> chết. Có vb -> check node đúng cấp;
    thiếu đúng cấp -> fallback §4.3 (chỉ khi điều chưa có khoản nào
    + có text thân) nếu không -> UNRESOLVED (missing cụ thể)."""
    so = _vb_in_registry(vb, registry)
    if so is None:
        return AnchorCheck(
            ok=False, vb=vb, missing="vb_khong_ton_tai"
        )
    vk = _vb_key(so)
    for lv in levels:
        dieu = str(lv.get("dieu") or "")
        kid = f"{vk}_D{dieu}"
        dnode = source.node(kid)
        if dnode is None:
            return AnchorCheck(
                ok=False, vb=so, level=lv["level"], vb_in_registry=True,
                missing="dieu_khong_ton_tai",
            )
        if lv["level"] == "article":
            return AnchorCheck(
                ok=True, vb=so, level="article", node_id=kid,
                vb_in_registry=True,
            )
        cid = f"{kid}_K{lv['khoan']}"
        if lv["level"] == "clause":
            if source.node(cid) is not None:
                return AnchorCheck(
                    ok=True, vb=so, level="clause", node_id=cid,
                    vb_in_registry=True,
                )
        else:  # point
            pid = f"{cid}_P{lv['diem']}"
            if source.node(pid) is not None:
                return AnchorCheck(
                    ok=True, vb=so, level="point", node_id=pid,
                    vb_in_registry=True,
                )
            if source.node(cid) is not None:
                # §4.2 (loại 3): parser không tách điểm riêng -> dừng ở
                # clause cha, KHÔNG bịa node point.
                return AnchorCheck(
                    ok=True, vb=so, level="clause", node_id=cid,
                    fallback=True, vb_in_registry=True,
                )
        # Không có đúng cấp -> fallback §4.3 (promote điều cha):
        has_children = bool(source.children(kid))
        has_body = bool((getattr(dnode, "text", "") or "").strip())
        if not has_children and has_body:
            return AnchorCheck(
                ok=True, vb=so,
                level=f"article_body({lv['level']})",
                node_id=kid, fallback=True, vb_in_registry=True,
            )
        missing = (
            "khoan_khong_ton_tai" if lv["level"] == "clause"
            else "diem_khong_ton_tai"
        )
        return AnchorCheck(
            ok=False, vb=so, level=lv["level"],
            vb_in_registry=True, missing=missing,
        )
    # levels rỗng (chỉ có vb) — caller tự quyết định (vb làm filter).
    return AnchorCheck(ok=True, vb=so, vb_in_registry=True)


def _follow_unresolved(checks: list[AnchorCheck]) -> str:
    """Câu hỏi lại GỘP cho phần chưa verify được (1 câu, không bịa)."""
    parts: list[str] = []
    for c in checks:
        if not c.ok:
            label = _MISSING_LABELS.get(c.missing, c.missing or "?")
            parts.append(label.replace("{vb}", c.vb or "?"))
    parts = list(dict.fromkeys(parts))
    if not parts:
        return ""
    return (
        "Không tra được: " + "; ".join(parts)
        + ". Bạn có thể nêu rõ lại số hiệu văn bản hoặc điều/khoản "
        "cần tra không?"
    )


def _llm_classify(llm: Any, prompt: str) -> str:
    """1 LLM call (purpose route_classify); lỗi/parse không được ->
    ANCHOR_SEMANTIC (an toàn: anchor vẫn pin, semantic chỉ bổ sung)."""
    try:
        _sp = getattr(llm, "set_purpose", None)
        if _sp:
            _sp("route_classify")
        raw = llm.complete(prompt)
    except Exception:
        return ROUTE_ANCHOR_SEMANTIC
    try:
        out = str(json.loads(raw).get("route") or "").lower().strip()
        if out in (ROUTE_EXACT, ROUTE_ANCHOR_SEMANTIC):
            return out
    except Exception:
        pass
    return ROUTE_ANCHOR_SEMANTIC


def route_classify_prompt(query: str, verified_desc: str) -> str:
    """Prompt duy nhất của PC-1 (1 JSON). Không hỏi nội dung pháp lý —
    chỉ phân loại MỤC ĐÍCH (EXACT vs mở rộng)."""
    return "\n".join([
        "Bạn là bộ định tuyến câu hỏi pháp lý. Đọc câu hỏi, xác định "
        "MỤC ĐÍCH truy vấn (không trả lời nội dung pháp lý):",
        '- "exact": user hỏi ĐÚNG nội dung provision được nêu đích danh.',
        '- "anchor_semantic": user hỏi MỞ RỘNG quanh provision (nội '
        'dung rải ở nhiều điều/khoản khác: "quy định những ... gì", '
        '"còn những ... nào").',
        "",
        f"CÂU HỎI: {query.strip()}",
        f"PROVISION NÊU ĐÍCH DANH (đã verify hợp lệ): {verified_desc}",
        'Trả DUY NHẤT 1 JSON: {"route": "exact" | "anchor_semantic"}',
    ])


def _verified_desc(checks: list[AnchorCheck]) -> str:
    bits: list[str] = []
    for c in checks:
        if c.ok and c.node_id:
            head, _, tail = c.node_id.partition("_D")
            bits.append(f"{head} {tail}".strip())
        elif c.ok and c.vb:
            bits.append(c.vb)
    return " + ".join(dict.fromkeys(bits))


# ---------------------------------------------------------------------------
# API chính
# ---------------------------------------------------------------------------
def classify_query(
    query: str,
    u: Any,
    *,
    llm: Any | None = None,
    source: Any | None = None,
) -> AnchorVerifyResult:
    """Verify-by-level + phân loại 5-route cho 1 câu / 1 intent con.

    Args:
        query: câu hỏi (hoặc text intent con đã rewriter).
        u: ``QueryUnderstanding`` của câu đó (reuse — 0 LLM mới).
        llm: BaseLLM (None / NoLLM name="none" -> không gọi, 0 call).
        source: ``AnchorVerifySource`` (KgAccess). Không verify được
            (None / thiếu `registry`) -> không bịa: có anchor thì
            UNRESOLVED (follow-up), không anchor thì SEMANTIC
            (giữ behavior vector+OOD gate).

    LLM call: 0 khi rule xác định được (EXACT/GRAPH/vb-only/SEMANTIC);
    1 (purpose ``route_classify``) khi intent mở + anchor hợp lệ còn
    chưa rõ EXACT vs ANCHOR_SEMANTIC. Ghi trong llm_calls_added.

    **Anchor = vb người dùng nêu RÕ** (số hiệu "NN/YYYY[/QH..]"), không
    gồm vb suy ra từ keyword/alias (vd "thuế GTGT" -> 48/2024 là nguồn
    đoán, không phải anchor — R6). Provision (điều/khoản/điểm) thì
    LUÔN rõ (user gõ "điều N").
    """
    # Load registry TRƯỚC (cần cho _explicit_doc_numbers + verify).
    registry: dict[str, dict[str, Any]] = {}
    if source is not None:
        reg_fn = getattr(source, "registry", None)
        if reg_fn is not None:
            try:
                registry = dict(reg_fn() or {})
            except Exception:
                registry = {}
    # Không có nguồn cấu trúc để verify -> degrade an toàn (không bịa,
    # không đổi đường cũ): giữ SEMANTIC + OOD gate (tránh chặn oan).
    if not registry:
        return AnchorVerifyResult(
            route=ROUTE_SEMANTIC,
            reason="không verify được cấu trúc (registry trống/không có "
                   "source) -> giữ semantic + OOD gate (không bịa, không "
                   "chặn oan)",
        )

    anchors = getattr(u, "anchors", None)
    # vb rõ ràng (user viết số hiệu) — KHÔNG tính vb alias/keyword.
    docs = _explicit_doc_numbers(query, registry)
    levels = _anchor_levels(
        list(getattr(anchors, "articles", []) or []),
        list(getattr(anchors, "clauses", []) or []),
        list(getattr(anchors, "points", []) or []),
    )

    # (0) Không vb rõ + không điều/khoản -> SEMANTIC (R6/R8, giữ behavior).
    if not docs and not levels:
        return AnchorVerifyResult(
            route=ROUTE_SEMANTIC,
            reason="không anchor (vb chỉ suy ra từ keyword nếu có) -> "
                   "semantic (vector + rerank, OOD gate)",
        )
    if not registry:
        return AnchorVerifyResult(
            route=ROUTE_SEMANTIC,
            checks=[AnchorCheck(ok=False, vb=v) for v in docs],
            reason="không verify được cấu trúc (registry trống/không có "
                   "source) -> giữ semantic + OOD gate (không bịa, không "
                   "chặn oan)",
        )

    # (1) Registry (R3) + (2) node đúng cấp (R5): check MỖI vb một lần,
    # phân loại kết quả: vb chết (vb_khong_ton_tai) vs lỗi cấp cấu trúc
    # (điều/khoản/điểm thiếu). vb không tồn tại -> KHÔNG semantic thay
    # anchor chết.
    dead_docs: list[AnchorCheck] = []
    level_checks: list[AnchorCheck] = []
    for vb in docs:
        c = _check_one(vb, levels, registry, source)
        if not c.ok and c.missing == "vb_khong_ton_tai":
            dead_docs.append(c)
        else:
            level_checks.append(c)
    # Provision mà thiếu vb (S3/DN_003/DN_051...): KHÔNG tự map văn bản,
    # NHƯNG KHÔNG chặn: route SEMANTIC (giữ behavior cũ 0-diff cho bộ
    # REF_12 NoLLM: B1 kéo đúng vb qua quan hệ + OOD gate như trước).
    # Ở luồng orchestrated, refiner (has_sufficient_anchor=False) là
    # người hỏi vb (S3 clarify_expected=true) — KHÔNG phải PC-1 chặn.
    # UNRESOLVED (R3/R4) chỉ khi vb NÊU RÕ mà verify chết.
    if levels and not docs:
        return AnchorVerifyResult(
            route=ROUTE_SEMANTIC,
            checks=[
                AnchorCheck(
                    ok=False, vb="?", level=levels[0]["level"],
                    missing="thieu_van_ban",
                )
            ],
            reason="provision thiếu vb -> không tự map, giữ semantic "
                   "(refiner hỏi vb ở luồng orchestrated; OOD gate giữ "
                   "câu ngoài phạm vi)",
        )

    # (3) Có phần chết (R3/R4, chỉ khi docs NÊU RỖI mà verify chết) ->
    # UNRESOLVED (R12: phần sống vẫn giữ trong checks — anti pull-down).
    all_checks = level_checks + dead_docs
    if dead_docs or (level_checks and any(not c.ok for c in level_checks)):
        return AnchorVerifyResult(
            route=ROUTE_ANCHOR_UNRESOLVED,
            checks=all_checks,
            follow_up=_follow_unresolved(
                [c for c in all_checks if not c.ok]
            ),
            reason="anchor không verify được -> không semantic thay, "
                   "hỏi lại phần thiếu",
        )

    # (4) Mọi anchor hợp lệ -> EXACT / ANCHOR_SEMANTIC.
    intent = str(getattr(u, "intent", ""))
    if intent in _GRAPH_INTENTS:
        # Nhánh relation/đồ thị (S8/R7): giữ cơ chế hiện có, 0 LLM.
        return AnchorVerifyResult(
            route=ROUTE_EXACT,
            checks=all_checks,
            reason=f"intent '{intent}' (quan hệ/thời gian) -> nhánh "
                   "đồ thị (fast-track relation, 0 LLM)",
        )
    # R2 (expansion signal): "quy định những ... gì?" = hỏi MỞ RỘNG
    # (nội dung rải nhiều điều) -> rule EXACT KHÔNG quyết được -> LLM
    # route_classify (1 call). Kiểm tra TRƯỚC rule EXACT + intent
    # chinh_xac (fallback "has_provision" của QueryUnderstanding gắn
    # intent chin_xac cho cả câu R2). NoLLM -> ANCHOR_SEMANTIC an toàn
    # (anchor vẫn pin, 0 call).
    if _EXPANSION_RE.search(query or ""):
        if llm is None or getattr(llm, "name", "") in ("none", ""):
            return AnchorVerifyResult(
                route=ROUTE_ANCHOR_SEMANTIC,
                checks=all_checks,
                llm_calls_added=0,
                reason="hỏi mở rộng + không LLM -> anchor_semantic "
                       "an toàn (anchor pin, semantic bổ sung)",
            )
        verdict = _llm_classify(
            llm, route_classify_prompt(query, _verified_desc(all_checks))
        )
        return AnchorVerifyResult(
            route=verdict,
            checks=all_checks,
            llm_calls_added=1,
            reason="hỏi mở rộng (expansion signal) -> LLM route_classify "
                   f"-> {verdict} (1 call)",
        )
    for m in _EXACT_RE:
        if m.search(query or ""):
            return AnchorVerifyResult(
                route=ROUTE_EXACT,
                checks=all_checks,
                reason="rule EXACT khớp (user hỏi đúng provision)",
            )
    if intent == "chinh_xac":
        return AnchorVerifyResult(
            route=ROUTE_EXACT,
            checks=all_checks,
            reason="intent chinh_xac (provision nêu đích danh, đã "
                   "verify) -> EXACT (0 LLM)",
        )
    if not levels:
        # Chỉ có vb (đã verify) + hỏi nội dung -> vb làm filter,
        # semantic bổ sung trong đúng vb (0 LLM).
        return AnchorVerifyResult(
            route=ROUTE_ANCHOR_SEMANTIC,
            checks=all_checks,
            reason="có vb (verify được), chưa chỉ định điều/khoản -> "
                   "anchor_semantic (vb filter, 0 LLM)",
        )

    # Intent mở (R2): 1 LLM call (purpose route_classify). NoLLM ->
    # ANCHOR_SEMANTIC an toàn, 0 call (giữ đường test không LLM).
    if llm is None or getattr(llm, "name", "") in ("none", ""):
        return AnchorVerifyResult(
            route=ROUTE_ANCHOR_SEMANTIC,
            checks=all_checks,
            llm_calls_added=0,
            reason="không LLM (NoLLM/None) -> anchor_semantic an toàn "
                   "(anchor vẫn pin, 0 call)",
        )
    verdict = _llm_classify(
        llm, route_classify_prompt(query, _verified_desc(all_checks))
    )
    return AnchorVerifyResult(
        route=verdict,
        checks=all_checks,
        llm_calls_added=1,
        reason=f"LLM route_classify -> {verdict} (1 call)",
    )


__all__ = [
    "AnchorCheck",
    "AnchorVerifyResult",
    "ROUTE_ANCHOR_SEMANTIC",
    "ROUTE_ANCHOR_UNRESOLVED",
    "ROUTE_EXACT",
    "ROUTE_FULL",
    "ROUTE_SEMANTIC",
    "classify_query",
    "route_classify_prompt",
]
