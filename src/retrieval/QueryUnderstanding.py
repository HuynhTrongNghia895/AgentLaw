"""Query Understanding — phân loại ý định + nhóm pháp lý của câu hỏi.

P3 (04/11). **Rule-based (keyword), KHÔNG dùng LLM** — tức thời, miễn phí,
và đủ cho 4 nhóm pháp lý + 5 ý định. (Nếu sau này rule không đủ thì mới
thêm lớp LLM phía trên, chứ không thay thế.)

2 lớp thông tin:

1. **Nhóm pháp lý** → quyết định search dataset nào (giảm nhiễu).
   - `doanh_nghiep` / `thue_gtgt` / `thue_tndn` / `quan_ly_thue`
   - Không khớp gì → `None` = search tất cả (câu OOD).

2. **Ý định** → quyết định nhánh xử lý:
   - `tra_cuu`     : hỏi nội dung quy định
   - `hieu_luc`    : hỏi còn hiệu lực / bãi bỏ chưa  → **luôn** đi đồ thị
   - `so_sanh`     : so sánh 2 văn bản / 2 quy định
   - `time_travel` : hỏi trạng thái tại một mốc thời gian → **luôn** đi đồ thị
   - `chinh_xac`   : có số điều/khoản/điểm rõ → có thể fast-track (KG tất định)

Anchor: các định danh bắt được trong câu hỏi (số hiệu vb, điều, khoản,
điểm) — dùng để tra KG tất định, không cần vector.
"""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Anchor: số hiệu văn bản / điều / khoản / điểm
# ---------------------------------------------------------------------------

# "59/2020/QH14", "168/2025/NĐ-CP", "09/2026/QH16"
# (?<!/) : từ chối khi phần "số/năm" nằm trong ngày dd/mm/yyyy
# (vd "23/07/2026" KHÔNG sinh "07/2026").
RE_DOC_NUMBER = re.compile(
    r"\b(?<!/)(\d{1,3}/\d{4}/(?:QH\d{1,2}|NĐ-?CP))\b",
    re.IGNORECASE,
)

# Dạng rút gọn hay gặp: "NĐ 168/2025", "168/2025", "Luật 59/2020".
# Chỉ giữ phần "số/năm" — sẽ khớp registry bằng tiền tố.
# (?<!/) : không nhận "mm/yyyy" của ngày dd/mm/yyyy làm số hiệu vb.
RE_DOC_SHORT = re.compile(
    r"\b(?<!/)(\d{1,3})/(\d{4})\b",
)

# "Điều 100", "điều 5 khoản 3 điểm c" (nhiều biến thể)
RE_ARTICLE = re.compile(r"\bđiều\s*(\d+[a-z]?)\b", re.IGNORECASE)
RE_CLAUSE = re.compile(r"\bkhoản\s*(\d+[a-z]?)\b", re.IGNORECASE)
RE_POINT = re.compile(r"\bđiểm\s*([a-zđ])\b", re.IGNORECASE)

# Ngày trong câu hỏi (dd/mm/yyyy) -> gợi ý time-travel
RE_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


@dataclass
class Anchors:
    """Định danh bắt được từ câu hỏi."""

    doc_numbers: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)
    points: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)

    @property
    def has_provision(self) -> bool:
        """Có trỏ được tới điều/khoản/điểm không."""
        return bool(self.articles or self.clauses or self.points)

    @property
    def has_doc(self) -> bool:
        return bool(self.doc_numbers)

    def __bool__(self) -> bool:
        return bool(
            self.doc_numbers
            or self.articles
            or self.clauses
            or self.points
            or self.dates
        )


def extract_anchors(query: str) -> Anchors:
    """Bắt định danh (số hiệu vb / điều / khoản / điểm / ngày) trong câu hỏi."""
    q = query or ""
    a = Anchors()

    a.doc_numbers = [m.group(1).upper() for m in RE_DOC_NUMBER.finditer(q)]
    a.articles = [m.group(1) for m in RE_ARTICLE.finditer(q)]
    a.clauses = [m.group(1) for m in RE_CLAUSE.finditer(q)]
    a.points = [m.group(1).lower() for m in RE_POINT.finditer(q)]
    a.dates = [
        f"{int(d)}/{int(m)}/{y}"
        for d, m, y in RE_DATE.findall(q)
    ]
    return a


def parse_temporal_constraint(query: str) -> dict[str, object] | None:
    """Parse explicit legal-time language into an inclusive ISO target date."""
    q = query or ""
    has_legal_scope = bool(
        extract_anchors(q).has_doc
        or extract_anchors(q).has_provision
        or re.search(
            r"(?i)quy định|văn bản|pháp luật|luật|nghị định|điều khoản|"
            r"hiệu lực|áp dụng|sửa đổi|bổ sung|bãi bỏ|thay thế|đăng ký doanh nghiệp|thuế",
            q,
        )
    )
    if not has_legal_scope:
        return None
    patterns = (
        ("UNTIL", r"\bđến\s+trước\s+(?:ngày\s+)?", False, -1),
        ("BEFORE", r"\btrước\s+(?:ngày\s+)?", False, -1),
        ("AFTER", r"\bsau\s+(?:ngày\s+)?", False, 1),
        ("FROM", r"\b(?:từ|kể\s+từ)\s+(?:ngày\s+)?", True, 0),
        ("UNTIL", r"\bđến\s+(?:hết\s+)?(?:ngày\s+)?", True, 0),
        ("AT", r"\b(?:tại|vào)\s+(?:thời\s+điểm\s+|ngày\s+)?", True, 0),
        ("UNTIL", r"\btính\s+đến\s+(?:ngày\s+)?", True, 0),
    )
    for operator, prefix, inclusive, offset in patterns:
        match = re.search(prefix + r"(\d{1,2})/(\d{1,2})/(\d{4})", q, re.I)
        if not match:
            continue
        day, month, year = map(int, match.groups())
        try:
            raw = date(year, month, day)
            target = raw + timedelta(days=offset)
        except ValueError:
            return None
        return {
            "operator": operator,
            "raw_date": match.group(0),
            "temporal_expression": match.group(0),
            "normalized_target_date": target.isoformat(),
            "inclusive": inclusive,
            "temporal_constraint_source": "query_understanding",
        }
    current = re.search(
        r"(?i)hi\u1ec7n\s+nay|hi\u1ec7n\s+t\u1ea1i|\u0111ang\s+\u00e1p\s+d\u1ee5ng|hi\u1ec7n\s+h\u00e0nh",
        q,
    )
    if current:
        return {
            "operator": "CURRENT",
            "raw_date": None,
            "temporal_expression": current.group(0),
            "normalized_target_date": None,
            "inclusive": True,
            "temporal_constraint_source": "query_understanding",
        }
    return None


def resolve_doc_numbers(query: str) -> list[str]:
    """Số hiệu vb trong câu hỏi, đã chuẩn hoá về dạng registry.

    "168/2025/NĐ-CP" -> ["168/2025/NĐ-CP"]
    "NĐ 168/2025"    -> ["168/2025/NĐ-CP"]   (tra registry)
    "Luật 59/2020"   -> ["59/2020/QH14"]      (tra registry)
    "Luật Doanh nghiệp 2020" -> ["59/2020/QH14"] (bản alias tên->số hiệu)

    Alias: khớp theo dạng KHÔNG DẤU + lowercase (không cần liệt kê từng
    biến thể có dấu/không dấu). Nếu 2 alias khác nhau trỏ 2 văn bản
    khác -> MÂU THUẪN -> trả [] (không đoán, để skill hỏi lại).
    """
    found = [m.group(1).upper() for m in RE_DOC_NUMBER.finditer(query)]

    # Dạng rút gọn: tra registry theo tiền tố "số/năm"
    registry = _load_registry_prefixes()
    short_out: list[str] = []
    for num, year in RE_DOC_SHORT.findall(query or ""):
        key = f"{num}/{year}"
        match = registry.get(key)
        if match and match not in short_out:
            short_out.append(match)

    # Tên thông thường ("Luật Doanh nghiệp", "thuế GTGT"...) -> alias
    alias_out = _resolve_doc_aliases(query)

    # Gộp 3 nguồn; nếu alias và nguồn khác cho ra số hiệu KHÁC nhau
    # trong cùng câu hỏi -> mâu thuẫn -> chỉ tin nguồn dạng số (rõ hơn).
    out = list(dict.fromkeys(found + short_out + alias_out))
    return out


# ---------------------------------------------------------------------------
# Alias tên văn bản -> số hiệu (bảng JSON, khớp dạng không dấu)
# ---------------------------------------------------------------------------

_DOC_ALIASES: dict[str, str] | None = None


def _load_doc_aliases() -> dict[str, str]:
    """{ 'luat doanh nghiep': '59/2020/QH14', ... } từ JSON.

    Chỉ đọc 1 lần. File thiếu/hoại -> {} (không chết — degrade an toàn).
    """
    global _DOC_ALIASES
    if _DOC_ALIASES is not None:
        return _DOC_ALIASES
    out: dict[str, str] = {}
    path = Path(__file__).resolve().parents[2] / "data" / "doc_name_aliases.json"
    if path.exists():
        import json as _json

        try:
            raw = _json.loads(path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                out[str(k).strip().lower()] = str(v).strip()
        except Exception:
            out = {}
    _DOC_ALIASES = out
    return out


def _strip_accents_local(text: str) -> str:
    """Bỏ dấu tiếng Việt (stdlib-only, không cần underthesea)."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    ).replace("đ", "d").replace("Đ", "D")


def _resolve_doc_aliases(query: str) -> list[str]:
    """Khớp alias tên văn bản (không dấu) trong câu hỏi -> số hiệu.

    2 alias khác nhau trỏ 2 văn bản khác nhau trong cùng câu ->
    trả [] (mâu thuẫn, để skill hỏi lại) — không đoán bừa.
    """
    aliases = _load_doc_aliases()
    if not aliases:
        return []
    haystack = _strip_accents_local(query).lower()
    hits: set[str] = set()
    for alias, target in aliases.items():
        if alias and alias in haystack:
            hits.add(target)
    return [] if len(hits) > 1 else sorted(hits)


_REGISTRY_PREFIXES: dict[str, str] | None = None


def _load_registry_prefixes() -> dict[str, str]:
    """{ '168/2025': '168/2025/NĐ-CP', ... } từ DocumentRegistry.csv."""
    global _REGISTRY_PREFIXES
    if _REGISTRY_PREFIXES is not None:
        return _REGISTRY_PREFIXES

    out: dict[str, str] = {}
    csv_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "metadata"
        / "DocumentRegistry.csv"
    )
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                so_hieu = (row.get("so_hieu") or "").strip()
                if not so_hieu:
                    continue
                num_year = "/".join(so_hieu.split("/")[:2])
                out.setdefault(num_year, so_hieu)
    _REGISTRY_PREFIXES = out
    return out


# ---------------------------------------------------------------------------
# Nhóm pháp lý (dataset)
# ---------------------------------------------------------------------------

# Thứ tự quan trọng: dataset cụ thể hơn phải được check trước. Ví dụ
# "thuế TNDN" chứa từ khóa "thuế" chung -> phải match TNDN trước GTGT.
LEGAL_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "thue_tndn": (
        "tndn", "thu nhập doanh nghiệp", "thu nhập doanh nghiệp",
        "lợi nhuận trước thuế", "an sinh kinh tế", "chi phí được trừ",
        "giải trừ", "lập báo cáo thuế", "nộp thuế tndn",
    ),
    "thue_gtgt": (
        "gtgt", "giá trị gia tăng", "hàng hóa dịch vụ", "mua bán",
        "xuất khẩu", "nhập khẩu", "hóa đơn", "hàng hóa tự sản xuất",
    ),
    "quan_ly_thue": (
        "quản lý thuế", "chuyển giá", "truy thu", "miễn giảm thuế",
        "miễn thuế", "giảm thuế", "kê khai thuế", "cơ quan thuế",
        "chế tộ thuế", "tiền chậm nộp",
    ),
    "doanh_nghiep": (
        "doanh nghiệp", "hộ kinh doanh", "công ty", "vốn điều lệ",
        "đăng ký doanh nghiệp", "người đại diện", "chủ tịch hội đồng",
        "thành viên công ty", "giấy chứng nhận đăng ký doanh nghiệp",
        "thủ tục đăng ký", "điều lệ",
    ),
}


def detect_legal_group(query: str) -> str | None:
    """Đoán nhóm pháp lý của câu hỏi (dataset để search).

    2 nguồn, nguồn mạnh hơn thắng:
    1. **Văn bản được nhắc tới** -> lấy `dataset` trong registry
       (vd "NĐ 168/2025" -> doanh_nghiep). Chính xác nhất.
    2. **Keyword trong câu** -> điểm theo nhóm (fallback).

    Trả `None` nếu không khớp gì -> search toàn bộ (câu OOD).
    """
    # 1. Văn bản được nhắc tới
    for so_hieu in resolve_doc_numbers(query):
        group = _registry_dataset(so_hieu)
        if group:
            return group

    # 2. Keyword
    q = (query or "").lower()
    hits: list[tuple[int, str]] = []
    for group, keys in LEGAL_GROUP_KEYWORDS.items():
        score = sum(1 for k in keys if k in q)
        if score:
            hits.append((score, group))
    if not hits:
        return None
    return max(hits)[1]


_REGISTRY_DATASETS: dict[str, str] | None = None


def _registry_dataset(so_hieu: str) -> str | None:
    """Dataset của 1 văn bản (từ registry)."""
    global _REGISTRY_DATASETS
    if _REGISTRY_DATASETS is None:
        _REGISTRY_DATASETS = {}
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "metadata"
            / "DocumentRegistry.csv"
        )
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    sh = (row.get("so_hieu") or "").strip()
                    ds = (row.get("dataset") or "").strip()
                    if sh and ds:
                        _REGISTRY_DATASETS[sh] = ds.split(";")[0].strip()
    return _REGISTRY_DATASETS.get(so_hieu)


# ---------------------------------------------------------------------------
# Ý định
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hieu_luc": (
        "hiệu lực", "còn hiệu lực", "mất hiệu lực", "hết hiệu lực",
        "bãi bỏ", "bãi bỏ chưa", "ngừng hiệu lực", "còn áp dụng",
        "còn dùng được không", "áp dụng từ",
    ),
    "time_travel": (
        "tại thời điểm", "vào thời điểm", "tính đến", "tính đến ngày",
        "thời điểm nào", "khi nào thì", "trước ngày", "sau ngày",
        "tại ngày", "vào ngày",
    ),
    "so_sanh": (
        "so sánh", "khác nhau", "khác gì", "giống nhau", "đối chiếu",
        "với ... của", "so với",
    ),
    "chinh_xac": (
        "quy định tại", "theo điều", "căn cứ", "xem điều", "mở điều",
        "trích điều", "nội dung điều", "điều ... khoản",
    ),
}

INTENT_TRA_CUU = "tra_cuu"
INTENT_HIEU_LUC = "hieu_luc"
INTENT_TIME_TRAVEL = "time_travel"
INTENT_SO_SANH = "so_sanh"
INTENT_CHINH_XAC = "chinh_xac"

# Từ khóa cho biết câu hỏi đang hỏi về **QUAN HỆ** giữa văn bản
# (ai sửa ai, còn hiệu lực không, văn bản nào hướng dẫn) chứ không
# phải hỏi **NỘI DUNG** quy định.
#
# Vì sao cần (P7 X1): câu quan hệ có cosine FAISS RẤT THẤP (~0.0) vì
# text nội dung provision không khớp ngữ nghĩa với câu hỏi quan hệ.
# Nếu áp ngưỡng cosine 0.70 cho fast-track thì câu quan hệ KHÔNG BAO
# GIỜ fast-track -> rơi vào vector -> kéo nhầm văn bản khác vào
# evidence. Câu quan hệ phải đi thẳng đồ thị (0 embedding).
#
# Đồng bộ với `GraphitiBridge.RELATION_KEYWORDS` (loại quan hệ pháp lý).
RELATION_QUERY_KEYWORDS: tuple[str, ...] = (
    "sửa đổi", "sửa", "bổ sung", "bãi bỏ", "thay thế",
    "hiệu lực", "hướng dẫn", "được hướng dẫn",
    "còn áp dụng", "mất hiệu lực",
)


def is_relation_query(query: str) -> bool:
    """Câu hỏi này hỏi về QUAN HỆ giữa văn bản không?

    True -> nên tra đồ thị (Cypher/Graphiti), không cần vector.
    """
    q = (query or "").lower()
    return any(k in q for k in RELATION_QUERY_KEYWORDS)


# "Điều 100" -> `chinh_xac` kèm anchor đầy đủ
# Thứ tự ưu tiên. `time_travel` đứng trước `hieu_luc` vì câu có MỐC THỜI
# GIAN cụ thể ("tại thời điểm 01/09/2025 ... còn hiệu lực không?") là
# time-travel query (cần truy vấn theo as_of_date) — đó là ý định chính
# xác hơn "hỏi hiệu lực" chung chung.
INTENT_PRIORITY = (
    "time_travel",
    "hieu_luc",
    "so_sanh",
    "chinh_xac",
)


def detect_intent(query: str, anchors: Anchors | None = None) -> str:
    """Đoán ý định của câu hỏi.

    Thứ tự ưu tiên: `hieu_luc` > `time_travel` > `so_sanh` > `chinh_xac`
    > `tra_cuu`. Vì hỏi "còn hiệu lực không" cũng có thể chứa "điều 5",
    nhưng ý định thật là hỏi hiệu lực.
    """
    a = anchors if anchors is not None else extract_anchors(query)
    q = (query or "").lower()

    if parse_temporal_constraint(query):
        return "time_travel"

    for intent in INTENT_PRIORITY:
        keys = INTENT_KEYWORDS.get(intent)
        if keys and any(k in q for k in keys):
            # `chinh_xac` chỉ hợp lệ khi có anchor điều/khoản/điểm thật.
            if intent == "chinh_xac" and not a.has_provision:
                continue
            return intent

    # Không có từ khóa nào nhưng có anchor đầy đủ (vd "Điều 100 NĐ 168")
    # -> vẫn là tra cứu chính xác, có thể fast-track.
    if a.has_provision:
        return "chinh_xac"
    return INTENT_TRA_CUU


# ---------------------------------------------------------------------------
# Kết quả tổng hợp
# ---------------------------------------------------------------------------
@dataclass
class QueryUnderstanding:
    """Kết quả phân tích câu hỏi đầu vào."""

    query: str
    legal_group: str | None = None   # dataset để search (None = tất cả)
    intent: str = INTENT_TRA_CUU
    anchors: Anchors = field(default_factory=Anchors)

    @property
    def needs_graph(self) -> bool:
        """Ý định nào BẮT BUỘC phải đi đồ thị (vector không biết thời gian)."""
        return self.intent in ("hieu_luc", "time_travel", "so_sanh")

    @property
    def can_fast_track(self) -> bool:
        """Có thể tra KG tất định (không cần vector) không."""
        return self.intent == "chinh_xac" and self.anchors.has_provision

    @property
    def is_relation(self) -> bool:
        """Câu hỏi đang hỏi về QUAN HỆ giữa văn bản (sửa/bãi bỏ/hiệu lực).

        Loại câu này phải đi đồ thị, KHÔNG được dùng ngưỡng cosine FAISS
        để quyết định fast-track (cosine luôn thấp ~0.0 -> nếu áp ngưỡng
        thì fast-track không bao giờ bật -> rơi vào vector -> nhiễu).
        """
        return self.needs_graph or is_relation_query(self.query)

    @property
    def should_ask_web(self) -> bool:
        """Có cần hỏi user làm rõ không (thiếu nhóm PL + thiếu anchor)."""
        return not self.anchors.has_provision

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "legal_group": self.legal_group,
            "intent": self.intent,
            "anchors": {
                "doc_numbers": self.anchors.doc_numbers,
                "articles": self.anchors.articles,
                "clauses": self.anchors.clauses,
                "points": self.anchors.points,
                "dates": self.anchors.dates,
            },
            "needs_graph": self.needs_graph,
            "can_fast_track": self.can_fast_track,
            "is_relation": self.is_relation,
            "should_ask_web": self.should_ask_web,
        }


def understand(query: str) -> QueryUnderstanding:
    """Phân tích 1 câu hỏi: nhóm pháp lý + ý định + anchor."""
    anchors = extract_anchors(query)
    # Số hiệu vb dạng rút gọn ("NĐ 168/2025") -> chuẩn hoá bằng registry.
    full_docs = resolve_doc_numbers(query)
    anchors.doc_numbers = (
        full_docs
        if full_docs
        else anchors.doc_numbers
    )
    return QueryUnderstanding(
        query=query,
        legal_group=detect_legal_group(query),
        intent=detect_intent(query, anchors),
        anchors=anchors,
    )


__all__ = [
    "Anchors",
    "INTENT_CHINH_XAC",
    "INTENT_HIEU_LUC",
    "INTENT_SO_SANH",
    "INTENT_TIME_TRAVEL",
    "INTENT_TRA_CUU",
    "LEGAL_GROUP_KEYWORDS",
    "QueryUnderstanding",
    "detect_intent",
    "detect_legal_group",
    "extract_anchors",
    "understand",
]
