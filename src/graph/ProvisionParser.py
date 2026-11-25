from __future__ import annotations

"""Parse provision strings từ DocumentRelations.csv sang cấu trúc.

Nguyên tắc (CLAUDE.md §6.4): CHỈ parse định dạng đã biết, parse fail
giữ nguyên chuỗi thô (`ok=False`, `raw` đầy đủ) — KHÔNG bịa số điều/khoản.

Định dạng trong CSV (đã rà 20 quan hệ):
- "Điều 7 (khoản 1-7)"
- "Điều 4 (khoản 5, 14, 16, bổ sung 35); Điều 8 (khoản 5a)"
- "Điều 4 (bổ sung khoản 3a; sửa điểm đ khoản 4; sửa khoản 14)"
- "Điều 2 (khoản 1, 2)"
- "Phụ lục I, Phụ lục II (thay thế)"
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProvisionRef:
    """1 provision tham chiếu được parse.

    label: "7" (điều) hoặc "Phụ lục I".
    kind: "article" | "annex".
    clauses: danh sách token khoản SÁCH (giữ nguyên chữ, ví dụ
      "1-7", "5a", "bổ sung 35", "điểm 2 điểm e" -> "2 điểm e").
    ok: False khi segment không parse được (giữ raw, không bịa).
    """

    label: str
    kind: str
    clauses: list[str] = field(default_factory=list)
    raw: str = ""
    ok: bool = True

    @property
    def unique_key_part(self) -> str:
        """Đoạn để ghép vào Provision.unique_key ("<so_hieu>|<label>")."""
        return self.label


_ARTICLE_RE = re.compile(
    r"^Điều\s+(\d+)\s*(?:\((.*)\))?$", re.IGNORECASE
)
_ANNEX_SPLIT_RE = re.compile(r"Phụ lục\s+([IVXLCDM\d]+)", re.IGNORECASE)


def _split_clauses(inner: str) -> list[str]:
    """'khoản 5, 14, 16, bổ sung 35' -> ['khoản 5','14','16','bổ sung 35'];
    'bổ sung khoản 3a; sửa khoản 14' -> ['bổ sung khoản 3a','sửa khoản 14'].
    Giữ nguyên chữ token (không bịa); chỉ tách , và ;."""
    out: list[str] = []
    for part in re.split(r"[;,]", inner):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _split_segments(text: str) -> list[str]:
    """Tách ';' ở NGOÀI ngoặc (REL_144 có 'Điều 4 (a; b; c); Điều 9 (d)'
    -> ';' trong ngoặc KHÔNG phải phân cách provision)."""
    segments: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == ";" and depth == 0:
            segments.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def parse_provision_ref(segment: str) -> list[ProvisionRef]:
    """Parse 1 segment (đã tách ';'). Trả LIST vì 'Phụ lục I, Phụ lục II
    (thay thế)' chứa 2 provision phụ lục."""
    segment = segment.strip()
    if not segment:
        return []

    # --- Phụ lục: 1 segment có thể chứa nhiều ("Phụ lục I, Phụ lục II")
    if "Phụ lục" in segment or "phụ lục" in segment:
        found = _ANNEX_SPLIT_RE.findall(segment)
        if found:
            # Ghi chú trong ngoặc (vd "(thay thế)") gắn lên ref cuối
            note_m = re.search(r"\((.*)\)\s*$", segment)
            refs = [
                ProvisionRef(
                    label=f"Phụ lục {g}", kind="annex", raw=segment
                )
                for g in found
            ]
            if note_m:
                # "(thay thế)" là ngữ cảnh quan hệ (REPEALS đã cover),
                # lưu làm clause token để KHÔNG MẤT thông tin.
                refs[-1].clauses = [note_m.group(1).strip()]
            return refs
        # "Phụ lục" nhưng không parse được label
        return [
            ProvisionRef(label=segment, kind="annex", raw=segment, ok=False)
        ]

    # --- Điều (article)
    m = _ARTICLE_RE.match(segment)
    if m:
        article = m.group(1)
        inner = (m.group(2) or "").strip()
        clauses = _split_clauses(inner) if inner else []
        return [
            ProvisionRef(
                label=article,
                kind="article",
                clauses=clauses,
                raw=segment,
            )
        ]

    # --- Không parse được: giữ raw, ok=False (KHÔNG bịa)
    return [
        ProvisionRef(label=segment, kind="unknown", raw=segment, ok=False)
    ]


def parse_provision_list(text: str | None) -> list[ProvisionRef]:
    """Parse cả chuỗi provision. Tách ';' ở NGOÀI ngoặc (xem
    _split_segments)."""
    if not text:
        return []
    refs: list[ProvisionRef] = []
    for segment in _split_segments(text):
        refs.extend(parse_provision_ref(segment))
    return refs


def unique_key(so_hieu: str, ref: ProvisionRef) -> str:
    """Khóa node Provision: '<so_hieu>|<label>' (vd '67/2025/QH15|11')."""
    return f"{so_hieu.strip()}|{ref.label}"


_EFF_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EFF_ENTRY_RE = re.compile(r"^(.+?):(\d{4}-\d{2}-\d{2})$")


def parse_provision_effective_dates(
    text: str | None,
) -> dict[str, Any]:
    """Parse cột `provision_effective_dates` (map provision->ngày riêng,
    phương án A - hiệu lực 3 lớp).

    Định dạng: "17:2026-01-01; 5.25:2026-01-01; Phụ lục IV:2026-07-01"
      - "17:..."         -> cả Điều 17 có ngày riêng (article_dates)
      - "5.25:..."       -> CHỈ khoản 25 của Điều 5 (clause_dates)
      - "Phụ lục IV:..." -> phụ lục (annex_dates)
    Rỗng/None -> dict rỗng (mọi điều theo ngày chung văn bản - không lỗi).

    Trả:
      {
        "article_dates": {"17": "2026-01-01"},
        "clause_dates":  {"5": {"25": "2026-01-01"}},
        "annex_dates":   {"Phụ lục IV": "2026-07-01"},
        "unparsed":      ["..."],   # entry không parse được (giữ raw,
                                     #  không bịa - §6.4)
      }
    """
    result: dict[str, Any] = {
        "article_dates": {},
        "clause_dates": {},
        "annex_dates": {},
        "unparsed": [],
    }
    if not text or not text.strip():
        return result

    for entry in text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        m = _EFF_ENTRY_RE.match(entry)
        if not m or not _EFF_DATE_RE.match(m.group(2)):
            result["unparsed"].append(entry)
            continue
        key, date = m.group(1).strip(), m.group(2)
        if key.lower().startswith("phụ lục") or key.lower().startswith(
            "phu luc"
        ):
            result["annex_dates"][key] = date
        elif "." in key:  # "5.25" -> khoản 25 của Điều 5
            art, clause = key.split(".", 1)
            art = art.strip()
            clause = clause.strip()
            if art.isdigit() and clause:
                result["clause_dates"].setdefault(art, {})[clause] = date
            else:
                result["unparsed"].append(entry)
        elif key.isdigit():
            result["article_dates"][key] = date
        else:
            result["unparsed"].append(entry)
    return result


def provision_unparsed(refs: list[ProvisionRef]) -> list[str]:
    """Danh sách raw chưa parse được (phải RỖNG cho bộ dữ liệu hiện tại)."""
    return [r.raw for r in refs if not r.ok]


__all__ = [
    "ProvisionRef",
    "parse_provision_ref",
    "parse_provision_list",
    "unique_key",
    "provision_unparsed",
]
