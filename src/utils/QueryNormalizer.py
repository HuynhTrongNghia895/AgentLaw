from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
from underthesea import text_normalize


# File nằm ở src/utils/ -> gốc repo là cha bậc 2 (src -> AgentLaw).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_ALIASES_PATH = PROJECT_ROOT / "data" / "query_aliases.json"


def load_query_aliases() -> dict[str, str]:
    if not QUERY_ALIASES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy query alias lexicon: {QUERY_ALIASES_PATH}"
        )

    with QUERY_ALIASES_PATH.open("r", encoding="utf-8") as file:
        raw_aliases = json.load(file)

    if not isinstance(raw_aliases, dict):
        raise ValueError("query_aliases.json phải là object")

    aliases: dict[str, str] = {}
    for category, values in raw_aliases.items():
        if not isinstance(category, str) or not isinstance(values, dict):
            raise ValueError("Mỗi nhóm alias phải là object")

        for alias, replacement in values.items():
            if not isinstance(alias, str) or not isinstance(replacement, str):
                raise ValueError("Alias và replacement phải là chuỗi")
            aliases[alias.lower()] = replacement

    return aliases


TEENCODE_MAP = load_query_aliases()

VIETNAMESE_DIACRITICS_PATTERN = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    flags=re.IGNORECASE,
)

TOKEN_PATTERN = re.compile(r"\b[\wđĐ]+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    normalized: str
    accentless: str | None
    is_accentless: bool
    has_teencode: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in decomposed
        if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")


def _replace_teencode(text: str) -> tuple[str, bool]:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return text, False

    has_teencode = False
    replacements: dict[str, str] = {}

    for token in tokens:
        replacement = TEENCODE_MAP.get(token.lower())
        if replacement:
            replacements[token] = replacement
            has_teencode = True

    if not replacements:
        return text, False

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(token) for token in sorted(replacements, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        return TEENCODE_MAP.get(value.lower(), value)

    return pattern.sub(replace, text), has_teencode


def looks_accentless(text: str) -> bool:
    """Detect a mostly alphabetic Vietnamese query typed without diacritics.

    The detector is conservative: legal references/numbers alone do not trigger it,
    and a query containing Vietnamese diacritics is treated as accented.
    """
    normalized = normalize_whitespace(text)
    if not normalized:
        return False

    if VIETNAMESE_DIACRITICS_PATTERN.search(normalized):
        return False

    alphabetic_tokens = [
        token for token in TOKEN_PATTERN.findall(normalized)
        if any(char.isalpha() for char in token)
    ]

    return len(alphabetic_tokens) >= 3


class QueryNormalizer:
    def normalize(self, question: str) -> NormalizedQuery:
        original = normalize_whitespace(str(question or ""))
        if not original:
            raise ValueError("Câu hỏi không được để trống")

        # underthesea normalizes Vietnamese Unicode/typing representation without
        # rewriting the legal meaning of the user's question.
        normalized = normalize_whitespace(text_normalize(original))
        normalized, has_teencode = _replace_teencode(normalized)
        normalized = normalize_whitespace(normalized)

        is_accentless = looks_accentless(original)
        accentless = strip_accents(normalized).lower() if is_accentless else None

        return NormalizedQuery(
            original=original,
            normalized=normalized,
            accentless=accentless,
            is_accentless=is_accentless,
            has_teencode=has_teencode,
        )
