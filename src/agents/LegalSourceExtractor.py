"""Conservative, quote-backed extraction from full official legal text."""
from __future__ import annotations

import re
import hashlib
from datetime import date
from html import unescape
from html.parser import HTMLParser
from typing import Any


_DOC_RE = re.compile(
    r"\b\d{1,5}/\d{4}/(?:NĐ-CP|NĐ|QH\d+|TT-[A-Z0-9-]+|UBTVQH\d+)\b",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-z]?)\s*[.\-–:]?", re.IGNORECASE)
_EFFECTIVE_RE = re.compile(
    r"(?P<quote>.{0,100}có hiệu lực(?: thi hành)?(?: kể từ| từ)? ngày\s+"
    r"(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})\s+năm\s+(?P<year>\d{4})[^\n.]*[.]?)",
    re.IGNORECASE,
)
_NUMERIC_EFFECTIVE_RE = re.compile(
    r"(?P<quote>.{0,100}có hiệu lực(?: thi hành)?(?: kể từ| từ)? ngày\s+"
    r"(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{4})[^\n.]*[.]?)",
    re.IGNORECASE,
)
_PUBLICATION_RE = re.compile(
    r"(?:ngày\s+ban hành\s*[:：]?|ban hành\s+ngày)\s*"
    r"(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})\s+năm\s+(?P<year>\d{4})",
    re.IGNORECASE,
)


class _TextExtractor(HTMLParser):
    _BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section"}
    _SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self.skip_depth += 1
        elif not self.skip_depth and tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw or "")
    visible = "".join(parser.parts)

    script_text: list[str] = []
    for match in re.finditer(r"<script\b[^>]*>(.*?)</script>", raw or "", re.I | re.S):
        script = match.group(1)
        if not script.strip():
            continue
        script = unescape(script)
        script = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda m: chr(int(m.group(1), 16)),
            script,
        )
        script = re.sub(r"</(?:p|div|li|tr|h[1-6]|section)>", "\n", script, flags=re.I)
        script = re.sub(r"<(?:p|div|li|tr|h[1-6]|section)\b[^>]*>", "\n", script, flags=re.I)
        script = re.sub(r"<br\s*/?>", "\n", script, flags=re.I)
        script = re.sub(r"<[^>]+>", " ", script)
        script_text.append(unescape(script))

    combined = visible + "\n" + "\n".join(script_text)
    combined = re.sub(r"[ \t]+", " ", combined)
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined.strip()


def _effective_date(text: str) -> tuple[str, str]:
    json_ld_match = re.search(
        r'"legislationDate"\s*:\s*"(\d{4})-(\d{2})-(\d{2})',
        text or "",
        re.I,
    )
    if json_ld_match:
        try:
            value = date(
                int(json_ld_match.group(1)),
                int(json_ld_match.group(2)),
                int(json_ld_match.group(3)),
            ).isoformat()
            start = max(0, json_ld_match.start() - 100)
            quote = (text or "")[start:json_ld_match.end() + 100].strip()
            return value, quote
        except ValueError:
            pass

    for pattern in (_EFFECTIVE_RE, _NUMERIC_EFFECTIVE_RE):
        match = pattern.search(text or "")
        if not match:
            continue
        try:
            value = date(
                int(match.group("year")), int(match.group("month")),
                int(match.group("day")),
            ).isoformat()
        except ValueError:
            return "", match.group("quote").strip()
        return value, match.group("quote").strip()
    return "", ""


def _publication_date(text: str) -> str:
    match = _PUBLICATION_RE.search(text or "")
    if not match:
        return ""
    try:
        return date(int(match.group("year")), int(match.group("month")),
                    int(match.group("day"))).isoformat()
    except ValueError:
        return ""


def _article_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_ARTICLE_RE.finditer(text or ""))
    if not matches:
        return [("", text or "")]
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1), text[match.start():end].strip()))
    return sections


def _repealed_provisions(section: str, document_number: str) -> list[dict[str, str]]:
    """Extract explicit article/clause repeal lists from a full-text section."""
    provisions: list[dict[str, str]] = []
    bãi_match = re.search(r"bãi bỏ\s+(.{0,600}?)(?:[.;]\s|$)", section, re.I)
    if not bãi_match:
        return provisions

    clause_prefix = re.match(r"\s*Điều\s+(\d+[a-z]?)", bãi_match.group(1), re.I)
    if clause_prefix:
        article = clause_prefix.group(1)
        quote = clause_prefix.group(0).strip()
        provisions.append({
            "document_number": document_number,
            "article": article,
            "clause": "",
            "scope": "article_repealed",
            "quote": quote,
        })

    for match in re.finditer(
        r"(?:khoản\s+(\d+[a-z]?)\s+)?Điều\s+(\d+[a-z]?)",
        bãi_match.group(1),
        re.I,
    ):
        clause = match.group(1) or ""
        article = match.group(2)
        provisions.append({
            "document_number": document_number,
            "article": article,
            "clause": clause,
            "scope": "clause_repealed" if clause else "article_repealed",
            "quote": match.group(0).strip(),
        })

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for provision in provisions:
        key = (provision["article"], provision["clause"])
        if key not in seen:
            seen.add(key)
            unique.append(provision)
    return unique


def extract_legal_evidence(
    text: str,
    *,
    title: str,
    url: str,
    official: bool,
    full_text: bool,
) -> dict[str, Any]:
    """Extract candidates; mark a relation verified only from exact full-text quotes."""
    text = re.sub(r"\r\n?", "\n", text or "")
    all_numbers = list(dict.fromkeys(_DOC_RE.findall(text)))
    title_numbers = list(dict.fromkeys(_DOC_RE.findall(title or "")))
    own_number = title_numbers[0] if len(title_numbers) == 1 else (
        all_numbers[0] if len(all_numbers) == 1 else ""
    )
    effective_date, effective_quote = _effective_date(text)
    publication_date = _publication_date(text)
    force_match = re.search(
        r'"legislationLegalForce"\s*:\s*"([^"]+)"',
        text or "",
        re.I,
    )
    document_effective_status = (
        "in_force" if force_match and force_match.group(1).lower() in {
            "inforce", "in_force", "in force", "currentlyinforce"
        } else None
    )
    relations: list[dict[str, Any]] = []

    for source_article, section in _article_sections(text):
        lower = section.lower()
        if not re.search(r"sửa đổi|bổ sung|bãi bỏ|thay thế|hướng dẫn thi hành", lower):
            continue
        targets = [number for number in dict.fromkeys(_DOC_RE.findall(section))
                   if number.lower() != own_number.lower()]
        if not targets:
            continue
        relation_types = set()
        if re.search(r"bãi bỏ", lower):
            relation_types.add("REPEALS")
        if re.search(r"hướng dẫn thi hành", lower):
            relation_types.add("GUIDES")
        if re.search(r"sửa đổi|bổ sung|thay thế", lower):
            relation_types.add("AMENDS")
        relation_type = next(iter(relation_types)) if len(relation_types) == 1 else "MIXED"

        # A provision can only be attributed safely when this amendment section
        # identifies exactly one target document. Keep multi-target sections
        # as candidates without assigning one target's provisions to another.
        provisions: list[dict[str, str]] = []
        patterns = (
            (r"bổ sung\s+Điều\s+(\d+[a-z]?)", "article_added"),
            (r"sửa đổi\s+(?:toàn bộ\s+)?Điều\s+(\d+[a-z]?)", "article_amended"),
            (r"bãi bỏ\s+Điều\s+(\d+[a-z]?)", "article_repealed"),
            (r"thay thế\s+Điều\s+(\d+[a-z]?)", "article_replaced"),
            (r"bổ sung\s+(?:khoản|điểm)\s+(\d+[a-z]?)\b[^\n.]{0,240}?Điều\s+(\d+[a-z]?)", "clause_added"),
            (r"sửa đổi\s+(?:khoản|điểm)\s+(\d+[a-z]?)\b[^\n.]{0,240}?Điều\s+(\d+[a-z]?)", "clause_amended"),
            (r"bãi bỏ\s+(?:khoản|điểm)\s+(\d+[a-z]?)\b[^\n.]{0,240}?Điều\s+(\d+[a-z]?)", "clause_repealed"),
            (r"thay thế\s+(?:khoản|điểm)\s+(\d+[a-z]?)\b[^\n.]{0,240}?Điều\s+(\d+[a-z]?)", "clause_replaced"),
        )
        for pattern, scope in patterns if len(targets) == 1 else ():
            for match in re.finditer(pattern, section, re.IGNORECASE):
                quote = match.group(0).strip()
                if quote and quote in section:
                    if scope.startswith("article_"):
                        article, clause = match.group(1), ""
                    else:
                        clause, article = match.group(1), match.group(2)
                    provisions.append({
                        "document_number": targets[0],
                        "article": article,
                        "clause": clause,
                        "scope": scope,
                        "quote": quote,
                    })
        if relation_type == "REPEALS" and len(targets) == 1:
            provisions.extend(_repealed_provisions(section, targets[0]))
        if not provisions and len(targets) == 1:
            article_refs = list(re.finditer(r"\bĐiều\s+(\d+[a-z]?)", section, re.IGNORECASE))
            for ref in article_refs:
                quote = section[max(0, ref.start() - 100):min(len(section), ref.end() + 180)].strip()
                if ref.group(1) != source_article and quote:
                    provisions.append({
                        "document_number": targets[0], "article": ref.group(1),
                        "scope": "article_scope_unclassified", "quote": quote,
                    })

        relation_quote = section[:6000].strip()
        for target in targets:
            rel = {
                "relation_type": relation_type,
                "from_document": own_number,
                "to_document": target,
                "source_location": f"Điều {source_article}" if source_article else "unknown",
                "quote": relation_quote,
                "affected_provisions": [p for p in provisions
                                         if p["document_number"] == target],
                "status": "candidate",
            }
            complete = bool(
                official and full_text and own_number
                and own_number.lower() in text.lower() and effective_date
                and len(targets) == 1 and relation_type in {"AMENDS", "REPEALS", "GUIDES"}
                and rel["affected_provisions"]
                and all(p["scope"] != "article_scope_unclassified"
                        for p in rel["affected_provisions"])
                and target in relation_quote
                and relation_quote in text
                and effective_quote in text
                and all(p["quote"] in text for p in rel["affected_provisions"])
            )
            if complete:
                rel["status"] = "verified"
                rel["verified_relationship"] = {
                    "verified": True,
                    "relation_type": relation_type,
                    "from_document": own_number,
                    "to_document": target,
                    "effective_date": effective_date,
                    "relationship_checked": True,
                    "affected_provisions": rel["affected_provisions"],
                    "source": {
                        "official": True, "url": url,
                        "citation": rel["source_location"],
                        "excerpt": relation_quote,
                        "effective_date_excerpt": effective_quote,
                        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    },
                }
            relations.append(rel)

    return {
        "source_document": own_number or None,
        "publication_date": publication_date or None,
        "effective_date": effective_date or None,
        "effective_date_excerpt": effective_quote or None,
        "document_effective_status": document_effective_status,
        "provision_effective_status": None,
        "relations": relations,
        "status": (
            "verified_relationship" if any(r["status"] == "verified" for r in relations)
            else "incomplete" if relations or effective_date else "not_found"
        ),
        "full_text": bool(full_text),
    }
