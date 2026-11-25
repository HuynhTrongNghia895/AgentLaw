from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any


SUBCHUNK_CHAR_THRESHOLD = 2500

DOCUMENT_NUMBER_PATTERN = re.compile(
    r"\b\d+/\d{4}/(?:QH\d+|NĐ-CP|ND-CP|TT-[A-ZĐ]+|NQ-[A-ZĐ]+|QĐ-[A-ZĐ]+)\b",
    re.IGNORECASE
)

VIETNAMESE_DATE_PATTERN = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE
)

ARTICLE_REFERENCE_PATTERN = re.compile(
    r"\bĐiều\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE
)

CLAUSE_REFERENCE_PATTERN = re.compile(
    r"\bkhoản\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE
)

POINT_REFERENCE_PATTERN = re.compile(
    r"(?<!thời )\bđiểm\s+([a-zđ]\d*)\b",
    re.IGNORECASE
)

AFTER_CLAUSE_PATTERN = re.compile(
    r"\bvào\s+sau\s+khoản\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE
)

BEFORE_CLAUSE_PATTERN = re.compile(
    r"\bvào\s+trước\s+khoản\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE
)

AFTER_POINT_PATTERN = re.compile(
    r"\bvào\s+sau\s+điểm\s+([a-zđ]\d*)",
    re.IGNORECASE
)

BEFORE_POINT_PATTERN = re.compile(
    r"\bvào\s+trước\s+điểm\s+([a-zđ]\d*)",
    re.IGNORECASE
)
INSTRUCTION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[a-zđ]\d*|\d+)[.)]\s*",
    re.IGNORECASE
)

AMENDMENT_INSTRUCTION_PATTERN = re.compile(
    r"^(?:sửa đổi|bổ sung|bãi bỏ|thay thế)\b",
    re.IGNORECASE
)

ARTICLE_TITLE_COMPONENT_PATTERN = re.compile(
    r"\btên\s+Điều\b",
    re.IGNORECASE
)

OPENING_PARAGRAPH_COMPONENT_PATTERN = re.compile(
    r"\bđoạn\s+mở\s+đầu\b",
    re.IGNORECASE
)

def normalize_id_part(value: str) -> str:
    normalized = str(value).strip()

    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")

    return re.sub(
        r"[^0-9a-zA-ZÀ-ỹ_]+",
        "",
        normalized
    )


def unique_values(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


def convert_vietnamese_date(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    match = VIETNAMESE_DATE_PATTERN.search(text)

    if not match:
        return None

    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))

    try:
        parsed_date = date(year, month, day)
    except ValueError:
        return None

    return parsed_date.isoformat()


def join_content_blocks(blocks: list[str]) -> str:
    return "\n".join(
        normalize_string_list(blocks)
    )


def extract_document_type(
    header: list[str]
) -> str | None:
    header_text = "\n".join(
        normalize_string_list(header)
    ).upper()

    document_types = (
        "LUẬT",
        "NGHỊ ĐỊNH",
        "THÔNG TƯ",
        "NGHỊ QUYẾT",
        "QUYẾT ĐỊNH",
    )

    for document_type in document_types:
        pattern = (
            rf"(^|\n)\s*"
            rf"{re.escape(document_type)}"
            rf"\s*($|\n)"
        )

        if re.search(pattern, header_text):
            return document_type.title()

    return None


def extract_document_title(
    header: list[str]
) -> str | None:
    ignored_values = {
        "LUẬT",
        "NGHỊ ĐỊNH",
        "THÔNG TƯ",
        "NGHỊ QUYẾT",
        "QUYẾT ĐỊNH",
        "QUỐC HỘI",
        "CHÍNH PHỦ",
    }

    candidates: list[str] = []

    for block in normalize_string_list(header):
        for line in block.splitlines():
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            upper_line = cleaned_line.upper()

            if upper_line in ignored_values:
                continue

            if (
                "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
                in upper_line
            ):
                continue

            if (
                "ĐỘC LẬP - TỰ DO - HẠNH PHÚC"
                in upper_line
            ):
                continue

            if upper_line.startswith("LUẬT SỐ:"):
                continue

            if (
                upper_line == cleaned_line
                and len(cleaned_line) >= 10
            ):
                candidates.append(cleaned_line)

    if not candidates:
        return None

    return candidates[-1]


def extract_issuing_authority(
    header: list[str]
) -> str | None:
    header_text = "\n".join(
        normalize_string_list(header)
    ).upper()

    authorities = (
        "QUỐC HỘI",
        "CHÍNH PHỦ",
        "BỘ TÀI CHÍNH",
        "CHỦ TỊCH NƯỚC",
        "THỦ TƯỚNG CHÍNH PHỦ",
    )

    for authority in authorities:
        if authority in header_text:
            return authority.title()

    return None


def extract_signer(
    signature: list[str]
) -> str | None:
    lines: list[str] = []

    for block in normalize_string_list(signature):
        lines.extend(
            line.strip()
            for line in block.splitlines()
            if line.strip()
        )

    if not lines:
        return None

    title_prefixes = (
        "CHỦ TỊCH",
        "THỦ TƯỚNG",
        "PHÓ THỦ TƯỚNG",
        "BỘ TRƯỞNG",
        "KT.",
        "TM.",
        "TUQ.",
    )

    for line in reversed(lines):
        if not line.upper().startswith(
            title_prefixes
        ):
            return line

    return None


def extract_issued_date(
    header: list[str]
) -> str | None:
    header_text = "\n".join(
        normalize_string_list(header)
    )

    return convert_vietnamese_date(
        header_text
    )


def extract_effective_date(
    document: dict[str, Any]
) -> str | None:
    for article in document.get("articles", []):
        article_title = str(
            article.get("article_title", "")
        ).strip()

        article_parts = [
            article_title,
            *normalize_string_list(
                article.get("body_blocks", [])
            ),
        ]

        for clause in article.get("clauses", []):
            clause_title = str(
                clause.get("clause_title", "")
            ).strip()

            if clause_title:
                article_parts.append(
                    clause_title
                )

            article_parts.extend(
                normalize_string_list(
                    clause.get(
                        "content_blocks",
                        []
                    )
                )
            )

        article_text = "\n".join(
            article_parts
        )

        if (
            "hiệu lực" in article_title.lower()
            or "có hiệu lực" in article_text.lower()
        ):
            effective_date = (
                convert_vietnamese_date(
                    article_text
                )
            )

            if effective_date:
                return effective_date

    return None


def extract_amended_document_numbers(
    header: list[str],
    current_document_number: str
) -> list[str]:
    header_text = "\n".join(
        normalize_string_list(header)
    )

    document_numbers = (
        DOCUMENT_NUMBER_PATTERN.findall(
            header_text
        )
    )

    normalized_current = (
        current_document_number.upper()
    )

    return unique_values([
        number
        for number in document_numbers
        if number.upper() != normalized_current
    ])


def extract_document_metadata(
    document: dict[str, Any],
    document_number: str
) -> dict[str, Any]:
    header = normalize_string_list(
        document.get("header", [])
    )

    signature = normalize_string_list(
        document.get("signature", [])
    )

    return {
        "document_number": document_number,
        "document_type":
            extract_document_type(header),
        "document_title":
            extract_document_title(header),
        "issuing_authority":
            extract_issuing_authority(header),
        "issued_date":
            extract_issued_date(header),
        "effective_date":
            extract_effective_date(document),
        "signer":
            extract_signer(signature),
        "related_document_numbers":
            extract_amended_document_numbers(
                header=header,
                current_document_number=
                    document_number
            ),
        "legal_relationships":
            extract_legal_relationships(
                document=document,
                current_document_number=
                    document_number
        ),
    }

def extract_target_document_name(
    evidence: str,
    document_number: str
) -> str | None:
    pattern = re.compile(
        rf"((?:Luật|Nghị định|Thông tư|Nghị quyết|Quyết định)"
        rf"(?:\s+[^.;:\n]{{0,140}}?)?)"
        rf"\s+số\s+{re.escape(document_number)}",
        re.IGNORECASE
    )

    match = pattern.search(evidence)

    if not match:
        return None

    document_name = re.sub(
        r"^\s*Căn\s+cứ\s+",
        "",
        match.group(1),
        flags=re.IGNORECASE
    ).strip(" ,;:")

    generic_names = {
        "luật",
        "nghị định",
        "thông tư",
        "nghị quyết",
        "quyết định",
    }

    if document_name.lower() in generic_names:
        return None

    return document_name


def add_legal_relationship(
    relationships: list[dict[str, Any]],
    relation_type: str,
    target_document_number: str,
    evidence: str,
    target_document_name: str | None = None
) -> None:
    for relationship in relationships:
        same_type = (
            relationship.get("relation_type")
            == relation_type
        )

        same_document = (
            str(
                relationship.get(
                    "target_document_number",
                    ""
                )
            ).upper()
            == target_document_number.upper()
        )

        if same_type and same_document:
            return

    relationship: dict[str, Any] = {
        "relation_type": relation_type,
        "target_document_number":
            target_document_number,
        "evidence": evidence.strip(),
    }

    if target_document_name:
        relationship["target_document_name"] = (
            target_document_name
        )

    relationships.append(relationship)


def extract_legal_relationships(
    document: dict[str, Any],
    current_document_number: str
) -> list[dict[str, Any]]:
    header = normalize_string_list(
        document.get("header", [])
    )

    relationships: list[dict[str, Any]] = []

    normalized_current = (
        current_document_number.upper()
    )

    evidence_blocks: list[str] = []

    for block in header:
        parts = re.split(
            r"(?<=[.;])\s+|\n+",
            block
        )

        evidence_blocks.extend(
            part.strip()
            for part in parts
            if part.strip()
        )

    previous_amendment_pattern = re.compile(
        r"đã\s+được\s+sửa\s+đổi"
        r"(?:\s*,\s*bổ\s+sung)?"
        r".{0,180}?"
        r"(?:theo|bởi)\s+"
        r"(?:Luật|Nghị định|Thông tư|"
        r"Nghị quyết|Quyết định)?"
        r"\s*(?:số\s+)?"
        r"("
        + DOCUMENT_NUMBER_PATTERN.pattern
        + r")",
        re.IGNORECASE
    )

    primary_amended_pattern = re.compile(
        r"(?:của|cho)\s+"
        r"((?:Luật|Nghị định|Thông tư|"
        r"Nghị quyết|Quyết định)"
        r"[^.;\n]{0,120}?)"
        r"\s+số\s+"
        r"("
        + DOCUMENT_NUMBER_PATTERN.pattern
        + r")",
        re.IGNORECASE
    )

    for evidence in evidence_blocks:
        document_numbers = unique_values([
            number
            for number
            in DOCUMENT_NUMBER_PATTERN.findall(
                evidence
            )
            if number.upper()
            != normalized_current
        ])

        if not document_numbers:
            continue

        lower_evidence = evidence.lower()

        if lower_evidence.startswith("căn cứ"):
            for number in document_numbers:
                add_legal_relationship(
                    relationships=relationships,
                    relation_type="cites",
                    target_document_number=number,
                    target_document_name=None,
                    evidence=evidence
                )

            continue

        previous_numbers = unique_values(
            previous_amendment_pattern.findall(
                evidence
            )
        )

        primary_match = (
            primary_amended_pattern.search(
                evidence
            )
        )

        if primary_match:
            target_name = (
                primary_match.group(1)
                .strip(" ,;:")
            )

            target_number = (
                primary_match.group(2)
            )

            target_name = re.sub(
                r"^(?:Luật|Nghị định|Thông tư|"
                r"Nghị quyết|Quyết định)\s+"
                r"sửa đổi,\s*bổ sung\s+"
                r"một\s+số\s+điều\s+của\s+",
                "",
                target_name,
                flags=re.IGNORECASE
            )

            add_legal_relationship(
                relationships=relationships,
                relation_type="amends",
                target_document_number=
                    target_number,
                target_document_name=
                    target_name,
                evidence=evidence
            )

        for number in previous_numbers:
            add_legal_relationship(
                relationships=relationships,
                relation_type=
                    "previously_amended_by",
                target_document_number=number,
                target_document_name=None,
                evidence=evidence
            )

        known_numbers = {
            str(
                relationship.get(
                    "target_document_number",
                    ""
                )
            ).upper()
            for relationship in relationships
        }

        for number in document_numbers:
            if number.upper() not in known_numbers:
                add_legal_relationship(
                    relationships=relationships,
                    relation_type="cites",
                    target_document_number=number,
                    target_document_name=None,
                    evidence=evidence
                )

    return relationships

def detect_amendment_action(
    text: str
) -> str | None:
    normalized = str(text).strip().lower()

    if normalized.startswith("bãi bỏ"):
        return "repeal"

    if normalized.startswith("thay thế"):
        return "replace"

    if normalized.startswith(
        "sửa đổi, bổ sung"
    ):
        return "amend_and_add"

    if normalized.startswith("bổ sung"):
        return "add"

    if normalized.startswith("sửa đổi"):
        return "amend"

    return None


def extract_amendment_targets(
    clause_title: str
) -> dict[str, Any]:
    title = str(clause_title).strip()

    target_articles = unique_values(
        ARTICLE_REFERENCE_PATTERN.findall(
            title
        )
    )

    all_clauses = unique_values(
        CLAUSE_REFERENCE_PATTERN.findall(
            title
        )
    )

    all_points = unique_values([
        point.lower()
        for point in
        POINT_REFERENCE_PATTERN.findall(
            title
        )
    ])

    after_clauses = unique_values(
        AFTER_CLAUSE_PATTERN.findall(
            title
        )
    )

    before_clauses = unique_values(
        BEFORE_CLAUSE_PATTERN.findall(
            title
        )
    )

    after_points = unique_values([
        point.lower()
        for point in
        AFTER_POINT_PATTERN.findall(
            title
        )
    ])

    before_points = unique_values([
        point.lower()
        for point in
        BEFORE_POINT_PATTERN.findall(
            title
        )
    ])

    anchor_clauses = unique_values(
        after_clauses
        + before_clauses
    )

    anchor_points = unique_values(
        after_points
        + before_points
    )

    target_clauses = [
        clause
        for clause in all_clauses
        if clause not in anchor_clauses
    ]

    target_points = [
        point
        for point in all_points
        if point not in anchor_points
    ]

    anchor_relation: str | None = None

    if after_clauses or after_points:
        anchor_relation = "after"

    elif before_clauses or before_points:
        anchor_relation = "before"

    return {
        "amendment_action":
            detect_amendment_action(title),
        "target_articles":
            target_articles,
        "target_clauses":
            target_clauses,
        "target_points":
            target_points,
        "anchor_relation":
            anchor_relation,
        "anchor_articles":
            target_articles
            if anchor_relation
            else [],
        "anchor_clauses":
            anchor_clauses,
        "anchor_points":
            anchor_points,
    }

def strip_instruction_prefix(
    text: str
) -> str:
    return INSTRUCTION_PREFIX_PATTERN.sub(
        "",
        str(text).strip()
    ).strip()


def is_amendment_instruction(
    text: str
) -> bool:
    normalized = strip_instruction_prefix(text)

    return bool(
        AMENDMENT_INSTRUCTION_PATTERN.match(
            normalized
        )
    )


def extract_primary_target_document_number(
    document_metadata: dict[str, Any]
) -> str | None:
    relationships = document_metadata.get(
        "legal_relationships",
        []
    )

    if not isinstance(relationships, list):
        return None

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue

        if relationship.get("relation_type") != "amends":
            continue

        document_number = str(
            relationship.get(
                "target_document_number",
                ""
            )
        ).strip()

        if document_number:
            return document_number

    return None


def extract_operation_context(
    clause_title: str
) -> dict[str, str | None]:
    title = str(clause_title).strip()

    article_numbers = unique_values(
        ARTICLE_REFERENCE_PATTERN.findall(
            title
        )
    )

    clause_numbers = unique_values(
        CLAUSE_REFERENCE_PATTERN.findall(
            title
        )
    )

    anchor_clauses = unique_values(
        AFTER_CLAUSE_PATTERN.findall(title)
        + BEFORE_CLAUSE_PATTERN.findall(title)
    )

    context_clauses = [
        clause_number
        for clause_number in clause_numbers
        if clause_number not in anchor_clauses
    ]

    return {
        "target_article": (
            article_numbers[-1]
            if article_numbers
            else None
        ),
        "target_clause": (
            context_clauses[-1]
            if context_clauses
            else None
        ),
    }


def build_amendment_operation(
    action: str,
    target_document_number: str | None,
    target_article: str | None,
    target_clause: str | None,
    target_point: str | None,
    evidence: str,
    anchor_relation: str | None = None,
    anchor_article: str | None = None,
    anchor_clause: str | None = None,
    anchor_point: str | None = None,
    target_component: str | None = None,
    confidence: float = 1.0,
    needs_review: bool = False
) -> dict[str, Any]:
    return {
        "operation_id": None,
        "action": action,
        "target_document_number":
            target_document_number,
        "target_article": target_article,
        "target_clause": target_clause,
        "target_point": target_point,
        "target_component": target_component,
        "anchor_relation": anchor_relation,
        "anchor_article": anchor_article,
        "anchor_clause": anchor_clause,
        "anchor_point": anchor_point,
        "evidence": evidence.strip(),
        "confidence": confidence,
        "needs_review": needs_review,
    }



DATASET_TARGET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "thue_tndn": (
        "thuế thu nhập doanh nghiệp",
    ),
    "thue_gtgt": (
        "thuế giá trị gia tăng",
    ),
    "doanh_nghiep": (
        "luật doanh nghiệp",
    ),
    "dau_tu": (
        "luật đầu tư",
    ),
    "quan_ly_thue": (
        "quản lý thuế",
    ),
}

POINT_CLAUSE_ARTICLE_TARGET_PATTERN = re.compile(
    r"\bđiểm\s+([a-zđ]\d*)\s+"
    r"khoản\s+(\d+[a-zA-Z]?)\s+"
    r"Điều\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE
)

CLAUSE_ARTICLE_TARGET_PATTERN = re.compile(
    r"\bkhoản\s+(\d+[a-zA-Z]?)\s+"
    r"Điều\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE
)

MULTI_ARTICLE_TARGET_PATTERN = re.compile(
    r"\bcác\s+điều\s+"
    r"((?:\d+[a-zA-Z]?)(?:\s*,\s*\d+[a-zA-Z]?)*"
    r"(?:\s+và\s+\d+[a-zA-Z]?)?)",
    re.IGNORECASE
)


def mask_spans(
    text: str,
    spans: list[tuple[int, int]]
) -> str:
    characters = list(text)

    for start, end in spans:
        for index in range(start, end):
            characters[index] = " "

    return "".join(characters)


def extract_segment_targets(
    reference_text: str
) -> list[dict[str, str | None]]:
    targets: list[dict[str, str | None]] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in POINT_CLAUSE_ARTICLE_TARGET_PATTERN.finditer(
        reference_text
    ):
        targets.append({
            "target_article": match.group(3),
            "target_clause": match.group(2),
            "target_point": match.group(1).lower(),
        })
        occupied_spans.append(match.span())

    remaining_text = mask_spans(
        reference_text,
        occupied_spans
    )
    occupied_spans = []

    for match in CLAUSE_ARTICLE_TARGET_PATTERN.finditer(
        remaining_text
    ):
        targets.append({
            "target_article": match.group(2),
            "target_clause": match.group(1),
            "target_point": None,
        })
        occupied_spans.append(match.span())

    remaining_text = mask_spans(
        remaining_text,
        occupied_spans
    )
    occupied_spans = []

    for match in MULTI_ARTICLE_TARGET_PATTERN.finditer(
        remaining_text
    ):
        article_numbers = re.findall(
            r"\d+[a-zA-Z]?",
            match.group(1)
        )

        for article_number in article_numbers:
            targets.append({
                "target_article": article_number,
                "target_clause": None,
                "target_point": None,
            })

        occupied_spans.append(match.span())

    remaining_text = mask_spans(
        remaining_text,
        occupied_spans
    )

    for article_number in ARTICLE_REFERENCE_PATTERN.findall(
        remaining_text
    ):
        targets.append({
            "target_article": article_number,
            "target_clause": None,
            "target_point": None,
        })

    unique_targets: list[dict[str, str | None]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for target in targets:
        key = (
            target.get("target_article"),
            target.get("target_clause"),
            target.get("target_point"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_targets.append(target)

    return unique_targets


def filter_segments_for_dataset(
    segments: list[str],
    dataset_target: str
) -> list[str]:
    keywords = DATASET_TARGET_KEYWORDS.get(
        str(dataset_target).strip().lower(),
        ()
    )

    if not keywords:
        return segments

    matched_segments = [
        segment
        for segment in segments
        if any(
            keyword in segment.lower()
            for keyword in keywords
        )
    ]

    return matched_segments or segments


def parse_multi_document_instruction(
    evidence: str,
    dataset_target: str
) -> list[dict[str, Any]] | None:
    original_evidence = str(evidence).strip()
    instruction = strip_instruction_prefix(
        original_evidence
    )
    action = detect_amendment_action(
        instruction
    )

    if action is None:
        return None

    segments = [
        segment.strip()
        for segment in re.split(
            r";\s*",
            instruction
        )
        if segment.strip()
    ]

    target_segments = [
        segment
        for segment in segments
        if DOCUMENT_NUMBER_PATTERN.search(segment)
    ]

    if len(target_segments) < 2:
        return None

    target_segments = filter_segments_for_dataset(
        target_segments,
        dataset_target
    )

    operations: list[dict[str, Any]] = []

    for segment in target_segments:
        document_match = DOCUMENT_NUMBER_PATTERN.search(
            segment
        )

        if document_match is None:
            continue

        target_document_number = document_match.group(0)
        reference_text = segment[:document_match.start()]
        targets = extract_segment_targets(
            reference_text
        )

        for target in targets:
            operations.append(
                build_amendment_operation(
                    action=action,
                    target_document_number=
                        target_document_number,
                    target_article=target.get(
                        "target_article"
                    ),
                    target_clause=target.get(
                        "target_clause"
                    ),
                    target_point=target.get(
                        "target_point"
                    ),
                    evidence=segment,
                    confidence=1.0,
                    needs_review=False
                )
            )

    return operations


def parse_amendment_instruction(
    evidence: str,
    context_article: str | None,
    context_clause: str | None,
    target_document_number: str | None
) -> list[dict[str, Any]]:
    original_evidence = str(evidence).strip()
    instruction = strip_instruction_prefix(
        original_evidence
    )

    if not instruction:
        return []

    action = detect_amendment_action(
        instruction
    )

    if action is None:
        return []

    article_numbers = unique_values(
        ARTICLE_REFERENCE_PATTERN.findall(
            instruction
        )
    )

    target_article = (
        article_numbers[-1]
        if article_numbers
        else context_article
    )

    clause_numbers = unique_values(
        CLAUSE_REFERENCE_PATTERN.findall(
            instruction
        )
    )

    point_numbers = unique_values([
        point.lower()
        for point in
        POINT_REFERENCE_PATTERN.findall(
            instruction
        )
    ])

    after_clauses = unique_values(
        AFTER_CLAUSE_PATTERN.findall(
            instruction
        )
    )

    before_clauses = unique_values(
        BEFORE_CLAUSE_PATTERN.findall(
            instruction
        )
    )

    after_points = unique_values([
        point.lower()
        for point in
        AFTER_POINT_PATTERN.findall(
            instruction
        )
    ])

    before_points = unique_values([
        point.lower()
        for point in
        BEFORE_POINT_PATTERN.findall(
            instruction
        )
    ])

    anchor_relation: str | None = None

    if after_clauses or after_points:
        anchor_relation = "after"

    elif before_clauses or before_points:
        anchor_relation = "before"

    anchor_clause = None

    if after_clauses:
        anchor_clause = after_clauses[-1]

    elif before_clauses:
        anchor_clause = before_clauses[-1]

    anchor_point = None

    if after_points:
        anchor_point = after_points[-1]

    elif before_points:
        anchor_point = before_points[-1]

    target_clauses = [
        clause_number
        for clause_number in clause_numbers
        if clause_number not in (
            after_clauses
            + before_clauses
        )
    ]

    target_points = [
        point_number
        for point_number in point_numbers
        if point_number not in (
            after_points
            + before_points
        )
    ]

    operations: list[dict[str, Any]] = []

    if target_points:
        parent_clause = (
            target_clauses[-1]
            if target_clauses
            else context_clause
        )

        for target_point in target_points:
            operations.append(
                build_amendment_operation(
                    action=action,
                    target_document_number=
                        target_document_number,
                    target_article=
                        target_article,
                    target_clause=
                        parent_clause,
                    target_point=
                        target_point,
                    anchor_relation=
                        anchor_relation,
                    anchor_article=(
                        target_article
                        if anchor_relation
                        else None
                    ),
                    anchor_clause=(
                        parent_clause
                        if anchor_relation
                        else None
                    ),
                    anchor_point=
                        anchor_point,
                    evidence=
                        original_evidence
                )
            )

        return operations

    if target_clauses:
        for target_clause in target_clauses:
            operations.append(
                build_amendment_operation(
                    action=action,
                    target_document_number=
                        target_document_number,
                    target_article=
                        target_article,
                    target_clause=
                        target_clause,
                    target_point=None,
                    anchor_relation=
                        anchor_relation,
                    anchor_article=(
                        target_article
                        if anchor_relation
                        else None
                    ),
                    anchor_clause=
                        anchor_clause,
                    anchor_point=None,
                    evidence=
                        original_evidence
                )
            )

        return operations

    target_component = None

    if ARTICLE_TITLE_COMPONENT_PATTERN.search(
        instruction
    ):
        target_component = "article_title"

    elif OPENING_PARAGRAPH_COMPONENT_PATTERN.search(
        instruction
    ):
        target_component = "opening_paragraph"

    if target_component:
        component_action = action

        if action == "amend_and_add":
            component_action = "amend"

        operations.append(
            build_amendment_operation(
                action=component_action,
                target_document_number=
                    target_document_number,
                target_article=
                    target_article,
                target_clause=None,
                target_point=None,
                target_component=
                    target_component,
                evidence=
                    original_evidence
            )
        )

        return operations

    if (
        target_article
        and (
            action == "replace"
            or instruction.lower().startswith(
                "bãi bỏ điều"
            )
        )
    ):
        operations.append(
            build_amendment_operation(
                action=action,
                target_document_number=
                    target_document_number,
                target_article=
                    target_article,
                target_clause=None,
                target_point=None,
                evidence=
                    original_evidence
            )
        )

        return operations

    operations.append(
        build_amendment_operation(
            action=action,
            target_document_number=
                target_document_number,
            target_article=
                target_article,
            target_clause=
                context_clause,
            target_point=None,
            evidence=
                original_evidence,
            confidence=0.4,
            needs_review=True
        )
    )

    return operations


def remove_duplicate_operations(
    operations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    unique_operations: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    for operation in operations:
        operation_key = (
            operation.get("action"),
            operation.get(
                "target_document_number"
            ),
            operation.get("target_article"),
            operation.get("target_clause"),
            operation.get("target_point"),
            operation.get("target_component"),
            operation.get("anchor_relation"),
            operation.get("anchor_article"),
            operation.get("anchor_clause"),
            operation.get("anchor_point"),
            operation.get("evidence"),
        )

        if operation_key in seen_keys:
            continue

        seen_keys.add(operation_key)
        unique_operations.append(operation)

    return unique_operations


def extract_amendment_operations(
    record_id: str,
    clause_title: str,
    content_blocks: list[str],
    document_metadata: dict[str, Any],
    dataset_target: str
) -> list[dict[str, Any]]:
    context = extract_operation_context(
        clause_title
    )

    context_article = context.get(
        "target_article"
    )

    context_clause = context.get(
        "target_clause"
    )

    primary_target_document_number = (
        extract_primary_target_document_number(
            document_metadata
        )
    )

    clause_title_document_numbers = (
        DOCUMENT_NUMBER_PATTERN.findall(
            clause_title
        )
    )

    clause_target_document_number = (
        clause_title_document_numbers[0]
        if clause_title_document_numbers
        else primary_target_document_number
    )

    content_lines = normalize_string_list(
        content_blocks
    )

    detailed_instruction_lines = [
        line
        for line in content_lines
        if is_amendment_instruction(line)
    ]

    if detailed_instruction_lines:
        evidence_lines = (
            detailed_instruction_lines
        )
    else:
        evidence_lines = [clause_title]

    operations: list[dict[str, Any]] = []

    for evidence in evidence_lines:
        if not is_amendment_instruction(
            evidence
        ):
            continue

        multi_document_operations = (
            parse_multi_document_instruction(
                evidence=evidence,
                dataset_target=dataset_target
            )
        )

        if multi_document_operations is not None:
            operations.extend(
                multi_document_operations
            )
            continue

        evidence_document_numbers = (
            DOCUMENT_NUMBER_PATTERN.findall(
                evidence
            )
        )

        effective_target_document_number = (
            evidence_document_numbers[0]
            if evidence_document_numbers
            else clause_target_document_number
        )

        if not effective_target_document_number:
            continue

        parsed_operations = (
            parse_amendment_instruction(
                evidence=evidence,
                context_article=
                    context_article,
                context_clause=
                    context_clause,
                target_document_number=
                    effective_target_document_number
            )
        )

        operations.extend(
            parsed_operations
        )

    operations = remove_duplicate_operations(
        operations
    )

    for index, operation in enumerate(
        operations,
        start=1
    ):
        operation["operation_id"] = (
            f"{record_id}_OP{index}"
        )

    return operations


def summarize_amendment_operations(
    operations: list[dict[str, Any]],
    fallback_targets: dict[str, Any]
) -> dict[str, Any]:
    if not operations:
        return fallback_targets

    def collect_values(
        field_name: str,
        lowercase: bool = False
    ) -> list[str]:
        values: list[str] = []

        for operation in operations:
            raw_value = operation.get(
                field_name
            )

            if raw_value is None:
                continue

            value = str(raw_value).strip()

            if not value:
                continue

            if lowercase:
                value = value.lower()

            values.append(value)

        return unique_values(values)

    actions = collect_values(
        "action"
    )

    target_articles = collect_values(
        "target_article"
    )

    target_clauses = collect_values(
        "target_clause"
    )

    target_points = collect_values(
        "target_point",
        lowercase=True
    )

    anchor_relations = collect_values(
        "anchor_relation"
    )

    anchor_articles = collect_values(
        "anchor_article"
    )

    anchor_clauses = collect_values(
        "anchor_clause"
    )

    anchor_points = collect_values(
        "anchor_point",
        lowercase=True
    )

    if len(actions) == 1:
        amendment_action = actions[0]

    elif len(actions) > 1:
        amendment_action = "mixed"

    else:
        amendment_action = None

    if len(anchor_relations) == 1:
        anchor_relation = (
            anchor_relations[0]
        )

    elif len(anchor_relations) > 1:
        anchor_relation = "mixed"

    else:
        anchor_relation = None

    return {
        "amendment_action":
            amendment_action,
        "target_articles":
            target_articles,
        "target_clauses":
            target_clauses,
        "target_points":
            target_points,
        "anchor_relation":
            anchor_relation,
        "anchor_articles":
            anchor_articles,
        "anchor_clauses":
            anchor_clauses,
        "anchor_points":
            anchor_points,
    }

def build_article_text(
    article_number: str,
    article_title: str,
    body_blocks: list[str]
) -> str:
    parts = [
        (
            f"Điều {article_number}. "
            f"{article_title}"
        ).strip()
    ]

    parts.extend(
        normalize_string_list(body_blocks)
    )

    return "\n".join(parts)


def build_clause_text(
    article_number: str,
    article_title: str,
    clause_number: str,
    clause_title: str,
    content_blocks: list[str]
) -> str:
    parts = [
        (
            f"Điều {article_number}. "
            f"{article_title}"
        ).strip(),
        (
            f"Khoản {clause_number}. "
            f"{clause_title}"
        ).strip(),
    ]

    content_text = join_content_blocks(
        content_blocks
    )

    if content_text:
        parts.append(content_text)

    return "\n".join(parts)


def create_article_record(
    article: dict[str, Any],
    document_metadata: dict[str, Any],
    dataset_target: str,
    source_filename: str
) -> dict[str, Any]:
    document_number = str(
        document_metadata["document_number"]
    )

    article_number = str(
        article.get("article_number", "")
    ).strip()

    article_title = str(
        article.get("article_title", "")
    ).strip()

    body_blocks = normalize_string_list(
        article.get("body_blocks", [])
    )

    text = build_article_text(
        article_number=article_number,
        article_title=article_title,
        body_blocks=body_blocks
    )

    record_id = (
        f"{normalize_id_part(document_number)}"
        f"_D{normalize_id_part(article_number)}"
    )

    storage_id = (
        f"{normalize_id_part(dataset_target)}"
        f"_{record_id}"
    )

    return {
        "id": record_id,
        "storage_id": storage_id,
        "document_number": document_metadata.get(
            "document_number"
        ),
        "document_type": document_metadata.get(
            "document_type"
        ),
        "document_title": document_metadata.get(
            "document_title"
        ),
        "issuing_authority": document_metadata.get(
            "issuing_authority"
        ),
        "issued_date": document_metadata.get(
            "issued_date"
        ),
        "effective_date": document_metadata.get(
            "effective_date"
        ),
        "signer": document_metadata.get(
            "signer"
        ),
        "related_document_numbers":
            document_metadata.get(
                "related_document_numbers",
                []
            ),
        "legal_relationships":
            document_metadata.get(
                "legal_relationships",
                []
            ),
        "dataset": dataset_target,
        "source_filename": source_filename,
        "record_type": "article",
        "citation": (
            f"Điều {article_number}, "
            f"văn bản {document_number}"
        ),
        "part": article.get("part"),
        "chapter": article.get("chapter"),
        "section": article.get("section"),
        "article_number": article_number,
        "article_title": article_title,
        "article_body": "\n".join(body_blocks).strip(),
        "clause_number": None,
        "clause_title": None,
        "content_blocks": body_blocks,
        "amendment_action": None,
        "target_articles": [],
        "target_clauses": [],
        "target_points": [],
        "anchor_relation": None,
        "anchor_articles": [],
        "anchor_clauses": [],
        "anchor_points": [],
        "amendment_operations": [],
        "char_count": len(text),
        "needs_subchunk": (
            len(text)
            > SUBCHUNK_CHAR_THRESHOLD
        ),
        "text": text,
    }


def create_clause_record(
    article: dict[str, Any],
    clause: dict[str, Any],
    document_metadata: dict[str, Any],
    dataset_target: str,
    source_filename: str
) -> dict[str, Any]:
    document_number = str(
        document_metadata["document_number"]
    )

    article_number = str(
        article.get("article_number", "")
    ).strip()

    article_title = str(
        article.get("article_title", "")
    ).strip()

    clause_number = str(
        clause.get("clause_number", "")
    ).strip()

    clause_title = str(
        clause.get("clause_title", "")
    ).strip()

    content_blocks = normalize_string_list(
        clause.get("content_blocks", [])
    )

    # --- Phần chữ TRƯỚC điểm a) -> text của node Khoản -------------------
    # clause_title đã là câu dẫn ngay sau "Khoản N."; lead_blocks là text
    # thân giữa dòng khoản và điểm đầu tiên; tail_blocks là text sau điểm
    # cuối (vẫn thuộc khoản, không thuộc điểm).
    clause_body = "\n".join(
        part for part in [
            clause_title,
            *normalize_string_list(
                clause.get("lead_blocks", [])
            ),
            *normalize_string_list(
                clause.get("tail_blocks", [])
            ),
        ] if part
    ).strip()

    # --- Điểm: ký hiệu đã bỏ, text = nội dung điểm -----------------------
    clause_points = [
        {
            "marker": str(point.get("marker", "")).strip(),
            "text": str(point.get("text", "")).strip(),
        }
        for point in clause.get("points", [])
        if str(point.get("text", "")).strip()
    ]

    # --- Chữ TRƯỚC khoản -> text của node Điều --------------------------
    article_body = "\n".join(
        normalize_string_list(
            article.get("body_blocks", [])
        )
    ).strip()

    text = build_clause_text(
        article_number=article_number,
        article_title=article_title,
        clause_number=clause_number,
        clause_title=clause_title,
        content_blocks=content_blocks
    )

    amendment_targets = (
        extract_amendment_targets(
            clause_title
        )
    )

    record_id = (
        f"{normalize_id_part(document_number)}"
        f"_D{normalize_id_part(article_number)}"
        f"_K{normalize_id_part(clause_number)}"
    )

    storage_id = (
        f"{normalize_id_part(dataset_target)}"
        f"_{record_id}"
    )

    amendment_operations = (
        extract_amendment_operations(
            record_id=record_id,
            clause_title=clause_title,
            content_blocks=content_blocks,
            document_metadata=document_metadata,
            dataset_target=dataset_target
        )
    )

    amendment_summary = (
        summarize_amendment_operations(
            operations=amendment_operations,
            fallback_targets=amendment_targets
        )
    )

    return {
        "id": record_id,
        "storage_id": storage_id,
        "document_number": document_metadata.get(
            "document_number"
        ),
        "document_type": document_metadata.get(
            "document_type"
        ),
        "document_title": document_metadata.get(
            "document_title"
        ),
        "issuing_authority": document_metadata.get(
            "issuing_authority"
        ),
        "issued_date": document_metadata.get(
            "issued_date"
        ),
        "effective_date": document_metadata.get(
            "effective_date"
        ),
        "signer": document_metadata.get(
            "signer"
        ),
        "related_document_numbers":
            document_metadata.get(
                "related_document_numbers",
                []
            ),
        "legal_relationships":
            document_metadata.get(
                "legal_relationships",
                []
            ),
        "dataset": dataset_target,
        "source_filename": source_filename,
        "record_type": "clause",
        "citation": (
            f"Điều {article_number}, "
            f"Khoản {clause_number}, "
            f"văn bản {document_number}"
        ),
        "part": article.get("part"),
        "chapter": article.get("chapter"),
        "section": article.get("section"),
        "article_number": article_number,
        "article_title": article_title,
        "article_body": article_body,
        "clause_number": clause_number,
        "clause_title": clause_title,
        "clause_body": clause_body,
        "clause_points": clause_points,
        "content_blocks": content_blocks,
        "amendment_action":
            amendment_summary.get(
                "amendment_action"
            ),
        "target_articles":
            amendment_summary.get(
                "target_articles",
                []
            ),
        "target_clauses":
            amendment_summary.get(
                "target_clauses",
                []
            ),
        "target_points":
            amendment_summary.get(
                "target_points",
                []
            ),
        "anchor_relation":
            amendment_summary.get(
                "anchor_relation"
            ),
        "anchor_articles":
            amendment_summary.get(
                "anchor_articles",
                []
            ),
        "anchor_clauses":
            amendment_summary.get(
                "anchor_clauses",
                []
            ),
        "anchor_points":
            amendment_summary.get(
                "anchor_points",
                []
            ),
        "amendment_operations":
            amendment_operations,
        "char_count": len(text),
        "needs_subchunk": (
            len(text)
            > SUBCHUNK_CHAR_THRESHOLD
        ),
        "text": text,
    }

AMENDS_RELATION_TYPES = {
    "AMENDS",
    "SUPPLEMENTS",
    "REPEALS",
    "REPLACES",
}


def load_relation_index(
    relations_csv: str | Path,
    registry_csv: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Đọc DocumentRelations.csv + DocumentRegistry.csv thành index
    theo so_hieu văn bản BỊ TÁC ĐỘNG (to_so_hieu), dùng để gắn
    trường amended_by vào record (Bug 5 - phương án A).

    Chỉ lấy quan hệ sửa đổi/thay thế/bãi bỏ (AMENDS, SUPPLEMENTS,
    REPEALS, REPLACES) và chỉ khi from_so_hieu khác to_so_hieu
    (liên kết tới chính văn bản của mình không mang thông tin).

    Không bịa dữ liệu: mọi thông tin lấy nguyên vẹn từ CSV nguồn.
    """
    import csv as _csv

    index: dict[str, list[dict[str, Any]]] = {}

    relations_path = Path(relations_csv)
    registry_path = Path(registry_csv)

    if not relations_path.exists():
        return index

    effective_dates: dict[str, str] = {}
    if registry_path.exists():
        with registry_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as registry_file:
            for row in _csv.DictReader(registry_file):
                so_hieu = str(row.get("so_hieu", "")).strip()
                if so_hieu:
                    effective_dates[so_hieu] = str(
                        row.get("effective_date", "")
                    ).strip()

    with relations_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as relations_file:
        for row in _csv.DictReader(relations_file):
            relation_type = str(
                row.get("relation_type", "")
            ).strip().upper()
            if relation_type not in AMENDS_RELATION_TYPES:
                continue

            from_so = str(row.get("from_so_hieu", "")).strip()
            to_so = str(row.get("to_so_hieu", "")).strip()
            if not to_so or to_so == from_so:
                continue

            entry = {
                "from_so_hieu": from_so,
                "relation_type": relation_type,
                "to_provision": str(
                    row.get("to_provision", "")
                ).strip(),
                "from_provision": str(
                    row.get("from_provision", "")
                ).strip(),
                "effective_date": effective_dates.get(
                    from_so, ""
                ),
            }

            index.setdefault(to_so, []).append(entry)

    return index


def parse_provision_segments(to_provision: str) -> list[dict[str, Any]]:
    """Tách to_provision thành các đoạn "Điều N (khoản M1, M2...)" theo dấu ';'.

    to_provision có thể liệt kê NHIỀU điều trong 1 quan hệ
    (vd. "Điều 12 (khoản 3); Điều 15 (khoản 9 điểm c)") — mỗi đoạn
    mô tả 1 điều bị tác động. Mỗi đoạn:
      - article: số điều nêu trong đoạn
      - clauses: tập khoản nêu trong đoạn (rỗng nếu đoạn không nêu khoản
        -> tác động cả điều)
    Quan hệ khớp record khi CÓ ÍT NHẤT MỘT đoạn khớp (article = record
    article, và nếu đoạn nêu khoản thì record clause phải nằm trong tập
    khoản đó). Chỉ phân tách theo ký hiệu có thật trong dữ liệu
    Relations (';' và '(...)'); không suy diễn nội dung.
    """
    segments: list[dict[str, Any]] = []

    for part in re.split(r";", to_provision):
        part = part.strip()
        if not part:
            continue

        article_match = re.search(
            r"Điều\s+(\d+[a-zA-Z]?)",
            part,
            re.IGNORECASE
        )

        if article_match is None:
            continue

        clause_numbers = set(
            re.findall(
                r"khoản\s+(\d+[a-zA-Z]?)",
                part,
                re.IGNORECASE
            )
        )

        segments.append({
            "article": article_match.group(1),
            "clauses": clause_numbers,
        })

    return segments


def match_amended_by(
    amended_by: list[dict[str, Any]],
    article_number: str,
    clause_number: str,
) -> list[dict[str, Any]]:
    """Lọc quan hệ ở level điều/khoản: quan hệ khớp khi MỌI đoạn
    to_provision (xem parse_provision_segments) khớp với record.

    - Quan hệ không nêu điều nào (cấp văn bản): không khớp ở level
      điều/khoản (đã được gắn ở level văn bản qua document_amended_by).
    - Quan hệ nêu đoạn điều/khoản khác record: không khớp.
    - Không bịa khớp: chỉ so khớp ký hiệu điều/khoản, không suy luận.
    """
    matched: list[dict[str, Any]] = []

    for entry in amended_by:
        segments = parse_provision_segments(
            entry.get("to_provision", "")
        )

        if not segments:
            continue

        has_article_match = False

        for segment in segments:
            if segment["article"] != str(article_number):
                continue

            # Đoạn khớp về điều. Nếu đoạn nêu khoản cụ thể, record
            # (level khoản) phải nằm trong tập khoản đó; đoạn không
            # nêu khoản = tác động cả điều, luôn khớp.
            if (
                str(clause_number)
                and segment["clauses"]
                and str(clause_number) not in segment["clauses"]
            ):
                continue

            has_article_match = True
            break

        if has_article_match:
            matched.append(entry)

    return matched


def build_annex_text(
    annex: dict[str, Any],
    document_number: str
) -> str:
    parts = [str(annex.get("title", "")).strip()]
    parts.extend(normalize_string_list(annex.get("content_blocks", [])))
    return "\n".join(part for part in parts if part).strip()


def create_annex_record(
    annex: dict[str, Any],
    document_metadata: dict[str, Any],
    dataset_target: str,
    source_filename: str,
    amended_by: list[dict[str, Any]],
) -> dict[str, Any]:
    document_number = str(
        document_metadata["document_number"]
    )

    annex_title = str(annex.get("title", "")).strip()
    content_blocks = normalize_string_list(
        annex.get("content_blocks", [])
    )

    text = build_annex_text(
        annex=annex,
        document_number=document_number
    )

    record_id = (
        f"{normalize_id_part(document_number)}"
        f"_A{int(annex.get('number', 0))}"
    )

    storage_id = (
        f"{normalize_id_part(dataset_target)}"
        f"_{record_id}"
    )

    return {
        "id": record_id,
        "storage_id": storage_id,
        "document_number": document_metadata.get(
            "document_number"
        ),
        "document_type": document_metadata.get(
            "document_type"
        ),
        "document_title": document_metadata.get(
            "document_title"
        ),
        "issuing_authority": document_metadata.get(
            "issuing_authority"
        ),
        "issued_date": document_metadata.get(
            "issued_date"
        ),
        "effective_date": document_metadata.get(
            "effective_date"
        ),
        "signer": document_metadata.get(
            "signer"
        ),
        "related_document_numbers":
            document_metadata.get(
                "related_document_numbers",
                []
            ),
        "legal_relationships":
            document_metadata.get(
                "legal_relationships",
                []
            ),
        "dataset": dataset_target,
        "source_filename": source_filename,
        "record_type": "annex",
        "citation": (
            f"{annex_title}, văn bản {document_number}"
        ),
        "article_number": None,
        "clause_number": None,
        "annex_number": annex.get("number"),
        "annex_title": annex_title,
        "content_blocks": content_blocks,
        "amended_by": amended_by,
        "document_amended_by": amended_by,
        "char_count": len(text),
        "needs_subchunk": (
            len(text)
            > SUBCHUNK_CHAR_THRESHOLD
        ),
        "text": text,
    }


def document_to_records(
    document: dict[str, Any],
    document_number: str,
    dataset_target: str,
    source_filename: str,
    relation_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    amended_by = (
        list(relation_index.get(document_number, []))
        if relation_index is not None
        else []
    )

    document_metadata = (
        extract_document_metadata(
            document=document,
            document_number=document_number
        )
    )

    for article in document.get(
        "articles",
        []
    ):
        clauses = article.get(
            "clauses",
            []
        )

        article_number = str(
            article.get("article_number", "")
        ).strip()

        if clauses:
            for clause in clauses:
                record = create_clause_record(
                    article=article,
                    clause=clause,
                    document_metadata=
                        document_metadata,
                    dataset_target=
                        dataset_target,
                    source_filename=
                        source_filename
                )

                clause_number = str(
                    clause.get("clause_number", "")
                ).strip()

                record["amended_by"] = match_amended_by(
                    amended_by=amended_by,
                    article_number=article_number,
                    clause_number=clause_number
                )

                record["document_amended_by"] = amended_by

                records.append(record)

        else:
            record = create_article_record(
                article=article,
                document_metadata=
                    document_metadata,
                dataset_target=
                    dataset_target,
                source_filename=
                    source_filename
            )

            record["amended_by"] = match_amended_by(
                amended_by=amended_by,
                article_number=article_number,
                clause_number=""
            )

            record["document_amended_by"] = amended_by

            records.append(record)

    for annex in document.get(
        "annexes",
        []
    ):
        record = create_annex_record(
            annex=annex,
            document_metadata=document_metadata,
            dataset_target=dataset_target,
            source_filename=source_filename,
            amended_by=amended_by
        )

        records.append(record)

    return records


def validate_records(
    records: list[dict[str, Any]]
) -> None:
    if not records:
        raise ValueError(
            "Không có record nào để xuất."
        )

    seen_ids: set[str] = set()
    seen_storage_ids: set[str] = set()

    required_fields = (
        "id",
        "storage_id",
        "document_number",
        "dataset",
        "source_filename",
        "record_type",
        "article_number",
        "text",
    )

    valid_operation_actions = {
        "amend",
        "amend_and_add",
        "add",
        "repeal",
        "replace",
    }

    for index, record in enumerate(
        records,
        start=1
    ):
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"Record {index} "
                    f"thiếu field: {field}"
                )

        record_id = str(
            record.get("id", "")
        ).strip()

        if not record_id:
            raise ValueError(
                f"Record {index} không có id."
            )

        if record_id in seen_ids:
            raise ValueError(
                f"ID bị trùng: {record_id}"
            )

        seen_ids.add(record_id)

        storage_id = str(
            record.get("storage_id", "")
        ).strip()

        if not storage_id:
            raise ValueError(
                f"Record {record_id} không có storage_id."
            )

        if storage_id in seen_storage_ids:
            raise ValueError(
                f"storage_id bị trùng: {storage_id}"
            )

        seen_storage_ids.add(storage_id)

        text = str(
            record.get("text", "")
        ).strip()

        if not text:
            raise ValueError(
                f"Record {record_id} "
                "không có text."
            )

        record_type = record.get(
            "record_type"
        )

        if record_type not in {
            "article",
            "clause",
            "annex",
        }:
            raise ValueError(
                f"Record {record_id} có "
                f"record_type không hợp lệ: "
                f"{record_type}"
            )

        operations = record.get(
            "amendment_operations",
            []
        )

        if not isinstance(
            operations,
            list
        ):
            raise ValueError(
                f"Record {record_id} có "
                "amendment_operations "
                "không phải danh sách."
            )

        seen_operation_ids: set[str] = set()

        for operation in operations:
            if not isinstance(
                operation,
                dict
            ):
                raise ValueError(
                    f"Record {record_id} có "
                    "operation không hợp lệ."
                )

            operation_id = str(
                operation.get(
                    "operation_id",
                    ""
                )
            ).strip()

            if not operation_id:
                raise ValueError(
                    f"Record {record_id} có "
                    "operation thiếu "
                    "operation_id."
                )

            if operation_id in seen_operation_ids:
                raise ValueError(
                    "Operation ID bị trùng: "
                    f"{operation_id}"
                )

            seen_operation_ids.add(
                operation_id
            )

            action = operation.get(
                "action"
            )

            if action not in valid_operation_actions:
                raise ValueError(
                    f"Operation {operation_id} "
                    f"có action không hợp lệ: "
                    f"{action}"
                )

            target_document_number = str(
                operation.get(
                    "target_document_number",
                    ""
                )
            ).strip()

            if not target_document_number:
                raise ValueError(
                    f"Operation {operation_id} "
                    "thiếu target_document_number."
                )

            target_article = str(
                operation.get(
                    "target_article",
                    ""
                )
            ).strip()

            target_clause = str(
                operation.get(
                    "target_clause",
                    ""
                )
            ).strip()

            target_point = str(
                operation.get(
                    "target_point",
                    ""
                )
            ).strip()

            target_component = str(
                operation.get(
                    "target_component",
                    ""
                )
            ).strip()

            if not any([
                target_article,
                target_clause,
                target_point,
                target_component,
            ]):
                raise ValueError(
                    f"Operation {operation_id} "
                    "không có đối tượng bị tác động."
                )

            anchor_relation = operation.get(
                "anchor_relation"
            )

            if anchor_relation not in {
                None,
                "after",
                "before",
            }:
                raise ValueError(
                    f"Operation {operation_id} "
                    "có anchor_relation "
                    f"không hợp lệ: "
                    f"{anchor_relation}"
                )

            if anchor_relation:
                anchor_article = str(
                    operation.get(
                        "anchor_article",
                        ""
                    )
                ).strip()

                anchor_clause = str(
                    operation.get(
                        "anchor_clause",
                        ""
                    )
                ).strip()

                anchor_point = str(
                    operation.get(
                        "anchor_point",
                        ""
                    )
                ).strip()

                if not any([
                    anchor_article,
                    anchor_clause,
                    anchor_point,
                ]):
                    raise ValueError(
                        f"Operation {operation_id} "
                        "có anchor_relation nhưng "
                        "không có anchor."
                    )

            evidence = str(
                operation.get(
                    "evidence",
                    ""
                )
            ).strip()

            if not evidence:
                raise ValueError(
                    f"Operation {operation_id} "
                    "thiếu evidence."
                )

            confidence = operation.get(
                "confidence"
            )

            if not isinstance(
                confidence,
                (int, float)
            ):
                raise ValueError(
                    f"Operation {operation_id} "
                    "có confidence không hợp lệ."
                )

            if not 0 <= confidence <= 1:
                raise ValueError(
                    f"Operation {operation_id} "
                    "có confidence ngoài khoảng "
                    "từ 0 đến 1."
                )

            needs_review = operation.get(
                "needs_review"
            )

            if not isinstance(
                needs_review,
                bool
            ):
                raise ValueError(
                    f"Operation {operation_id} "
                    "có needs_review "
                    "không phải boolean."
                )

        if record_type == "article":
            if operations:
                raise ValueError(
                    f"Record Điều {record_id} "
                    "không nên có "
                    "amendment_operations."
                )

        if record_type == "clause":
            clause_number = str(
                record.get(
                    "clause_number",
                    ""
                )
            ).strip()

            if not clause_number:
                raise ValueError(
                    f"Record {record_id} "
                    "thiếu clause_number."
                )
def export_jsonl(
    records: list[dict[str, Any]],
    output_path: str | Path
) -> Path:
    validate_records(records)

    file_path = Path(output_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if file_path.exists():
        file_path.unlink()

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:
        for record in records:
            json_line = json.dumps(
                record,
                ensure_ascii=False
            )

            output_file.write(
                json_line + "\n"
            )

    return file_path

def print_export_summary(
    records: list[dict[str, Any]],
    output_path: str | Path
) -> None:
    article_count = sum(
        1
        for record in records
        if record.get("record_type")
        == "article"
    )

    clause_count = sum(
        1
        for record in records
        if record.get("record_type")
        == "clause"
    )

    large_record_count = sum(
        1
        for record in records
        if record.get("needs_subchunk")
    )

    print("\n===== XUẤT JSONL =====")
    print(f"Tổng record: {len(records)}")
    print(f"Record Điều: {article_count}")
    print(f"Record Khoản: {clause_count}")
    print(
        "Record cần xem xét chia nhỏ: "
        f"{large_record_count}"
    )
    print(
        "Ngưỡng chia nhỏ: "
        f"{SUBCHUNK_CHAR_THRESHOLD} ký tự"
    )
    print(
        f"File đầu ra: "
        f"{Path(output_path).resolve()}"
    )
