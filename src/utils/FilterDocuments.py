from __future__ import annotations

import csv
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any


SUPPORTED_FILTER_MODES = {
    "full",
    "include_articles",
    "include_clauses",
    "relation_only",
    "custom",
}


def normalize_document_number(document_number: str) -> str:
    """Chuyển số hiệu văn bản thành tên file ASCII ổn định trên mọi hệ điều hành.

    Ví dụ:
        168/2025/NĐ-CP -> 168_2025_ND_CP
        59/2020/QH14  -> 59_2020_QH14
    """
    value = document_number.strip().replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return value.strip("_")


def normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().lower()


def parse_boolean(value: str, default: bool = True) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "có", "co"}:
        return True

    if normalized in {"false", "0", "no", "không", "khong"}:
        return False

    return default


def parse_list_field(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def parse_dataset_targets(value: str) -> list[str]:
    return [
        normalize_dataset_name(item)
        for item in value.split(";")
        if item.strip()
    ]


def parse_annexes_field(value) -> list[str]:
    """Cột annex_giu_lai (nếu có): danh sách số phụ lục cần giữ, phân cách ';'.

    Rỗng = chưa chỉ định. Hành vi giữ/bỏ phụ lục theo chế độ lọc được định
    nghĩa trong filter_document (chỉ áp dụng với include_*)."""
    if value is None:
        return []
    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def validate_registry_columns(fieldnames: list[str] | None) -> None:
    required_columns = {
        "so_hieu",
        "ten_file",
        "dataset",
        "che_do_loc",
        "dieu_giu_lai",
        "khoan_giu_lai",
        "config_status",
    }

    available_columns = set(fieldnames or [])
    missing_columns = required_columns - available_columns

    if missing_columns:
        raise ValueError(
            "DocumentRegistry.csv thiếu các cột: "
            + ", ".join(sorted(missing_columns))
        )


def load_filter_rules(
    csv_path: str | Path
) -> list[dict[str, Any]]:
    file_path = Path(csv_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file registry: {file_path}"
        )

    rules: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        validate_registry_columns(reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            document_number = row.get("so_hieu", "").strip()
            dataset_targets = parse_dataset_targets(
                row.get("dataset", "")
            )
            filter_mode = row.get(
                "che_do_loc",
                ""
            ).strip().lower()

            if not document_number:
                continue

            if not dataset_targets:
                raise ValueError(
                    f"Dòng {row_number}: thiếu dataset"
                )

            if filter_mode not in SUPPORTED_FILTER_MODES:
                raise ValueError(
                    f"Dòng {row_number}: che_do_loc không hợp lệ: "
                    f"{filter_mode}"
                )

            for dataset_target in dataset_targets:
                rules.append({
                    "row_number": row_number,
                    "document_number": document_number,
                    "document_key": normalize_document_number(
                        document_number
                    ),
                    "dataset_target": dataset_target,
                    "mode": filter_mode,
                    "included_articles": parse_list_field(
                        row.get("dieu_giu_lai", "")
                    ),
                    "included_clauses": parse_list_field(
                        row.get("khoan_giu_lai", "")
                    ),
                    "scope_description": row.get(
                        "relation_summary",
                        ""
                    ).strip(),
                    "keep_header": True,
                    "keep_footer": True,
                    "config_status": row.get(
                        "config_status",
                        ""
                    ).strip().lower(),
                    "note": row.get("ghi_chu", "").strip(),
                    "source_filename": row.get(
                        "ten_file",
                        ""
                    ).strip(),
                    "kept_annexes": parse_annexes_field(
                        row.get("annex_giu_lai")
                    ),
                })

    return rules


def find_filter_rule(
    rules: list[dict[str, Any]],
    document_number: str,
    dataset_target: str
) -> dict[str, Any]:
    document_key = normalize_document_number(document_number)
    dataset_key = normalize_dataset_name(dataset_target)

    matched_rules = [
        rule
        for rule in rules
        if rule["document_key"] == document_key
        and rule["dataset_target"] == dataset_key
    ]

    if not matched_rules:
        raise KeyError(
            "Không tìm thấy rule cho văn bản "
            f"{document_number} trong dataset {dataset_target}"
        )

    if len(matched_rules) > 1:
        raise ValueError(
            "Có nhiều rule trùng nhau cho văn bản "
            f"{document_number} trong dataset {dataset_target}"
        )

    return matched_rules[0]


def filter_articles(
    articles: list[dict[str, Any]],
    included_article_numbers: list[str]
) -> list[dict[str, Any]]:
    included_numbers = {
        number.strip().lower()
        for number in included_article_numbers
    }

    return [
        deepcopy(article)
        for article in articles
        if str(
            article.get("article_number", "")
        ).strip().lower() in included_numbers
    ]


def filter_clauses(
    articles: list[dict[str, Any]],
    article_number: str,
    included_clause_numbers: list[str]
) -> list[dict[str, Any]]:
    article_key = article_number.strip().lower()
    included_clause_keys = {
        number.strip().lower()
        for number in included_clause_numbers
    }

    filtered_articles: list[dict[str, Any]] = []

    for article in articles:
        current_article_number = str(
            article.get("article_number", "")
        ).strip().lower()

        if current_article_number != article_key:
            continue

        filtered_article = deepcopy(article)
        filtered_article["clauses"] = [
            clause
            for clause in filtered_article.get("clauses", [])
            if str(
                clause.get("clause_number", "")
            ).strip().lower() in included_clause_keys
        ]

        filtered_articles.append(filtered_article)

    return filtered_articles


def apply_header_footer_rules(
    document: dict[str, Any],
    keep_header: bool,
    keep_footer: bool
) -> None:
    if not keep_header:
        document["header"] = []

    if not keep_footer:
        document["footer"] = []
        document["signature"] = []


def filter_document(
    document: dict[str, Any],
    rule: dict[str, Any]
) -> dict[str, Any]:
    filtered_document = deepcopy(document)
    mode = rule["mode"]
    config_status = rule.get("config_status", "")

    if config_status != "ready":
        raise ValueError(
            "Rule chưa ở trạng thái ready: "
            f"{rule['document_number']} - "
            f"{rule['dataset_target']} - "
            f"{config_status or 'unknown'}"
        )

    if mode == "full":
        pass

    elif mode == "include_articles":
        article_numbers = rule.get("included_articles", [])

        if not article_numbers:
            raise ValueError(
                "Rule include_articles không có Điều giữ lại"
            )

        filtered_document["articles"] = filter_articles(
            filtered_document.get("articles", []),
            article_numbers
        )

    elif mode == "include_clauses":
        article_numbers = rule.get("included_articles", [])
        clause_numbers = rule.get("included_clauses", [])

        if len(article_numbers) != 1:
            raise ValueError(
                "Rule include_clauses hiện yêu cầu đúng một Điều"
            )

        if not clause_numbers:
            raise ValueError(
                "Rule include_clauses không có Khoản giữ lại"
            )

        filtered_document["articles"] = filter_clauses(
            filtered_document.get("articles", []),
            article_numbers[0],
            clause_numbers
        )

    elif mode == "relation_only":
        filtered_document["articles"] = []
        filtered_document["relation_metadata"] = {
            "document_number": rule["document_number"],
            "dataset_target": rule["dataset_target"],
            "description": rule.get(
                "scope_description",
                ""
            ),
            "note": rule.get("note", ""),
        }

    elif mode == "custom":
        raise ValueError(
            "Rule custom cần cấu hình bộ lọc riêng: "
            f"{rule['document_number']} - "
            f"{rule['dataset_target']}"
        )

    else:
        raise ValueError(
            f"Chế độ lọc chưa được hỗ trợ: {mode}"
        )

    # Phụ lục: chế độ include_* chỉ giữ phụ lục nằm trong annex_giu_lai
    # (danh sách số phụ lục; rỗng = bỏ hết). full/relation_only giữ nguyên
    # vì phụ lục là nội dung chính thức của văn bản đó.
    if mode in {"include_articles", "include_clauses"}:
        kept_numbers = {
            str(number)
            for number in rule.get("kept_annexes", [])
        }
        filtered_document["annexes"] = [
            annex
            for annex in filtered_document.get("annexes", [])
            if str(annex.get("number", "")) in kept_numbers
        ]

    apply_header_footer_rules(
        filtered_document,
        keep_header=rule.get("keep_header", True),
        keep_footer=rule.get("keep_footer", True)
    )

    filtered_document["filter_info"] = {
        "document_number": rule["document_number"],
        "dataset_target": rule["dataset_target"],
        "mode": mode,
        "included_articles": rule.get(
            "included_articles",
            []
        ),
        "included_clauses": rule.get(
            "included_clauses",
            []
        ),
        "scope_description": rule.get(
            "scope_description",
            ""
        ),
        "config_status": config_status,
        "note": rule.get("note", ""),
    }

    return filtered_document


def load_and_filter_document(
    document: dict[str, Any],
    registry_path: str | Path,
    document_number: str,
    dataset_target: str
) -> dict[str, Any]:
    rules = load_filter_rules(registry_path)
    rule = find_filter_rule(
        rules,
        document_number,
        dataset_target
    )
    return filter_document(document, rule)


def print_filter_summary(
    original_document: dict[str, Any],
    filtered_document: dict[str, Any]
) -> None:
    filter_info = filtered_document.get("filter_info", {})
    original_article_count = len(
        original_document.get("articles", [])
    )
    filtered_article_count = len(
        filtered_document.get("articles", [])
    )

    print("\n===== KẾT QUẢ LỌC =====")
    print(
        "Số hiệu: "
        f"{filter_info.get('document_number', 'unknown')}"
    )
    print(
        "Dataset: "
        f"{filter_info.get('dataset_target', 'unknown')}"
    )
    print(
        "Chế độ: "
        f"{filter_info.get('mode', 'unknown')}"
    )
    print(
        "Phạm vi: "
        f"{filter_info.get('scope_description', '')}"
    )
    print(
        "Trạng thái: "
        f"{filter_info.get('config_status', '')}"
    )
    print(f"Số Điều trước lọc: {original_article_count}")
    print(f"Số Điều sau lọc: {filtered_article_count}")