import re
from typing import Any


ARTICLE_PATTERN = re.compile(
    r"^Điều\s+(\d+[a-zA-Z]?)\.\s*(.*)$",
    re.IGNORECASE
)

PART_PATTERN = re.compile(
    r"^Phần\s+((?:[IVXLCDM]+|\d+)|(?:thứ\s+[^\W\d_]+))\.?\s*(.*)$",
    re.IGNORECASE
)

CHAPTER_PATTERN = re.compile(
    r"^Chương\s+([IVXLCDM\d]+)\.?\s*(.*)$",
    re.IGNORECASE
)

SECTION_PATTERN = re.compile(
    r"^Mục\s+(\d+[a-zA-Z]?)\.?\s*(.*)$",
    re.IGNORECASE
)

# Số khoản tối đa 2 chữ số (khoản thật lớn nhất trong tập dữ liệu là 27).
# Cho phép KHÔNG có khoảng trắng sau dấu chấm (vd. "16.Thi hành án...").
# Giới hạn 1-2 số tự loại các dòng bảng/danh mục dạng "2823. 00.00.40..."
# (số thứ tự danh mục, mã ngành) bị nhầm thành khoản.
CLAUSE_PATTERN = re.compile(
    r"^(\d{1,2}[a-zA-Z]?)\.\s*(.+)$"
)

# Phụ lục / danh mục ban hành kèm theo: nội dung từ marker này về sau
# KHÔNG phải điều-khoản, không gán vào điều cuối.
# Dữ liệu: marker thật đều HOA (PHỤ LỤC I, DANH MỤC HÓA CHẤT..., BẢNG KÊ...).
# Dùng case-sensitive để KHÔNG khớp nội dung điều viết thường/mixed-case
# (vd. dòng "Danh mục máy móc, thiết bị..." nằm trong khoản của một Điều).
ANNEX_PATTERN = re.compile(
    r"^(?:PHỤ LỤC|DANH MỤC|MỤC LỤC|BẢNG KÊ)\b"
)

POINT_PATTERN = re.compile(
    r"^([a-zđ])\)\s*(.+)$",
    re.IGNORECASE
)

# Điểm nằm GIỮA 1 paragraph (mẫu: "e) Sửa đổi ... như sau:đ) Bổ sung ...").
# Chỉ khớp khi ký hiệu điểm đứng ngay sau ':' hoặc ';' -> mẫu "sửa đổi,
# bổ sung ... như sau:" của văn bản sửa đổi. Tránh cắt nhầm chữ trong câu.
POINT_INLINE_PATTERN = re.compile(
    r"(?<=[:;])\s*([a-zđ])\)\s+",
    re.IGNORECASE
)

FOOTER_PREFIXES = (
    "Luật này được Quốc hội",
    "Nghị định này được Chính phủ",
    "Thông tư này có hiệu lực",
)

SIGNATURE_PREFIXES = (
    "CHỦ TỊCH QUỐC HỘI",
    "CHỦ TỊCH NƯỚC",
    "THỦ TƯỚNG CHÍNH PHỦ",
    "PHÓ THỦ TƯỚNG CHÍNH PHỦ",
    "BỘ TRƯỞNG",
    "KT. BỘ TRƯỞNG",
)


def classify_block(text: str) -> dict[str, Any]:
    """
    Phân loại một khối văn bản độc lập.

    Việc theo dõi nội dung trích dẫn kéo dài qua nhiều khối
    được xử lý trong classify_blocks().
    """
    stripped_text = text.strip()

    part_match = PART_PATTERN.match(stripped_text)

    if part_match:
        return {
            "type": "part",
            "number": part_match.group(1),
            "title": part_match.group(2).strip(),
            "text": stripped_text
        }

    chapter_match = CHAPTER_PATTERN.match(stripped_text)

    if chapter_match:
        return {
            "type": "chapter",
            "number": chapter_match.group(1),
            "title": chapter_match.group(2).strip(),
            "text": stripped_text
        }

    section_match = SECTION_PATTERN.match(stripped_text)

    if section_match:
        return {
            "type": "section",
            "number": section_match.group(1),
            "title": section_match.group(2).strip(),
            "text": stripped_text
        }

    article_match = ARTICLE_PATTERN.match(stripped_text)

    if article_match:
        return {
            "type": "article",
            "number": article_match.group(1),
            "title": article_match.group(2).strip(),
            "text": stripped_text
        }

    if stripped_text.startswith(FOOTER_PREFIXES):
        return {
            "type": "footer",
            "text": stripped_text
        }

    if stripped_text.startswith(SIGNATURE_PREFIXES):
        return {
            "type": "signature",
            "text": stripped_text
        }

    annex_match = ANNEX_PATTERN.match(stripped_text)

    if annex_match:
        return {
            "type": "annex_title",
            "text": stripped_text
        }

    clause_match = CLAUSE_PATTERN.match(stripped_text)

    if clause_match:
        return {
            "type": "clause",
            "number": clause_match.group(1),
            "text": clause_match.group(2).strip(),
            "raw_text": stripped_text
        }

    point_match = POINT_PATTERN.match(stripped_text)

    if point_match:
        return {
            "type": "point",
            "number": point_match.group(1).lower(),
            "text": point_match.group(2).strip(),
            "raw_text": stripped_text
        }

    return {
        "type": "body_text",
        "text": stripped_text
    }


def classify_blocks(blocks: list[str]) -> list[dict[str, Any]]:
    """
    Phân loại toàn bộ danh sách khối.

    Hàm theo dõi trạng thái dấu ngoặc kép để tránh tách nhầm
    khoản và điểm nằm bên trong phần nội dung được trích dẫn.
    """
    classified_blocks: list[dict[str, Any]] = []
    inside_quote = False

    for text in blocks:
        stripped_text = text.strip()

        if not stripped_text:
            continue

        opening_quotes = stripped_text.count("“")
        closing_quotes = stripped_text.count("”")

        # Khối đang tiếp tục một trích dẫn từ khối trước
        if inside_quote:
            classified_blocks.append({
                "type": "quoted_content",
                "text": stripped_text
            })

            if closing_quotes > opening_quotes:
                inside_quote = False

            continue

        # Khối mở trích dẫn nhưng chưa đóng trong cùng khối
        if opening_quotes > closing_quotes:
            classified_blocks.append({
                "type": "quoted_content",
                "text": stripped_text
            })

            inside_quote = True
            continue

        # Trích dẫn mở và đóng trong cùng một khối
        if stripped_text.startswith(("“", '"')):
            classified_blocks.append({
                "type": "quoted_content",
                "text": stripped_text
            })
            continue

        classified_blocks.append(classify_block(stripped_text))

    return classified_blocks


def count_block_types(
    classified_blocks: list[dict[str, Any]]
) -> dict[str, int]:
    """
    Đếm số lượng từng loại khối để kiểm tra parser.
    """
    counts: dict[str, int] = {}

    for block in classified_blocks:
        block_type = block["type"]
        counts[block_type] = counts.get(block_type, 0) + 1

    return counts


def build_document_structure(
    classified_blocks: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Dựng cấu trúc tài liệu từ các khối đã phân loại.
    """
    document: dict[str, Any] = {
        "header": [],
        "articles": [],
        "annexes": [],
        "footer": [],
        "signature": []
    }

    current_chapter: dict[str, str] | None = None
    current_section: dict[str, str] | None = None
    current_part: dict[str, str] | None = None
    current_article: dict[str, Any] | None = None
    current_clause: dict[str, Any] | None = None
    current_point: dict[str, str] | None = None
    current_annex: dict[str, Any] | None = None

    found_first_article = False
    inside_footer = False
    inside_signature = False
    pending_title: dict[str, str] | None = None

    def save_current_clause() -> None:
        nonlocal current_clause, current_point

        if current_article is not None and current_clause is not None:
            current_article["clauses"].append(current_clause)
            current_clause = None
            current_point = None

    def save_current_article() -> None:
        nonlocal current_article

        save_current_clause()

        if current_article is not None:
            document["articles"].append(current_article)
            current_article = None

    for block in classified_blocks:
        block_type = block["type"]
        block_text = block.get("raw_text", block.get("text", ""))

        # Sau marker Phần/Chương/Mục không có tiêu đề cùng dòng, gắn dòng
        # in hoa ngay sau làm tiêu đề. Chỉ áp dụng khi marker tường minh vừa gặp.
        if pending_title is not None:
            letters = [char for char in block_text if char.isalpha()]
            if block_type == "body_text" and letters and block_text.isupper():
                pending_title["title"] = block_text
                pending_title = None
                continue
            pending_title = None

        # Marker phụ lục luôn ưu tiên: nó có thể xuất hiện SAU phần chữ ký
        # (phụ lục ban hành kèm theo đứng sau chữ ký, ví dụ 143/2025).
        # Khi gặp annex, thoát khỏi trạng thái chữ ký/footer.
        if block_type == "annex_title":
            save_current_article()
            inside_footer = False
            inside_signature = False
            current_annex = {
                "number": len(document["annexes"]) + 1,
                "title": block_text,
                "content_blocks": []
            }
            document["annexes"].append(current_annex)
            continue

        # Sau khi gặp chữ ký, mọi khối còn lại thuộc phần chữ ký
        if inside_signature:
            document["signature"].append(block_text)
            continue

        if block_type == "signature":
            save_current_clause()
            inside_footer = False
            inside_signature = True
            document["signature"].append(block_text)
            continue

        # Sau khi gặp footer, các khối tiếp theo vẫn thuộc footer
        # cho đến khi gặp phần chữ ký hoặc phụ lục
        if inside_footer:
            document["footer"].append(block_text)
            continue

        if block_type == "footer":
            save_current_clause()
            inside_footer = True
            document["footer"].append(block_text)
            continue

        # Từ marker phụ lục về sau: mọi khối (kể cả dòng bảng) thuộc phụ lục,
        # không gán vào điều/khoản. Chỉ kết thúc khi gặp chữ ký.
        if current_annex is not None:
            if block_type == "signature":
                document["signature"].append(block_text)
                inside_signature = True
                current_annex = None
                continue
            current_annex["content_blocks"].append(block_text)
            continue

        if block_type == "part":
            # A structural heading closes the preceding Article. Otherwise an
            # uppercase Part/Chapter title can leak into that Article's body.
            save_current_article()
            current_part = {
                "number": block["number"],
                "title": block["title"]
            }
            current_chapter = None
            current_section = None
            pending_title = current_part if not current_part["title"] else None
            continue

        if block_type == "chapter":
            save_current_article()
            current_chapter = {
                "number": block["number"],
                "title": block["title"]
            }
            current_section = None
            pending_title = current_chapter if not current_chapter["title"] else None
            continue

        if block_type == "section":
            save_current_article()
            current_section = {
                "number": block["number"],
                "title": block["title"]
            }
            pending_title = current_section if not current_section["title"] else None
            continue

        if block_type == "article":
            found_first_article = True
            save_current_article()

            current_article = {
                "article_number": block["number"],
                "article_title": block["title"],
                "part": current_part,
                "chapter": current_chapter,
                "section": current_section,
                "body_blocks": [],
                "clauses": []
            }
            continue

        # Nội dung trước Điều đầu tiên thuộc header
        if not found_first_article:
            document["header"].append(block_text)
            continue

        if current_article is None:
            continue

        if block_type == "clause":
            save_current_clause()

            current_clause = {
                "clause_number": block["number"],
                "clause_title": block["text"],
                "content_blocks": [],
                "points": [],
                "lead_blocks": [],
                "tail_blocks": []
            }
            current_point = None
            continue

        # Điểm thật (ngoài ngoặc kép) -> 1 node Điểm (đơn vị nhỏ nhất).
        # 1 dòng có thể chứa nhiều điểm (vd "e) ... như sau:đ) Bổ sung ...")
        if block_type == "point":
            if current_clause is None:
                continue
            current_clause["content_blocks"].append(block_text)
            content = block["text"]
            inline = list(POINT_INLINE_PATTERN.finditer(content))
            if not inline:
                current_point = {
                    "marker": block["number"],
                    "text": content.strip()
                }
                current_clause["points"].append(current_point)
            else:
                # 1 dòng chứa nhiều điểm: "e) ... như sau:đ) Bổ sung ..."
                cursor = 0
                for index, match in enumerate(inline):
                    marker = (
                        block["number"]
                        if index == 0
                        else match.group(1)
                    )
                    segment = content[cursor:match.start()].strip()
                    if segment:
                        current_point = {
                            "marker": marker,
                            "text": segment
                        }
                        current_clause["points"].append(current_point)
                    cursor = match.end()
                segment = content[cursor:].strip()
                if segment:
                    current_point = {
                        "marker": inline[-1].group(1),
                        "text": segment
                    }
                    current_clause["points"].append(current_point)
            continue

        # Điểm, trích dẫn và body text được giữ trong khoản cha
        if current_clause is not None:
            current_clause["content_blocks"].append(block_text)
            if block_type == "quoted_content":
                # Trích dẫn nằm NGAY SAU điểm -> thuộc điểm đó;
                # không có điểm nào trước -> thuộc khoản (trích dẫn khoản/điều).
                if current_point is not None:
                    current_point["text"] += "\n" + block_text
                else:
                    current_clause["lead_blocks"].append(block_text)
            elif current_point is None:
                # Text trước điểm đầu tiên -> phần dẫn của khoản.
                current_clause["lead_blocks"].append(block_text)
            else:
                # Text sau điểm cuối -> vẫn thuộc khoản (không phải điểm).
                current_clause["tail_blocks"].append(block_text)
        else:
            current_article["body_blocks"].append(block_text)

    save_current_article()

    return document


def print_document_structure(
    document: dict[str, Any],
    content_limit: int = 2
) -> None:
    """
    In cấu trúc tài liệu để kiểm tra trực tiếp trên Terminal.
    """
    print("\n===== HEADER =====")

    for block in document["header"]:
        print(f"- {block}")

    print("\n===== NỘI DUNG =====")

    for article in document["articles"]:
        part = article.get("part")
        chapter = article.get("chapter")
        section = article.get("section")

        if part:
            print(
                f"\nPhần {part['number']}: "
                f"{part['title']}"
            )

        if chapter:
            print(
                f"\nChương {chapter['number']}: "
                f"{chapter['title']}"
            )

        if section:
            print(
                f"Mục {section['number']}: "
                f"{section['title']}"
            )

        print(
            f"\nĐiều {article['article_number']}: "
            f"{article['article_title']}"
        )

        for block in article["body_blocks"]:
            print(f"  - {block}")

        for clause in article["clauses"]:
            print(
                f"  Khoản {clause['clause_number']}: "
                f"{clause['clause_title']}"
            )

            content_blocks = clause["content_blocks"]

            for content in content_blocks[:content_limit]:
                print(f"    - {content}")

            if len(content_blocks) > content_limit:
                remaining = len(content_blocks) - content_limit
                print(f"    ... còn {remaining} khối")

    print("\n===== FOOTER =====")

    for block in document["footer"]:
        print(f"- {block}")

    print("\n===== SIGNATURE =====")

    for block in document["signature"]:
        print(f"- {block}")
