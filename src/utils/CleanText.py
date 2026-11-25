import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Làm sạch một khối văn bản nhưng không thay đổi nội dung pháp lý.
    """
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi")

    # Chuẩn hóa Unicode tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # Chuẩn hóa các loại khoảng trắng
    text = text.replace("\u00A0", " ")
    text = text.replace("\t", " ")

    # Chuẩn hóa ký tự xuống dòng
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    cleaned_lines: list[str] = []

    for line in text.split("\n"):
        line = line.strip()

        # Gộp nhiều khoảng trắng liên tiếp
        line = re.sub(r"[ ]{2,}", " ", line)

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_blocks(raw_blocks: list[str]) -> list[str]:
    """
    Làm sạch danh sách khối và loại bỏ các khối rỗng.
    """
    cleaned_blocks: list[str] = []

    for block in raw_blocks:
        cleaned = clean_text(block)

        if cleaned:
            cleaned_blocks.append(cleaned)

    return cleaned_blocks