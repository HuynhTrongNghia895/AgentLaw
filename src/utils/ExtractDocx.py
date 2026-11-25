from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_document_blocks(
    document: DocumentObject
) -> Iterator[Paragraph | Table]:
    """
    Duyệt đoạn văn và bảng theo đúng thứ tự xuất hiện trong DOCX.
    """
    body = document.element.body

    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)

        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_table_rows(table: Table) -> list[str]:
    """
    Trích xuất nội dung từng hàng của bảng.
    Các ô trong cùng một hàng được nối bằng dấu |.
    """
    extracted_rows: list[str] = []

    for row in table.rows:
        cell_texts: list[str] = []

        for cell in row.cells:
            paragraphs = [
                paragraph.text.strip()
                for paragraph in cell.paragraphs
                if paragraph.text.strip()
            ]

            cell_text = "\n".join(paragraphs).strip()

            if cell_text:
                cell_texts.append(cell_text)

        if cell_texts:
            # Loại nội dung lặp do ô gộp trong Word
            unique_cell_texts = list(dict.fromkeys(cell_texts))
            extracted_rows.append(" | ".join(unique_cell_texts))

    return extracted_rows


def extract_docx(docx_path: str | Path) -> list[str]:
    """
    Đọc toàn bộ nội dung DOCX.

    Hàm này chỉ thực hiện trích xuất dữ liệu thô,
    không làm sạch và không phân tích cấu trúc pháp luật.
    """
    file_path = Path(docx_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if file_path.suffix.lower() != ".docx":
        raise ValueError(f"File không phải DOCX: {file_path}")

    document = Document(file_path)
    raw_blocks: list[str] = []

    for block in iter_document_blocks(document):
        if isinstance(block, Paragraph):
            raw_blocks.append(block.text)

        elif isinstance(block, Table):
            raw_blocks.extend(extract_table_rows(block))

    return raw_blocks


