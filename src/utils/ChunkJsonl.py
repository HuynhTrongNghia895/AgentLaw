from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200


def read_jsonl(
    file_path: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{file_path.name}, dòng "
                    f"{line_number}: JSON không hợp lệ"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{file_path.name}, dòng "
                    f"{line_number}: record không phải object"
                )

            records.append(record)

    return records


def split_long_block(
    text: str,
    max_chars: int
) -> list[str]:
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            pieces.append(remaining.strip())
            break

        split_position = remaining.rfind(
            ". ",
            0,
            max_chars
        )

        if split_position < max_chars // 2:
            split_position = remaining.rfind(
                "; ",
                0,
                max_chars
            )

        if split_position < max_chars // 2:
            split_position = remaining.rfind(
                ", ",
                0,
                max_chars
            )

        if split_position < max_chars // 2:
            split_position = remaining.rfind(
                " ",
                0,
                max_chars
            )

        if split_position <= 0:
            split_position = max_chars
        else:
            split_position += 1

        piece = remaining[
            :split_position
        ].strip()

        if piece:
            pieces.append(piece)

        remaining = remaining[
            split_position:
        ].strip()

    return pieces


def prepare_text_blocks(
    record: dict[str, Any]
) -> list[str]:
    text = str(
        record.get("text", "")
    ).strip()

    if not text:
        return []

    blocks = [
        block.strip()
        for block in text.splitlines()
        if block.strip()
    ]

    prepared_blocks: list[str] = []

    for block in blocks:
        prepared_blocks.extend(
            split_long_block(
                block,
                MAX_CHUNK_CHARS
            )
        )

    return prepared_blocks


def build_chunks_from_blocks(
    blocks: list[str]
) -> list[str]:
    if not blocks:
        return []

    chunks: list[str] = []
    current_blocks: list[str] = []

    for block in blocks:
        candidate_blocks = [
            *current_blocks,
            block
        ]

        candidate_text = "\n".join(
            candidate_blocks
        ).strip()

        if (
            current_blocks
            and len(candidate_text)
            > MAX_CHUNK_CHARS
        ):
            current_text = "\n".join(
                current_blocks
            ).strip()

            if current_text:
                chunks.append(current_text)

            # Overlap lấy các BLOCK (điểm/câu) NGUYÊN VẪN, không cắt giữa
            # block (chống "nửa dòng rác" làm nhiễu embedding). Lấy từ block
            # gần nhất về trước, tổng overlap bị giới hạn bằng
            # CHUNK_OVERLAP_CHARS.
            overlap_blocks: list[str] = []
            for previous_block in reversed(current_blocks):
                if overlap_blocks:
                    combined = "\n".join(
                        [previous_block, *overlap_blocks]
                    )
                    if len(combined) > CHUNK_OVERLAP_CHARS:
                        break
                overlap_blocks = [previous_block, *overlap_blocks]

            candidate_text = "\n".join(
                [*overlap_blocks, block]
            ).strip()

            if len(candidate_text) <= MAX_CHUNK_CHARS:
                current_blocks = [*overlap_blocks, block]
            else:
                # Khối mới + overlap không vừa ngưỡng: giữ lại tối đa số
                # block overlap (tính từ gần nhất) sao cho khối kế tiếp
                # không vượt MAX_CHUNK_CHARS. Block đơn lẻ quá dài được
                # tier-2 (split_long_block) xử lý ở cuối.
                kept_blocks: list[str] = []
                for previous_block in reversed(overlap_blocks):
                    candidate = "\n".join(
                        [previous_block, *kept_blocks, block]
                    ).strip()
                    if len(candidate) > MAX_CHUNK_CHARS:
                        break
                    kept_blocks = [previous_block, *kept_blocks]
                current_blocks = [*kept_blocks, block]

        else:
            current_blocks.append(block)

    if current_blocks:
        chunk_text = "\n".join(
            current_blocks
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

    final_chunks: list[str] = []

    for chunk_text in chunks:
        if len(chunk_text) <= MAX_CHUNK_CHARS:
            final_chunks.append(
                chunk_text
            )
        else:
            final_chunks.extend(
                split_long_block(
                    chunk_text,
                    MAX_CHUNK_CHARS
                )
            )

    return final_chunks

def create_chunk_record(
    source_record: dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    chunk_count: int
) -> dict[str, Any]:
    chunk_record = deepcopy(
        source_record
    )

    source_id = str(
        source_record.get("id", "")
    ).strip()

    source_storage_id = str(
        source_record.get(
            "storage_id",
            source_id
        )
    ).strip()

    chunk_suffix = (
        f"_C{chunk_index}"
    )

    chunk_record["parent_id"] = source_id
    chunk_record["parent_storage_id"] = (
        source_storage_id
    )

    chunk_record["id"] = (
        source_id + chunk_suffix
    )

    chunk_record["storage_id"] = (
        source_storage_id
        + chunk_suffix
    )

    chunk_record["chunk_index"] = (
        chunk_index
    )

    chunk_record["chunk_count"] = (
        chunk_count
    )

    chunk_record["is_chunk"] = (
        chunk_count > 1
    )

    chunk_record["text"] = chunk_text
    chunk_record["char_count"] = len(
        chunk_text
    )

    chunk_record["needs_subchunk"] = False

    return chunk_record


def chunk_record(
    record: dict[str, Any]
) -> list[dict[str, Any]]:
    text = str(
        record.get("text", "")
    ).strip()

    if not text:
        raise ValueError(
            f"Record {record.get('id')} "
            "không có text"
        )

    if len(text) <= MAX_CHUNK_CHARS:
        chunks = [text]
    else:
        blocks = prepare_text_blocks(
            record
        )

        chunks = build_chunks_from_blocks(
            blocks
        )

    if not chunks:
        raise ValueError(
            f"Không tạo được chunk cho "
            f"{record.get('id')}"
        )

    chunk_count = len(chunks)

    return [
        create_chunk_record(
            source_record=record,
            chunk_text=chunk_text,
            chunk_index=index,
            chunk_count=chunk_count
        )
        for index, chunk_text in enumerate(
            chunks,
            start=1
        )
    ]


def validate_chunks(
    chunks: list[dict[str, Any]]
) -> None:
    if not chunks:
        raise ValueError(
            "Không có chunk nào được tạo"
        )

    seen_storage_ids: set[str] = set()

    for chunk in chunks:
        storage_id = str(
            chunk.get(
                "storage_id",
                ""
            )
        ).strip()

        text = str(
            chunk.get(
                "text",
                ""
            )
        ).strip()

        if not storage_id:
            raise ValueError(
                "Chunk thiếu storage_id"
            )

        if storage_id in seen_storage_ids:
            raise ValueError(
                f"storage_id bị trùng: "
                f"{storage_id}"
            )

        seen_storage_ids.add(
            storage_id
        )

        if not text:
            raise ValueError(
                f"Chunk {storage_id} "
                "không có text"
            )

        if len(text) > MAX_CHUNK_CHARS:
            raise ValueError(
                f"Chunk {storage_id} quá dài: "
                f"{len(text)} ký tự"
            )


def write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


def process_jsonl_file(
    input_path: Path,
    input_root: Path,
    output_root: Path
) -> dict[str, Any]:
    records = read_jsonl(
        input_path
    )

    chunks: list[dict[str, Any]] = []

    for record in records:
        chunks.extend(
            chunk_record(record)
        )

    validate_chunks(chunks)

    relative_path = input_path.relative_to(
        input_root
    )

    output_path = (
        output_root / relative_path
    )

    write_jsonl(
        chunks,
        output_path
    )

    split_record_count = sum(
        1
        for record in records
        if len(
            str(record.get("text", ""))
        ) > MAX_CHUNK_CHARS
    )

    return {
        "input_file": input_path,
        "output_file": output_path,
        "record_count": len(records),
        "chunk_count": len(chunks),
        "split_record_count":
            split_record_count,
    }


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    input_root = (
        project_root
        / "data"
        / "processed"
    )

    output_root = (
        project_root
        / "data"
        / "chunked"
    )

    jsonl_files = sorted(
        file_path
        for file_path in input_root.rglob(
            "*.jsonl"
        )
        if "dau_tu" not in {
            part.lower()
            for part in file_path.parts
        }
    )

    if not jsonl_files:
        raise FileNotFoundError(
            "Không tìm thấy file JSONL "
            "trong data/processed"
        )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for input_path in jsonl_files:
        try:
            result = process_jsonl_file(
                input_path=input_path,
                input_root=input_root,
                output_root=output_root
            )

            results.append(result)

            print(
                f"[OK] {input_path.name}: "
                f"{result['record_count']} record "
                f"→ {result['chunk_count']} chunk"
            )

        except Exception as error:
            failures.append({
                "file": str(input_path),
                "error": str(error),
            })

            print(
                f"[LỖI] {input_path.name}: "
                f"{error}"
            )

    print("\n===== TỔNG KẾT CHUNKING =====")
    print(
        f"Số file thành công: "
        f"{len(results)}"
    )
    print(
        f"Số file lỗi: "
        f"{len(failures)}"
    )
    print(
        "Tổng record đầu vào: "
        f"{sum(
            result['record_count']
            for result in results
        )}"
    )
    print(
        "Tổng chunk đầu ra: "
        f"{sum(
            result['chunk_count']
            for result in results
        )}"
    )
    print(
        "Record đã được chia: "
        f"{sum(
            result['split_record_count']
            for result in results
        )}"
    )
    print(
        f"Thư mục đầu ra: "
        f"{output_root.resolve()}"
    )

    if failures:
        print("\nFile lỗi:")

        for failure in failures:
            print(
                f"- {failure['file']}: "
                f"{failure['error']}"
            )


if __name__ == "__main__":
    main()