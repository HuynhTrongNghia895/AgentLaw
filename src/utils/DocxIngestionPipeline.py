from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from CleanText import clean_blocks
from ExtractDocx import extract_docx
from FilterDocuments import (
    filter_document,
    load_filter_rules,
    normalize_document_number,
)
from ParseLegalStructure import (
    build_document_structure,
    classify_blocks,
    count_block_types,
)
from ExportJsonl import (
    document_to_records,
    export_jsonl,
    load_relation_index,
)


PIPELINE_VERSION = "docx-ingestion-v2.0"


@dataclass(slots=True)
class IngestionResult:
    status: str
    document_number: str
    dataset_target: str
    source_file: str
    processed_file: str | None = None
    chunked_file: str | None = None
    raw_blocks: int = 0
    cleaned_blocks: int = 0
    removed_blocks: int = 0
    records: int = 0
    chunks: int = 0
    split_records: int = 0
    block_types: dict[str, int] | None = None
    file_hash: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_fingerprint(rule: dict[str, Any]) -> str:
    relevant_rule = {
        "document_number": rule.get("document_number"),
        "dataset_target": rule.get("dataset_target"),
        "mode": rule.get("mode"),
        "included_articles": rule.get("included_articles", []),
        "included_clauses": rule.get("included_clauses", []),
        "scope_description": rule.get("scope_description"),
        "keep_header": rule.get("keep_header"),
        "keep_footer": rule.get("keep_footer"),
        "kept_annexes": sorted(
            str(n) for n in rule.get("kept_annexes", [])
        ),
        "config_status": rule.get("config_status"),
        "source_filename": rule.get("source_filename"),
        "pipeline_version": PIPELINE_VERSION,
    }
    serialized = json.dumps(
        relevant_rule,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ManifestStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "files": {},
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as input_file:
            loaded = json.load(input_file)
        if isinstance(loaded, dict):
            self.data.update(loaded)
            self.data.setdefault("files", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(self.data, output_file, ensure_ascii=False, indent=2)
        temporary_path.replace(self.path)

    @staticmethod
    def key(rule: dict[str, Any]) -> str:
        return (
            f"{rule['dataset_target']}::"
            f"{normalize_document_number(rule['document_number'])}"
        )

    def is_current(
        self,
        rule: dict[str, Any],
        file_hash: str,
        processed_file: Path,
        chunked_file: Path | None = None,
    ) -> bool:
        entry = self.data["files"].get(self.key(rule), {})
        return bool(
            entry.get("file_hash") == file_hash
            and entry.get("rule_hash") == rule_fingerprint(rule)
            and entry.get("pipeline_version") == PIPELINE_VERSION
            and processed_file.exists()
        )

    def update(
        self,
        rule: dict[str, Any],
        file_hash: str,
        result: IngestionResult,
    ) -> None:
        self.data["pipeline_version"] = PIPELINE_VERSION
        self.data["files"][self.key(rule)] = {
            "source_file": result.source_file,
            "document_number": result.document_number,
            "dataset_target": result.dataset_target,
            "file_hash": file_hash,
            "rule_hash": rule_fingerprint(rule),
            "pipeline_version": PIPELINE_VERSION,
            "processed_file": result.processed_file,
            "chunked_file": result.chunked_file,
            "records": result.records,
            "chunks": result.chunks,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()


class DocxIngestionPipeline:
    """Pipeline duy nhất: DOCX -> làm sạch -> parse -> JSONL -> chunk JSONL."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.raw_root = self.project_root / "data" / "raw" / "docx"
        self.processed_root = self.project_root / "data" / "processed"
        self.chunked_root = self.project_root / "data" / "chunked"
        self.registry_file = (
            self.project_root / "data" / "metadata" / "DocumentRegistry.csv"
        )
        self.relations_file = (
            self.project_root / "data" / "metadata" / "DocumentRelations.csv"
        )
        self._relation_index: dict[str, list[dict[str, Any]]] | None = None
        self.manifest = ManifestStore(
            self.project_root / "data" / "manifests" / "docx_ingestion.json"
        )

    def load_relation_index(self) -> dict[str, list[dict[str, Any]]]:
        """Đọc index quan hệ (số hiệu bị tác động -> danh sách quan hệ sửa
        đổi/thay thế/bãi bỏ). Cất giữa để chỉ đọc 2 file CSV metadata 1 lần
        trong phiên chạy pipeline."""
        if self._relation_index is None:
            self._relation_index = load_relation_index(
                relations_csv=self.relations_file,
                registry_csv=self.registry_file,
            )
        return self._relation_index

    def load_ready_rules(self) -> list[dict[str, Any]]:
        return [
            rule
            for rule in load_filter_rules(self.registry_file)
            if rule.get("config_status") == "ready"
        ]

    def build_docx_index(self) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        for file_path in self.raw_root.rglob("*.docx"):
            index.setdefault(file_path.name.casefold(), []).append(file_path)
        return index

    @staticmethod
    def resolve_docx(
        rule: dict[str, Any],
        docx_index: dict[str, list[Path]],
    ) -> Path:
        source_filename = str(rule.get("source_filename", "")).strip()
        if not source_filename:
            raise ValueError(
                f"Registry thiếu Tên File cho {rule['document_number']}"
            )
        matches = docx_index.get(source_filename.casefold(), [])
        if not matches:
            raise FileNotFoundError(f"Không tìm thấy DOCX: {source_filename}")
        if len(matches) > 1:
            raise ValueError(
                f"Có nhiều file cùng tên {source_filename}: "
                + ", ".join(str(path) for path in matches)
            )
        return matches[0]

    def output_paths(self, rule: dict[str, Any]) -> tuple[Path, Path]:
        """(processed, chunked) — chunked giữ lại để tương thích, nhưng
        KHÔNG còn được sinh (đã bỏ bước chunk; KG đọc thẳng processed)."""
        filename = normalize_document_number(rule["document_number"]) + ".jsonl"
        return (
            self.processed_root / rule["dataset_target"] / filename,
            self.chunked_root / rule["dataset_target"] / filename,
        )

    def ingest_rule(
        self,
        rule: dict[str, Any],
        docx_file: Path,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> IngestionResult:
        processed_file, chunked_file = self.output_paths(rule)
        file_hash = sha256_file(docx_file)

        if not force and self.manifest.is_current(
            rule, file_hash, processed_file, chunked_file
        ):
            return IngestionResult(
                status="skipped",
                document_number=rule["document_number"],
                dataset_target=rule["dataset_target"],
                source_file=docx_file.name,
                processed_file=str(processed_file),
                file_hash=file_hash,
                message="DOCX và cấu hình không thay đổi.",
            )

        raw_blocks = extract_docx(docx_file)
        cleaned_blocks = clean_blocks(raw_blocks)
        classified_blocks = classify_blocks(cleaned_blocks)
        block_types = count_block_types(classified_blocks)
        document = build_document_structure(classified_blocks)

        if not document.get("articles"):
            raise ValueError(
                "Không nhận diện được Điều nào. Kiểm tra định dạng DOCX hoặc regex parser."
            )

        filtered_document = filter_document(document, rule)
        records = document_to_records(
            document=filtered_document,
            document_number=rule["document_number"],
            dataset_target=rule["dataset_target"],
            source_filename=docx_file.name,
            relation_index=self.load_relation_index(),
        )
        if not records:
            raise ValueError("Không tạo được record sau bước lọc.")

        result = IngestionResult(
            status="validated" if dry_run else "success",
            document_number=rule["document_number"],
            dataset_target=rule["dataset_target"],
            source_file=docx_file.name,
            processed_file=str(processed_file),
            raw_blocks=len(raw_blocks),
            cleaned_blocks=len(cleaned_blocks),
            removed_blocks=max(0, len(raw_blocks) - len(cleaned_blocks)),
            records=len(records),
            block_types=block_types,
            file_hash=file_hash,
            message=(
                "Đã kiểm tra, chưa ghi file." if dry_run
                else "Đã tạo processed JSONL (đã bỏ bước chunk)."
            ),
        )

        if dry_run:
            return result

        export_jsonl(records, processed_file)
        result.chunks = 0
        result.split_records = 0
        result.chunked_file = None
        self.manifest.update(rule, file_hash, result)
        return result

    def ingest(
        self,
        *,
        filenames: Iterable[str] | None = None,
        dataset: str | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> tuple[list[IngestionResult], list[dict[str, str]]]:
        rules = self.load_ready_rules()
        selected_names = {
            name.casefold() for name in (filenames or []) if name.strip()
        }
        if selected_names:
            rules = [
                rule
                for rule in rules
                if str(rule.get("source_filename", "")).casefold()
                in selected_names
            ]
        if dataset:
            rules = [
                rule for rule in rules if rule["dataset_target"] == dataset
            ]
        if not rules:
            raise ValueError("Không có rule ready phù hợp để xử lý.")

        docx_index = self.build_docx_index()
        results: list[IngestionResult] = []
        failures: list[dict[str, str]] = []
        for rule in rules:
            try:
                docx_file = self.resolve_docx(rule, docx_index)
                results.append(
                    self.ingest_rule(
                        rule,
                        docx_file,
                        force=force,
                        dry_run=dry_run,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "document_number": str(rule.get("document_number", "")),
                        "dataset_target": str(rule.get("dataset_target", "")),
                        "source_file": str(rule.get("source_filename", "")),
                        "error": str(error),
                    }
                )
        return results, failures
