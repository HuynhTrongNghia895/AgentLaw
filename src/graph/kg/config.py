"""Hằng số KG 4 tầng — 1 chỗ duy nhất sửa mốc cảnh báo as_of_date.

CLAUDE.md §5: tập trung config, tránh rải hằng số.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -- Đường dẫn nguồn (chỉ-đọc) -------------------------------------------
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CHUNKED = PROJECT_ROOT / "data" / "chunked"
REGISTRY_CSV = PROJECT_ROOT / "data" / "metadata" / "DocumentRegistry.csv"
RELATIONS_CSV = PROJECT_ROOT / "data" / "metadata" / "DocumentRelations.csv"

# -- Mốc cảnh báo (dòng "dữ liệu tính đến …") ----------------------------
# CHƯA GV chốt. Mặc định = ngày rà soát pháp lý (khớp registry ghi_chu 29/10/2026).
# Đây KHÔNG phải "góc nhìn người hỏi" — chỉ dùng in cảnh báo + mốc mặc định
# khi user không chỉ mốc. Câu time-travel (4 case GĐ3) dùng mốc hỏi T riêng.
AS_OF_DATE = "2026-10-29"

# -- Neo4j ----------------------------------------------------------------
NEO4J_URI = "bolt://localhost:7687"
# db neo4j (Community = 1 db, không tách db riêng được — đã verify)
NEO4J_DATABASE = "neo4j"


# -- Tầng / label node ----------------------------------------------------
LABEL_DOCUMENT = "Document"
LABEL_ARTICLE = "Article"
LABEL_CLAUSE = "Clause"
LABEL_POINT = "Point"
LABEL_ANNEX = "Annex"
LABEL_PART = "Part"
LABEL_CHAPTER = "Chapter"
LABEL_SECTION = "Section"

# -- Loại cạnh (mỗi tầng CON quyết định 1 loại cạnh, hướng cha -> con) ----
EDGE_HAS_PART = "HAS_PART"        # (Document|Part)      -> Part
EDGE_HAS_CHAPTER = "HAS_CHAPTER"  # (Document|Part)      -> Chapter
EDGE_HAS_SECTION = "HAS_SECTION"  # (Part|Chapter)       -> Section
EDGE_HAS_ARTICLE = "HAS_ARTICLE"  # (Document|Section..) -> Article
EDGE_HAS_CLAUSE = "HAS_CLAUSE"    # Article              -> Clause
EDGE_HAS_POINT = "HAS_POINT"      # Clause               -> Point
EDGE_HAS_ANNEX = "HAS_ANNEX"      # Document             -> Annex
EDGE_AMENDS = "AMENDS"
EDGE_REPEALS = "REPEALS"
EDGE_GUIDES = "GUIDES"
