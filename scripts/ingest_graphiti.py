"""Nạp KG pháp lý (P5) vào Graphiti (P6) — ingest DETERMINISTIC, không LLM.

Vì sao không dùng `add_episode` cho toàn bộ?
  `add_episode` gọi LLM để trích xuất entity/edge từ text -> chậm, tốn
  token, và kết quả KHÔNG deterministic (mỗi lần chạy ra entity khác).
  Trong khi đó, KG pháp lý P5 đã có sẵn node/cạnh đã verify, có tầng
  đúng (Article/Clause/Point/Annex) và quan hệ AMENDS/REPEALS/GUIDES đã
  trỏ đúng tầng.

  -> Nạp thẳng node/cạnh vào Graphiti (embedding BGE-M3 local, không LLM):
     nhanh, deterministic, kiểm định được, tái lập được.
  -> Phần LLM (`add_episode`) chỉ dành cho văn bản MỚI chưa có trong KG
     (xem `GraphitiBridge.sync_episode`).

Dùng:
    python scripts/ingest_graphiti.py [--clear] [--limit N]

Kết quả ghi: data/logs/graphiti_ingest.log + báo cáo trên stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "utils",
    PROJECT_ROOT / "src" / "retrieval",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv  # noqa: E402

from agents.GraphitiBridge import GROUP_ID, GraphitiBridge  # noqa: E402

if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")

# Node nào -> label của graphiti. Giữ nguyên tên tầng pháp lý để search
# hiểu ngữ nghĩa (không phải "Generic").
LABEL_MAP = {
    "Document": "LegalDocument",
    "Article": "LegalArticle",
    "Clause": "LegalClause",
    "Point": "LegalPoint",
    "Annex": "LegalAnnex",
    "Chapter": "LegalChapter",
    "Section": "LegalSection",
    "Part": "LegalPart",
}

# Quan hệ pháp lý -> tên edge của graphiti.
REL_MAP = {
    "AMENDS": "AMENDS",
    "REPEALS": "REPEALS",
    "GUIDES": "GUIDES",
    "HAS_ARTICLE": "HAS_ARTICLE",
    "HAS_CLAUSE": "HAS_CLAUSE",
    "HAS_POINT": "HAS_POINT",
    "HAS_ANNEX": "HAS_ANNEX",
    "HAS_SECTION": "HAS_SECTION",
    "HAS_CHAPTER": "HAS_CHAPTER",
}

MAX_TEXT_CHARS = 2000
NODE_LABELS = list(LABEL_MAP)


def fetch_nodes(driver: Any) -> list[dict[str, Any]]:
    """Lấy toàn bộ node KG pháp lý."""
    label_clause = " OR ".join(f"n:{lab}" for lab in NODE_LABELS)
    query = f"""
    MATCH (n) WHERE ({label_clause})
      AND n.kind4tier = true
    RETURN n.uuid AS uuid, n.name AS name, n.id AS id,
           n.text AS text, n.title AS title,
           labels(n) AS labels,
           n.so_hieu AS so_hieu,
           n.effective_date AS effective_date,
           n.dieu AS dieu, n.khoan AS khoan, n.diem AS diem,
           n.phu_luc AS phu_luc,
           n.nhom_pl AS nhom_pl
    """
    out: list[dict[str, Any]] = []
    with driver.session(database="neo4j") as session:
        for rec in session.run(query):
            labels = [l for l in rec["labels"] if l in LABEL_MAP]
            if not labels:
                continue
            text = rec["text"] or ""
            summary = (rec["title"] or "").strip()
            if not summary and text:
                summary = text[:300].strip()
            out.append({
                "uuid": rec["uuid"],
                "name": rec["name"],
                "labels": [LABEL_MAP[l] for l in labels],
                "summary": summary,
                "text": text[:MAX_TEXT_CHARS],
                "so_hieu": rec["so_hieu"],
                "effective_date": rec["effective_date"],
                "dieu": rec["dieu"],
                "khoan": rec["khoan"],
                "diem": rec["diem"],
                "phu_luc": rec["phu_luc"],
                "nhom_pl": rec["nhom_pl"],
                "kg_id": rec["id"],
            })
    return out


def fetch_edges(driver: Any) -> list[dict[str, Any]]:
    """Lấy cạnh quan hệ pháp lý (chỉ cạnh có ý nghĩa pháp lý)."""
    rel_clause = " OR ".join(f"r:{rel}" for rel in REL_MAP)
    rel_list = ", ".join(f"'{k}'" for k in REL_MAP)
    query = f"""
    MATCH (a)-[r]->(b)
    WHERE type(r) IN [{rel_list}]
      AND a.uuid IS NOT NULL AND b.uuid IS NOT NULL
      AND a.kind4tier = true AND b.kind4tier = true
    RETURN a.uuid AS src, b.uuid AS dst, type(r) AS rel,
           r.relation_effective_date AS eff,
           r.clauses AS clauses,
           r.source_reference AS ref
    """
    out: list[dict[str, Any]] = []
    with driver.session(database="neo4j") as session:
        for rec in session.run(query):
            out.append({
                "source_uuid": rec["src"],
                "target_uuid": rec["dst"],
                "rel": rec["rel"],
                "eff": rec["eff"],
                "clauses": rec["clauses"],
                "ref": rec["ref"],
            })
    return out


def build_nodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chuyển node KG -> spec cho GraphitiBridge."""
    out = []
    for r in rows:
        attrs = {k: v for k, v in (
            ("kg_id", r["kg_id"]),
            ("so_hieu", r["so_hieu"]),
            ("effective_date", r["effective_date"]),
            ("dieu", r["dieu"]),
            ("khoan", r["khoan"]),
            ("diem", r["diem"]),
            ("phu_luc", r["phu_luc"]),
            ("nhom_pl", r["nhom_pl"]),
            ("text", r["text"]),
        ) if v not in (None, "", [])}
        out.append({
            "uuid": r["uuid"],
            "name": r["name"],
            "labels": r["labels"],
            "summary": r["summary"],
            "attributes": attrs,
        })
    return out


def build_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chuyển cạnh KG -> spec cho GraphitiBridge."""
    out = []
    for i, r in enumerate(rows):
        fact = f"{r['rel']}"
        if r.get("clauses"):
            fact += f" ({r['clauses']})"
        out.append({
            # uuid ổn định để nạp lại không sinh trùng.
            "uuid": _edge_uuid(r, i),
            "name": r["rel"],
            "fact": fact,
            "source_uuid": r["source_uuid"],
            "target_uuid": r["target_uuid"],
            "attributes": {
                k: v for k, v in (
                    ("relation_effective_date", r.get("eff")),
                    ("clauses", r.get("clauses")),
                    ("source_reference", r.get("ref")),
                ) if v not in (None, "")
            },
        })
    return out


def _edge_uuid(r: dict[str, Any], i: int) -> str:
    """UUID 32-hex ổn định cho cạnh (tính từ nội dung cạnh)."""
    import hashlib

    key = f"{r['source_uuid']}|{r['target_uuid']}|{r['rel']}|{r.get('eff') or ''}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true",
                    help="Xoá group graphiti cũ trước khi nạp lại.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Giới hạn số node (debug). 0 = tất cả.")
    args = ap.parse_args()

    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    print("=== Lấy dữ liệu từ KG pháp lý ===", flush=True)
    node_rows = fetch_nodes(driver)
    edge_rows = fetch_edges(driver)
    driver.close()
    print(f"  node: {len(node_rows)} | cạnh: {len(edge_rows)}", flush=True)

    if args.limit:
        node_rows = node_rows[:args.limit]
        keep = {n["uuid"] for n in node_rows}
        edge_rows = [
            e for e in edge_rows
            if e["source_uuid"] in keep and e["target_uuid"] in keep
        ]

    bridge = GraphitiBridge(group_id=GROUP_ID, use_llm=False)
    print("=== Trạng thái P6 ===", flush=True)
    for k, v in bridge.status().items():
        print(f"  {k}: {v}", flush=True)

    if args.clear:
        n = bridge.clear_group()
        print(f"=== Đã xoá {n} node cũ của group '{GROUP_ID}' ===", flush=True)

    print("=== Ingest (deterministic, không LLM) ===", flush=True)
    started = time.perf_counter()
    report = bridge.ingest_deterministic(
        build_nodes(node_rows),
        build_edges(edge_rows),
    )
    report.duration_seconds = time.perf_counter() - started
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if report.errors:
        print("\n--- 10 lỗi đầu ---")
        for e in report.errors[:10]:
            print("  ", e[:200])
    return 0 if report.nodes_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
