from __future__ import annotations

"""Build graph pháp lý vào Neo4j từ 2 CSV metadata (nguồn sự thật).

Schema:
- (:Document {so_hieu, dataset, document_type, vai_tro, issued_date,
  effective_date, effective_status, superseded_by,
  legal_review_status, ghichu})
- (:Provision {unique_key='<so_hieu>|<label>', document, label, kind,
  dataset, document_title})
- (:Provision)-[:BELONGS_TO]->(:Document)
- (:Provision)-[:AMENDS|REPEALS|GUIDES {clauses, relation_id,
  effective_date, verification_status, source_reference, relation_notes}]->
  (:Provision)

Mỗi dòng DocumentRelations.csv (1 relation) có thể mở rộng thành N cạnh
(1 cạnh cho mỗi provision bên "to"). `effective_date` cạnh = effective_date
văn bản FROM trong registry (văn bản sửa/bãi bỏ/hướng dẫn có hiệu lực
ngày nào) - nguồn rõ, không suy diễn (CLAUDE.md §6.4).

Idempotent: MERGE on uniqueness constraints. Chạy 2 lần -> graph không
thay đổi.
"""

import csv
from pathlib import Path
from typing import Any

from Neo4jClient import Neo4jClient
from ProvisionParser import (
    parse_provision_effective_dates,
    parse_provision_list,
    provision_unparsed,
    unique_key,
)

RELATION_CSV = "data/metadata/DocumentRelations.csv"
REGISTRY_CSV = "data/metadata/DocumentRegistry.csv"


def _eff_maps(
    registry: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """so_hieu -> parsed provision_effective_dates (phương án A)."""
    out: dict[str, dict[str, Any]] = {}
    for row in registry:
        so = (row.get("so_hieu") or "").strip()
        if so:
            out[so] = parse_provision_effective_dates(
                row.get("provision_effective_dates", "")
            )
    return out


def _resolve_eff(
    so_hieu: str,
    label: str,
    eff_maps: dict[str, dict[str, Any]],
    reg: dict[str, dict[str, str]],
) -> tuple[str, str]:
    """(effective_date, source) cho 1 provision node.
    source: 'provision' (ngày riêng, lớp 2) | 'document' (fallback ngày
    chung văn bản, lớp 1). Lớp 3 (quan hệ riêng) CHƯA dùng dữ liệu -
    giữ hook: sau này cột relation_effective_date sẽ set source='relation'."""
    m = eff_maps.get(so_hieu, {})
    if label in m.get("article_dates", {}):
        return m["article_dates"][label], "provision"
    if label in m.get("annex_dates", {}):
        return m["annex_dates"][label], "provision"
    return (reg.get(so_hieu) or {}).get("effective_date", "").strip(), "document"


def _clause_dates(
    so_hieu: str,
    label: str,
    eff_maps: dict[str, dict[str, Any]],
) -> list[str]:
    """Danh sách khoản->ngày RIÊNG của điều này (phương án A, chỉ khoản
    có ngày khác ngày mặc định). Neo4j property KHÔNG lưu map (chỉ
    primitive + array) -> lưu dạng array chuỗi "khoản:YYYY-MM-DD";
    client parse lại khi query khoản-level."""
    m = eff_maps.get(so_hieu, {}).get("clause_dates", {}).get(label, {})
    return [f"{k}:{v}" for k, v in sorted(m.items())]


def read_registry(project_root: Path) -> list[dict[str, str]]:
    path = project_root / REGISTRY_CSV
    with path.open("r", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("so_hieu")]


def read_relations(project_root: Path) -> list[dict[str, str]]:
    path = project_root / RELATION_CSV
    with path.open("r", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("relation_id")]


def _registry_by_number(
    registry: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    return {row["so_hieu"].strip(): row for row in registry}


def plan_documents(
    registry: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Nghiệp vụ: Document nodes (từ registry)."""
    docs: list[dict[str, Any]] = []
    for row in registry:
        so_hieu = row["so_hieu"].strip()
        docs.append(
            {
                "so_hieu": so_hieu,
                "dataset": row.get("dataset", "").strip(),
                "document_type": (
                    "QH" if "QH" in (row.get("ban_hanh_boi") or "")
                    else "CP" if "CP" in (row.get("ban_hanh_boi") or "")
                    else (row.get("ban_hanh_boi") or "").strip()
                ),
                "vai_tro": row.get("vai_tro", "").strip(),
                "issued_date": row.get("ngay_ban_hanh", "").strip(),
                "effective_date": row.get("effective_date", "").strip(),
                "effective_status": row.get("effective_status", "").strip(),
                "superseded_by": row.get("superseded_by", "").strip(),
                "legal_review_status": row.get(
                    "legal_review_status", ""
                ).strip(),
                "ghi_chu": row.get("ghi_chu", "").strip(),
            }
        )
    return docs


def plan_provisions(
    relations: list[dict[str, str]],
    registry: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Nghiệp vụ: Provision nodes (CHỈ provision xuất hiện trong
    quan hệ đã verified - đúng CLAUDE.md §6.4). Trả (provisions,
    danh_sach_unparsed)."""
    reg = _registry_by_number(registry)
    eff_maps = _eff_maps(registry)
    provisions: dict[str, dict[str, Any]] = {}

    def add(so_hieu: str, label: str, kind: str) -> None:
        so = so_hieu.strip()
        key = f"{so}|{label}"
        if key in provisions:
            return
        doc = reg.get(so, {})
        eff_date, eff_source = _resolve_eff(so, label, eff_maps, reg)
        provisions[key] = {
            "unique_key": key,
            "document": so,
            "label": label,
            "kind": kind,
            "dataset": doc.get("dataset", ""),
            "document_title": doc.get("ten_file", ""),
            # Hiệu lực 3 lớp (phương án A):
            "effective_date": eff_date,          # lớp 2 (riêng) / lớp 1 (fallback)
            "effective_date_source": eff_source, # 'provision' | 'document'
            "clause_dates": _clause_dates(so, label, eff_maps),  # map khoản->ngày
        }

    unparsed: list[str] = []
    for rel in relations:
        from_so = (rel.get("from_so_hieu") or "").strip()
        to_so = (rel.get("to_so_hieu") or "").strip()
        for side, so in (("from", from_so), ("to", to_so)):
            refs = parse_provision_list(rel.get(f"{side}_provision"))
            unparsed.extend(provision_unparsed(refs))
            for ref in refs:
                if not ref.ok:
                    continue
                add(so, ref.label, ref.kind)

    return list(provisions.values()), unparsed


def plan_edges(
    relations: list[dict[str, str]],
    registry: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Nghiệp vụ: relation edges, MỞ RỘNG per to-provision (1 dòng CSV
    -> N cạnh). Parse fail bên to -> BỎ qua (đã report qua
    plan_provisions), không bịa."""
    reg = _registry_by_number(registry)
    edges: list[dict[str, Any]] = []
    for rel in relations:
        from_so = (rel.get("from_so_hieu") or "").strip()
        to_so = (rel.get("to_so_hieu") or "").strip()
        rtype = (rel.get("relation_type") or "").strip().upper()
        if rtype not in ("AMENDS", "REPEALS", "GUIDES"):
            continue

        from_refs = parse_provision_list(rel.get("from_provision"))
        to_refs = parse_provision_list(rel.get("to_provision"))
        eff_maps = _eff_maps(registry)
        doc_eff = reg.get(from_so, {}).get("effective_date", "").strip()

        for fref in from_refs:
            if not fref.ok:
                continue
            from_key = unique_key(from_so, fref)
            # Cạnh hiệu lực theo provision NGUỒN (việc sửa xuất phát từ
            # provision này của văn bản nguồn) - lớp 2; fallback lớp 1
            # (ngày chung văn bản) nếu provision không có ngày riêng.
            f_eff, f_src = _resolve_eff(from_so, fref.label, eff_maps, reg)
            for tref in to_refs:
                if not tref.ok:
                    continue
                to_key = unique_key(to_so, tref)
                edges.append(
                    {
                        "relation_type": rtype,
                        "from_key": from_key,
                        "to_key": to_key,
                        "clauses": "; ".join(tref.clauses)
                        or None,
                        "relation_id": (
                            rel.get("relation_id") or ""
                        ).strip(),
                        "effective_date": f_eff or doc_eff or None,
                        # Lớp 3 hook (chưa dùng dữ liệu): 'provision' khi
                        # lấy ngày riêng provision nguồn; 'document' khi
                        # fallback ngày chung. Sau này 'relation' khi có
                        # cột relation_effective_date.
                        "effective_date_source": f_src,
                        "verification_status": (
                            rel.get("verification_status") or ""
                        ).strip(),
                        "source_reference": (
                            rel.get("source_reference") or ""
                        ).strip(),
                        "relation_notes": (
                            rel.get("notes") or ""
                        ).strip(),
                    }
                )
    return edges


class Neo4jBuilder:
    """Build (MERGE, idempotent) từ 2 CSV + client Neo4j."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def ensure_schema(self) -> None:
        """Constraints uniqueness + index query thường dùng."""
        queries = (
            "CREATE CONSTRAINT document_so_hieu IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.so_hieu IS UNIQUE",
            "CREATE CONSTRAINT provision_unique_key IF NOT EXISTS "
            "FOR (p:Provision) REQUIRE p.unique_key IS UNIQUE",
        )
        for q in queries:
            self.client._run_write(q)  # noqa: SLF001 (nội bộ module)

    def build(
        self,
        registry: list[dict[str, str]],
        relations: list[dict[str, str]],
    ) -> dict[str, int]:
        self.ensure_schema()
        docs = plan_documents(registry)
        provisions, unparsed = plan_provisions(relations, registry)
        edges = plan_edges(relations, registry)

        for doc in docs:
            self.client._run_write(  # noqa: SLF001
                "MERGE (d:Document {so_hieu: $so_hieu}) "
                "SET d += $props",
                so_hieu=doc["so_hieu"], props=doc,
            )
        for prov in provisions:
            self.client._run_write(  # noqa: SLF001
                "MERGE (p:Provision {unique_key: $key}) "
                "SET p += $props "
                "WITH p "
                "MATCH (d:Document {so_hieu: p.document}) "
                "MERGE (p)-[:BELONGS_TO]->(d)",
                key=prov["unique_key"],
                props={
                    "unique_key": prov["unique_key"],
                    "document": prov["document"],
                    "label": prov["label"],
                    "kind": prov["kind"],
                    "dataset": prov["dataset"],
                    "document_title": prov["document_title"],
                    "effective_date": prov["effective_date"] or None,
                    "effective_date_source": prov["effective_date_source"],
                    # map khoản->ngày riêng (phương án A). Neo4j lưu
                    # map; query khoản-level đọc p.clause_dates.
                    "clause_dates": prov["clause_dates"] or None,
                },
            )
        for edge in edges:
            self.client._run_write(  # noqa: SLF001
                "MATCH (a:Provision {unique_key: $from_key}), "
                "(b:Provision {unique_key: $to_key}) "
                f"MERGE (a)-[r:{edge['relation_type']}]->(b) "
                "SET r.clauses = $clauses, "
                "r.relation_id = $relation_id, "
                "r.effective_date = $effective_date, "
                "r.effective_date_source = $eff_source, "
                "r.verification_status = $verification_status, "
                "r.source_reference = $source_reference, "
                "r.relation_notes = $relation_notes",
                from_key=edge["from_key"], to_key=edge["to_key"],
                clauses=edge["clauses"],
                relation_id=edge["relation_id"],
                effective_date=edge["effective_date"],
                eff_source=edge["effective_date_source"],
                verification_status=edge["verification_status"],
                source_reference=edge["source_reference"],
                relation_notes=edge["relation_notes"],
            )

        return {
            "documents": len(docs),
            "provisions": len(provisions),
            "edges": len(edges),
            "unparsed": len(unparsed),
        }

    def clear(self) -> None:
        """Xóa graph do builder tạo (chỉ 2 label của ta) - dùng --force."""
        for label in ("Provision", "Document"):
            self.client._run_write(  # noqa: SLF001
                f"MATCH (n:{label}) DETACH DELETE n"
            )


__all__ = [
    "Neo4jBuilder",
    "read_registry",
    "read_relations",
    "plan_documents",
    "plan_provisions",
    "plan_edges",
]
