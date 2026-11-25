"""KgAccess — lớp truy cập KG dùng chung cho các Agent (P7).

Cung cấp:
  - `lookup(node_ids)` -> node + cây con 1 tầng (để materialize evidence)
  - `neighbors(node_id)` -> quan hệ sửa/bãi/hướng dẫn + anh-em
  - `provision_status()` -> tình trạng hiệu lực tại thời điểm

Nguồn dữ liệu:
  - Cây 4 tầng (Article/Clause/Point/Annex): đọc từ `data/processed/`
    qua `graph.kg.build.assemble()` (cache 1 lần trong process).
  - Quan hệ + thời gian: Neo4j qua `Neo4jClient` (schema 4-tầng P5).

Không ghi gì vào DB — chỉ đọc.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from graph.kg.build import assemble
from graph.kg.loader import KgNode, load_registry
from graph.kg.materialize import Evidence, materialize


class KgAccess:
    """Truy cập KG (cây trong bộ nhớ + quan hệ/time trên Neo4j)."""

    def __init__(self, use_neo4j: bool = True) -> None:
        self._lock = threading.Lock()
        self._graph: Any = None
        self._children: dict[str, list[KgNode]] | None = None
        self._registry: dict[str, dict] | None = None
        self._client: Any = None
        self._client_failed = False
        self._use_neo4j = use_neo4j

    # -- Cây 4 tầng (bộ nhớ) ---------------------------------------------

    def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        with self._lock:
            if self._graph is not None:
                return
            g = assemble()
            children: dict[str, list[KgNode]] = {}
            for n in g.nodes.values():
                if n.parent_id:
                    children.setdefault(n.parent_id, []).append(n)
            for kids in children.values():
                kids.sort(key=lambda x: (x.khoan, x.diem_index))
            self._graph = g
            self._children = children
            self._registry = load_registry()

    def node(self, node_id: str) -> KgNode | None:
        self._ensure_graph()
        return self._graph.nodes.get(node_id)

    def children(self, node_id: str) -> list[KgNode]:
        self._ensure_graph()
        return list((self._children or {}).get(node_id, []))

    def doc_meta(self, so_hieu: str) -> dict[str, Any]:
        self._ensure_graph()
        return dict((self._registry or {}).get(so_hieu, {}))

    def registry(self) -> dict[str, dict[str, Any]]:
        """Toàn bộ DocumentRegistry {so_hieu: row} (PC-1 verify-by-level).

        Nguồn: `load_registry()` (DocumentRegistry.csv — T1 cấu trúc,
        đã verify). Trả {} nếu chưa load (không bịa)."""
        self._ensure_graph()
        return dict(self._registry or {})

    def all_nodes(self) -> dict[str, KgNode]:
        self._ensure_graph()
        return dict(self._graph.nodes)

    # -- Evidence ----------------------------------------------------------

    def evidence_for(self, node_id: str) -> Evidence | None:
        """Materialize 1 node (+ cây con 1 tầng) thành evidence cho LLM."""
        node = self.node(node_id)
        if node is None:
            return None
        return materialize(
            node,
            children=self._children,
            doc_meta=self.doc_meta(node.so_hieu),
        )

    def evidence_for_many(
        self, node_ids: list[str]
    ) -> list[Evidence]:
        out: list[Evidence] = []
        seen: set[str] = set()
        for nid in node_ids:
            if nid in seen:
                continue
            seen.add(nid)
            ev = self.evidence_for(nid)
            if ev is not None:
                out.append(ev)
        return out

    def lookup(self, node_ids: list[str]) -> list[dict[str, Any]]:
        """Adapter cho `QueryRouter.graph_lookup`: trả node + text + metadata."""
        out: list[dict[str, Any]] = []
        for nid in node_ids:
            node = self.node(nid)
            if node is None:
                continue
            out.append({
                "node_id": nid,
                "text": node.text,
                "kind": node.kind,
                "metadata": {
                    "so_hieu": node.so_hieu,
                    "dieu": node.dieu,
                    "khoan": node.khoan,
                    "diem": node.diem,
                },
            })
        return out

    # -- Neo4j: quan hệ + thời gian --------------------------------------

    def _neo4j(self) -> Any | None:
        if not self._use_neo4j or self._client_failed:
            return None
        if self._client is None:
            try:
                from graph.Neo4jClient import Neo4jClient

                self._client = Neo4jClient()
            except Exception:
                self._client_failed = True
                return None
        return self._client

    def relations_of(
        self,
        so_hieu: str,
        article: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Quan hệ AMENDS/REPEALS/GUIDES có hướng tới (so_hieu, article)."""
        client = self._neo4j()
        if client is None:
            return []
        try:
            return client.amendments_of(
                so_hieu, article=article, as_of_date=as_of_date
            )
        except Exception:
            return []

    def status_of(
        self,
        so_hieu: str,
        article: str,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        """Trạng thái hiệu lực của 1 điều tại thời điểm (time-travel)."""
        client = self._neo4j()
        if client is None:
            return {
                "so_hieu": so_hieu,
                "article": article,
                "status": "unknown",
                "is_repealed": False,
                "is_amended": False,
                "relations": [],
            }
        try:
            return client.provision_status(
                so_hieu, article=article, as_of_date=as_of_date
            )
        except Exception:
            return {
                "so_hieu": so_hieu,
                "article": article,
                "status": "unknown",
                "is_repealed": False,
                "is_amended": False,
                "relations": [],
            }

    def governing(
        self,
        so_hieu: str,
        article: str,
        as_of_date: str | None = None,
        clause: str | None = None,
    ) -> dict[str, Any]:
        """Nội dung nào đang hiệu lực cho (vb, điều) tại thời điểm T."""
        client = self._neo4j()
        if client is None:
            return {
                "so_hieu": so_hieu,
                "article": article,
                "status": "unknown",
                "provision_in_effect_at_T": None,
                "governing_amendments": [],
                "effective_date_used": None,
                "effective_date_source": "unknown",
            }
        try:
            return client.governing_provision(
                so_hieu, article, as_of_date=as_of_date, clause=clause
            )
        except Exception:
            return {
                "so_hieu": so_hieu,
                "article": article,
                "status": "unknown",
                "provision_in_effect_at_T": None,
                "governing_amendments": [],
                "effective_date_used": None,
                "effective_date_source": "unknown",
            }

    def has_amendment_relations(
        self,
        so_hieu: str,
        article: str | None = None,
    ) -> bool:
        """Node này có bị VĂN BẢN KHÁC sửa/bãi bỏ/hướng dẫn không?

        Dùng cho OOD gate (P7): nếu đồ thị khẳng định "văn bản X sửa
        điều Y" thì câu hỏi về điều Y CHẮC CHẮN có trong DB, dù
        reranker (đo mức giống văn bản) chấm thấp vì không hiểu quan
        hệ sửa ("Điều 1 của NĐ sửa" trông không giống "điều 213").
        """
        return bool(self.relations_of(so_hieu, article=article))

    def related_documents(
        self, so_hieu: str, depth: int = 1
    ) -> list[str]:
        client = self._neo4j()
        if client is None:
            return []
        try:
            return client.related_documents(so_hieu, depth=depth)
        except Exception:
            return []


__all__ = ["KgAccess"]
