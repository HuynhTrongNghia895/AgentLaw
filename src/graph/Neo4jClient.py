from __future__ import annotations

"""Client Neo4j cho đồ án pháp lý.

Nối qua env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) - config trong
`.env` (git-ignored). Query helpers trả DỮ LIỆU THẬT (verification_status
từ CSV), không suy diễn quan hệ (CLAUDE.md §6.4).

Khi Neo4j chưa bật: `ping()` trả False, các query khác ném GraphNotReady
lỗi MÔ TẢ RÕ (không fake kết quả rỗng).
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GraphNotReady(RuntimeError):
    """Neo4j chưa bật / chưa build graph. Hướng dẫn rõ trong message."""

    def __init__(self) -> None:
        super().__init__(
            "Neo4j chưa sẵn sàng (chưa bật service hoặc chưa build graph). "
            "Windows: đảm bảo service Neo4j đang chạy "
            "(trình duyệt: http://localhost:7474), rồi chạy "
            ".venv\\Scripts\\python scripts\\build_graph.py"
        )


def _env(name: str, default: str | None = None) -> str:
    return os.environ.get(name, default or "")


class Neo4jClient:
    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        # Lazy import neo4j (có thể chưa cài / chưa cần)
        from neo4j import GraphDatabase  # type: ignore[import-not-found]

        load_dotenv(PROJECT_ROOT / ".env")
        self._driver = GraphDatabase.driver(
            uri or _env("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                username or _env("NEO4J_USERNAME", "neo4j"),
                password or _env("NEO4J_PASSWORD", ""),
            ),
        )

    # -- Lifecycle ---------------------------------------------------------

    def ping(self) -> bool:
        """Neo4j đạt được không? (không kiểm tra graph có dữ liệu)."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- Query helpers (data thật từ graph, không suy diễn) ---------------

    def _require_ready(self) -> None:
        if not self.ping():
            raise GraphNotReady()

    def _run_write(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Chạy query WRITE (MERGE/SET) trên 1 transaction.
        Ghi chú: phải đọc hết record TRƯỚC khi tx đóng (Neo4j driver
        ném ResultConsumedError nếu đọc sau commit)."""
        self._require_ready()
        with self._driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(query, **params)
                records = [dict(record) for record in result]
                tx.commit()
                return records

    def _query(
        self, query: str, **params: Any
    ) -> list[dict[str, Any]]:
        self._require_ready()
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def amendments_of(
        self,
        so_hieu: str,
        article: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Các quan hệ AMENDS/REPEALS/GUIDES có hướng TỚI văn bản này
        (điều/khoản/điểm này bị sửa/bãi bỏ/hướng dẫn bởi đâu).

        Schema 4-tầng mới (P5): node provision = Article/Clause/Point (có
        property `so_hieu`, `dieu`, `khoan`, `diem` trực tiếp); không còn
        `Provision`/`BELONGS_TO`. `article` lọc theo `dieu` (điều này,
        mọi khoản/điểm con của nó).
        `as_of_date` (ISO): chỉ quan hệ hiệu lực <= ngày này; None = tất cả."""
        anchor = "{so_hieu: $so_hieu" + (", dieu: $article" if article else "") + "}"
        q = (
            f"MATCH (src:Article|Clause|Point)-[r:AMENDS|REPEALS|GUIDES]->"
            f"(dst:Article|Clause|Point {anchor}) "
            "WHERE $as_of IS NULL "
            "OR r.relation_effective_date IS NULL "
            "OR r.relation_effective_date <= $as_of "
            "RETURN DISTINCT type(r) AS relation_type, "
            "src.dieu AS by_article, "
            "coalesce(r.detail_source_clause, src.khoan) AS by_clause, "
            "src.so_hieu AS by_document, "
            "r.clauses AS clauses, r.relation_effective_date AS effective_date, "
            "r.relation_id AS relation_id, "
            "r.source_reference AS source_reference "
            "ORDER BY r.relation_effective_date IS NULL, "
            "r.relation_effective_date DESC"
        )
        params: dict[str, Any] = {
            "so_hieu": so_hieu,
            "as_of": as_of_date,
        }
        if article:
            params["article"] = article
        return self._query(q, **params)

    def provision_status(
        self,
        so_hieu: str,
        article: str,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        """Trạng thái pháp lý của 1 provision tại thời điểm (đúng ngữ nghĩa
        pháp lý, không suy diễn - CLAUDE.md §6.4):

        - `is_repealed`: True nếu có quan hệ REPEALS hiệu lực <= as_of_date
          (bị bãi bỏ -> không còn áp dụng).
        - `is_amended`: True nếu có quan hệ AMENDS (bị sửa -> vẫn áp dụng
          theo BẢN HỢP NHẤT, không phải văn bản gốc).
        - `status`: 'repealed' | 'amended' | 'active' (repealed ưu tiên).

        Chỉ trả quan hệ đã verified có trong graph; không bịa.
        """
        relations = self.amendments_of(
            so_hieu, article=article, as_of_date=as_of_date
        )
        types = {row.get("relation_type") for row in relations}
        is_repealed = "REPEALS" in types
        is_amended = "AMENDS" in types
        if is_repealed:
            status = "repealed"
        elif is_amended:
            status = "amended"
        else:
            status = "active"
        return {
            "so_hieu": so_hieu,
            "article": article,
            "as_of_date": as_of_date,
            "status": status,
            "is_repealed": is_repealed,
            "is_amended": is_amended,
            "relations": relations,
        }

    def governing_provision(
        self,
        so_hieu: str,
        article: str,
        as_of_date: str | None = None,
        clause: str | None = None,
    ) -> dict[str, Any]:
        """Nội dung nào đang hiệu lực cho (so_hieu, article) tại thời
        điểm T (time-travel, hiệu lực 3 lớp - CLAUDE.md §6.2/§6.4).

        Trả:
          - provision_in_effect_at_T: True/False (điều này đã có hiệu
            lực chưa tại T; so provision.effective_date <= T, fallback
            document.effective_date nếu provision rỗng).
          - status: 'chưa_hiệu_lực' | 'gốc' | 'amended' | 'repealed'
          - governing_amendments: các quan hệ AMENDS/REPEALS có
            effective_date <= T (văn bản + điều mục) - để GĐ4 chọn
            đúng văn bản khi retrieval nội dung.
          - effective_date_used + source: ngày hiệu lực provision +
            nguồn ('provision' | 'document' | 'unknown').

        Nguyên tắc: chỉ trả dữ liệu graph (cờ source audit); provision
        chưa hiệu lực tại T -> không xét quan hệ (chưa tồn tại để sửa).
        """
        # Ngày hiệu lực provision (lớp 2, fallback lớp 1). Nếu hỏi ở
        # cấp KHOẢN (clause), tra map p.clause_dates trước (cấu trúc
        # A: node điều + map khoản->ngày riêng).
        q_prov = (
            "MATCH (p:Article {so_hieu: $so, dieu: $art}) "
            "OPTIONAL MATCH (d:Document {so_hieu: $so}) "
            "RETURN p.provision_effective_date AS pdate, "
            "p.effective_date AS pfallback, "
            "d.effective_date AS ddate, "
            "p.invalid_at AS invalid_at"
        )
        rows = self._query(q_prov, so=so_hieu, art=article)
        if not rows:
            return {
                "so_hieu": so_hieu,
                "article": article,
                "clause": clause,
                "as_of_date": as_of_date,
                "provision_in_effect_at_T": None,
                "status": "unknown",
                "governing_amendments": [],
                "effective_date_used": None,
                "effective_date_source": "unknown",
            }
        # Lớp 2 (ngày riêng của provision) -> lớp 1 (ngày chung văn bản).
        # Node Clause có ngày riêng theo khoản -> tra node khoản trước.
        pdate = rows[0]["pdate"]
        psource = "provision" if pdate else "unknown"
        ddate = rows[0]["ddate"]
        if clause is not None and psource == "unknown":
            crows = self._query(
                "MATCH (p:Clause {so_hieu: $so, dieu: $art, khoan: $cl}) "
                "RETURN p.provision_effective_date AS pdate",
                so=so_hieu, art=article, cl=str(clause),
            )
            if crows and crows[0]["pdate"]:
                pdate = crows[0]["pdate"]
                psource = "provision"
        eff_used = pdate or ddate
        eff_source = psource if pdate else ("document" if ddate else "unknown")

        if as_of_date is None:
            # Không có T: trả trạng thái "hiện tại" = cả 3 lớp, provision
            # tính hiệu lực (có ngày).
            in_effect = True
        else:
            in_effect = bool(eff_used and eff_used <= as_of_date)

        if not in_effect:
            return {
                "so_hieu": so_hieu,
                "article": article,
                "clause": clause,
                "as_of_date": as_of_date,
                "provision_in_effect_at_T": False,
                "status": "chưa_hiệu_lực",
                "governing_amendments": [],
                "effective_date_used": eff_used,
                "effective_date_source": eff_source,
            }

        # Provision đã hiệu lực tại T -> xét quan hệ sửa/bãi hiệu lực <= T
        rels = self.amendments_of(
            so_hieu, article=article, as_of_date=as_of_date
        )
        types = {r.get("relation_type") for r in rels}
        gov = [
            r for r in rels
            if r.get("relation_type") in ("AMENDS", "REPEALS")
        ]
        if "REPEALS" in types:
            status = "repealed"
        elif "AMENDS" in types:
            status = "amended"
        else:
            status = "gốc"
        return {
            "so_hieu": so_hieu,
            "article": article,
            "clause": clause,
            "as_of_date": as_of_date,
            "provision_in_effect_at_T": True,
            "status": status,
            "governing_amendments": gov,
            "effective_date_used": eff_used,
            "effective_date_source": eff_source,
        }

    def related_documents(
        self,
        so_hieu: str,
        depth: int = 1,
    ) -> list[str]:
        """Văn bản liên quan (amends/repeals/guides, cả 2 chiều) trong
        `depth` bước - cho tăng cường candidate retrieval (GĐ4).

        Schema 4-tầng: node provision (Article/Clause/Point) có property
        `so_hieu` trực tiếp -> không cần join qua BELONGS_TO. Đường đi:
        a <-(HAS_*)- (ap) -[REL*depth]-> (bp) -(HAS_*)-> b
        """
        depth = max(1, int(depth))
        q = (
            f"MATCH (a:Document {{so_hieu: $so_hieu}}) "
            f"MATCH p = (a)<-[:HAS_ARTICLE|HAS_CLAUSE|HAS_POINT*1..2]-"
            f"(ap:Article|Clause|Point)"
            f"-[:AMENDS|REPEALS|GUIDES*1..{depth}]->(bp:Article|Clause|Point)"
            f"-[:HAS_ARTICLE|HAS_CLAUSE|HAS_POINT*1..2]->(b:Document) "
            "WHERE a <> b "
            "RETURN DISTINCT b.so_hieu AS related"
        )
        rows = self._query(q, so_hieu=so_hieu)
        return [row["related"] for row in rows if row.get("related")]

    def document_graph_stats(self) -> dict[str, int]:
        rows = self._query(
            "MATCH (n) "
            "RETURN labels(n) AS labels, count(*) AS c"
        )
        out: dict[str, int] = {}
        for row in rows:
            labels = " + ".join(row["labels"]) or "Node"
            out[labels] = out.get(labels, 0) + row["c"]
        edge_rows = self._query(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c"
        )
        for row in edge_rows:
            out[f"rel:{row['t']}"] = row["c"]
        return out
