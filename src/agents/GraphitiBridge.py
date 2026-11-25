"""GraphitiBridge — lớp nối P6 (đồng bộ tri thức) với P7 (Agent).

Bối cảnh (đọc lại để tránh hiểu sai):
  - Spec P6 giả định "Redis local + Ollama Qwen2.5-7B (2 GPU) + DB
    riêng `graphiti`". Trên máy thực tế:
      * chỉ có 1 GPU RTX 3050 4GB -> 7B không chạy được, dùng
        Qwen2.5-3B (Q4) qua Ollama.
      * graphiti-core 0.31 **đã bỏ Redis** -> mọi thứ lưu Neo4j.
      * Neo4j Community 5.26 **không tạo được DB riêng** -> dùng
        chung DB `neo4j`, tách bằng `group_id` (xem GROUP_ID).
  - Vì vậy module này:
      1. Dùng **BGE-M3 local (CPU)** làm embedder -> embedding nhất quán
         với P1 (vector DB), offline 100%, không cần API key.
      2. Dùng **Ollama (OpenAI-compatible)** làm LLM cho phần extract
         tri thức mới; nếu Ollama không sẵn sàng -> degrade (không chết).
      3. Cung cấp `ingest_deterministic()` để nạp node/cạnh đã có sẵn
         trong DB mà **không cần LLM** (nhanh, deterministic, kiểm định
         được), và `sync_new_episodes()` cho phần LLM (chậm, tốn token).

Ranh giới quan trọng: nội dung lấy từ văn bản là DỮ LIỆU KHÔNG TIN CẬY
(CLAUDE.md §6.3) -> mọi episode ingest đều đi qua prompt có delimiters.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from graphiti_core.cross_encoder import CrossEncoderClient  # noqa: E402
from graphiti_core.embedder import EmbedderClient  # noqa: E402
from graphiti_core.llm_client.client import LLMClient  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV = PROJECT_ROOT / ".env"
if _ENV.exists():
    load_dotenv(_ENV)

# group_id để tách graphiti khỏi KG pháp lý chính trong cùng DB.
GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "agentlaw-p6")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
# Ollama không cần key thật nhưng OpenAI SDK bắt buộc có.
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

EMBEDDING_DIM = 1024   # BGE-M3
EMBED_BATCH = 256      # số text mỗi lần forward BGE-M3 (CPU)


# ---------------------------------------------------------------------------
# Embedder: BGE-M3 local -> khớp P1, offline
# ---------------------------------------------------------------------------
class BgeM3Embedder(EmbedderClient):
    """EmbedderClient của graphiti, dùng BGE-M3 local (CPU).

    Graphiti cần dimension cố định; BGE-M3 cho 1024. Vector DB P1 cũng
    dùng BGE-M3 -> cùng không gian embedding, search chéo được.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            import embedder as _emb

            self._model = _emb
        return self._model

    async def create(
        self, input_data: str | list[str] | Any
    ) -> list[float]:
        """Trả embedding PHẲNG của phần tử đầu tiên (đúng contract của
        graphiti: `name_embedding` là mảng float 1 chiều)."""
        emb = self._load()
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = list(input_data)
        if not texts:
            return []
        vectors, _ = emb.embed_texts(texts)
        return [float(x) for x in vectors[0]]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embedding cho nhiều text -> list[mảng phẳng]."""
        if not input_data_list:
            return []
        emb = self._load()
        vectors, _ = emb.embed_texts(list(input_data_list))
        return [[float(x) for x in v] for v in vectors]

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# LLM: Ollama (OpenAI-compatible)
# ---------------------------------------------------------------------------
def ollama_available() -> bool:
    """Ollama có sẵn sàng không?"""
    try:
        import urllib.request

        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


class NullLLMClient(LLMClient):
    """LLMClient giả — dùng khi ingest deterministic (không LLM).

    Graphiti bắt buộc phải có llm_client (mặc định OpenAIClient, đòi
    OPENAI_API_KEY). Với ingest node/cạnh đã biết trước ta không cần
    LLM -> dùng client này để tránh phải có key. Gọi LLM thì raise
    (để rõ sai, không im lặng).
    """

    def __init__(self) -> None:
        super().__init__(config=None)

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        group_id=None,
        prompt_name=None,
        **kwargs,
    ):
        raise RuntimeError(
            "NullLLMClient: ingest deterministic không dùng LLM. "
            "Bật use_llm=True (cần Ollama) để extract tri thức."
        )


class NullCrossEncoder(CrossEncoderClient):
    """CrossEncoderClient giả — không rerank (giữ thứ tự gốc).

    Graphiti mặc định tạo `OpenAIRerankerClient` (đòi OPENAI_API_KEY).
    Với ingest deterministic ta không cần rerank -> dùng client này.
    rank() trả [(passage, 0.0)] giữ nguyên thứ tự.
    """

    async def rank(
        self, query: str, passages: list[str]
    ) -> list[tuple[str, float]]:
        return [(p, 0.0) for p in passages]


def build_llm_client() -> Any:
    """OpenAIClient của graphiti trỏ về Ollama.

    Graphiti gọi LLM với 2 kích cỡ: `model` (kích cỡ medium/large) và
    `small_model` (kích cỡ small — dùng cho summarise/decide). Nếu
    `small_model=None`, graphiti fallback về model OpenAI mặc định
    (`gpt-6-luna`) -> 404 với Ollama. -> set CẢ HAI về Ollama model.
    """
    from graphiti_core.llm_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig

    config = LLMConfig(
        api_key=OLLAMA_API_KEY,
        model=OLLAMA_MODEL,
        small_model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        max_tokens=4096,
    )
    return OpenAIClient(config)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
@dataclass
class IngestReport:
    """Kết quả ingest."""

    nodes_written: int = 0
    edges_written: int = 0
    episodes_written: int = 0
    errors: list[str] = field(default_factory=list)
    llm_used: bool = False
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_written": self.nodes_written,
            "edges_written": self.edges_written,
            "episodes_written": self.episodes_written,
            "errors": self.errors,
            "llm_used": self.llm_used,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class GraphitiBridge:
    """Nối Graphiti với hệ thống: ingest + search.

    Dùng:
        bridge = GraphitiBridge()
        if bridge.available():
            bridge.ingest_deterministic(nodes, edges)
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        group_id: str = GROUP_ID,
        use_llm: bool = True,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.group_id = group_id
        self.use_llm = use_llm
        self._graphiti: Any = None
        self._init_error: str = ""
        # Neo4j async driver gắn connection pool vào event loop đầu tiên
        # dùng. Nếu mỗi thao tác gọi asyncio.run() riêng, các lần sau
        # dùng pool của loop đã đóng -> "'NoneType' object has no
        # attribute 'send'". -> Giữ MỘT loop suốt vòng đời bridge.
        self._loop: Any = None

    # -- Event loop dùng chung ---------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Chạy coroutine trên MỘT event loop dùng chung.

        Neo4j async driver giữ connection pool gắn với loop đầu tiên.
        Dùng `asyncio.run` cho từng thao tác sẽ tạo loop mới mỗi lần ->
        pool thuộc loop đã đóng -> lỗi "'NoneType' object has no
        attribute 'send'". Vì vậy giữ một loop sống suốt vòng đời bridge.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    # -- Khởi tạo -----------------------------------------------------------

    @property
    def graphiti(self) -> Any:
        """Graphiti instance (lazy)."""
        if self._graphiti is None:
            from graphiti_core import Graphiti

            if self.use_llm and ollama_available():
                llm_client = build_llm_client()
            else:
                # Ingest deterministic: không LLM.
                llm_client = NullLLMClient()
            # LUÔN dùng NullCrossEncoder: graphiti chỉ cần rerank khi
            # muốn gọi OpenAI reranker (đòi OPENAI_API_KEY). Tìm kiếm
            # pháp lý của P6/P7 dùng vector cosine (dense) -> không
            # cần cross-encoder. Dùng client null giữ thứ tự gốc.
            cross_encoder = NullCrossEncoder()

            self._graphiti = Graphiti(
                uri=self.uri,
                user=self.user,
                password=self.password,
                llm_client=llm_client,
                embedder=BgeM3Embedder(),
                cross_encoder=cross_encoder,
                store_raw_episode_content=True,
            )
        return self._graphiti

    def available(self) -> bool:
        """Graphiti dùng được không?"""
        try:
            _ = self.graphiti
            return True
        except Exception as exc:
            self._init_error = str(exc)
            return False

    def status(self) -> dict[str, Any]:
        """Trạng thái P6 để báo cáo."""
        return {
            "graphiti_available": self.available(),
            "llm_backend": (
                f"ollama/{OLLAMA_MODEL}"
                if (self.use_llm and ollama_available())
                else "none (ingest deterministic vẫn chạy)"
            ),
            "embedder": "BGE-M3 local (CPU, dim=1024)",
            "group_id": self.group_id,
            "neo4j_database": "neo4j",
            "note": (
                "Neo4j Community không tạo được DB riêng -> dùng chung "
                "DB 'neo4j', tách bằng group_id"
            ),
            "error": self._init_error,
        }

    # -- Ingest deterministic (không LLM) ----------------------------------

    def ingest_deterministic(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> IngestReport:
        """Nạp node/cạnh đã biết vào Graphiti — KHÔNG cần LLM.

        `nodes`: [{name, uuid, labels, summary, attributes}]
        `edges`: [{name, fact, source_uuid, target_uuid, valid_at}]
        """
        import time

        started = time.perf_counter()
        report = IngestReport()
        if not self.available():
            report.errors.append(self._init_error or "graphiti không khả dụng")
            return report

        async def _run() -> None:
            from graphiti_core.edges import EntityEdge
            from graphiti_core.nodes import EntityNode

            g = self.graphiti
            driver = g.driver
            now = datetime.now(timezone.utc)

            # Graphiti chỉ schedule build_indices_and_constraints() khi
            # khởi tạo CÓ event loop đang chạy. Bridge tạo Graphiti ngoài
            # loop -> index (fulltext + vector) không được tạo -> search
            # lỗi "no such fulltext schema index". -> Build tường minh.
            await g.build_indices_and_constraints()

            # --- Batch embedding -------------------------------------------
            # Gọi embedder cho TỪNG node tốn ~0.58s/node (CPU forward
            # pass mỗi lần) -> 5.269 node mất ~51 phút. Batch 1 lần cho
            # cả node + edge -> nhanh hơn nhiều.
            node_texts = [
                (s.get("name") or "").replace("\n", " ")
                for s in nodes
            ]
            node_vecs: list[list[float]] = []
            for i in range(0, len(node_texts), EMBED_BATCH):
                chunk = [t for t in node_texts[i:i + EMBED_BATCH] if t]
                if chunk:
                    node_vecs.extend(await g.embedder.create_batch(chunk))
                else:
                    node_vecs.append([])

            edge_texts = [
                (s.get("fact") or s.get("name") or "").replace("\n", " ")
                for s in edges
            ]
            edge_vecs: list[list[float]] = []
            for i in range(0, len(edge_texts), EMBED_BATCH):
                chunk = [t for t in edge_texts[i:i + EMBED_BATCH] if t]
                if chunk:
                    edge_vecs.extend(await g.embedder.create_batch(chunk))
                else:
                    edge_vecs.append([])

            for idx, spec in enumerate(nodes):
                try:
                    node = EntityNode(
                        uuid=spec["uuid"],
                        name=spec["name"],
                        group_id=self.group_id,
                        labels=spec.get("labels") or ["LegalProvision"],
                        created_at=now,
                        summary=spec.get("summary", ""),
                        attributes=spec.get("attributes") or {},
                    )
                    # Embedding đã tính sẵn ở trên (batch).
                    node.name_embedding = node_vecs[idx]
                    await node.save(driver)
                    report.nodes_written += 1
                except Exception as exc:
                    report.errors.append(f"node {spec.get('name')}: {exc}")

            for idx, spec in enumerate(edges):
                try:
                    edge = EntityEdge(
                        uuid=spec.get("uuid"),
                        group_id=self.group_id,
                        source_node_uuid=spec["source_uuid"],
                        target_node_uuid=spec["target_uuid"],
                        name=spec.get("name", ""),
                        fact=spec.get("fact", ""),
                        created_at=now,
                        valid_at=spec.get("valid_at") or now,
                        attributes=spec.get("attributes") or {},
                    )
                    edge.fact_embedding = edge_vecs[idx]
                    await edge.save(driver)
                    report.edges_written += 1
                except Exception as exc:
                    report.errors.append(f"edge {spec.get('name')}: {exc}")

        try:
            self._run(_run())
        except Exception as exc:
            report.errors.append(f"ingest lỗi: {exc}")

        report.duration_seconds = time.perf_counter() - started
        return report

    def sync_episode(
        self,
        name: str,
        body: str,
        source_description: str,
        reference_time: datetime | None = None,
    ) -> IngestReport:
        """Nạp 1 episode (văn bản) qua LLM để trích xuất tri thức mới.

        Cần Ollama. Nếu không có -> report lỗi, KHÔNG chết.
        """
        import time

        started = time.perf_counter()
        report = IngestReport()
        if not self.available():
            report.errors.append(self._init_error or "graphiti không khả dụng")
            return report
        if not ollama_available():
            report.errors.append(
                "Ollama không khả dụng -> không thể extract tri thức bằng LLM"
            )
            return report

        report.llm_used = True

        async def _run() -> None:
            from graphiti_core.nodes import EpisodeType

            res = await self.graphiti.add_episode(
                name=name,
                episode_body=body,
                source_description=source_description,
                reference_time=reference_time or datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id=self.group_id,
                update_communities=False,
            )
            report.episodes_written = 1
            # graphiti 0.31: AddEpisodeResults có fields `nodes`/`edges`
            # (KHÔNG phải `entity_nodes`/`edge_results` như API cũ).
            report.nodes_written = len(res.nodes or [])
            report.edges_written = len(res.edges or [])

        try:
            self._run(_run())
        except Exception as exc:
            report.errors.append(f"add_episode lỗi: {exc}")

        report.duration_seconds = time.perf_counter() - started
        return report

    # -- Search -------------------------------------------------------------

    def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Search trong graphiti (cần embedder, không cần LLM)."""
        if not self.available():
            return []

        async def _run() -> list[dict[str, Any]]:
            edges = await self.graphiti.search(
                query,
                group_ids=[self.group_id],
                num_results=num_results,
            )
            return [
                {
                    "fact": e.fact,
                    "name": e.name,
                    "source": e.source_node_uuid,
                    "target": e.target_node_uuid,
                    "valid_at": e.valid_at.isoformat() if e.valid_at else None,
                }
                for e in edges
            ]

        try:
            return self._run(_run())
        except Exception:
            return []

    def search_with_nodes(
        self, query: str, num_results: int = 5
    ) -> list[dict[str, Any]]:
        """Search + giải xuất node 2 đầu cạnh -> đủ citation/text/kg_id.

        Dùng cho P7: Agent muốn evidence có nguồn rõ (số hiệu văn bản,
        điều/khoản, text) chứ không chỉ "fact" trần. Mỗi kết quả gộp
        1 fact (cạnh) + 2 node 2 đầu (bên nguồn & bên đích).
        """
        base = self.search(query, num_results)
        if not base:
            return []

        async def _run() -> list[dict[str, Any]]:
            from graphiti_core.nodes import EntityNode

            uuids: list[str] = []
            for r in base:
                for k in ("source", "target"):
                    u = r.get(k)
                    if u and u not in uuids:
                        uuids.append(u)
            nodes = await EntityNode.get_by_uuids(
                self.graphiti.driver, uuids, group_id=self.group_id
            )
            by_uuid = {n.uuid: n for n in nodes}

            out = []
            for r in base:
                src = by_uuid.get(r.get("source"))
                dst = by_uuid.get(r.get("target"))
                out.append({
                    "fact": r.get("fact"),
                    "name": r.get("name"),
                    "source": r.get("source"),
                    "target": r.get("target"),
                    "valid_at": r.get("valid_at"),
                    "source_node": self._node_brief(src),
                    "target_node": self._node_brief(dst),
                })
            return out

        try:
            return self._run(_run())
        except Exception:
            return base

    # -- Exact-by-anchor (P7, ưu tiên #1) ---------------------------------
    #
    # Vì sao: search() bằng vector trên TOÀN BỘ 5.499 cạnh -> câu hỏi
    # "Điều 49 Luật Doanh nghiệp" ra cả fact của 296/2026, 144/2026...
    # (gần nghĩa trong không gian vector, KHÔNG liên quan pháp lý).
    # Khi câu hỏi ĐÃ có số hiệu + điều thì không cần vector: tra đồ thị
    # tất định theo anchor -> chính xác, nhanh, không nhiễu.

    # Từ khóa trong câu hỏi -> loại quan hệ graphiti tương ứng.
    RELATION_KEYWORDS: dict[str, tuple[str, ...]] = {
        "AMENDS": ("sửa", "sửa đổi", "bổ sung", "amend", "thay đổi"),
        "REPEALS": ("bãi bỏ", "bỏ", "hết hiệu lực", "repeal", "ngưng"),
        "GUIDES": ("hướng dẫn", "hướng dẫn thực hiện", "guide", "giải thích"),
    }

    # Chỉ 3 quan hệ PHÁP LÝ được đưa vào evidence cho LLM. Cạnh HAS_*
    # (HAS_ARTICLE/HAS_CLAUSE/HAS_POINT/HAS_CHAPTER/HAS_SECTION/HAS_ANNEX)
    # là cạnh cấu trúc nội bộ của graph, không phải quan hệ pháp lý ->
    # luôn bị loại khỏi kết quả search.
    LEGAL_RELATIONS: tuple[str, ...] = ("AMENDS", "REPEALS", "GUIDES")

    def _relation_filter(self, query: str) -> list[str]:
        """Đoán loại quan hệ người dùng đang hỏi.

        - Có từ khóa (sửa/bãi bỏ/hướng dẫn) -> chỉ loại tương ứng.
        - Không có từ khóa -> trả 3 quan hệ PHÁP LÝ (AMENDS/REPEALS/GUIDES).

        Luôn trả danh sách KHÔNG RỖNG (trong 3 quan hệ pháp lý) để Cypher
        không bao giờ leak cạnh HAS_* vào evidence.
        """
        q = (query or "").lower()
        hits: list[str] = []
        for rel, keys in self.RELATION_KEYWORDS.items():
            if any(k in q for k in keys):
                hits.append(rel)
        if hits:
            return hits
        return list(self.LEGAL_RELATIONS)

    def search_exact(
        self,
        doc_numbers: list[str],
        articles: list[str],
        query: str = "",
        num_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Tra cạnh graphiti theo anchor (số hiệu vb + điều). KHÔNG embedding.

        Trả về cùng shape với `search_with_nodes()` để `graphiti_block()`
        dùng chung được.

        - `doc_numbers`: so_hieu trong registry, vd ['59/2020/QH14'].
        - `articles`: so điều, vd ['49'].
        - `query`: dùng để đoán loại quan hệ (sửa/bãi bỏ/hướng dẫn).
        """
        if not self.available():
            return []
        docs = [d for d in (doc_numbers or []) if d]
        arts = [a for a in (articles or []) if a]
        if not docs and not arts:
            return []

        rels = self._relation_filter(query)

        async def _run() -> list[dict[str, Any]]:
            driver = self.graphiti.driver

            # Graphiti lưu attributes thành PROPERTIES TOP-LEVEL của node
            # (n.so_hieu, n.dieu, n.kg_id...), KHÔNG phải map lồng.
            match_clauses: list[str] = []
            params: dict[str, Any] = {"gid": self.group_id}
            if docs:
                match_clauses.append(
                    "(n.so_hieu IN $docs OR m.so_hieu IN $docs)"
                )
                params["docs"] = docs
            if arts:
                match_clauses.append(
                    "(n.dieu IN $arts OR m.dieu IN $arts)"
                )
                params["arts"] = arts
            if not match_clauses:
                return []

            where = " AND ".join(match_clauses)
            rel_clause = ""
            if rels:
                rel_clause = " AND e.name IN $rels"
                params["rels"] = rels

            cypher = (
                "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
                "WHERE e.group_id = $gid "
                f"AND {where}{rel_clause} "
                "RETURN e.fact AS fact, e.name AS name, "
                "e.source_uuid AS source, e.target_uuid AS target, "
                "e.valid_at AS valid_at, "
                "n.uuid AS src_uuid, m.uuid AS tgt_uuid, "
                "n.name AS src_name, m.name AS tgt_name, "
                "n.kg_id AS src_kg, n.so_hieu AS src_so, n.dieu AS src_dieu, "
                "n.effective_date AS src_eff, "
                "m.kg_id AS tgt_kg, m.so_hieu AS tgt_so, m.dieu AS tgt_dieu, "
                "m.effective_date AS tgt_eff "
                "LIMIT $limit"
            )
            params["limit"] = num_results

            async with driver.session() as session:
                res = await session.run(cypher, params)
                records = await res.data()

            out: list[dict[str, Any]] = []
            for rec in records:
                valid_at = rec.get("valid_at")
                out.append({
                    "fact": rec.get("fact"),
                    "name": rec.get("name"),
                    "source": rec.get("source"),
                    "target": rec.get("target"),
                    "valid_at": valid_at.isoformat() if valid_at else None,
                    "source_node": self._brief_from_node_props(
                        rec.get("src_uuid"),
                        rec.get("src_name"),
                        {
                            "kg_id": rec.get("src_kg"),
                            "so_hieu": rec.get("src_so"),
                            "dieu": rec.get("src_dieu"),
                            "effective_date": rec.get("src_eff"),
                        },
                    ),
                    "target_node": self._brief_from_node_props(
                        rec.get("tgt_uuid"),
                        rec.get("tgt_name"),
                        {
                            "kg_id": rec.get("tgt_kg"),
                            "so_hieu": rec.get("tgt_so"),
                            "dieu": rec.get("tgt_dieu"),
                            "effective_date": rec.get("tgt_eff"),
                        },
                    ),
                    "match": "exact",
                })
            return out

        try:
            return self._run(_run())
        except Exception:
            return []

    @classmethod
    def _brief_from_node_props(
        cls, uuid: Any, name: Any, props: dict[str, Any]
    ) -> dict[str, Any]:
        """Dựng node brief từ properties đã rút gọn (kết quả Cypher)."""
        so = props.get("so_hieu")
        dieu = props.get("dieu")
        parts: list[str] = []
        if so:
            parts.append(so)
        if dieu:
            parts.append(f"Điều {dieu}")
        loc = " ".join(parts)
        return {
            "uuid": uuid,
            "name": name,
            "citation": f"{loc} ({so})" if (loc and so) else (loc or name or ""),
            "text": "",
            "kg_id": props.get("kg_id"),
            "so_hieu": so,
            "effective_date": props.get("effective_date"),
        }

    @staticmethod
    def _node_brief_from_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
        """Tóm tắt attributes thô -> citation/text/kg_id (bản không cần node)."""
        so = attrs.get("so_hieu")
        dieu = attrs.get("dieu")
        khoan = attrs.get("khoan")
        diem = attrs.get("diem")
        parts: list[str] = []
        if so:
            parts.append(so)
        if dieu:
            parts.append(f"Điều {dieu}")
        if khoan:
            parts.append(f"Khoản {khoan}")
        if diem:
            parts.append(f"Điểm {diem}")
        loc = " ".join(parts)
        citation = f"{loc} ({so})" if (loc and so) else (loc or "")
        return {
            "citation": citation,
            "text": (attrs.get("text") or "")[:800],
            "kg_id": attrs.get("kg_id"),
            "so_hieu": so,
            "nhom_pl": attrs.get("nhom_pl"),
        }

    @staticmethod
    def _node_brief(node: Any) -> dict[str, Any]:
        """Tóm tắt node thành citation/text/kg_id để cho prompt."""
        if node is None:
            return {}
        brief = GraphitiBridge._node_brief_from_attrs(node.attributes or {})
        if not brief.get("citation"):
            brief["citation"] = node.name or ""
        brief.setdefault("uuid", node.uuid)
        brief.setdefault("name", node.name)
        brief["summary"] = node.summary
        return brief

    # -- Dọn dẹp -----------------------------------------------------------

    def clear_group(self) -> int:
        """Xoá toàn bộ node/edge của group_id này (để nạp lại)."""
        if not self.available():
            return 0

        async def _run() -> int:
            driver = self.graphiti.driver
            async with driver.session() as session:
                res = await session.run(
                    "MATCH (n) WHERE n.group_id = $gid "
                    "DETACH DELETE n RETURN count(n) AS c",
                    {"gid": self.group_id},
                )
                rec = await res.single()
                return int(rec["c"]) if rec else 0

        try:
            return self._run(_run())
        except Exception:
            return 0


__all__ = [
    "BgeM3Embedder",
    "EMBEDDING_DIM",
    "GROUP_ID",
    "GraphitiBridge",
    "IngestReport",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "build_llm_client",
    "ollama_available",
]
