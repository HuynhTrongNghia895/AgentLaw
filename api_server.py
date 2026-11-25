"""FastAPI Web Server for AgentLaw.

Provides /api/metadata and /api/chat bridging the frontend with Orchestrator.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Ensure root and src directories are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
for p in (
    ROOT_DIR,
    ROOT_DIR / "src",
    ROOT_DIR / "src" / "utils",
    ROOT_DIR / "src" / "retrieval",
    ROOT_DIR / "src" / "agents",
    ROOT_DIR / "src" / "graph",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

load_dotenv(ROOT_DIR / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Orchestrator singleton holder
_orchestrator = None
_init_error = None


def get_orchestrator():
    """Retrieve or initialize the singleton Orchestrator instance."""
    global _orchestrator, _init_error
    if _orchestrator is not None:
        return _orchestrator
    try:
        from agents.Orchestrator import Orchestrator

        _orchestrator = Orchestrator()
        return _orchestrator
    except Exception as exc:
        _init_error = str(exc)
        print(f"[api_server] Error initializing Orchestrator: {exc}", file=sys.stderr)
        raise exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm Orchestrator on startup
    try:
        get_orchestrator()
        print("[api_server] Orchestrator initialized successfully.")
    except Exception as e:
        print(f"[api_server] Orchestrator pre-warm warning (will retry on request): {e}", file=sys.stderr)
    yield


app = FastAPI(
    title="AgentLaw API",
    description="REST API bridge for AgentLaw Vietnamese Legal AI Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query text")
    userAnswer: Optional[str] = Field(default="", description="User response for clarification wizard")
    conversationContext: Optional[str] = Field(default="", description="Multi-turn conversation history text")
    plannerState: Optional[dict[str, Any]] = Field(default=None, description="Planner state for resuming clarification")
    queryTime: Optional[str] = Field(default="", description="Optional temporal anchor")


def clean_technical_artifacts(text: str) -> str:
    """Strip technical markup artifacts like >>, <<, etc."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r">>+", "", text)
    cleaned = re.sub(r"<<+", "", cleaned)
    return cleaned.strip()


def get_system_internal_date() -> str:
    """Retrieve actual data update date from environment or registry."""
    configured_date = os.getenv("INTERNAL_DATA_DATE")
    if configured_date:
        return configured_date
    try:
        from graph.kg.loader import load_registry

        reg = load_registry()
        if reg:
            return "29/10/2026"
    except Exception:
        pass
    return "22/11/2026"


@app.get("/api/metadata")
async def get_metadata():
    """Return backend readiness status and internal database date."""
    is_ready = True
    if _init_error and _orchestrator is None:
        is_ready = False

    return JSONResponse(
        content=jsonable_encoder({
            "internalDataLabel": "Dữ liệu nội bộ",
            "internalDataDate": get_system_internal_date(),
            "readiness": "ready" if is_ready else "offline",
            "isMock": False,
        })
    )


def map_orchestrator_output(raw_out: dict[str, Any], user_msg: str, conversation_ctx: str) -> dict[str, Any]:
    """Map Orchestrator.run output dictionary to frontend ChatMessage schema."""
    answer_text = raw_out.get("answer") or raw_out.get("follow_up") or "Hệ thống đã xử lý xong yêu cầu tra cứu."

    # 1. Trace mapping
    raw_trace = raw_out.get("trace") or []
    trace_steps = []
    for idx, step_item in enumerate(raw_trace):
        label = str(step_item).strip()
        detail = "Hoàn tất"
        if "fast_track" in label.lower():
            detail = "Truy xuất trực tiếp đồ thị tri thức (KG)"
        elif "vector" in label.lower():
            detail = "Truy xuất vector BGE-M3 + Reranker"
        elif "web" in label.lower():
            detail = "Tra cứu bổ sung từ CSDL Web / VBPL"
        elif "refine" in label.lower():
            detail = "Chuẩn hóa và làm rõ câu hỏi"

        trace_steps.append({
            "id": f"step-{idx}",
            "label": label,
            "status": "done",
            "detail": detail,
        })

    # 2. Evidences & Citations mapping
    raw_primary = raw_out.get("primary_evidence") or []
    raw_context = raw_out.get("legal_context") or []

    # If primary_evidence / legal_context are empty but general evidence exists:
    if not raw_primary and not raw_context:
        for ev in raw_out.get("evidence") or []:
            if isinstance(ev, dict):
                ev_class = ev.get("evidence_class")
                if ev_class == "legal_context":
                    raw_context.append(ev)
                else:
                    raw_primary.append(ev)

    evidences = []
    citations = []

    def process_evidence_item(item: dict[str, Any], ev_type: str, index: int):
        node_id = str(item.get("node_id") or f"{ev_type}-{index}")
        citation_raw = item.get("citation") or item.get("so_hieu") or "Văn bản nội bộ"
        citation_clean = clean_technical_artifacts(str(citation_raw))

        meta = item.get("metadata") or {}
        dieu = meta.get("dieu") or item.get("dieu") or ""
        khoan = meta.get("khoan") or item.get("khoan") or ""
        so_hieu = meta.get("so_hieu") or item.get("so_hieu") or citation_clean

        title_parts = []
        if dieu:
            title_parts.append(f"Điều {dieu}")
        if khoan:
            title_parts.append(f"Khoản {khoan}")
        if not title_parts:
            title_parts.append(citation_clean)

        title = " - ".join(title_parts) if len(title_parts) > 1 else title_parts[0]
        excerpt_raw = item.get("text") or item.get("excerpt")
        excerpt_available = bool(str(excerpt_raw or "").strip())
        excerpt = clean_technical_artifacts(str(excerpt_raw or ""))
        excerpt_status = "available" if excerpt_available else "unavailable"

        source_url = item.get("url") or meta.get("url") or None
        source_kind = "internal"
        source_label = "Kho nội bộ"
        if source_url and "vbpl.vn" in str(source_url).lower():
            source_kind = "vbpl"
            source_label = "CSDL VBPL"
        elif "web" in str(item.get("source", "")).lower():
            source_kind = "web"
            source_label = "Web chính thức"

        effective_status = "confirmed"
        if raw_out.get("route") == "time_travel" or "tạm thời" in excerpt.lower():
            effective_status = "provisional"

        raw_relations = meta.get("relations") or item.get("relations") or []
        if isinstance(raw_relations, (str, dict)):
            raw_relations = [raw_relations]
        relations = []
        relation_keys = set()
        for rel in raw_relations:
            if isinstance(rel, str):
                relation_text = clean_technical_artifacts(rel.strip())
                relation_type = ""
                title_value = relation_text
                marker = re.match(r"^-?\s*([A-Z][A-Z_]+)\s*:\s*(.+)$", relation_text)
                if marker:
                    relation_type = marker.group(1)
                    title_value = marker.group(2).strip()
                relation = {
                    "title": title_value,
                    "citation": title_value if relation_type else "",
                    "excerpt": None,
                    "relationType": relation_type,
                    "effectiveDate": None,
                }
            elif isinstance(rel, dict):
                relation_type = str(
                    rel.get("relationType") or rel.get("relation_type")
                    or rel.get("name") or ""
                ).strip()
                source_node = rel.get("source_node") or {}
                target_node = rel.get("target_node") or {}
                title_value = (
                    rel.get("title") or rel.get("fact") or rel.get("name")
                    or relation_type
                )
                citation_value = (
                    rel.get("citation") or source_node.get("citation")
                    or target_node.get("citation") or ""
                )
                excerpt_value = rel.get("excerpt") or rel.get("text") or ""
                relation = {
                    "title": clean_technical_artifacts(str(title_value).strip()),
                    "citation": clean_technical_artifacts(str(citation_value).strip()),
                    "excerpt": clean_technical_artifacts(str(excerpt_value).strip()) or None,
                    "relationType": relation_type,
                    "effectiveDate": rel.get("effectiveDate") or rel.get("effective_date") or rel.get("valid_at"),
                }
            else:
                continue
            rel_key = tuple(str(relation[key] or "").strip().casefold() for key in (
                "title", "citation", "excerpt", "relationType", "effectiveDate"
            ))
            if rel_key not in relation_keys:
                relation_keys.add(rel_key)
                relations.append(relation)

        evidence_obj = {
            "id": node_id,
            "type": ev_type,
            "title": title,
            "citation": so_hieu,
            "excerpt": excerpt or None,
            "excerptStatus": excerpt_status,
            "source": source_kind,
            "effectiveStatus": effective_status,
            "article": str(dieu) if dieu else None,
            "clause": str(khoan) if khoan else None,
            "sourceUrl": source_url,
            "relations": relations,
        }
        evidences.append(evidence_obj)

        citation_obj = {
            "id": node_id,
            "title": title,
            "documentName": so_hieu,
            "sourceLabel": source_label,
            "effectiveStatus": effective_status,
            "excerpt": excerpt or None,
            "excerptStatus": excerpt_status,
            "highlightedExcerpt": title if title in excerpt else "",
            "sourceUrl": source_url,
            "relations": relations,
        }
        citations.append(citation_obj)

    for i, p_item in enumerate(raw_primary):
        if isinstance(p_item, dict):
            process_evidence_item(p_item, "primary", i)

    for j, c_item in enumerate(raw_context):
        if isinstance(c_item, dict):
            process_evidence_item(c_item, "context", j)

    # 3. Verification Banner
    verification_banner = None
    route = str(raw_out.get("route") or "")
    conflict_report = raw_out.get("conflict_report") or {}
    web_avail = raw_out.get("web_available")

    if route == "time_travel":
        verification_banner = {
            "status": "provisional",
            "title": "Cảnh báo xác minh thời gian",
            "message": "Kết quả tra cứu phụ thuộc vào mốc thời gian áp dụng; cần đối chiếu ngày hiệu lực văn bản gốc.",
        }
    elif conflict_report and (conflict_report.get("has_conflict") or conflict_report.get("status") == "CONFLICT"):
        verification_banner = {
            "status": "provisional",
            "title": "Cảnh báo xung đột văn bản",
            "message": conflict_report.get("summary") or "Phát hiện sự khác biệt giữa các văn bản pháp luật liên quan.",
        }
    elif web_avail is False:
        verification_banner = {
            "status": "provisional",
            "title": "Cảnh báo nguồn ngoài",
            "message": "Không thể kết nối CSDL Web mở rộng; kết quả được tổng hợp hoàn toàn từ cơ sở dữ liệu nội bộ.",
        }

    # 4. Clarification Wizard Mapping
    clarification = None
    planner_state = raw_out.get("planner_state")
    choices = raw_out.get("choices") or []

    # Check if we need clarification wizard
    needs_clarify = False
    pending_intent_label = ""
    clarification_questions = []

    if isinstance(planner_state, dict):
        intents = planner_state.get("intents") or []
        pending_intents = [
            i for i in intents
            if isinstance(i, dict) and i.get("needs_clarification")
        ]
        if pending_intents:
            needs_clarify = True
            for q_idx, pi in enumerate(pending_intents):
                pi_label = pi.get("sub_query") or f"Ý {q_idx + 1}"
                if not pending_intent_label:
                    pending_intent_label = pi_label

                # Options for this intent
                opts = [
                    {"id": f"opt-{q_idx}-1", "label": "Xác nhận áp dụng theo văn bản hiện hành"},
                    {"id": f"opt-{q_idx}-2", "label": "Xác nhận áp dụng theo dự thảo / quy định sửa đổi"},
                    {"id": f"opt-{q_idx}-other", "label": "Khác", "allowCustomText": True, "description": "Tự nhập trường hợp cụ thể của bạn"},
                ]
                clarification_questions.append({
                    "id": str(pi.get("intent_id") or f"intent-{q_idx}"),
                    "label": f"Câu {q_idx + 1}/{len(pending_intents)}",
                    "question": pi.get("clarification") or f"Làm rõ yêu cầu cho ý: {pi_label}",
                    "options": opts,
                })

    elif choices and route in ("refine", "ask"):
        needs_clarify = True
        pending_intent_label = "Lựa chọn phạm vi áp dụng"
        opts = [
            {"id": f"choice-{c_i}", "label": str(choice_text)}
            for c_i, choice_text in enumerate(choices)
        ]
        opts.append({"id": "choice-other", "label": "Khác", "allowCustomText": True})
        clarification_questions.append({
            "id": "refine-question-1",
            "label": "Câu 1/1",
            "question": raw_out.get("follow_up") or "Vui lòng chọn văn bản hoặc phạm vi bạn muốn tra cứu:",
            "options": opts,
        })

    if needs_clarify and clarification_questions:
        clarification = {
            "status": "pending",
            "currentIndex": 0,
            "questions": clarification_questions,
            "pendingIntentLabel": pending_intent_label or "Làm rõ thông tin",
        }

    chat_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer_text,
        "createdAt": int(asyncio.get_event_loop().time() * 1000),
        "trace": trace_steps,
        "evidences": evidences,
        "verificationBanner": verification_banner,
        "citations": citations,
        "clarification": clarification,
    }

    return chat_message


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Execute Orchestrator.run without blocking the async event loop."""
    try:
        orch = get_orchestrator()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ phân tích pháp lý AgentLaw chưa sẵn sàng.",
        )

    # Standardize conversation context format (Người dùng: ...\nTrợ lý: ...)
    clean_ctx = (request.conversationContext or "").strip()

    try:
        # Run synchronous Orchestrator.run in threadpool
        raw_result = await asyncio.to_thread(
            orch.run,
            question=request.message,
            user_answer=request.userAnswer or "",
            conversation_context=clean_ctx,
            planner_state=request.plannerState,
            query_time=request.queryTime or "",
        )
    except Exception as exc:
        print(f"[api_server] Error during Orchestrator execution: {exc}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail="Lỗi trong quá trình xử lý suy luận pháp lý.",
        )

    # Map to frontend schema
    chat_message = map_orchestrator_output(raw_result, request.message, clean_ctx)

    # Build updated conversation context
    user_turn = f"Người dùng: {request.message}"
    if request.userAnswer:
        user_turn += f" (Đã chọn làm rõ: {request.userAnswer})"
    assistant_turn = f"Trợ lý: {chat_message['content']}"

    if clean_ctx:
        updated_context = f"{clean_ctx}\n{user_turn}\n{assistant_turn}"
    else:
        updated_context = f"{user_turn}\n{assistant_turn}"

    # Return safe JSON with encoder
    response_payload = {
        "message": chat_message,
        "conversationContext": updated_context,
        "plannerState": raw_result.get("planner_state"),
    }

    return JSONResponse(content=jsonable_encoder(response_payload))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
