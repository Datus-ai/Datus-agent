"""API routes for knowledge base bootstrap with SSE streaming."""

import json
import os
import uuid

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.kb_models import BootstrapDocInput, BootstrapKbInput
from datus.api.utils.path_utils import safe_resolve
from datus.api.utils.stream_cancellation import (
    cancel_stream,
    cleanup_cancel_token,
    create_cancel_token,
)
from datus.utils.exceptions import DatusException

router = APIRouter(prefix="/api/v1/kb", tags=["knowledge-base"])


@router.post(
    "/bootstrap",
    summary="Bootstrap Knowledge Base",
    description="Start KB bootstrap with SSE progress streaming",
)
async def bootstrap_kb(
    request: BootstrapKbInput,
    svc: ServiceDep,
):
    """Start KB bootstrap with SSE progress streaming."""
    stream_id = str(uuid.uuid4())
    cancel_event = create_cancel_token(stream_id)

    # Derive project_files_root from AgentConfig.home (= project dir)
    project_files_root = os.path.join(svc.agent_config.home, "files")

    # Validate user-supplied paths against the project root
    try:
        _validate_paths(request, project_files_root)
    except DatusException as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    async def generate_sse():
        try:
            async for event in svc.kb.bootstrap_stream(request, stream_id, cancel_event, project_files_root):
                data = json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False)
                yield f"id: {stream_id}\nevent: {event.stage}\ndata: {data}\n\n"
        finally:
            cleanup_cancel_token(stream_id)

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@router.post(
    "/bootstrap/{stream_id}/cancel",
    response_model=Result[dict],
    summary="Cancel Bootstrap",
    description="Cancel a running bootstrap stream",
)
async def cancel_bootstrap(
    svc: ServiceDep,  # noqa: ARG001 — triggers auth
    stream_id: str = Path(..., description="Stream ID to cancel"),
):
    """Cancel a running bootstrap stream."""
    cancelled = cancel_stream(stream_id)
    return Result(success=cancelled, data={"stream_id": stream_id, "cancelled": cancelled})


def _validate_paths(request: BootstrapKbInput, project_root: str) -> None:
    """Validate that user-supplied file paths don't escape the project root."""
    from pathlib import Path as P

    base = P(project_root)
    if request.success_story:
        safe_resolve(base, request.success_story)
    if request.sql_dir:
        safe_resolve(base, request.sql_dir)
    if request.ext_knowledge:
        safe_resolve(base, request.ext_knowledge)


# ======================================================================
# Platform document bootstrap
# ======================================================================


@router.post(
    "/bootstrap-docs",
    summary="Bootstrap Platform Documentation",
    description="Start platform documentation bootstrap with SSE progress streaming",
)
async def bootstrap_docs(
    request: BootstrapDocInput,
    svc: ServiceDep,
):
    """Start platform doc bootstrap with SSE progress streaming."""
    stream_id = str(uuid.uuid4())
    cancel_event = create_cancel_token(stream_id)

    # Validate: platform must exist in config OR request must supply source
    platform = request.platform
    doc_cfg = svc.agent_config.document_configs.get(platform)
    if not doc_cfg and not request.source:
        raise HTTPException(
            status_code=422,
            detail=f"Platform '{platform}' not found in agent config and no source provided",
        )

    # Path validation for local sources
    source_type = request.source_type or (doc_cfg.type if doc_cfg else None)
    if request.source and source_type == "local":
        from pathlib import Path as P

        project_files_root = os.path.join(svc.agent_config.home, "files")
        try:
            safe_resolve(P(project_files_root), request.source)
        except DatusException as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    async def generate_sse():
        try:
            async for event in svc.kb.bootstrap_doc_stream(request, stream_id, cancel_event):
                data = json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False)
                yield f"id: {stream_id}\nevent: {event.stage}\ndata: {data}\n\n"
        finally:
            cleanup_cancel_token(stream_id)

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@router.post(
    "/bootstrap-docs/{stream_id}/cancel",
    response_model=Result[dict],
    summary="Cancel Doc Bootstrap",
    description="Cancel a running platform doc bootstrap stream",
)
async def cancel_doc_bootstrap(
    svc: ServiceDep,  # noqa: ARG001 — triggers auth
    stream_id: str = Path(..., description="Stream ID to cancel"),
):
    """Cancel a running platform doc bootstrap stream."""
    cancelled = cancel_stream(stream_id)
    return Result(success=cancelled, data={"stream_id": stream_id, "cancelled": cancelled})
