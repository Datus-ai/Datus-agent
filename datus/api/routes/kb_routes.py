"""API routes for knowledge base bootstrap with SSE streaming."""

import json
import os
import uuid

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.kb_models import BootstrapKbInput
from datus.api.utils.path_utils import safe_resolve
from datus.api.utils.stream_cancellation import (
    cancel_stream,
    cleanup_cancel_token,
    create_cancel_token,
)

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
    except ValueError as e:
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
