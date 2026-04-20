"""API routes for success-story persistence.

Thin HTTP wrapper around :class:`SuccessStoryService`. The service resolves
``subagent_id`` to a safe directory name and appends a row to
``{benchmark_dir}/{subagent}/success_story.csv``.
"""

from fastapi import APIRouter, HTTPException

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.success_story_models import SuccessStoryData, SuccessStoryInput
from datus.api.services.success_story_service import SubagentNotFoundError
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/success-stories", tags=["success-stories"])


@router.post(
    "",
    summary="Save Success Story",
    description=(
        "Append a success story row to `{benchmark_dir}/{subagent}/success_story.csv`. "
        "The subagent directory is derived from `subagent_id` (builtin name, "
        "agentic_nodes key, or custom sub-agent DB UUID); omit to use 'default'."
    ),
    response_model=Result[SuccessStoryData],
)
async def save_success_story(payload: SuccessStoryInput, svc: ServiceDep) -> Result[SuccessStoryData]:
    try:
        data = svc.success_story.save(payload)
    except SubagentNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OSError as e:
        logger.exception("Failed to write success story")
        return Result[SuccessStoryData](
            success=False,
            errorCode="SUCCESS_STORY_WRITE_FAILED",
            errorMessage=str(e),
        )
    return Result[SuccessStoryData](success=True, data=data)
