"""
API routes for Explorer endpoints.
"""

from fastapi import APIRouter

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.explorer_models import (
    CreateDirectoryInput,
    CreateKnowledgeInput,
    DeleteSubjectInput,
    EditKnowledgeInput,
    EditMetricInput,
    EditSemanticModelInput,
    KnowledgeInfo,
    MetricInfo,
    ReferenceSQLInfo,
    ReferenceSQLInput,
    RenameSubjectInput,
    SubjectListData,
    SubjectPathInput,
)

router = APIRouter(prefix="/api/v1", tags=["explorer"])


# ========== Subject Endpoints ==========


@router.get(
    "/subject/list",
    response_model=Result[SubjectListData],
    summary="Get Subject List",
    description="Get nested subject tree structure with directories, metrics, and reference SQL items",
)
async def get_subject_list(
    svc: ServiceDep,
) -> Result[SubjectListData]:
    """Get subject tree."""
    return await svc.explorer.get_subject_list()


@router.post(
    "/subject/create",
    response_model=Result[dict],
    summary="Create Directory",
    description="Create a new directory in the subject tree at the specified path",
)
async def create_directory(
    request: CreateDirectoryInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Create directory."""
    return await svc.explorer.create_directory(request)


@router.post(
    "/subject/rename",
    response_model=Result[dict],
    summary="Rename or Move Subject",
    description="Rename a subject node or move it to a different location in the tree",
)
async def rename_subject(
    request: RenameSubjectInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Rename/move subject."""
    return await svc.explorer.rename_subject(request)


@router.delete(
    "/subject/delete",
    response_model=Result[dict],
    summary="Delete Subject",
    description="Delete a subject node (directory, metric, reference SQL, or knowledge) from the tree",
)
async def delete_subject(
    request: DeleteSubjectInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Delete subject."""
    return await svc.explorer.delete_subject(request)


@router.post(
    "/subject/metric",
    response_model=Result[MetricInfo],
    summary="Get Metric",
    description="Get metric information including YAML configuration by subject path",
)
async def get_metric(
    request: SubjectPathInput,
    svc: ServiceDep,
) -> Result[MetricInfo]:
    """Get metric info."""
    return await svc.explorer.get_metric(request.subject_path)


@router.post(
    "/subject/reference_sql",
    response_model=Result[ReferenceSQLInfo],
    summary="Get Reference SQL",
    description="Get reference SQL details including summary, comment, and SQL query",
)
async def get_reference_sql(
    request: SubjectPathInput,
    svc: ServiceDep,
) -> Result[ReferenceSQLInfo]:
    """Get reference SQL."""
    return await svc.explorer.get_reference_sql(request.subject_path)


@router.post(
    "/subject/reference_sql/create",
    response_model=Result[dict],
    summary="Create Reference SQL",
    description="Create a new reference SQL entry in the subject tree",
)
async def create_reference_sql(
    request: ReferenceSQLInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Create reference SQL."""
    return await svc.explorer.create_reference_sql(request)


@router.post(
    "/subject/reference_sql/edit",
    response_model=Result[dict],
    summary="Edit Reference SQL",
    description="Update reference SQL summary, comment, and SQL query",
)
async def edit_reference_sql(
    request: ReferenceSQLInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Edit reference SQL."""
    return await svc.explorer.edit_reference_sql(request)


@router.post(
    "/subject/metric/create",
    response_model=Result[dict],
    summary="Create Metric",
    description="Create a new metric from YAML definition",
)
async def create_metric(
    request: EditMetricInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Create metric from YAML."""
    return await svc.explorer.create_metric(request)


@router.post(
    "/subject/metric/edit",
    response_model=Result[dict],
    summary="Edit Metric",
    description="Update an existing metric's YAML definition",
)
async def edit_metric(
    request: EditMetricInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Edit metric YAML."""
    return await svc.explorer.edit_metric(request)


@router.post(
    "/subject/semantic_model/edit",
    response_model=Result[dict],
    summary="Edit Semantic Model",
    description="Update a semantic model entry (table or column) by entry ID",
)
async def edit_semantic_model(
    request: EditSemanticModelInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Edit semantic model entry."""
    return await svc.explorer.edit_semantic_model(request)


# ========== Knowledge Endpoints ==========


@router.post(
    "/subject/knowledge/create",
    response_model=Result[dict],
    summary="Create Knowledge",
    description="Create a new knowledge entry in the subject tree",
)
async def create_knowledge(
    request: CreateKnowledgeInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Create knowledge."""
    return await svc.explorer.create_knowledge(request)


@router.post(
    "/subject/knowledge",
    response_model=Result[KnowledgeInfo],
    summary="Get Knowledge",
    description="Get knowledge details by subject path",
)
async def get_knowledge(
    request: SubjectPathInput,
    svc: ServiceDep,
) -> Result[KnowledgeInfo]:
    """Get knowledge."""
    return await svc.explorer.get_knowledge(request.subject_path)


@router.post(
    "/subject/knowledge/edit",
    response_model=Result[dict],
    summary="Edit Knowledge",
    description="Update knowledge search_text and explanation",
)
async def edit_knowledge(
    request: EditKnowledgeInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Edit knowledge."""
    return await svc.explorer.edit_knowledge(request)
