"""
API routes for configuration status and metadata.

This module provides endpoints for initialization status checks
and supported provider/database type listings.
"""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from datus.api import deps
from datus.api.deps import AppContextDep, ServiceDep
from datus.api.models.base_models import Result
from datus.configuration.agent_config import _SAFE_NAME_RE, DbConfig, load_model_config
from datus.configuration.agent_config_loader import configuration_manager
from datus.models.base import LLMBaseModel
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["configuration"])


class UpdateDatabasesRequest(BaseModel):
    """Full desired state for `services.databases`.

    Any existing database key absent from `databases` will be deleted.
    """

    databases: Dict[str, Dict[str, Any]]


class UpdateModelsRequest(BaseModel):
    """Optional full-replace for `models` and/or update to `target`.

    At least one of `models` or `target` must be provided.
    """

    models: Optional[Dict[str, Dict[str, Any]]] = None
    target: Optional[str] = None


class ProbeModelRequest(BaseModel):
    """Single LLM model config dict — flat shape matching IModelInfo."""

    model_config = {"extra": "allow"}

    type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ProbeDatabaseRequest(BaseModel):
    """Single database config dict — flat shape matching IDatabaseConfig."""

    model_config = {"extra": "allow"}

    type: str


def _probe_llm_sync(payload: Dict[str, Any]) -> None:
    """Build a one-shot LLM client from a raw dict and send a tiny probe."""
    model_cfg = load_model_config(payload)
    model_class_name = LLMBaseModel.MODEL_TYPE_MAP.get(model_cfg.type)
    if model_class_name is None:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"Unsupported model type: {model_cfg.type}",
        )
    module = __import__(f"datus.models.{model_cfg.type}_model", fromlist=[model_class_name])
    model_class = getattr(module, model_class_name)
    client = model_class(model_config=model_cfg)
    client.generate("Hello")


def _probe_database_sync(payload: Dict[str, Any]) -> None:
    """Build a one-shot connector from a raw dict and run a SELECT 1 probe."""
    from datus.tools.db_tools.db_manager import DBManager

    kwargs = dict(payload)
    kwargs.setdefault("name", "_probe_")
    db_config = DbConfig.filter_kwargs(DbConfig, kwargs)

    manager = DBManager({"_probe_": {"_probe_": db_config}})
    try:
        conn = manager.get_conn("_probe_")
        conn.test_connection()
    finally:
        manager.close()


def _validate_keys(entries: Dict[str, Any], kind: str) -> None:
    """Ensure every key matches the naming policy used by AgentConfig."""
    for name in entries.keys():
        if not _SAFE_NAME_RE.match(name):
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=(
                    f"Invalid {kind} name '{name}'. Only alphanumeric characters, underscores, and hyphens are allowed."
                ),
            )


async def _evict_current_project(project_id: str) -> None:
    """Drop the cached DatusService so the next request reloads from YAML."""
    cache = deps._service_cache
    if cache is None:
        return
    try:
        await cache.evict(project_id)
    except Exception:
        logger.exception(f"Failed to evict service cache for project {project_id}")


@router.get(
    "/config/agent",
    response_model=Result[dict],
    summary="Get Agent Configuration",
    description="Get the current project's agent configuration (models, namespace, agentic_nodes)",
)
async def get_agent_config_endpoint(
    svc: ServiceDep,
) -> Result[dict]:
    """Return the project's loaded AgentConfig summary."""
    config = svc.agent_config
    flat_databases: dict = {}

    for db_name, inner in config.namespaces.items():
        if not inner:
            continue
        db_config = inner.get(db_name)
        if db_config is None:
            db_config = next(iter(inner.values()))
        flat_databases[db_name] = db_config

    return Result(
        success=True,
        data={
            "target": config.target,
            "models": config.models,
            "current_database": config.current_namespace,
            "databases": flat_databases,
            "home": config.home,
        },
    )


@router.put(
    "/config/databases",
    response_model=Result[dict],
    summary="Update Databases",
    description="Replace the databases (services.databases) block in agent.yml.",
)
async def update_databases_endpoint(
    body: UpdateDatabasesRequest,
    svc: ServiceDep,  # noqa: ARG001  # populates request.state.app_context; must resolve before AppContextDep
    ctx: AppContextDep,
) -> Result[dict]:
    """Full-replace `services.databases` with the provided databases."""
    _validate_keys(body.databases, kind="database")

    cm = configuration_manager()
    services = cm.data.setdefault("services", {})
    services["databases"] = dict(body.databases)
    cm.save()

    await _evict_current_project(ctx.project_id or "default")

    return Result(success=True, data={"updated": True})


@router.put(
    "/config/models",
    response_model=Result[dict],
    summary="Update Models and Target",
    description="Replace the models block and/or update the default target in agent.yml.",
)
async def update_models_endpoint(
    body: UpdateModelsRequest,
    svc: ServiceDep,  # noqa: ARG001
    ctx: AppContextDep,
) -> Result[dict]:
    """Optional full-replace `models`, optional update `target`. One must be set."""
    if body.models is None and body.target is None:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message="At least one of 'models' or 'target' must be provided.",
        )

    if body.models is not None:
        _validate_keys(body.models, kind="model")

    cm = configuration_manager()

    if body.target is not None:
        effective_models = body.models if body.models is not None else cm.data.get("models") or {}
        if body.target not in effective_models:
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=f"target '{body.target}' does not exist in models.",
            )

    if body.models is not None:
        cm.data["models"] = dict(body.models)
    if body.target is not None:
        cm.data["target"] = body.target
    cm.save()

    await _evict_current_project(ctx.project_id or "default")

    return Result(success=True, data={"updated": True})


@router.post(
    "/config/models/test",
    response_model=Result[dict],
    summary="Test Model Connectivity",
    description="Send a tiny probe to verify an LLM model config is reachable.",
)
async def probe_model_connectivity_endpoint(
    body: ProbeModelRequest,
    svc: ServiceDep,  # noqa: ARG001
) -> Result[dict]:
    """Return `{ok: True}` if the probe succeeds, else `{ok: False, message: ...}`."""
    payload = body.model_dump()
    try:
        await asyncio.to_thread(_probe_llm_sync, payload)
        return Result(success=True, data={"ok": True})
    except Exception as e:
        logger.info(f"Model connectivity probe failed: {e}")
        return Result(success=True, data={"ok": False, "message": str(e)})


@router.post(
    "/config/databases/test",
    response_model=Result[dict],
    summary="Test Database Connectivity",
    description="Run SELECT 1 against a database config to verify reachability and credentials.",
)
async def probe_database_connectivity_endpoint(
    body: ProbeDatabaseRequest,
    svc: ServiceDep,  # noqa: ARG001
) -> Result[dict]:
    """Return `{ok: True}` if the probe succeeds, else `{ok: False, message: ...}`."""
    payload = body.model_dump()
    try:
        await asyncio.to_thread(_probe_database_sync, payload)
        return Result(success=True, data={"ok": True})
    except Exception as e:
        logger.info(f"Database connectivity probe failed: {e}")
        return Result(success=True, data={"ok": False, "message": str(e)})
