"""
API routes for configuration status and metadata.

This module provides endpoints for initialization status checks
and supported provider/database type listings.
"""

from fastapi import APIRouter

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.config_models import (
    DatabaseTypeInfo,
    DatabaseTypesData,
    LLMProviderInfo,
    LLMProvidersData,
)

router = APIRouter(prefix="/api/v1", tags=["configuration"])


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
    return Result(
        success=True,
        data={
            "target": config.target,
            "models": list(config.models.keys()),
            "current_namespace": config.current_namespace,
            "namespaces": list(config.namespaces.keys()),
            "agentic_nodes": list(config.agentic_nodes.keys()) if config.agentic_nodes else [],
            "home": config.home,
        },
    )


@router.get(
    "/config/llm/providers",
    response_model=Result[LLMProvidersData],
    summary="Get LLM Providers",
    description="Get supported LLM provider templates",
)
async def get_llm_providers() -> Result[LLMProvidersData]:
    """Get supported LLM providers."""
    providers = {
        "openai": LLMProviderInfo(
            type="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            description="OpenAI GPT models",
        ),
        "deepseek": LLMProviderInfo(
            type="openai",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            description="DeepSeek models",
        ),
        "claude": LLMProviderInfo(
            type="claude",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-5-20250929",
            description="Anthropic Claude models",
        ),
        "qwen": LLMProviderInfo(
            type="openai",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-max",
            description="Alibaba Qwen models",
        ),
        "kimi": LLMProviderInfo(
            type="openai",
            base_url="https://api.moonshot.cn/v1",
            model="moonshot-v1-auto",
            description="Moonshot Kimi models",
        ),
    }
    return Result(
        success=True,
        data=LLMProvidersData(providers=providers, default="openai"),
    )


@router.get(
    "/config/database/types",
    response_model=Result[DatabaseTypesData],
    summary="Get Database Types",
    description="Get supported database type templates",
)
async def get_database_types() -> Result[DatabaseTypesData]:
    """Get supported database types."""
    database_types = [
        DatabaseTypeInfo(
            type="postgresql",
            name="PostgreSQL",
            description="Open-source relational database",
            connection_method="asyncpg",
            required_fields=["host", "port", "database", "user", "password"],
        ),
        DatabaseTypeInfo(
            type="mysql",
            name="MySQL",
            description="Popular open-source relational database",
            connection_method="aiomysql",
            required_fields=["host", "port", "database", "user", "password"],
        ),
        DatabaseTypeInfo(
            type="snowflake",
            name="Snowflake",
            description="Cloud data warehouse",
            connection_method="snowflake-connector",
            required_fields=["account", "user", "password", "database", "warehouse"],
            default_catalog="SNOWFLAKE",
        ),
        DatabaseTypeInfo(
            type="starrocks",
            name="StarRocks",
            description="High-performance analytical database",
            connection_method="pymysql",
            required_fields=["host", "port", "database", "user", "password"],
        ),
        DatabaseTypeInfo(
            type="duckdb",
            name="DuckDB",
            description="In-process analytical database",
            connection_method="duckdb",
            required_fields=["database"],
        ),
    ]
    return Result(
        success=True,
        data=DatabaseTypesData(database_types=database_types, default="postgresql"),
    )
