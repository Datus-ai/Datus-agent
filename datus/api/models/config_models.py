"""Data models for unified agent configuration API endpoints."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(str, Enum):
    """Error codes for configuration API."""

    SYSTEM_NOT_READY = "SYSTEM_NOT_READY"
    PROVIDER_CONFIG_ERROR = "PROVIDER_CONFIG_ERROR"
    LLM_CONNECTIVITY_FAILED = "LLM_CONNECTIVITY_FAILED"
    DATABASE_CONNECTIVITY_FAILED = "DATABASE_CONNECTIVITY_FAILED"
    INVALID_SESSION = "INVALID_SESSION"
    CONFIGURATION_SAVE_FAILED = "CONFIGURATION_SAVE_FAILED"
    CONFIGURATION_UPDATE_FAILED = "CONFIGURATION_UPDATE_FAILED"
    CONFIGURATION_GET_FAILED = "CONFIGURATION_GET_FAILED"
    MCP_SERVER_ERROR = "MCP_SERVER_ERROR"
    SQL_EXECUTION_ERROR = "SQL_EXECUTION_ERROR"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    CONTEXT_COMMAND_ERROR = "CONTEXT_COMMAND_ERROR"
    CHAT_COMMAND_ERROR = "CHAT_COMMAND_ERROR"
    INTERNAL_COMMAND_ERROR = "INTERNAL_COMMAND_ERROR"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INVALID_TASK_PARAMETERS = "INVALID_TASK_PARAMETERS"
    TASK_NOT_READY = "TASK_NOT_READY"
    TASK_EXECUTION_FAILED = "TASK_EXECUTION_FAILED"
    AGENT_NOT_AVAILABLE = "AGENT_NOT_AVAILABLE"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class ModelConfig(BaseModel):
    """LLM model configuration."""

    type: str = Field(..., description="Model type")
    vendor: str = Field(..., description="Model vendor")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    model: str = Field(..., description="Model name")


class DatasourceConfig(BaseModel):
    """Database datasource configuration."""

    model_config = ConfigDict(exclude_none=True)

    type: str = Field(..., description="Database type")
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database: str = Field(..., description="Database name")
    catalog: Optional[str] = Field(None, description="Database catalog (for databases that support catalogs)")


class StorageConfig(BaseModel):
    """Storage configuration for RAG."""

    base_path: str = Field(..., description="Base path for storage")
    embedding_device_type: str = Field("cpu", description="Device type for embeddings")
    database: "StorageDatabaseConfig" = Field(..., description="Storage database configuration")


class StorageDatabaseConfig(BaseModel):
    """Storage database configuration."""

    registry_name: str = Field(..., description="Registry name")
    model_name: str = Field(..., description="Model name")
    dim_size: int = Field(..., description="Dimension size")
    batch_size: int = Field(..., description="Batch size")


class AgentConfigData(BaseModel):
    """Agent configuration data."""

    model_config = ConfigDict(exclude_none=True)

    target: Optional[str] = Field(None, description="Target model name")
    models: Optional[Dict[str, ModelConfig]] = Field(None, description="Model configurations")
    nodes: Optional[Dict[str, Dict]] = Field(None, description="Node configurations")
    benchmark: Optional[Dict[str, Dict]] = Field(None, description="Benchmark configurations")
    datasources: Optional[Dict[str, DatasourceConfig]] = Field(None, description="Datasource configurations")
    metrics: Optional[Dict[str, Dict]] = Field(None, description="Metrics configurations")
    storage: Optional[StorageConfig] = Field(None, description="Storage configuration")
    workflow: Optional[Dict] = Field(None, description="Workflow configuration")
    reflection_nodes: Optional[Dict[str, List]] = Field(None, description="Reflection nodes configuration")


class UpdateAgentConfigInput(BaseModel):
    """Request for updating agent configuration."""

    agent: AgentConfigData = Field(..., description="Agent configuration data")
    test_connectivity: bool = Field(True, description="Whether to test connectivity")


class ModelConnectivityTest(BaseModel):
    """Model connectivity test result."""

    status: str = Field(..., description="Test status (success/failed)")
    type: str = Field(..., description="Model type")
    model: str = Field(..., description="Model name")
    response_time: int = Field(..., description="Response time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if test failed")


class DatabaseInfo(BaseModel):
    """Database information from connectivity test."""

    version: str = Field(..., description="Database version")
    databases: List[str] = Field(..., description="Available databases")
    current_datasource: str = Field(..., description="Current database name")
    catalogs: Optional[List[str]] = Field(None, description="Available catalogs (if supported)")


class DatasourceConnectivityTest(BaseModel):
    """Datasource connectivity test result."""

    status: str = Field(..., description="Test status (success/failed)")
    type: str = Field(..., description="Database type")
    response_time: int = Field(..., description="Response time in milliseconds")
    database_info: Optional[DatabaseInfo] = Field(None, description="Database information")
    error_message: Optional[str] = Field(None, description="Error message if test failed")


class ConnectivityTests(BaseModel):
    """All connectivity test results."""

    models: Dict[str, ModelConnectivityTest] = Field(..., description="Model connectivity test results")
    datasources: Dict[str, DatasourceConnectivityTest] = Field(..., description="Datasource connectivity test results")


class ValidationSummary(BaseModel):
    """Validation summary."""

    total_tests: int = Field(..., description="Total number of tests")
    passed: int = Field(..., description="Number of passed tests")
    failed: int = Field(..., description="Number of failed tests")


class UpdateAgentConfigData(BaseModel):
    """Agent configuration update result data."""

    configuration_updated: bool = Field(..., description="Whether configuration was updated")
    connectivity_tests: ConnectivityTests = Field(..., description="Connectivity test results")
    validation_summary: ValidationSummary = Field(..., description="Validation summary")
    services_need_reload: bool = Field(default=False, description="Whether services need to be reloaded")


class FailedTest(BaseModel):
    """Failed test information."""

    component: str = Field(..., description="Component that failed (model/datasource)")
    name: str = Field(..., description="Name of the failed component")
    error: str = Field(..., description="Error message")
    suggestion: str = Field(..., description="Suggestion to fix the error")


class ConfigurationErrorDetails(BaseModel):
    """Configuration error details."""

    failed_tests: List[FailedTest] = Field(..., description="List of failed tests")


# LLM Provider Models
class LLMProviderInfo(BaseModel):
    """LLM provider information."""

    type: str = Field(..., description="Provider type")
    base_url: str = Field(..., description="Base URL")
    model: str = Field(..., description="Default model")
    description: str = Field(..., description="Provider description")


class LLMProvidersData(BaseModel):
    """LLM providers data."""

    providers: Dict[str, LLMProviderInfo] = Field(..., description="Available providers")
    default: str = Field(..., description="Default provider")


# Database Types Models
class DatabaseTypeInfo(BaseModel):
    """Database type information."""

    type: str = Field(..., description="Database type")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Database description")
    connection_method: str = Field(..., description="Connection method")
    required_fields: List[str] = Field(..., description="Required configuration fields")
    default_catalog: Optional[str] = Field(None, description="Default catalog name")


class DatabaseTypesData(BaseModel):
    """Database types data."""

    database_types: List[DatabaseTypeInfo] = Field(..., description="Available database types")
    default: str = Field(..., description="Default database type")


# Model Catalog Models
class ModelPricing(BaseModel):
    """Per-token pricing for a single model, in OpenRouter's native format.

    Values are preserved as strings to avoid floating-point rounding at the API
    boundary. Units are USD per token unless the upstream source overrides.
    """

    model_config = ConfigDict(exclude_none=True)

    prompt: Optional[str] = Field(None, description="Price per input token")
    completion: Optional[str] = Field(None, description="Price per output token")


class ModelInfo(BaseModel):
    """A single model entry returned by the catalog endpoint."""

    model_config = ConfigDict(exclude_none=True)

    provider: str = Field(..., description="Provider key from providers.yml, or 'custom' for agent.models entries")
    id: str = Field(..., description="Model slug as consumed by the SDK")
    model: Optional[str] = Field(
        None, description="Actual model name (same as id for provider models, ModelConfig.model for custom)"
    )
    name: Optional[str] = Field(None, description="Human-readable model name")
    context_length: Optional[int] = Field(None, description="Maximum context window in tokens")
    max_tokens: Optional[int] = Field(None, description="Maximum completion tokens")
    pricing: Optional[ModelPricing] = Field(None, description="Per-token pricing, when available")


class ModelsData(BaseModel):
    """Response payload for GET /api/v1/models."""

    model_config = ConfigDict(exclude_none=True)

    models: List[ModelInfo] = Field(..., description="Flat list of available models")
    providers: List[str] = Field(..., description="Provider keys represented in this response")
    fetched_at: Optional[str] = Field(None, description="ISO-8601 timestamp of the OpenRouter cache")
    source: str = Field(..., description="Where the data came from: cache or catalog")
