"""Pydantic models for CLI Command Type API endpoints."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# SQL Execution models
class ExecuteSQLInput(BaseModel):
    """Input model for SQL execution."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "database_name": "sales_db",
                "sql_query": "SELECT * FROM users WHERE status = 'active'",
                "result_format": "csv",
                "system": False,
            }
        }
    )

    database_name: Optional[str] = Field(None, description="Database name")
    sql_query: str = Field(..., description="SQL query to execute")
    result_format: str = Field("arrow", description="Result format (arrow, csv, json)")
    system: bool = Field(False, description="Whether this is a system command")


class ExecuteSQLData(BaseModel):
    """Data for SQL execution result."""

    execute_task_id: str = Field(..., description="Task ID for this SQL execution, can be used to stop it")
    sql_query: str = Field(..., description="Executed SQL query")
    row_count: Optional[int] = Field(None, description="Number of rows returned")
    sql_return: Optional[str] = Field(None, description="SQL result data")
    result_format: str = Field(..., description="Result format")
    execution_time: float = Field(..., description="Execution time in seconds")
    executed_at: str = Field(..., description="Execution timestamp")
    columns: Optional[List[str]] = Field(None, description="Column names")


class StopExecuteSQLInput(BaseModel):
    """Input model for stopping a SQL execution."""

    execute_task_id: str = Field(..., description="Task ID of the SQL execution to stop")


class StopExecuteSQLData(BaseModel):
    """Data for stop SQL execution result."""

    execute_task_id: str = Field(..., description="Task ID of the stopped SQL execution")
    stopped: bool = Field(..., description="Whether the execution was successfully stopped")


# Tool Commands models
class SchemaLinkingToolInput(BaseModel):
    """Input model for schema linking tool (sl)."""

    query_text: str = Field(..., description="Text to search for in table metadata")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    top_n: int = Field(5, description="Number of top results to return")


class SearchMetricsToolInput(BaseModel):
    """Input model for search metrics tool (sm)."""

    query_text: str = Field(..., description="Text to search for in metrics")
    domain: Optional[str] = Field(None, description="Metrics domain")
    layer1: Optional[str] = Field(None, description="First layer classification")
    layer2: Optional[str] = Field(None, description="Second layer classification")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    top_n: int = Field(5, description="Number of top results to return")


class SearchHistoryToolInput(BaseModel):
    """Input model for search history tool (sh)."""

    query_text: str = Field(..., description="Text to search for in SQL history")
    domain: Optional[str] = Field(None, description="Query domain")
    layer1: Optional[str] = Field(None, description="First layer classification")
    layer2: Optional[str] = Field(None, description="Second layer classification")
    top_n: int = Field(5, description="Number of top results to return")


class SaveToolInput(BaseModel):
    """Input model for save tool."""

    file_type: str = Field("all", description="File type to save (json/csv/sql/all)")
    target_dir: Optional[str] = Field(None, description="Target directory")
    file_name: Optional[str] = Field(None, description="File name")


class ExecuteToolInput(BaseModel):
    """Generic input model for tool execution - used for backward compatibility."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_name": "sl",
                "args": "users table analysis",
                "stream_output": False,
            }
        }
    )

    tool_name: str = Field(..., description="Name of the tool to execute")
    args: str = Field("", description="Arguments for the tool")
    stream_output: bool = Field(False, description="Whether to stream output")


# Tool-specific result models
class TableMetadata(BaseModel):
    """Table metadata information."""

    table_name: str = Field(..., description="Table name")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    definition: Optional[str] = Field(None, description="Table definition (DDL)")
    score: Optional[float] = Field(None, description="Relevance score")


class SampleData(BaseModel):
    """Sample data information."""

    table_name: str = Field(..., description="Table name")
    sample_rows: str = Field(..., description="Sample rows as formatted text")


class SchemaLinkingResult(BaseModel):
    """Schema linking tool result."""

    metadata: List[TableMetadata] = Field(default_factory=list, description="Table metadata")
    sample_data: List[SampleData] = Field(default_factory=list, description="Sample data")
    total_metadata: int = Field(0, description="Total metadata count")
    total_sample_data: int = Field(0, description="Total sample data count")


class Metric(BaseModel):
    """Metric information."""

    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    constraint: Optional[str] = Field(None, description="Metric constraint/formula")
    domain: Optional[str] = Field(None, description="Metric domain")
    layer1: Optional[str] = Field(None, description="First layer")
    layer2: Optional[str] = Field(None, description="Second layer")
    score: Optional[float] = Field(None, description="Relevance score")


class SearchMetricsResult(BaseModel):
    """Search metrics tool result."""

    metrics: List[Metric] = Field(default_factory=list, description="Found metrics")
    total_count: int = Field(0, description="Total metrics count")


class HistoricalQuery(BaseModel):
    """Historical query information."""

    sql_query: str = Field(..., description="SQL query")
    description: Optional[str] = Field(None, description="Query description")
    domain: Optional[str] = Field(None, description="Query domain")
    layer1: Optional[str] = Field(None, description="First layer")
    layer2: Optional[str] = Field(None, description="Second layer")
    timestamp: Optional[str] = Field(None, description="Query timestamp")
    score: Optional[float] = Field(None, description="Relevance score")


class SearchHistoryResult(BaseModel):
    """Search history tool result."""

    history: List[HistoricalQuery] = Field(default_factory=list, description="Historical queries")
    total_count: int = Field(0, description="Total queries count")


class SavedFile(BaseModel):
    """Saved file information."""

    file_type: str = Field(..., description="File type")
    file_path: str = Field(..., description="File path")
    file_size: str = Field(..., description="File size")


class SaveToolResult(BaseModel):
    """Save tool result."""

    files_saved: List[SavedFile] = Field(default_factory=list, description="Saved files")
    total_files: int = Field(0, description="Total files saved")
    total_size: str = Field("0B", description="Total size")
    zip_path: Optional[str] = Field(None, description="Path to zip archive")
    download_url: Optional[str] = Field(None, description="Download URL for zip archive")


class ExecuteToolData(BaseModel):
    """Data for tool execution result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str = Field(..., description="Tool name")
    query_text: Optional[str] = Field(None, description="Query text used")
    result: Any = Field(..., description="Tool execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    executed_at: str = Field(..., description="Execution timestamp")


# Context Commands models
class ExecuteContextInput(BaseModel):
    """Input model for context commands."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context_type": "tables",
                "database_name": "sales_db",
                "schema_name": "public",
                "args": "",
            }
        }
    )

    context_type: str = Field(..., description="Type of context command")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    args: str = Field("", description="Context command arguments")


class TableInfo(BaseModel):
    """Information about a database table."""

    table_name: str = Field(..., description="Table name")
    table_type: str = Field(..., description="Table type")
    row_count: Optional[int] = Field(None, description="Number of rows")
    columns_count: Optional[int] = Field(None, description="Number of columns")


class ContextResultData(BaseModel):
    """Generic context result data."""

    tables: Optional[List[TableInfo]] = Field(None, description="Tables information")
    total_count: Optional[int] = Field(None, description="Total count")
    context_info: Optional[Dict[str, Any]] = Field(None, description="Context information")
    output: Optional[Any] = Field(None, description="Context command output")


class ExecuteContextData(BaseModel):
    """Data for context execution result."""

    context_type: str = Field(..., description="Context type")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    result: ContextResultData = Field(..., description="Context result")


# Chat Commands models - Enhanced for ChatAgenticNode integration
class TableSchema(BaseModel):
    """Table schema reference for chat context."""

    table_name: str = Field(..., description="Table name")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    definition: Optional[str] = Field(None, description="Table definition/DDL")
    score: Optional[float] = Field(None, description="Relevance score")


class HistoricalSql(BaseModel):
    """Historical SQL reference for chat context."""

    sql_query: str = Field(..., description="SQL query")
    description: Optional[str] = Field(None, description="Query description")
    domain: Optional[str] = Field(None, description="Query domain")
    layer1: Optional[str] = Field(None, description="Layer 1 classification")
    layer2: Optional[str] = Field(None, description="Layer 2 classification")
    timestamp: Optional[str] = Field(None, description="When query was executed")
    score: Optional[float] = Field(None, description="Relevance score")


class ChatInput(BaseModel):
    """Enhanced input model for chat commands with ChatAgenticNode support."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Show me total sales for last month",
                "session_id": "session_123",
                "catalog": "main",
                "database": "sales_db",
                "db_schema": "public",
                "stream_response": True,
                "plan_mode": False,
                "subagent_id": None,
                "max_turns": 30,
            }
        }
    )

    # Core message fields
    message: str = Field(..., description="Chat message")
    session_id: Optional[str] = Field(None, description="Session ID")
    plan_mode: bool = Field(False, description="Whether in plan mode")
    source: Optional[str] = Field(None, description="chat source, web/vscode")
    interactive: Optional[bool] = Field(
        default=None,
        description="Override server default for ask_user interactive tool. None = use server default.",
    )

    # Database context fields
    catalog: Optional[str] = Field(None, description="Database catalog for context")
    database: Optional[str] = Field(None, description="Database name for context")
    db_schema: Optional[str] = Field(None, description="Database schema for context")

    # Chat configuration
    max_turns: int = Field(default=30, description="Maximum conversation turns per interaction")
    workspace_root: Optional[str] = Field(None, description="Root directory path for filesystem MCP server")

    # Context references (parsed from @ symbols)
    table_paths: Optional[List[str]] = Field(default=None, description="Table path identifiers for @Table references")
    metric_paths: Optional[List[str]] = Field(
        default=None, description="Metric path identifiers for @Metrics references"
    )
    sql_paths: Optional[List[str]] = Field(default=None, description="SQL path identifiers for @Sql references")
    knowledge_paths: Optional[List[str]] = Field(
        default=None, description="Knowledge path identifiers for @Knowledge references"
    )

    # Legacy fields (direct objects, deprecated in favor of paths)
    schemas: Optional[List[TableSchema]] = Field(default=None, description="Table schemas to reference (legacy)")
    metrics: Optional[List[Metric]] = Field(default=None, description="Metrics to reference (legacy)")
    historical_sql: Optional[List[HistoricalSql]] = Field(
        default=None, description="Historical SQL to reference (legacy)"
    )

    # Response control
    stream_response: bool = Field(False, description="Whether to stream response")

    # Legacy fields for backward compatibility
    context_id: Optional[str] = Field(None, description="Context ID (legacy)")


class ActionInfo(BaseModel):
    """Action execution information."""

    action_id: str = Field(..., description="Action ID")
    role: str = Field(..., description="Action role (user/assistant/tool)")
    function_name: Optional[str] = Field(None, description="Function name if tool action")
    status: str = Field(..., description="Action status")
    output: Optional[Dict[str, Any]] = Field(None, description="Action output")
    timestamp: str = Field(..., description="Action timestamp")


class SessionInfo(BaseModel):
    """Chat session information."""

    session_id: Optional[str] = Field(None, description="Session ID")
    token_count: int = Field(0, description="Total tokens used")
    action_count: int = Field(0, description="Total actions performed")
    conversation_count: int = Field(0, description="Total conversations")


class ExtractedData(BaseModel):
    """Extracted structured data from chat response."""

    sql: Optional[str] = Field(None, description="Extracted SQL query")
    clean_output: Optional[str] = Field(None, description="Clean text output")
    raw_response: Optional[str] = Field(None, description="Raw AI response")
    context_references: Dict[str, List[str]] = Field(default_factory=dict, description="Referenced context items")


class ChatData(BaseModel):
    """Enhanced data for chat result."""

    # Basic response fields
    message_id: str = Field(..., description="Message ID")
    user_message: str = Field(..., description="User message")
    response: str = Field(..., description="AI response")
    session_id: Optional[str] = Field(None, description="Session ID")
    timestamp: str = Field(..., description="Response timestamp")

    # Enhanced fields from ChatAgenticNode
    sql: Optional[str] = Field(None, description="Generated or referenced SQL query")
    tokens_used: int = Field(0, description="Tokens used in this interaction")
    node_type: str = Field(default="chat", description="Node type used (chat/gensql)")

    # Execution details
    actions: List[ActionInfo] = Field(default_factory=list, description="Execution actions performed")
    session_info: Optional[SessionInfo] = Field(None, description="Session statistics")
    extracted_data: Optional[ExtractedData] = Field(None, description="Extracted structured data")

    # Context information
    database_context: Dict[str, Optional[str]] = Field(default_factory=dict, description="Database context used")
    context_updated: bool = Field(False, description="Whether SQL context was updated")

    # Execution metadata
    execution_time: Optional[float] = Field(None, description="Total execution time in seconds")


# Chat Session Management models
class ChatSessionItemInfo(BaseModel):
    """Chat session item information."""

    user_query: Optional[str] = Field(None, description="First user message in the session")
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Session creation time")
    last_updated: str = Field(..., description="Last update time")
    total_turns: int = Field(0, description="Total conversation turns")
    token_count: int = Field(0, description="Total tokens used")
    last_sql_queries: List[str] = Field(default_factory=list, description="Recent SQL queries")
    is_active: bool = Field(False, description="Whether session is currently active")


class ChatSessionData(BaseModel):
    """Chat session summary data."""

    sessions: List[ChatSessionItemInfo] = Field(default_factory=list, description="Chat session items")
    total_count: int = Field(0, description="Total session count")


class CompactSessionInput(BaseModel):
    """Input for session compaction."""

    session_id: str = Field(..., description="Session ID to compact")


class CompactSessionData(BaseModel):
    """Session compaction result data."""

    session_id: str = Field(..., description="Session ID")
    success: bool = Field(..., description="Whether compaction succeeded")
    new_token_count: Optional[int] = Field(None, description="Token count after compaction")
    tokens_saved: Optional[int] = Field(None, description="Tokens saved by compaction")
    compression_ratio: Optional[str] = Field(None, description="Compression ratio achieved")
    error: Optional[str] = Field(None, description="Error message if failed")


# Streaming Chat models
class StreamChatInput(ChatInput):
    """Input for streaming chat via /chat/stream."""

    stream_response: bool = Field(True, description="Whether to stream response (always true for this endpoint)")
    subagent_id: Optional[str] = Field(default=None, description="Subagent ID (builtin name or DB SubAgent id)")
    prompt_version: Optional[str] = Field(default=None, description="Prompt version")
    prompt_language: str = Field(default="en", description="Prompt language")


class UserInteractionInput(BaseModel):
    """Input for user interaction submission."""

    session_id: str = Field(..., description="Session ID for the active chat task")
    interaction_key: str = Field(..., description="Interaction key (action_id) for the interaction request")
    input: List[str] = Field(..., description="List of user answers, one per request")


class StreamChatChunk(BaseModel):
    """Individual chunk in streaming chat response."""

    type: str = Field(..., description="Chunk type (action/response/error/complete)")
    data: Dict[str, Any] = Field(..., description="Chunk data")
    timestamp: str = Field(..., description="Chunk timestamp")


# Internal Commands models
class InternalCommandInput(BaseModel):
    """Input model for internal commands."""

    model_config = ConfigDict(json_schema_extra={"example": {"command": "show", "args": "tables"}})

    command: str = Field(..., description="Internal command")
    args: str = Field("", description="Command arguments")


class InternalCommandResultData(BaseModel):
    """Internal command result data."""

    command_output: str = Field(..., description="Command output")
    action_taken: str = Field(..., description="Action taken")
    context_changed: bool = Field(False, description="Whether context changed")
    data: Optional[Any] = Field(None, description="Additional data")


class InternalCommandData(BaseModel):
    """Data for internal command result."""

    command: str = Field(..., description="Command executed")
    args: str = Field(..., description="Command arguments")
    result: InternalCommandResultData = Field(..., description="Command result")


# Tool Types Enum
class ToolType(str, Enum):
    """Supported tool types."""

    DARUN = "darun"
    DASTART = "dastart"
    CLEAN = "clean"
    REASON = "reason"
    REASON_STREAM = "reason_stream"
    GEN_METRICS = "gen_metrics"
    GEN_METRICS_STREAM = "gen_metrics_stream"
    GEN_SCHEMAS_DESCRIPTIONS = "gen_schemas_descriptions"
    GEN_QUERY = "gen_query"
    SHOW_RESULT_RAW = "show_result_raw"
    SHOW_RESULT_TABLE = "show_result_table"
    COMPARE = "compare"
    SUMMARY = "summary"
    EXPORT = "export"
    CALL_TOOL = "call_tool"
    RUN = "run"
    SL = "sl"
    GEN = "gen"
    FIX = "fix"
    SAVE = "save"
    BASH = "bash"


# Context Types Enum
class ContextType(str, Enum):
    """Supported context types."""

    CATALOGS = "catalogs"
    CONTEXT = "context"
    CONTEXT_SCREEN = "context_screen"
    TABLES = "tables"
    CATALOG = "catalog"
    SUBJECT = "subject"
    SQL = "sql"


# Internal Command Types Enum
class InternalCommandType(str, Enum):
    """Supported internal command types."""

    HELP = "help"
    EXIT = "exit"
    QUIT = "quit"
    CLEAR = "clear"
    SHOW = "show"
    DATABASES = "databases"
    DATABASE = "database"
    SESSIONS = "sessions"
    SESSION = "session"
    NEW = "new"
    EXPORT = "export"
    SAVE = "save"
    LOAD = "load"
    HISTORY = "history"
    CONFIG = "config"
    LOGS = "logs"
    MCP = "mcp"
    DEBUG = "debug"
    CHAT_INFO = "chat_info"
    COMPACT = "compact"
    TABLES = "tables"
    SCHEMAS = "schemas"
    SCHEMA = "schema"
    TABLE_SCHEMA = "table_schema"
    INDEXES = "indexes"
    NAMESPACE = "namespace"


# SSE (Server-Sent Events) Models for Streaming Chat


class SSEDataType(str, Enum):
    """SSE data types for message operations."""

    CREATE_MESSAGE = "createMessage"
    APPEND_MESSAGE = "appendMessage"
    UPDATE_MESSAGE = "updateMessage"


class MessageRole(str, Enum):
    """Message roles."""

    USER = "user"
    ASSISTANT = "assistant"


class ContentType(str, Enum):
    """Content types for messages."""

    MARKDOWN = "markdown"
    CODE = "code"
    CSV = "csv"


# Message Content Types
class IMessageContentPayload(BaseModel):
    """Base interface for message content payload."""

    pass


class ICsvPayload(IMessageContentPayload):
    """CSV content payload."""

    content: str = Field(..., description="CSV content")


class IMarkdownPayload(IMessageContentPayload):
    """Markdown content payload."""

    content: str = Field(..., description="Markdown content")


class ICodePayload(IMessageContentPayload):
    """Code content payload."""

    code_type: str = Field(..., description="Code type (json, xml, sql, etc.)")
    content: str = Field(..., description="Code content")


class IMessageContent(BaseModel):
    """Message content with type and payload."""

    type: str = Field(..., description="Content type (markdown, code, csv, etc.)")
    payload: Dict[str, Any] = Field(..., description="Content payload")


# Message Operation Payloads
class ICreateMessagePayload(BaseModel):
    """Payload for creating a new message."""

    message_id: int = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user, assistant)")


class IAppendMessagePayload(BaseModel):
    """Payload for appending content to a message."""

    message_id: int = Field(..., description="Message ID")
    type: str = Field(..., description="Content type (markdown, code, etc.)")
    content: Dict[str, Any] = Field(..., description="Message content payload")


class IUpdateMessagePayload(BaseModel):
    """Payload for updating a complete message."""

    message_id: int = Field(..., description="Message ID")
    payload: Dict[str, List[IMessageContent]] = Field(..., description="Complete message payload")


# SSE Event Data Models
class SSEMessagePayload(BaseModel):
    """Payload for SSE message events."""

    message_id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user, assistant)")
    content: List[IMessageContent] = Field(default_factory=list, description="Message content list")
    depth: int = Field(default=0, description="Nesting depth (0=main, 1=sub-agent)")
    parent_action_id: Optional[str] = Field(default=None, description="Parent action ID for sub-agent grouping")


class SSEMessageData(BaseModel):
    """Data structure for SSE message events (createMessage, appendMessage, updateMessage)."""

    type: SSEDataType = Field(..., description="Message operation type")
    payload: SSEMessagePayload = Field(..., description="Message payload")


class SSESessionData(BaseModel):
    """Data structure for SSE session events."""

    session_id: str = Field(..., description="Service session ID")
    llm_session_id: Optional[str] = Field(None, description="LLM session ID")


class SSEEndData(BaseModel):
    """Data structure for SSE end events."""

    session_id: str = Field(..., description="Service session ID")
    llm_session_id: Optional[str] = Field(None, description="LLM session ID")
    total_events: int = Field(..., description="Total events sent")
    action_count: int = Field(..., description="Total actions performed")
    duration: float = Field(..., description="Duration in seconds")
    requests: int = Field(0, description="Number of LLM calls in this turn")
    input_tokens: int = Field(0, description="Turn input tokens")
    output_tokens: int = Field(0, description="Turn output tokens")
    total_tokens: int = Field(0, description="Turn total tokens")
    cached_tokens: int = Field(0, description="Cache hit tokens")
    session_total_tokens: int = Field(0, description="Current context window usage (last model call input_tokens)")
    context_length: int = Field(0, description="Model max context window")


class SSEPingData(BaseModel):
    """Data structure for SSE ping events."""

    pass


class SSEErrorData(BaseModel):
    """Data structure for SSE error events."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type name")
    session_id: Optional[str] = Field(None, description="Service session ID")
    llm_session_id: Optional[str] = Field(None, description="LLM session ID")


# Union type for SSE event data
SSEEventData = Union[SSEMessageData, SSESessionData, SSEEndData, SSEPingData, SSEErrorData]


# SSE Event Structure
class SSEEvent(BaseModel):
    """Server-Sent Event with proper ID and event type."""

    id: int = Field(..., description="Sequential event ID")
    event: str = Field(
        ...,
        description="Event type (start, thinking, action, message, response, error, ping, end)",
    )
    data: SSEEventData = Field(..., description="Event payload")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")


class SSEPayload(BaseModel):
    """Standard payload structure for SSE events (deprecated, use SSEMessageData instead)."""

    type: str = Field(..., description="Data type (createMessage, appendMessage, updateMessage)")
    payload: Dict[str, Any] = Field(..., description="Type-specific payload")


class ChatHistoryData(BaseModel):
    """Chat history data."""

    messages: List[SSEMessagePayload] = Field(default_factory=list, description="chat history messages")
