"""
Service for handling CLI Command operations.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import ValidationError

from datus.api.models.base_models import Result
from datus.api.models.cli_models import (
    ContextResultData,
    ExecuteContextData,
    ExecuteContextInput,
    ExecuteSQLData,
    ExecuteSQLInput,
    ExecuteToolData,
    HistoricalQuery,
    InternalCommandData,
    InternalCommandInput,
    InternalCommandResultData,
    Metric,
    SampleData,
    SavedFile,
    SaveToolInput,
    SaveToolResult,
    SchemaLinkingResult,
    SchemaLinkingToolInput,
    SearchHistoryResult,
    SearchHistoryToolInput,
    SearchMetricsResult,
    SearchMetricsToolInput,
    TableInfo,
    TableMetadata,
)
from datus.api.models.config_models import ErrorCode
from datus.api.services.chat_service import ChatService
from datus.configuration.agent_config_loader import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.output_tools.output import OutputTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CLIService:
    """Service for handling CLI command operations."""

    def __init__(self, agent_config: Optional[AgentConfig] = None, chat_service: Optional[ChatService] = None):
        """
        Initialize the CLI service.

        Args:
            agent_config: Datus agent configuration
        """
        self.agent_config = agent_config
        self.chat_service = chat_service
        # Initialize database manager and namespace only if agent_config is provided
        if self.agent_config:
            self.db_manager = DBManager(self.agent_config.namespaces)
            self.current_namespace = self.agent_config.current_namespace
        else:
            self.db_manager = None
            self.current_namespace = None

        # Initialize CLI context first (before _initialize_connection)
        from datus.cli.cli_context import CliContext

        self.current_db_name = None
        self.cli_context = CliContext(
            current_db_name="",
            current_catalog="",
            current_schema="",
        )

        # Initialize database connection
        self.current_db_connector = None
        if self.agent_config:
            self._initialize_connection()

        # Initialize agent (lazy loading)
        self.agent = None
        self.agent_ready = False

        # Initialize context search tools and output tool
        self.context_search_tools = None
        self.output_tool = None
        if self.agent_config:
            self.context_search_tools = ContextSearchTools(self.agent_config)
            self.output_tool = OutputTool(agent_config=self.agent_config)

    def _initialize_connection(self):
        """Initialize the current database connection."""
        if self.db_manager and self.current_namespace:
            try:
                db_name, connector = self.db_manager.first_conn_with_name(self.current_namespace)
                self.current_db_connector = connector
                self.current_db_name = db_name

                # Update CLI context with connection info
                if self.cli_context and connector:
                    self.cli_context.update_database_context(
                        catalog=getattr(connector, "catalog_name", ""),
                        db_name=db_name or "",
                        schema=getattr(connector, "schema_name", ""),
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize database connection: {e}")

    def _ensure_agent(self):
        """Ensure agent is initialized."""
        if not self.agent and not self.agent_ready:
            try:
                from argparse import Namespace

                from datus.agent.agent import Agent

                agent_args = Namespace(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=8000,
                    plan="reflection",
                    max_steps=20,
                    debug=False,
                    load_cp=False,
                    components=["metrics", "metadata", "table_lineage", "document"],
                )

                if self.agent_config:
                    self.agent = Agent(agent_args, self.agent_config)
                    self.agent_ready = True
                    return True
            except Exception as e:
                logger.error(f"Failed to initialize agent: {e}")
                return False
        return self.agent_ready

    def execute_sql(self, request: ExecuteSQLInput) -> Result[ExecuteSQLData]:
        """
        Execute SQL query.

        Args:
            request: SQL execution request

        Returns:
            ExecuteSQLResult with query result
        """
        try:
            if not self.current_db_connector:
                return Result(
                    success=False,
                    errorCode=ErrorCode.DATABASE_CONNECTION_ERROR,
                    errorMessage="No database connection available",
                )

            # Switch to the requested database/catalog context before executing.
            # The connector is initialized without an active database; callers
            # pass database_name per-request to select the correct one.
            if request.database_name:
                catalog = getattr(self.current_db_connector, "catalog_name", "") or ""
                self.current_db_connector.switch_context(
                    catalog_name=catalog,
                    database_name=request.database_name,
                )

            # Create action for SQL execution (local to avoid cross-request state)
            actions = ActionHistoryManager()
            sql_action = ActionHistory.create_action(
                role=ActionRole.USER,
                action_type="sql_execution",
                messages=(
                    f"Executing SQL: {request.sql_query[:100]}..."
                    if len(request.sql_query) > 100
                    else f"Executing SQL: {request.sql_query}"
                ),
                input_data={"sql": request.sql_query, "system": request.system},
                status=ActionStatus.PROCESSING,
            )
            actions.add_action(sql_action)

            # Execute the query
            start_time = time.time()
            result = self.current_db_connector.execute(
                input_params={"sql_query": request.sql_query},
                result_format=request.result_format,
            )
            end_time = time.time()
            exec_time = end_time - start_time

            if not result:
                # Update action with error
                actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": "No result from query"},
                    messages="SQL execution failed: No result from query",
                )
                return Result(
                    success=False,
                    errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                    errorMessage="No result from the query",
                )

            if result.success:
                # Convert result data based on format
                sql_return = None
                row_count = None
                columns = None

                if hasattr(result.sql_return, "column_names"):
                    # Arrow format
                    if request.result_format == "csv":
                        # Convert Arrow to CSV
                        import csv
                        import io

                        rows = result.sql_return.to_pylist()
                        output = io.StringIO()
                        if rows:
                            writer = csv.DictWriter(output, fieldnames=result.sql_return.column_names)
                            writer.writeheader()
                            writer.writerows(rows)
                        sql_return = output.getvalue()
                    elif request.result_format == "json":
                        # Convert Arrow to JSON
                        import json

                        rows = result.sql_return.to_pylist()
                        sql_return = json.dumps(rows)
                    else:
                        # Keep as Arrow string representation
                        sql_return = str(result.sql_return)

                    row_count = result.sql_return.num_rows
                    columns = result.sql_return.column_names
                else:
                    # Non-Arrow result
                    sql_return = str(result.sql_return) if result.sql_return else ""
                    row_count = result.row_count

                # Update action with success
                actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.SUCCESS,
                    output={
                        "row_count": row_count,
                        "execution_time": exec_time,
                        "columns": columns,
                        "success": True,
                    },
                    messages=f"SQL executed successfully: {row_count or 0} rows in {exec_time:.2f}s",
                )

                data = ExecuteSQLData(
                    sql_query=request.sql_query,
                    row_count=row_count,
                    sql_return=sql_return,
                    result_format=request.result_format,
                    execution_time=exec_time,
                    executed_at=datetime.now().isoformat() + "Z",
                    columns=columns,
                )

                return Result(success=True, data=data)
            else:
                error_msg = result.error or "Unknown SQL error"

                # Update action with error
                actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "sql_error": True},
                    messages=f"SQL error: {error_msg}",
                )

                return Result(
                    success=False,
                    errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                    errorMessage=error_msg,
                )

        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                errorMessage=str(e),
            )

    def execute_tool(self, tool_name: str, request: Any) -> Result[ExecuteToolData]:
        """Execute Tool Commands API as defined in the design."""
        try:
            if hasattr(request, "model_dump"):
                payload = request.model_dump(exclude_unset=True)
            elif isinstance(request, dict):
                payload = dict(request)
            else:
                return Result(
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Tool request must be a mapping or Pydantic model",
                )

            tool_key = tool_name.strip().lower()
            handlers: Dict[str, Any] = {
                "sl": (SchemaLinkingToolInput, self._execute_schema_linking_tool, True),
                "schema_linking": (
                    SchemaLinkingToolInput,
                    self._execute_schema_linking_tool,
                    True,
                ),
                "sm": (SearchMetricsToolInput, self._execute_search_metrics_tool, True),
                "search_metrics": (
                    SearchMetricsToolInput,
                    self._execute_search_metrics_tool,
                    True,
                ),
                "sh": (SearchHistoryToolInput, self._execute_search_history_tool, True),
                "search_history": (
                    SearchHistoryToolInput,
                    self._execute_search_history_tool,
                    True,
                ),
                "save": (SaveToolInput, self._execute_save_tool, False),
            }

            handler = handlers.get(tool_key)
            if not handler:
                return Result(
                    success=False,
                    errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                    errorMessage=(
                        "Tool not supported. Supported tools: "
                        "sl/schema_linking, sm/search_metrics, "
                        "sh/search_history, save"
                    ),
                )

            model_cls, executor, requires_agent = handler

            if requires_agent and not self._ensure_agent():
                return Result(
                    success=False,
                    errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                    errorMessage="Agent not available",
                )

            payload = {key: value for key, value in payload.items() if key not in {"tool_name", "stream_output"}}
            args = payload.pop("args", None)
            if args and "query_text" not in payload:
                payload["query_text"] = args

            try:
                tool_input = model_cls.model_validate(payload)
            except ValidationError as e:
                error_details = "; ".join(err.get("msg", "Invalid value") for err in e.errors())
                logger.error(f"Invalid input for tool {tool_key}: {e}")
                return Result(
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"Invalid input parameters: {error_details}",
                )

            start_time = time.time()
            result = executor(tool_input)
            exec_time = time.time() - start_time

            data = ExecuteToolData(
                tool_name=tool_key,
                query_text=getattr(tool_input, "query_text", None),
                result=result,
                execution_time=exec_time,
                executed_at=datetime.now().isoformat() + "Z",
            )

            return Result(success=True, data=data)

        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                errorMessage=str(e),
            )

    def _execute_schema_linking_tool(self, request: SchemaLinkingToolInput) -> SchemaLinkingResult:
        """Execute schema linking tool with structured parameters."""
        try:
            if not self.context_search_tools:
                raise Exception("Context search tools not available")

            result = self.context_search_tools.search_table_metadata(
                query_text=request.query_text,
                catalog_name=request.catalog_name or "",
                database_name=request.database_name or "",
                schema_name=request.schema_name or "",
                top_n=request.top_n,
                simple_sample_data=False,
            )

            if not result.success:
                raise Exception(result.error or "Schema linking search failed")

            metadata = []
            sample_data = []

            # Convert result to our model format
            if result.result and "metadata" in result.result:
                for item in result.result["metadata"]:
                    metadata.append(
                        TableMetadata(
                            table_name=item.get("table_name", ""),
                            catalog_name=item.get("catalog_name", ""),
                            database_name=item.get("database_name", ""),
                            schema_name=item.get("schema_name", ""),
                            definition=item.get("definition", ""),
                            score=1.0 - item.get("_distance", 0.0),  # Convert distance to score
                        )
                    )

            if result.result and "sample_data" in result.result:
                for item in result.result["sample_data"]:
                    sample_data.append(
                        SampleData(
                            table_name=item.get("table_name", ""),
                            sample_rows=item.get("sample_rows", ""),
                        )
                    )

            return SchemaLinkingResult(
                metadata=metadata,
                sample_data=sample_data,
                total_metadata=len(metadata),
                total_sample_data=len(sample_data),
            )
        except Exception as e:
            logger.error(f"Schema linking tool error: {e}")
            return SchemaLinkingResult(metadata=[], sample_data=[], total_metadata=0, total_sample_data=0)

    def _execute_search_metrics_tool(self, request: SearchMetricsToolInput) -> SearchMetricsResult:
        """Execute search metrics tool with structured parameters."""
        try:
            if not self.context_search_tools:
                raise Exception("Context search tools not available")

            result = self.context_search_tools.search_metrics(
                query_text=request.query_text,
                domain=request.domain or "",
                layer1=request.layer1 or "",
                layer2=request.layer2 or "",
                catalog_name=request.catalog_name or "",
                database_name=request.database_name or "",
                schema_name=request.schema_name or "",
                top_n=request.top_n,
            )

            if not result.success:
                raise Exception(result.error or "Metrics search failed")

            metrics = []

            # Convert result to our model format
            if result.result:
                for item in result.result:
                    metrics.append(
                        Metric(
                            name=item.get("name", ""),
                            description=item.get("description", ""),
                            constraint=item.get("constraint", ""),
                            domain=item.get("domain", ""),
                            layer1=item.get("layer1", ""),
                            layer2=item.get("layer2", ""),
                            score=1.0 - item.get("_distance", 0.0),  # Convert distance to score
                        )
                    )

            return SearchMetricsResult(metrics=metrics, total_count=len(metrics))
        except Exception as e:
            logger.error(f"Search metrics tool error: {e}")
            return SearchMetricsResult(metrics=[], total_count=0)

    def _execute_search_history_tool(self, request: SearchHistoryToolInput) -> SearchHistoryResult:
        """Execute search history tool with structured parameters."""
        try:
            if not self.context_search_tools:
                raise Exception("Context search tools not available")

            # Call real implementation
            result = self.context_search_tools.search_historical_sql(
                query_text=request.query_text,
                domain=request.domain or "",
                layer1=request.layer1 or "",
                layer2=request.layer2 or "",
                top_n=request.top_n,
            )

            if not result.success:
                raise Exception(result.error or "Historical SQL search failed")

            history = []

            # Convert result to our model format
            if result.result:
                for item in result.result:
                    history.append(
                        HistoricalQuery(
                            sql_query=item.get("sql", ""),
                            description=item.get("comment", ""),
                            domain=item.get("domain", ""),
                            layer1=item.get("layer1", ""),
                            layer2=item.get("layer2", ""),
                            timestamp=item.get("timestamp", ""),
                            score=1.0 - item.get("_distance", 0.0),  # Convert distance to score
                        )
                    )

            return SearchHistoryResult(history=history, total_count=len(history))
        except Exception as e:
            logger.error(f"Search history tool error: {e}")
            return SearchHistoryResult(history=[], total_count=0)

    def _execute_save_tool(self, request: SaveToolInput) -> SaveToolResult:
        """Execute save tool with structured parameters."""
        try:
            import os
            import zipfile
            from datetime import datetime

            from datus.schemas.node_models import OutputInput

            # Get last SQL context
            last_sql = self.cli_context.get_last_sql_context()
            if not last_sql:
                logger.error("No previous SQL result to save")
                return SaveToolResult(files_saved=[], total_files=0, total_size="0B")

            # Create target directory
            target_dir = request.target_dir or os.path.expanduser("~/.datus/output")
            os.makedirs(target_dir, exist_ok=True)

            file_name = request.file_name or datetime.now().strftime("%Y%m%d%H%M%S")

            # Initialize output tool if not exists
            if not self.output_tool:
                self.output_tool = OutputTool(agent_config=self.agent_config)

            # Execute output tool to save results
            output_input = OutputInput(
                task="",
                database_name=self.cli_context.current_db_name or "",
                task_id=file_name,
                gen_sql=last_sql.sql_query,
                sql_result=last_sql.sql_return or "",
                row_count=last_sql.row_count or 0,
                file_type=request.file_type,
                check_result=False,
                error=last_sql.sql_error,
                finished=not last_sql.sql_error,
                output_dir=target_dir,
            )

            result = self.output_tool.execute(output_input, self.current_db_connector)

            # Parse saved files from result.output
            files_saved = []
            total_size = 0
            file_paths = []

            # OutputTool returns paths in result.output
            output_paths = result.output.split("\n") if result.output else []
            for path in output_paths:
                path = path.strip()
                if path and os.path.exists(path):
                    file_type = path.split(".")[-1] if "." in path else "unknown"
                    file_size = os.path.getsize(path)
                    files_saved.append(
                        SavedFile(
                            file_type=file_type,
                            file_path=path,
                            file_size=f"{file_size}B",
                        )
                    )
                    total_size += file_size
                    file_paths.append(path)

            # Create zip archive
            zip_path = None
            download_url = None
            if file_paths:
                zip_filename = f"{file_name}.zip"
                zip_path = os.path.join(target_dir, zip_filename)

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in file_paths:
                        # Add file to zip with just the basename (no directory structure)
                        zipf.write(file_path, os.path.basename(file_path))

                # Calculate zip size
                zip_size = os.path.getsize(zip_path)
                logger.info(f"Created zip archive: {zip_path} ({zip_size} bytes)")

                # Generate download URL (relative path from output directory)
                # Frontend should map this to actual download endpoint
                download_url = f"/api/download/{zip_filename}"

            return SaveToolResult(
                files_saved=files_saved,
                total_files=len(files_saved),
                total_size=f"{total_size}B",
                zip_path=zip_path,
                download_url=download_url,
            )
        except Exception as e:
            logger.error(f"Save tool error: {e}")
            return SaveToolResult(files_saved=[], total_files=0, total_size="0B")

    def execute_context(self, context_type: str, request: ExecuteContextInput) -> Result[ExecuteContextData]:
        """
        Execute context command.

        Args:
            context_type: Type of context command
            request: Context execution request

        Returns:
            ExecuteContextResult with context result
        """
        try:
            result_data = ContextResultData()

            if context_type == "tables":
                # Get tables list
                if self.current_db_connector:
                    tables = self.current_db_connector.get_tables()
                    if tables:
                        table_info_list = []
                        for table in tables:
                            table_info = TableInfo(
                                table_name=table,
                                table_type="table",
                                row_count=None,  # Would need additional query
                                columns_count=None,  # Would need additional query
                            )
                            table_info_list.append(table_info)
                        result_data.tables = table_info_list
                        result_data.total_count = len(table_info_list)
                else:
                    result_data.tables = []
                    result_data.total_count = 0

            elif context_type == "catalogs":
                # Get real catalogs from database connection
                if self.current_db_connector:
                    try:
                        # Try to get actual catalogs from the database
                        catalogs = (
                            self.current_db_connector.get_catalogs()
                            if hasattr(self.current_db_connector, "get_catalogs")
                            else ["main"]
                        )
                        current_catalog = self.cli_context.current_catalog if self.cli_context else "main"
                        result_data.context_info = {
                            "catalogs": catalogs,
                            "current": current_catalog,
                            "total_count": len(catalogs),
                        }
                    except Exception as e:
                        logger.debug(f"Failed to get catalogs from database: {e}")
                        result_data.context_info = {
                            "catalogs": ["main"],
                            "current": "main",
                            "error": str(e),
                        }
                else:
                    result_data.context_info = {
                        "catalogs": [],
                        "current": None,
                        "error": "No database connection",
                    }

            elif context_type == "context":
                # Get real current context with more details
                db_info = {}
                if self.current_db_connector:
                    try:
                        # Get database type and details
                        db_type = getattr(self.current_db_connector, "db_type", "unknown")
                        db_name = getattr(
                            self.current_db_connector,
                            "database_name",
                            self.current_db_name,
                        )
                        host = getattr(self.current_db_connector, "host", None)
                        port = getattr(self.current_db_connector, "port", None)

                        db_info = {
                            "db_type": db_type,
                            "database_name": db_name,
                            "host": host,
                            "port": port,
                            "connection_status": "connected",
                        }
                    except Exception as e:
                        logger.debug(f"Failed to get database details: {e}")
                        db_info = {
                            "database_name": self.current_db_name,
                            "connection_status": "connected",
                            "error": str(e),
                        }
                else:
                    db_info = {"connection_status": "disconnected"}

                result_data.context_info = {
                    "current_namespace": self.current_namespace,
                    "current_database": self.current_db_name,
                    "current_catalog": getattr(self.cli_context, "current_catalog", None) if self.cli_context else None,
                    "current_schema": getattr(self.cli_context, "current_schema", None) if self.cli_context else None,
                    "database": db_info,
                    "timestamp": datetime.now().isoformat() + "Z",
                }

            elif context_type == "catalog":
                # Display database catalogs (@catalog command) - real implementation
                try:
                    if self.current_db_connector and hasattr(self, "agent_config") and self.agent_config:
                        # Use real catalog context similar to ContextCommands.cmd_catalog
                        db_type = getattr(self.agent_config, "db_type", "unknown")
                        catalog_name = (
                            getattr(self.cli_context, "current_catalog", "main") if self.cli_context else "main"
                        )

                        result_data.context_info = {
                            "db_type": db_type,
                            "catalog_name": catalog_name,
                            "database_name": self.current_db_name,
                            "connection_status": "connected",
                            "message": "Database catalog context displayed",
                            "tables_available": len(self.current_db_connector.get_tables())
                            if self.current_db_connector
                            else 0,
                        }
                    else:
                        result_data.context_info = {
                            "error": "No database connection or configuration available",
                            "message": "Catalog context not available",
                        }
                except Exception as e:
                    logger.error(f"Error getting catalog context: {e}")
                    result_data.context_info = {
                        "error": str(e),
                        "message": "Failed to get catalog context",
                    }

            elif context_type == "subject":
                # Display metrics (@subject command) - real implementation
                try:
                    # Check if agent_config is available for RAG functionality
                    if not self.agent_config:
                        result_data.context_info = {
                            "database_name": self.current_db_name,
                            "metrics_available": False,
                            "error": "No agent configuration available",
                            "message": "Metrics context not available - agent config required",
                        }
                    else:
                        # Use real metrics RAG similar to ContextCommands.cmd_subject
                        from datus.storage.metric.store import rag_by_configuration

                        metrics_rag = rag_by_configuration(self.agent_config)
                        metrics_count = metrics_rag.get_metrics_size()
                        rag_path = self.agent_config.rag_storage_path()

                        result_data.context_info = {
                            "database_name": self.current_db_name,
                            "metrics_available": metrics_count > 0,
                            "metrics_count": metrics_count,
                            "rag_storage_path": rag_path,
                            "message": f"Subject/metrics context displayed - {metrics_count} metrics found",
                        }
                except Exception as e:
                    logger.error(f"Error getting metrics context: {e}")
                    result_data.context_info = {
                        "database_name": self.current_db_name,
                        "metrics_available": False,
                        "error": str(e),
                        "message": "Failed to get metrics context",
                    }

            elif context_type == "sql":
                # Display historical SQL (@sql command) - real implementation
                try:
                    # Check if agent_config is available for RAG functionality
                    if not self.agent_config:
                        result_data.context_info = {
                            "database_name": self.current_db_name,
                            "historical_sql_available": False,
                            "error": "No agent configuration available",
                            "message": "SQL history context not available - agent config required",
                        }
                    else:
                        # Use real SQL history RAG similar to ContextCommands.cmd_historical_sql
                        from datus.storage.sql_history import (
                            sql_history_rag_by_configuration,
                        )

                        sql_rag = sql_history_rag_by_configuration(self.agent_config)
                        sql_count = sql_rag.get_sql_history_size()
                        rag_path = self.agent_config.rag_storage_path()

                        result_data.context_info = {
                            "database_name": self.current_db_name,
                            "historical_sql_available": sql_count > 0,
                            "sql_count": sql_count,
                            "rag_storage_path": rag_path,
                            "message": f"Historical SQL context displayed - {sql_count} queries found",
                        }
                except Exception as e:
                    logger.error(f"Error getting SQL history context: {e}")
                    result_data.context_info = {
                        "database_name": self.current_db_name,
                        "historical_sql_available": False,
                        "error": str(e),
                        "message": "Failed to get SQL history context",
                    }

            else:
                return Result(
                    success=False,
                    errorCode=ErrorCode.CONTEXT_COMMAND_ERROR,
                    errorMessage=f"Context type '{context_type}' not supported",
                )

            data = ExecuteContextData(
                context_type=context_type,
                database_name=request.database_name or self.current_db_name,
                schema_name=request.schema_name,
                result=result_data,
            )

            return Result(success=True, data=data)

        except Exception as e:
            logger.error(f"Failed to execute context command: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.CONTEXT_COMMAND_ERROR,
                errorMessage=str(e),
            )

    def execute_internal_command(self, command: str, request: InternalCommandInput) -> Result[InternalCommandData]:
        """
        Execute internal command.

        Args:
            command: Internal command name
            request: Internal command request

        Returns:
            InternalCommandResult with command result
        """
        try:
            result_data = InternalCommandResultData(command_output="", action_taken="none", context_changed=False)

            if command == "help":
                result_data.command_output = "Available commands: help, databases, tables, schemas, clear, exit"
                result_data.action_taken = "display_help"

            elif command in ["databases", "database"]:
                if self.db_manager:
                    connections = self.db_manager.get_connections(self.current_namespace)
                    # Handle both single connector and dict of connectors
                    if isinstance(connections, dict):
                        db_list = list(connections.keys())
                    else:
                        # Single connector - get database name from current context or config
                        db_list = [self.current_db_name] if self.current_db_name else ["default"]
                    result_data.command_output = f"Available databases: {', '.join(db_list)}"
                    result_data.data = {"databases": db_list}
                else:
                    result_data.command_output = "No database connections available"
                result_data.action_taken = "list_databases"

            elif command == "tables":
                if self.current_db_connector:
                    tables = self.current_db_connector.get_tables()
                    result_data.command_output = f"Tables: {', '.join(tables or [])}"
                    result_data.data = {"tables": tables or []}
                else:
                    result_data.command_output = "No database connection"
                result_data.action_taken = "list_tables"

            elif command == "clear":
                # Clear LLM-level session by service session ID
                # Args: service_session_id (finds and deletes corresponding LLM session)
                try:
                    service_session_id = request.args.strip() if request.args else None

                    if service_session_id:
                        # Call chat_service to delete LLM session for this service session
                        result = self.chat_service.delete_session(service_session_id)

                        if result.success:
                            result_data.command_output = f"Session {service_session_id} cleared successfully"
                            result_data.context_changed = True
                            result_data.data = {
                                "service_session_id": service_session_id,
                                "cleared": True,
                            }
                        else:
                            result_data.command_output = (
                                f"Failed to clear session {service_session_id}: "
                                f"{result.errorMessage or 'Unknown error'}"
                            )
                            result_data.data = {
                                "service_session_id": service_session_id,
                                "cleared": False,
                                "error": result.errorMessage,
                            }
                    else:
                        result_data.command_output = "No service session ID provided. Usage: clear <service_session_id>"
                        result_data.data = {"error": "Missing service_session_id parameter"}

                    result_data.action_taken = "clear_llm_session"

                except Exception as e:
                    logger.error(f"Error clearing LLM session: {e}")
                    result_data.command_output = f"Error clearing LLM session: {str(e)}"
                    result_data.action_taken = "clear_llm_session_error"
                    result_data.data = {"error": str(e)}

            elif command in ["exit", "quit"]:
                result_data.command_output = "Goodbye!"
                result_data.action_taken = "exit_program"

            elif command == "chat_info":
                # Real chat info implementation based on ChatCommands.cmd_chat_info
                try:
                    # Try to get session info from current context or session manager
                    current_session_id = getattr(self, "current_session_id", None)

                    if current_session_id:
                        # Use SessionManager to get detailed session info
                        from datus.models.session_manager import SessionManager

                        session_dir = self.chat_service._session_dir if self.chat_service else None
                        session_manager = SessionManager(session_dir=session_dir)
                        session_info = session_manager.get_session_info(current_session_id)

                        if session_info:
                            result_data.command_output = (
                                f"Current session: {current_session_id}\n"
                                f"  Token Count: {session_info.get('total_tokens', 0)}\n"
                                f"  Action Count: {session_info.get('action_count', 0)}\n"
                                f"  Created: {session_info.get('created_at', 'Unknown')}\n"
                                f"  Last Updated: {session_info.get('last_updated', 'Unknown')}"
                            )
                            result_data.data = {
                                "current_session_id": current_session_id,
                                "session_info": session_info,
                                "token_count": session_info.get("total_tokens", 0),
                                "action_count": session_info.get("action_count", 0),
                                "created_at": session_info.get("created_at"),
                                "last_updated": session_info.get("last_updated"),
                            }
                        else:
                            result_data.command_output = f"Session {current_session_id} info not available"
                            result_data.data = {
                                "current_session_id": current_session_id,
                                "error": "Session info not found",
                            }
                    else:
                        result_data.command_output = "No active session"
                        result_data.data = {"current_session_id": None}

                    result_data.action_taken = "show_chat_info"

                except Exception as e:
                    logger.error(f"Error getting chat info: {e}")
                    result_data.command_output = f"Error getting chat info: {str(e)}"
                    result_data.data = {"current_session_id": None, "error": str(e)}
                    result_data.action_taken = "show_chat_info_error"

            elif command == "sessions":
                # Use chat_service.list_sessions() for consistent session listing
                try:
                    sessions_result = self.chat_service.list_sessions()

                    if not sessions_result.success:
                        result_data.command_output = (
                            f"Error listing sessions: {sessions_result.errorMessage or 'Unknown error'}"
                        )
                        result_data.data = {
                            "sessions": [],
                            "error": sessions_result.errorMessage,
                        }
                        result_data.action_taken = "list_sessions_error"
                    elif not sessions_result.data:
                        result_data.command_output = "No chat sessions found"
                        result_data.data = {"sessions": []}
                        result_data.action_taken = "list_sessions"
                    else:
                        # Convert ChatSessionData to dict format
                        sessions_with_info = []
                        for session_data in sessions_result.data:
                            # Format timestamps to be readable
                            created = session_data.created_at
                            updated = session_data.last_updated
                            if isinstance(created, str) and len(created) > 19:
                                created = created[:19]
                            if isinstance(updated, str) and len(updated) > 19:
                                updated = updated[:19]

                            session_info = {
                                "session_id": session_data.session_id,
                                "created_at": created,
                                "last_updated": updated,
                                "total_turns": session_data.total_turns,
                                "token_count": session_data.token_count,
                                "is_active": session_data.is_active,
                            }
                            sessions_with_info.append(session_info)

                        session_list = [s["session_id"] for s in sessions_with_info]
                        result_data.command_output = f"Available sessions: {', '.join(session_list[:5])}"
                        if len(session_list) > 5:
                            result_data.command_output += f" ... and {len(session_list) - 5} more"

                        result_data.data = {
                            "sessions": sessions_with_info,
                            "total_count": len(sessions_with_info),
                        }
                        result_data.action_taken = "list_sessions"

                except Exception as e:
                    logger.error(f"Error listing sessions: {e}")
                    result_data.command_output = f"Error listing sessions: {str(e)}"
                    result_data.data = {"sessions": [], "error": str(e)}
                    result_data.action_taken = "list_sessions_error"

            else:
                return Result(
                    success=False,
                    errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                    errorMessage=f"Internal command '{command}' not supported",
                )

            data = InternalCommandData(command=command, args=request.args, result=result_data)

            return Result(success=True, data=data)

        except Exception as e:
            logger.error(f"Failed to execute internal command: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                errorMessage=str(e),
            )
