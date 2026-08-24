"""
Service for handling CLI Command operations.
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, Optional

from datus.api.models.base_models import Result
from datus.api.models.cli_models import (
    ContextResultData,
    ExecuteContextData,
    ExecuteContextInput,
    ExecuteSQLData,
    ExecuteSQLInput,
    InternalCommandData,
    InternalCommandInput,
    InternalCommandResultData,
    StopExecuteSQLData,
    TableInfo,
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
from datus.utils.config_utils import coerce_positive_seconds
from datus.utils.exceptions import DatusException
from datus.utils.exceptions import ErrorCode as DbErrorCode
from datus.utils.loggings import get_logger
from datus.utils.time_utils import now_utc_iso

logger = get_logger(__name__)

<<<<<<< HEAD
=======
#: Statement types the console may send straight to the connector.
#:
#: Deliberately an allow-list of *identified* writes. ``UNKNOWN`` is absent:
#: anything the parser could not place must fall to the enforced path, where it
#: is refused — the alternative is that a syntax the dialect accepts but sqlglot
#: does not becomes an unfiltered read.
# Fallback for ``agent.api.sql_queue_budget_seconds`` (see conf/agent.yml.example),
# used when a deployment's yaml predates the setting. Long enough to ride out a
# normal interactive query, short enough that a queue cannot pin to_thread
# workers indefinitely.
_DEFAULT_SQL_QUEUE_BUDGET_SECONDS = 30.0

_WRITE_SQL_TYPES = frozenset(
    {
        SQLType.INSERT,
        SQLType.UPDATE,
        SQLType.DELETE,
        SQLType.MERGE,
        SQLType.DDL,
        SQLType.CONTENT_SET,
    }
)

>>>>>>> b9f8253 ([Feature] Address table metadata and SQL execution per datasource (#1342))

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
        # Initialize database manager and datasource only if agent_config is provided
        if self.agent_config:
            self.db_manager = DBManager(self.agent_config.datasource_configs)
            self.current_datasource = self.agent_config.current_datasource
        else:
            self.db_manager = None
            self.current_datasource = None

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
<<<<<<< HEAD
=======
        # Connectors for the project's other bound datasources, opened on first
        # use. The IDE console addresses one by name per request, and opening
        # every bound warehouse at startup would cost seconds per project.
        self._datasource_connectors: Dict[str, Any] = {}
        # One lock per datasource, held across ``switch_context`` + execute.
        #
        # A connector is shared by every request on its datasource and carries
        # mutable catalog/database context, while ``_execute_sql_sync`` runs in
        # worker threads — so two interleaved requests could each execute under
        # the other's database. Serializing per datasource is the same trade
        # ``DatasourceService._schema_lock`` already makes for the same reason:
        # a queued query beats a query answered from the wrong database.
        self._connector_locks: Dict[str, threading.Lock] = {}
        self._db_tool_cache: Optional[func_tool_mod.DBFuncTool] = None
        # `_execute_sql_sync` runs under `asyncio.to_thread`, so two clicks on
        # Run can reach the lazy build below at the same time and each pay for
        # a DBManager and three RAG indexes.
        self._db_tool_lock = threading.Lock()
>>>>>>> b9f8253 ([Feature] Address table metadata and SQL execution per datasource (#1342))
        if self.agent_config:
            self._initialize_connection()

        # Track running SQL execution tasks: {task_id: asyncio.Task}
        self._sql_tasks: Dict[str, asyncio.Task] = {}
        # A thread-safe stop signal per task, because cancelling the asyncio task
        # cannot reach the worker thread it dispatched: a worker parked in
        # ``execution_lock.acquire()`` keeps waiting, then acquires and runs. For
        # a write that means the statement lands *after* stop_execute_sql told
        # the caller it stopped.
        self._sql_cancels: Dict[str, threading.Event] = {}
        self._sql_tasks_lock = threading.Lock()

    def _initialize_connection(self):
        """Initialize the current database connection."""
        if self.db_manager and self.current_datasource:
            try:
                db_name, connector = self.db_manager.first_conn_with_name(self.current_datasource)
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

    def _execution_target(self, datasource: Optional[str] = None) -> tuple[str, Any]:
        """``(datasource, connector)`` this statement should run against.

        Defaults to the project's current datasource. An unknown name is an
        error rather than a fall-through: silently running the statement on a
        different warehouse than the editor's tab is showing is how a query
        returns plausible rows from the wrong place.
        """
        key = (datasource or "").strip() or (self.current_datasource or "")
        if not key:
            raise DatusException(
                DbErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": "No database connection available"},
            )

        if key == self.current_datasource:
            if not self.current_db_connector:
                raise DatusException(
                    DbErrorCode.DB_CONNECTION_FAILED,
                    message_args={"error_message": "No database connection available"},
                )
            return key, self.current_db_connector

        configs = getattr(self.agent_config, "datasource_configs", {}) or {}
        if key not in configs:
            raise DatusException(
                DbErrorCode.COMMON_UNSUPPORTED, message_args={"field_name": "datasource", "your_value": key}
            )

        cached = self._datasource_connectors.get(key)
        if cached is None:
            # Whatever the db manager raises for a declared-but-unreachable
            # datasource reaches the caller as the structured database error, so
            # one exception type maps to one error response.
            try:
                _db_name, connector = self.db_manager.first_conn_with_name(key)
            except DatusException:
                raise
            except Exception as e:  # noqa: BLE001 — normalized for the caller
                raise DatusException(
                    DbErrorCode.DB_CONNECTION_FAILED,
                    message_args={"error_message": f"datasource '{key}' is unreachable: {e}"},
                ) from e
            if connector is None:
                raise DatusException(
                    DbErrorCode.DB_CONNECTION_FAILED,
                    message_args={"error_message": f"datasource '{key}' has no usable connection"},
                )
            # Only cached once known good. Caching a None turned the next
            # request's `connector.dialect` into an AttributeError.
            cached = connector
            self._datasource_connectors[key] = cached
        return key, cached

    def _connector_lock(self, datasource: str) -> threading.Lock:
        """The lock guarding one datasource's shared connector.

        Tolerates an instance built without ``__init__`` (which the policy tests
        do), and needs no lock of its own: ``dict.setdefault`` resolves the
        create-vs-reuse race itself, so two threads asking at once still get the
        same lock.
        """
        locks = getattr(self, "_connector_locks", None)
        if locks is None:
            locks = {}
            self._connector_locks = locks
        return locks.setdefault(datasource, threading.Lock())

    def _cleanup_sql_task(self, task_id: str) -> None:
        """Remove a completed SQL task from the tracking dicts."""
        with self._sql_tasks_lock:
            self._sql_tasks.pop(task_id, None)
            self._sql_cancels.pop(task_id, None)

<<<<<<< HEAD
    def _execute_sql_sync(self, request: ExecuteSQLInput, task_id: str) -> Result[ExecuteSQLData]:
=======
    def _execute_sql_sync(
        self,
        request: ExecuteSQLInput,
        task_id: str,
        policy_context: Optional[Dict[str, Any]] = None,
        cancelled: Optional[threading.Event] = None,
    ) -> Result[ExecuteSQLData]:
>>>>>>> b9f8253 ([Feature] Address table metadata and SQL execution per datasource (#1342))
        """Synchronous SQL execution logic (runs in a thread)."""
        try:
            try:
                datasource, connector = self._execution_target(request.datasource)
            except DatusException as e:
                return Result(
                    success=False,
                    errorCode=ErrorCode.DATABASE_CONNECTION_ERROR,
                    errorMessage=str(e),
                )

<<<<<<< HEAD
=======
            # Deployment-wide read-only posture. This route reaches the connector
            # directly (see ``.execute`` below) and never touches DBFuncTool, so
            # the ``execute_sql`` tool gate does not cover it — without this check
            # a hardened deployment would still expose arbitrary SQL here. The
            # rules come from the same helper the tool path uses so the two
            # entry points cannot disagree.
            #
            # Read per request rather than snapshotted: CLIService is built from
            # the shared service-level config, so a host hardening it at runtime
            # takes effect without rebuilding the service. Placed before
            # ``switch_context`` so a request about to be refused does not mutate
            # connector state.
            #
            # Still needed after the statement-type dispatch below, and this is
            # the reason: that dispatch sends identified single writes straight
            # to ``current_db_connector.execute`` and only reads through
            # ``execute_read_enforced``. So a write never reaches DBFuncTool and
            # never meets the tool-layer read-only gate — this check is the only
            # thing standing in front of it. Refusing here also means the write
            # branch is simply unreachable on a hardened deployment.
            if getattr(self.agent_config, "sql_read_only", False):
                from datus.utils.sql_utils import (
                    READ_ONLY_MULTI_STATEMENT,
                    READ_ONLY_NON_READ,
                    READ_ONLY_WRITABLE_PRAGMA,
                    parse_sql_statement_kind,
                    validate_read_only_sql,
                )

                dialect = getattr(connector, "dialect", "") or ""
                violation, sql_type = validate_read_only_sql(request.sql_query, dialect)
                if violation:
                    # The helper returns a code, not prose, so each entry point
                    # words its own refusal. This route answers an HTTP client
                    # rather than a model, so it names the setting that caused
                    # the refusal.
                    reason = {
                        READ_ONLY_MULTI_STATEMENT: (
                            "Multi-statement SQL is not allowed. Please submit one query at a time."
                        ),
                        READ_ONLY_NON_READ: (
                            f"Only read-only queries (SELECT, SHOW, DESCRIBE, EXPLAIN) are allowed. "
                            f"Detected SQL type: {sql_type.value}"
                        ),
                        READ_ONLY_WRITABLE_PRAGMA: "Writable PRAGMA statements are not allowed in read-only mode.",
                    }[violation]
                    # Same structured shape as the DBFuncTool refusal in
                    # ``database._refuse_write_if_read_only`` so an operator can
                    # filter both entry points on one field set — including the
                    # finer statement kind, since ``ddl`` alone cannot tell them
                    # whether a caller tried to CREATE or to DROP. ``source`` is
                    # always "deployment" here: this route has no per-agent
                    # read-only posture to distinguish it from.
                    logger.warning(
                        "POST /sql/execute rejected by read-only policy",
                        sql_type=parse_sql_statement_kind(request.sql_query, dialect) or sql_type.value,
                        database=request.database_name or "",
                        source="deployment",
                        rule=violation,
                    )
                    return Result(
                        success=False,
                        errorCode=ErrorCode.SQL_READ_ONLY,
                        errorMessage=(f"This deployment is read-only (agent.sql_read_only). {reason}"),
                    )

            # Held from the context switch through the execute below: the
            # connector is shared across requests on this datasource and its
            # catalog/database context is mutable, so releasing in between lets
            # another request run under this one's database (or this one under
            # theirs). Acquired after the read-only refusal above so a rejected
            # request never queues behind a running query.
            #
            # Bounded, because waiting here is not free: this runs on an
            # ``asyncio.to_thread`` worker, and the default executor has only
            # ``min(32, cpu + 4)`` of them. A few slow queries queued on one
            # datasource could otherwise starve every other to_thread caller in
            # the process — table detail, catalog listing, the agent's own tool
            # calls. Timing out also bounds how long a cancelled request keeps a
            # worker: ``stop_execute_sql`` cancels the asyncio task, which cannot
            # interrupt a thread parked in ``acquire()``.
            api_config = getattr(self.agent_config, "api_config", {}) or {}
            wait_budget = coerce_positive_seconds(
                api_config.get("sql_queue_budget_seconds"), _DEFAULT_SQL_QUEUE_BUDGET_SECONDS
            )
            execution_lock = self._connector_lock(datasource)
            if not execution_lock.acquire(timeout=wait_budget):
                logger.warning(
                    "POST /sql/execute gave up waiting for the datasource connector",
                    datasource=datasource,
                    waited_seconds=wait_budget,
                )
                return Result(
                    success=False,
                    errorCode=ErrorCode.DATASOURCE_BUSY,
                    errorMessage=(
                        f"Datasource '{datasource}' is busy with another statement "
                        f"(waited {wait_budget:g}s). Nothing was executed — try again."
                    ),
                )
            try:
                # The one point a queued statement can still be stopped. The
                # await in `execute_sql` is already gone by now — cancelling it
                # never reached this thread — so without this check a write the
                # caller was told had stopped would execute here.
                if cancelled is not None and cancelled.is_set():
                    logger.info("SQL execution cancelled while queued for the datasource connector", task_id=task_id)
                    return Result(
                        success=False,
                        errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                        errorMessage="SQL execution was cancelled",
                    )
                return self._execute_resolved_sql(request, task_id, policy_context, datasource, connector)
            finally:
                execution_lock.release()

        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                errorMessage=str(e),
            )

    def _execute_resolved_sql(
        self,
        request: ExecuteSQLInput,
        task_id: str,
        policy_context: Optional[Dict[str, Any]],
        datasource: str,
        connector: Any,
    ) -> Result[ExecuteSQLData]:
        """Run one statement on an already-resolved connector, under its lock."""
        try:
>>>>>>> b9f8253 ([Feature] Address table metadata and SQL execution per datasource (#1342))
            # Switch to the requested database/catalog context before executing.
            if request.database_name:
                catalog = getattr(connector, "catalog_name", "") or ""
                connector.switch_context(
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
<<<<<<< HEAD
            result = self.current_db_connector.execute(
                input_params={"sql_query": request.sql_query},
                result_format=request.result_format,
            )
=======
            sql_type = parse_sql_type(request.sql_query, connector.dialect)
            is_write = sql_type in _WRITE_SQL_TYPES and is_single_statement(request.sql_query)

            if is_write and policy_context:
                # A write can carry a read: `CREATE TABLE mine AS SELECT * FROM
                # orders` copies the rows a policy just filtered into a table no
                # policy covers. The plugin cannot help — it hooks reads only —
                # so on a project that has policies at all, a write that embeds
                # a query is refused here.
                reads = write_statement_reads_data(request.sql_query, connector.dialect)
                if reads:
                    message = (
                        "This project has row-level policies, so a write statement that "
                        "reads from a query is not allowed here — it would copy filtered "
                        "rows into a table no policy covers. Create views and derived "
                        "tables on the database side instead."
                    )
                    # Every other exit from here closes the action it opened;
                    # returning without doing so leaves it PROCESSING forever
                    # and the refusal never reaches the history.
                    actions.update_action_by_id(
                        sql_action.action_id,
                        status=ActionStatus.FAILED,
                        output={"error": message},
                        messages=f"SQL execution refused: {message}",
                    )
                    return Result(
                        success=False,
                        errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                        errorMessage=message,
                    )

            if is_write:
                result = connector.execute(
                    input_params={"sql_query": request.sql_query},
                    result_format=request.result_format,
                )
            else:
                result = self._db_tool().execute_read_enforced(
                    request.sql_query,
                    connector,
                    datasource=datasource,
                    result_format=request.result_format,
                    policy_context=policy_context,
                )
>>>>>>> b9f8253 ([Feature] Address table metadata and SQL execution per datasource (#1342))
            end_time = time.time()
            exec_time = end_time - start_time

            if not result:
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
                sql_return = None
                row_count = None
                columns = None

                if hasattr(result.sql_return, "column_names"):
                    if request.result_format == "csv":
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
                        import json

                        rows = result.sql_return.to_pylist()
                        sql_return = json.dumps(rows)
                    else:
                        sql_return = str(result.sql_return)

                    row_count = result.sql_return.num_rows
                    columns = result.sql_return.column_names
                else:
                    sql_return = str(result.sql_return) if result.sql_return else ""
                    row_count = result.row_count

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
                    execute_task_id=task_id,
                    sql_query=request.sql_query,
                    row_count=row_count,
                    sql_return=sql_return,
                    result_format=request.result_format,
                    execution_time=exec_time,
                    executed_at=now_utc_iso(),
                    columns=columns,
                )

                return Result(success=True, data=data)
            else:
                error_msg = result.error or "Unknown SQL error"

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

    async def execute_sql(self, request: ExecuteSQLInput) -> Result[ExecuteSQLData]:
        """Execute SQL query asynchronously with cancellation support.

        If ``request.execute_task_id`` is provided, it is used as-is and returned
        unchanged in ``ExecuteSQLData`` so the caller can cancel the execution
        via ``stop_execute_sql()``. Otherwise a server-generated UUID is used.
        """
        task_id = request.execute_task_id or str(uuid.uuid4())
        cancelled = threading.Event()

        async def _run() -> Result[ExecuteSQLData]:
            try:
<<<<<<< HEAD
                return await asyncio.to_thread(self._execute_sql_sync, request, task_id)
=======
                return await asyncio.to_thread(self._execute_sql_sync, request, task_id, policy_context, cancelled)
>>>>>>> b9f8253 ([Feature] Address table metadata and SQL execution per datasource (#1342))
            except asyncio.CancelledError:
                logger.info(f"SQL execution task cancelled: {task_id}")
                return Result(
                    success=False,
                    errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                    errorMessage="SQL execution was cancelled",
                )
            finally:
                self._cleanup_sql_task(task_id)

        with self._sql_tasks_lock:
            if task_id in self._sql_tasks:
                return Result(
                    success=False,
                    errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                    errorMessage=f"execute_task_id '{task_id}' is already in use",
                )
            task = asyncio.create_task(_run())
            self._sql_tasks[task_id] = task
            self._sql_cancels[task_id] = cancelled

        return await task

    async def stop_execute_sql(self, task_id: str) -> Result[StopExecuteSQLData]:
        """Stop a running SQL execution task.

        Args:
            task_id: The execute_task_id returned from execute_sql.

        Returns:
            Result indicating whether the task was stopped.
        """
        with self._sql_tasks_lock:
            task = self._sql_tasks.get(task_id)
            cancelled = self._sql_cancels.get(task_id)

        if not task:
            return Result(
                success=False,
                errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                errorMessage=f"No running SQL execution found for task ID: {task_id}",
                data=StopExecuteSQLData(execute_task_id=task_id, stopped=False),
            )

        if task.done():
            self._cleanup_sql_task(task_id)
            return Result(
                success=False,
                errorCode=ErrorCode.SQL_EXECUTION_ERROR,
                errorMessage="SQL execution has already completed",
                data=StopExecuteSQLData(execute_task_id=task_id, stopped=False),
            )

        # Raised before cancelling: the worker checks it once it owns the lock,
        # which is the only place a queued statement can still be stopped.
        if cancelled is not None:
            cancelled.set()
        task.cancel()
        logger.info(f"Cancellation requested for SQL execution task: {task_id}")
        return Result(
            success=True,
            data=StopExecuteSQLData(execute_task_id=task_id, stopped=True),
        )

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
                    "current_datasource": self.current_datasource,
                    "current_database": self.current_db_name,
                    "current_catalog": getattr(self.cli_context, "current_catalog", None) if self.cli_context else None,
                    "current_schema": getattr(self.cli_context, "current_schema", None) if self.cli_context else None,
                    "database": db_info,
                    "timestamp": now_utc_iso(),
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
                        from datus.storage.metric.store import MetricRAG

                        metrics_rag = MetricRAG(self.agent_config)
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
                        # Use real reference SQL RAG
                        from datus.storage.reference_sql.store import ReferenceSqlRAG

                        sql_rag = ReferenceSqlRAG(self.agent_config)
                        sql_count = sql_rag.get_reference_sql_size()
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
                    connections = self.db_manager.get_connections(self.current_datasource)
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
