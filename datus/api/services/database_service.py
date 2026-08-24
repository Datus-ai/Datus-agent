"""
Service for handling Database Management operations.
"""

import asyncio
import stat
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from datus_db_core import BaseSqlConnector

from datus.api.models.base_models import Result
from datus.api.models.config_models import ErrorCode
from datus.api.models.database_models import (
    DatabaseInfo,
    ListDatabasesData,
    ListDatabasesInput,
)
from datus.api.models.table_models import (
    ColumnInfo,
    GetSemanticModelData,
    GetTableDetailData,
    GetTablesColumnsData,
    SaveSemanticModelData,
    SaveSemanticModelInput,
    TableColumnBrief,
    TableColumns,
    TableDetailData,
    ValidateSemanticModelData,
    ValidateSemanticModelInput,
)
from datus.configuration.agent_config_loader import AgentConfig
from datus.storage.semantic_model.artifact_file import (
    artifact_revision,
    atomic_write_bytes,
    semantic_artifact_lock,
)
from datus.tools.db_tools.capabilities import supports_namespace
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.config_utils import coerce_positive_int, coerce_positive_seconds
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_table_name_parts
from datus.utils.text_utils import redact_uri
from datus.utils.time_utils import now_utc_iso

logger = get_logger(__name__)
# Database types that do NOT support schema switching
_NO_SCHEMA_TYPES = {"sqlite", "duckdb", "mysql"}
# Default cap on tables per /table/columns batch; override with
# ``api.max_prefetch_tables`` in agent.yml.
_DEFAULT_MAX_PREFETCH_TABLES = 500
# Wall-clock a single /table/columns batch may spend resolving uncached tables
# before it returns what it has; override with ``api.prefetch_budget_seconds``.
# A batch walks tables one at a time behind the connector lock, so on a slow
# source an unbounded one holds a to_thread worker for minutes.
_DEFAULT_PREFETCH_BUDGET_SECONDS = 3.0


def _brief_columns(columns: list[ColumnInfo]) -> list[TableColumnBrief]:
    """Slim the prefetch payload: no default_value, which no client reads."""
    return [TableColumnBrief(name=c.name, type=c.type, nullable=c.nullable, pk=c.pk) for c in columns]


class DatasourceService:
    """Service for handling datasource management operations."""

    def __init__(self, agent_config: Optional[AgentConfig] = None):
        """
        Initialize the database service.

        Args:
            agent_config: Datus agent configuration
        """
        self.agent_config = agent_config

        self.db_manager = DBManager(agent_config.datasource_configs)
        self.current_datasource = agent_config.current_datasource

        self.current_db_connector = None
        self.current_db_name = None
        # Connectors for the project's non-current datasources, resolved on
        # demand. A project can bind several, and the catalog tree, table detail
        # and SQL execution all address one by name — but only the current one is
        # worth opening up front, since a warehouse listing costs seconds.
        self._datasource_connectors: dict[str, tuple[BaseSqlConnector, str]] = {}
        # In-memory column cache keyed by datasource + resolved table identity,
        # so repeated table/detail + autocomplete prefetch requests don't re-hit
        # the source. The datasource has to be part of the key: two bound
        # warehouses routinely hold a same-named table, and without it the second
        # one would serve the first one's columns.
        # The lock serializes the not-thread-safe connector across concurrent
        # asyncio.to_thread detail/batch requests.
        self._columns_cache: dict[str, list[ColumnInfo]] = {}
        self._schema_lock = threading.Lock()
        # Only one prefetch batch resolves uncached tables at a time. Several in
        # parallel cannot go any faster — they serialize on _schema_lock anyway —
        # they just each pin a to_thread worker while queuing, which is what
        # starves interactive metadata reads.
        self._prefetch_gate = threading.BoundedSemaphore(1)
        self._initialize_connection()

    def _active_semantic_adapter(self) -> str:
        resolver = getattr(self.agent_config, "resolve_semantic_adapter", None)
        if callable(resolver):
            return str(resolver() or "").strip().lower()
        return ""

    def _is_osi_semantic_layer(self) -> bool:
        from datus.agent.node.semantic_authoring import is_osi_semantic_adapter

        return is_osi_semantic_adapter(self._active_semantic_adapter())

    @staticmethod
    def _validate_dosi_semantic_yaml(yaml_content: str) -> tuple[bool, List[str]]:
        try:
            document = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            return False, [str(exc)]
        if not isinstance(document, dict):
            return False, ["YAML document must be an object"]
        try:
            from datus_semantic_core.exceptions import SemanticCoreException
            from datus_semantic_dosi.authoring import validate_dosi_document
        except ImportError as exc:
            return False, [f"datus-semantic-dosi is required to validate Dosi semantic YAML: {exc}"]

        try:
            validate_dosi_document(document)
            return True, []
        except SemanticCoreException as exc:
            return False, [str(exc)]

    def _get_database_type(self, database_name: Optional[str] = None) -> tuple[str, str]:
        """
        Get database type from agent configuration.

        Args:
            database_name: Optional database name. If not provided, uses current database.

        Returns:
            Database type string (e.g., 'starrocks', 'mysql', etc.)
            db_name: Database name
        """
        db_type = "unknown"
        target_db = database_name or self.current_db_name

        try:
            if self.agent_config and self.current_datasource in self.agent_config.datasource_configs:
                db_config = self.agent_config.datasource_configs[self.current_datasource]
                db_type = db_config.type.value if hasattr(db_config.type, "value") else str(db_config.type)
        except Exception as e:
            logger.warning(f"Failed to get db type from config: {e}")

        return db_type, target_db

    def _initialize_connection(self):
        """Initialize the current database connection."""
        if self.db_manager and self.current_datasource:
            try:
                db_name, connector = self.db_manager.first_conn_with_name(self.current_datasource)
                self.current_db_connector = connector
                self.current_db_name = connector.database_name or db_name
            except Exception as e:
                logger.warning(f"Failed to initialize database connection: {e}")
                self.current_db_connector = None
                self.current_db_name = None

    def resolve_datasource(self, datasource: Optional[str] = None) -> str:
        """Normalize a requested datasource name, falling back to the current one."""
        return (datasource or "").strip() or (self.current_datasource or "")

    def _connector_for(self, datasource: Optional[str] = None) -> tuple[BaseSqlConnector, str]:
        """``(connector, database_name)`` for one of the project's datasources.

        The current datasource keeps using the connection opened at startup;
        anything else is opened on first use and remembered, because DBManager
        resolves a fresh wrapper per call and the column cache keys off the
        datasource name rather than the object.

        Raises ValueError for a name the config does not declare — callers turn
        that into an error response rather than silently answering from the
        wrong warehouse, which is what falling back to the current one would do.
        """
        key = self.resolve_datasource(datasource)
        if not key:
            raise ValueError("No datasource configured")

        if key == self.current_datasource:
            if self.current_db_connector is None:
                raise ValueError(f"Datasource {key!r} has no usable connection")
            return self.current_db_connector, self.current_db_name or ""

        configs = getattr(self.agent_config, "datasource_configs", {}) or {}
        if key not in configs:
            raise ValueError(f"Unknown datasource {key!r}")

        cached = self._datasource_connectors.get(key)
        if cached is not None:
            return cached

        try:
            # Declared but unreachable raises DatusException, not ValueError, so
            # normalize it — every caller of this method reports ValueError as a
            # clean error response.
            db_name, connector = self.db_manager.first_conn_with_name(key)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 — normalized for the callers
            raise ValueError(f"Datasource {key!r} has no usable connection: {e}") from e

        # Symmetric with the current-datasource branch above, which rejects a
        # missing connector rather than handing one back.
        if connector is None:
            raise ValueError(f"Datasource {key!r} has no usable connection")

        entry = (connector, connector.database_name or db_name or "")
        self._datasource_connectors[key] = entry
        return entry

    def _get_connection_info(
        self,
        connector,
        ds_id: str,
        request: ListDatabasesInput,
    ) -> List[DatabaseInfo]:
        """Get connection information for a database connector.

        Lists the database(s) this connector is scoped to — its configured
        database, or the whole server only when no database is configured —
        resolves schemas if supported, and marks the connector's configured
        database as ``current``. Request-level filters (database_name,
        schema_name, catalog_name) narrow the result set when provided.
        """
        dialect = getattr(connector, "dialect", "unknown")
        has_schema = supports_namespace("schema", connector=connector, dialect=dialect)
        catalog_name = request.catalog_name or getattr(connector, "catalog_name", None)
        now = now_utc_iso()

        def _disconnected(db_name: str) -> DatabaseInfo:
            return DatabaseInfo(
                name=db_name,
                uri=_get_uri(connector),
                type=dialect,
                current=(db_name == connector.database_name),
                catalog_name=catalog_name,
                schema_name=None,
                connection_status="disconnected",
                tables_count=None,
                last_accessed=now,
            )

        def _listing_failed(
            db_name: str, error: str, schema: Optional[str] = None, *, current: Optional[bool] = None
        ) -> DatabaseInfo:
            """The connection works but its objects could not be enumerated.

            Reporting this as ``disconnected`` used to hide the real cause and contradict
            the agent, which keeps querying the same database successfully.

            ``current`` overrides the database comparison for a root that is not a
            database: a schema-only dialect reports no ``database_name``, so the
            default comparison would call its own default schema "not current" on
            failure while the success path above calls it current.
            """
            return DatabaseInfo(
                name=db_name,
                uri=_get_uri(connector),
                type=dialect,
                current=(db_name == connector.database_name) if current is None else current,
                catalog_name=catalog_name,
                schema_name=schema,
                connection_status="connected",
                tables_count=None,
                last_accessed=now,
                error=error,
            )

        try:
            if not connector.test_connection():
                return [_disconnected(connector.database_name)]
        except Exception:
            logger.exception("Connection test failed for %s", connector.database_name)
            return [_disconnected(connector.database_name)]

        # 1) Resolve which databases to list — fatal if this fails since we have nothing
        # to iterate. A datasource is a connection profile scoped to its configured
        # database(s): get_connections() already yields one connector per config-known
        # database (a server datasource's configured ``database``, or one connector per
        # glob file). So list the database this connector is bound to, NOT every database
        # on the server — otherwise a single project datasource leaks the whole instance.
        try:
            if request.database_name:
                db_names = [request.database_name]
            elif connector.database_name:
                db_names = [connector.database_name]
            elif hasattr(connector, "get_databases"):
                # No database configured for this datasource — fall back to enumerating
                # the server so the user can still browse what the connection can reach.
                db_names = connector.get_databases(
                    catalog_name=catalog_name,
                    include_sys=request.include_sys_schemas,
                )
            else:
                db_names = []
        except Exception as e:
            logger.warning("Failed to enumerate databases for %s: %s", connector.database_name, e)
            return [_listing_failed(connector.database_name, str(e))]

        db_infos: List[DatabaseInfo] = []

        # 1b) A dialect with no database level (Oracle: schema is the only
        # namespace) yields nothing above — its connector reports no database and
        # its get_databases() is empty by design. Iterating that dropped the whole
        # datasource out of the catalog while its tables were perfectly listable.
        # Its schemas ARE the top level, so report them as the roots: a table then
        # addresses as schema.table, which is exactly the identifier such a dialect
        # accepts. Only when nothing else produced a root, so a dialect that does
        # report a database keeps its existing shape.
        if not db_names and has_schema and not supports_namespace("database", connector=connector, dialect=dialect):
            try:
                schemas = (
                    [request.schema_name]
                    if request.schema_name
                    else connector.get_schemas(
                        catalog_name=request.catalog_name,
                        include_sys=request.include_sys_schemas,
                    )
                )
            except Exception as e:
                logger.warning("Failed to get schemas for schema-only dialect=%s: %s", dialect, e)
                # The only root we can name here is the connector's own schema.
                return [_listing_failed(connector.schema_name, str(e), current=True)]

            for schema in schemas:
                try:
                    tables = connector.get_tables(catalog_name=catalog_name, schema_name=schema)
                    tables.sort()
                except Exception as e:
                    logger.warning("Failed to get tables for schema=%s dialect=%s: %s", schema, dialect, e)
                    db_infos.append(_listing_failed(schema, str(e), current=(schema == connector.schema_name)))
                    continue

                db_infos.append(
                    DatabaseInfo(
                        name=schema,
                        uri=_get_uri(connector),
                        type=dialect,
                        current=(schema == connector.schema_name),
                        catalog_name=catalog_name,
                        # Already the root: nesting it under itself would render the
                        # same name twice in a catalog tree.
                        schema_name=None,
                        connection_status="connected",
                        tables_count=len(tables),
                        last_accessed=now,
                        tables=tables,
                    )
                )
            return db_infos

        for db_name in db_names:
            if has_schema:
                # 2) Resolve schemas for this db — a single failing db must not
                # abort the whole listing. Report the db with its listing error
                # and keep going.
                if request.schema_name:
                    schemas = [request.schema_name]
                elif hasattr(connector, "get_schemas"):
                    try:
                        schemas = connector.get_schemas(
                            catalog_name=request.catalog_name,
                            database_name=db_name,
                            include_sys=request.include_sys_schemas,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to get schemas for db=%s dialect=%s: %s",
                            db_name,
                            dialect,
                            e,
                        )
                        db_infos.append(_listing_failed(db_name, str(e)))
                        continue
                else:
                    schemas = ["public"]

                for schema in schemas:
                    # 3) Fetch tables for this (db, schema). A failure here only
                    # invalidates this entry, not sibling schemas.
                    try:
                        tables = connector.get_tables(
                            catalog_name=catalog_name, database_name=db_name, schema_name=schema
                        )
                        tables.sort()
                    except Exception as e:
                        logger.warning(
                            "Failed to get tables for db=%s schema=%s: %s",
                            db_name,
                            schema,
                            e,
                        )
                        db_infos.append(_listing_failed(db_name, str(e), schema=schema))
                        continue

                    db_infos.append(
                        DatabaseInfo(
                            name=db_name,
                            uri=_get_uri(connector),
                            type=dialect,
                            current=(db_name == connector.database_name),
                            catalog_name=catalog_name,
                            schema_name=schema,
                            connection_status="connected",
                            tables_count=len(tables),
                            last_accessed=now,
                            tables=tables,
                        )
                    )
            else:
                # No schema support — get tables directly. Isolate per-db failures.
                try:
                    tables = connector.get_tables(
                        catalog_name=catalog_name, database_name=db_name, schema_name=request.schema_name
                    )
                    tables.sort()
                except Exception as e:
                    logger.warning("Failed to get tables for db=%s: %s", db_name, e)
                    db_infos.append(_listing_failed(db_name, str(e)))
                    continue

                db_infos.append(
                    DatabaseInfo(
                        name=db_name,
                        uri=_get_uri(connector),
                        type=dialect,
                        current=(db_name == connector.database_name),
                        catalog_name=catalog_name,
                        schema_name=None,
                        connection_status="connected",
                        tables_count=len(tables),
                        last_accessed=now,
                        tables=tables,
                    )
                )
        return db_infos

    def list_databases(self, request: ListDatabasesInput) -> Result[ListDatabasesData]:
        """
        List available databases.

        Args:
            request: List databases request

        Returns:
            ListDatabasesResult with databases list
        """
        # FIXME try use project_id
        try:
            if not self.db_manager:
                return Result(
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Database manager not initialized",
                )

            # Get connections from the specified datasource
            datasource = self.resolve_datasource(request.datasource_id)
            configs = getattr(self.agent_config, "datasource_configs", {}) or {}
            # Refuse an unknown name instead of quietly listing the current
            # datasource under it: a client rendering several datasources side by
            # side would file the answer under the wrong tree node.
            if datasource and datasource not in configs:
                return Result(
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"Unknown datasource '{datasource}'",
                )

            connections = self.db_manager.get_connections(datasource)

            databases = []
            # Handle both single connector and dictionary of connectors
            if isinstance(connections, dict):
                for _db_name, connector in connections.items():
                    db_info = self._get_connection_info(connector, _db_name, request)
                    databases.extend(db_info)
            else:
                # Single connector case
                db_info = self._get_connection_info(connections, datasource, request)
                databases.extend(db_info)

            # Stamped here rather than at each of the six DatabaseInfo call
            # sites: every row in this response came from the one datasource
            # resolved above.
            for info in databases:
                info.datasource = datasource

            data = ListDatabasesData(
                databases=databases,
                total_count=len(databases),
                current_database=self.current_db_name,
                current_datasource=datasource,
            )

            return Result(success=True, data=data)

        except Exception as e:
            logger.error(f"Failed to list databases: {e}", exc_info=True)
            return Result(
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    def _resolve_table_identity(
        self,
        full_path: str,
        connector: BaseSqlConnector,
        default_database: str,
    ) -> tuple[str, str, str, str]:
        """Resolve a dotted reference to (catalog, database, schema, table).

        Connector defaults fill in whatever the caller left out. For StarRocks
        that is catalog.database.table, with no schema level. The connector is
        passed in rather than read off ``self`` because the reference may address
        a datasource other than the current one — and the defaults that complete
        a partial name differ per connection profile.
        """
        name_parts = parse_table_name_parts(full_path, connector.get_type())
        catalog_name = name_parts["catalog_name"] or getattr(connector, "catalog_name", "")
        database_name = name_parts["database_name"] or default_database or getattr(connector, "database", "")
        schema_name = name_parts["schema_name"] or getattr(connector, "schema_name", "")
        return catalog_name, database_name, schema_name, name_parts["table_name"]

    def _cache_key(self, datasource: str, identity: tuple[str, str, str, str]) -> str:
        """Column-cache key. The datasource leads: two bound warehouses commonly
        hold a same-named table, and a shared key would serve one's columns for
        the other."""
        catalog_name, database_name, schema_name, table_name = identity
        return f"{datasource}\t{catalog_name}.{database_name}.{schema_name}.{table_name}"

    def _cached_columns(self, full_path: str, datasource: Optional[str] = None) -> Optional[list[ColumnInfo]]:
        """Columns already in the cache for this reference, without touching the source."""
        try:
            connector, default_database = self._connector_for(datasource)
            identity = self._resolve_table_identity(full_path, connector, default_database)
        except Exception:
            # Unparseable name or unresolvable datasource — let the per-table
            # path report it.
            return None
        return self._columns_cache.get(self._cache_key(self.resolve_datasource(datasource), identity))

    def get_table_schema(self, full_path: str, datasource: Optional[str] = None) -> Result[GetTableDetailData]:
        """
        Get table schema details.

        Args:
            full_path: table name, [catalog.][database.][schema.]table
            datasource: which configured datasource to resolve it against;
                defaults to the current one

        Returns:
            GetTableSchemaResult with table schema
        """
        try:
            try:
                connector, default_database = self._connector_for(datasource)
            except ValueError as e:
                return Result(
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=str(e),
                )

            try:
                identity = self._resolve_table_identity(full_path, connector, default_database)
                table_name = identity[3]
                cache_key = self._cache_key(self.resolve_datasource(datasource), identity)

                def _detail(cols: list[ColumnInfo]) -> Result[GetTableDetailData]:
                    return Result(
                        success=True,
                        data=GetTableDetailData(table=TableDetailData(name=table_name, columns=cols, indexes=[])),
                    )

                cached = self._columns_cache.get(cache_key)
                if cached is not None:
                    return _detail(cached)

                # Serialize the not-thread-safe connector; re-check the cache
                # inside the lock (double-checked) so each table is fetched once
                # even under concurrent detail/batch requests.
                with self._schema_lock:
                    cached = self._columns_cache.get(cache_key)
                    if cached is not None:
                        return _detail(cached)

                    schema_info = connector.get_schema(
                        catalog_name=identity[0],
                        database_name=identity[1],
                        schema_name=identity[2],
                        table_name=table_name,
                    )
                    if not schema_info:
                        return Result(
                            success=False,
                            errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                            errorMessage=f"Table '{table_name}' not found or schema not available",
                        )

                    # Convert schema info to ColumnInfo objects
                    columns: list[ColumnInfo] = []
                    if isinstance(schema_info, list):
                        for _i, col in enumerate(schema_info):
                            if isinstance(col, dict):
                                column_info = ColumnInfo(
                                    name=col.get("name", ""),
                                    type=col.get("type", ""),
                                    nullable=(
                                        bool(col["nullable"])
                                        if "nullable" in col
                                        else col.get("notnull", 1) == 0  # SQLite: notnull=0 means nullable
                                    ),
                                    default_value=col.get("default_value", col.get("dflt_value")),
                                    pk=bool(col.get("pk", False)),
                                )
                            else:
                                # Handle string or other formats
                                column_info = ColumnInfo(
                                    name=str(col),
                                    type="TEXT",
                                    nullable=True,
                                    default_value=None,
                                    pk=False,
                                )
                            columns.append(column_info)

                    self._columns_cache[cache_key] = columns
                    return _detail(columns)

            except Exception as e:
                return Result(
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Failed to get table schema: {str(e)}",
                )

        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    def get_tables_columns(self, tables: list[str], datasource: Optional[str] = None) -> Result[GetTablesColumnsData]:
        """Batch-fetch columns for multiple tables (autocomplete prefetch).

        Reuses get_table_schema (and its column cache) per table. Tables that
        fail to resolve are omitted rather than failing the whole batch.

        Prefetch is a convenience, so it yields rather than competes. Three
        bounds keep one caller from monopolising the datasource, because the
        columns of N tables cost N serial round-trips behind the connector lock —
        on a slow source (Oracle listing every schema) that is minutes of held
        lock and a to_thread worker, and everything else needing table metadata,
        interactive reads included, waits behind it:

        - the request size is capped at ``api.max_prefetch_tables``;
        - only one batch resolves uncached tables at a time, and a second does
          not queue — it returns what the cache already holds;
        - a batch stops once ``api.prefetch_budget_seconds`` is spent.

        Under the last two, tables are left out of the response. That is the
        contract a table failing to resolve already has, so a client re-asks for
        what it still needs — one table at a time, via /table/detail.
        """
        api_config = getattr(self.agent_config, "api_config", {}) or {}
        max_tables = coerce_positive_int(api_config.get("max_prefetch_tables"), _DEFAULT_MAX_PREFETCH_TABLES)
        if len(tables) > max_tables:
            return Result(
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=f"Too many tables requested ({len(tables)}); max is {max_tables}",
            )

        results: list[TableColumns] = []
        pending: list[str] = []
        for full_path in tables:
            cached = self._cached_columns(full_path, datasource)
            if cached is None:
                pending.append(full_path)
            else:
                results.append(TableColumns(table=full_path, columns=_brief_columns(cached)))

        # Cache hits cost nothing, so they are served regardless of the bounds
        # below — and a fully-warm batch never even takes the gate.
        if not pending:
            return Result(success=True, data=GetTablesColumnsData(tables=results))

        if not self._prefetch_gate.acquire(blocking=False):
            logger.info(
                "Prefetch already in flight; serving %d cached of %d tables and omitting %d",
                len(results),
                len(tables),
                len(pending),
            )
            return Result(success=True, data=GetTablesColumnsData(tables=results))

        try:
            budget = coerce_positive_seconds(
                api_config.get("prefetch_budget_seconds"), _DEFAULT_PREFETCH_BUDGET_SECONDS
            )
            deadline = time.monotonic() + budget
            for index, full_path in enumerate(pending):
                if time.monotonic() >= deadline:
                    logger.info(
                        "Prefetch budget of %.1fs spent after %d tables; omitting the remaining %d",
                        budget,
                        index,
                        len(pending) - index,
                    )
                    break

                detail = self.get_table_schema(full_path, datasource)
                if detail.success and detail.data is not None:
                    results.append(TableColumns(table=full_path, columns=_brief_columns(detail.data.table.columns)))
        finally:
            self._prefetch_gate.release()

        return Result(success=True, data=GetTablesColumnsData(tables=results))

    def _semantic_models_root(self) -> Optional[Path]:
        """Return the semantic-model root shared by every datasource."""

        from datus.agent.node.semantic_authoring import osi_semantic_models_root

        root = osi_semantic_models_root(self.agent_config)
        return root.expanduser().resolve(strict=False) if root is not None else None

    def _semantic_model_display_path(self, path: Path) -> str:
        """Return the stable project-relative path exposed by the API."""

        root = self._semantic_models_root()
        if root is None:
            return str(path)
        relative = path.resolve(strict=False).relative_to(root)
        return (Path("subject") / "semantic_models" / relative).as_posix()

    def _resolve_semantic_model_file(self, semantic_model_file: str) -> Path:
        """Resolve an API file selector without trusting client absolute paths.

        The selector is resolved against the whole ``subject/semantic_models``
        tree rather than the active datasource's subdirectory, so every
        datasource configured for the project is addressable.
        """

        selector = str(semantic_model_file or "").strip()
        if not selector:
            raise ValueError("semantic_model_file is required")
        raw_path = Path(selector).expanduser()
        if raw_path.is_absolute():
            raise ValueError("semantic_model_file must be a project-relative path")

        root = self._semantic_models_root()
        if root is None:
            raise ValueError("The project semantic-model directory is unavailable")
        # Only the canonical project-relative form is stripped; anything else is
        # taken as-is under the root. Accepting a bare ``semantic_models/`` alias
        # too would be ambiguous for a datasource named ``semantic_models``.
        parts = raw_path.parts
        if parts[:2] == ("subject", "semantic_models"):
            parts = parts[2:]
        if not parts:
            raise ValueError("semantic_model_file must identify a YAML file")

        candidate = root.joinpath(*parts).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("semantic_model_file escapes the semantic-model directory") from exc
        if candidate.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError("semantic_model_file must be a .yml or .yaml file")
        if not candidate.is_file():
            raise ValueError(f"Semantic model file not found: {selector}")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError("semantic_model_file resolves outside the semantic-model directory") from exc
        return candidate

    def _semantic_model_datasource(self, path: Path) -> str:
        """Return the datasource directory a resolved artifact sits under."""

        root = self._semantic_models_root()
        if root is None:
            return ""
        relative = path.resolve(strict=False).relative_to(root)
        return relative.parts[0] if len(relative.parts) > 1 else ""

    def _resolve_writable_semantic_model_file(self, semantic_model_file: str) -> Path:
        """Resolve a save/validate target, refusing another datasource's model.

        Reads span the whole tree, but every write-side stage is built around
        ``current_datasource``: metricflow validation takes it as an argument,
        the OSI adapter resolves the target inside the active datasource's
        inventory, and the knowledge-base sync stamps its rows with it. Writing
        another datasource's artifact would therefore file its rows under the
        wrong datasource, so reject it with a message that says what to do.
        Threading a per-request datasource through those stages is the real
        fix and needs its own change.
        """

        path = self._resolve_semantic_model_file(semantic_model_file)
        datasource = self._semantic_model_datasource(path)
        active = str(getattr(self.agent_config, "current_datasource", "") or "").strip()
        if datasource and active and datasource != active:
            raise ValueError(
                f"semantic_model_file belongs to datasource {datasource!r}, but {active!r} is active; "
                "switch the active datasource before saving this model"
            )
        return path

    @staticmethod
    def _osi_candidate_identity(yaml_content: str) -> tuple[Optional[str], bool, Optional[str]]:
        """Return ``(model_name, has_metrics, error)`` for one OSI API candidate."""

        try:
            document = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            return None, False, str(exc)
        models = document.get("semantic_model") if isinstance(document, dict) else None
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            return None, False, "OSI YAML must contain exactly one semantic_model object"
        model_name = str(models[0].get("name") or "").strip()
        if not model_name:
            return None, False, "semantic_model[0].name is required"
        metrics = models[0].get("metrics")
        if metrics is not None and not isinstance(metrics, list):
            return None, False, "semantic_model[0].metrics must be a list"
        return model_name, bool(metrics), None

    @staticmethod
    def _revision_matches(expected_revision: str, current_revision: str) -> bool:
        expected = str(expected_revision or "").strip().lower()
        if not expected:
            return True
        if not expected.startswith("sha256:"):
            expected = f"sha256:{expected}"
        return expected == current_revision.lower()

    def _validate_semantic_content(
        self,
        request: ValidateSemanticModelInput,
        semantic_model_path: Path,
    ) -> tuple[bool, List[str], Optional[str], bool]:
        """Validate submitted content without mutating the live artifact."""
        del semantic_model_path

        is_valid, errors = self._validate_dosi_semantic_yaml(request.yaml)
        model_name, has_metrics, identity_error = self._osi_candidate_identity(request.yaml)
        if identity_error:
            errors = [*errors, identity_error]
            is_valid = False
        requested_name = str(request.semantic_model_name or "").strip()
        if requested_name and model_name and requested_name != model_name:
            errors = [
                *errors,
                f"The YAML declares semantic model {model_name!r}, but the requested target is {requested_name!r}",
            ]
            is_valid = False
        return is_valid, errors, model_name, has_metrics

    @staticmethod
    def _full_osi_validation(
        agent_config: AgentConfig,
        *,
        semantic_model_name: str,
        has_metrics: bool,
    ) -> tuple[bool, Dict[str, Any], str]:
        """Run the adapter's complete checks against the newly written target."""

        from datus.tools.func_tool.semantic_tools import SemanticTools

        result = SemanticTools(agent_config).validate_semantic(
            scope="all" if has_metrics else "semantic_model",
            semantic_model_name=semantic_model_name,
        )
        payload = result.result if isinstance(result.result, dict) else {}
        return bool(result.success and payload.get("valid", False)), payload, str(result.error or "")

    def get_semantic_model(self, semantic_model_file: str) -> Result[GetSemanticModelData]:
        """Read one SemanticModel YAML artifact.

        Args:
            semantic_model_file: Project-relative path under ``subject/semantic_models``.

        Returns:
            Result[GetSemanticModelData] with the YAML plus the identity and
            revision a later save needs.
        """
        try:
            semantic_model_path = self._resolve_semantic_model_file(semantic_model_file)
        except ValueError as exc:
            return Result[GetSemanticModelData](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=str(exc),
            )

        try:
            content = semantic_model_path.read_bytes()
            yaml_content = content.decode("utf-8")
            declared_name = None
            try:
                document = yaml.safe_load(yaml_content)
                declared = document.get("semantic_model") if isinstance(document, dict) else None
                if isinstance(declared, list) and len(declared) == 1 and isinstance(declared[0], dict):
                    declared_name = str(declared[0].get("name") or "").strip() or declared_name
                elif isinstance(declared, dict):
                    declared_name = str(declared.get("name") or "").strip() or declared_name
            except yaml.YAMLError:
                pass

            return Result[GetSemanticModelData](
                success=True,
                data=GetSemanticModelData(
                    yaml=yaml_content,
                    semantic_model_name=declared_name,
                    semantic_model_file=self._semantic_model_display_path(semantic_model_path),
                    revision=artifact_revision(content),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get semantic model: {e}")
            return Result[GetSemanticModelData](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def save_semantic_model(self, request: SaveSemanticModelInput) -> Result[SaveSemanticModelData]:
        """Atomically save and reconcile one semantic model artifact."""

        try:
            return await asyncio.to_thread(self._save_semantic_model_sync, request)
        except Exception as exc:
            logger.error("Failed to save semantic model: %s", exc, exc_info=True)
            return Result[SaveSemanticModelData](
                success=False,
                errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                errorMessage=str(exc),
            )

    def _save_semantic_model_sync(self, request: SaveSemanticModelInput) -> Result[SaveSemanticModelData]:
        # Semantic authoring is Dosi-only: refuse before reading, validating,
        # or writing anything so no half-saved artifact can exist.
        if not self._is_osi_semantic_layer():
            from datus.agent.node.semantic_authoring import QUERY_ONLY_MIGRATION_MESSAGE

            return Result[SaveSemanticModelData](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=QUERY_ONLY_MIGRATION_MESSAGE,
            )
        try:
            semantic_model_path = self._resolve_writable_semantic_model_file(request.semantic_model_file)
        except ValueError as exc:
            return Result[SaveSemanticModelData](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=str(exc),
            )

        display_path = self._semantic_model_display_path(semantic_model_path)
        candidate_content = request.yaml.encode("utf-8")
        with semantic_artifact_lock(semantic_model_path):
            try:
                original_exists = semantic_model_path.exists()
                if original_exists:
                    original_content = semantic_model_path.read_bytes()
                    original_mode = stat.S_IMODE(semantic_model_path.stat().st_mode)
                else:
                    original_content = b""
                    original_mode = 0o644
            except OSError as exc:
                return Result[SaveSemanticModelData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Failed to read semantic model file: {exc}",
                )

            original_revision = artifact_revision(original_content)
            if not self._revision_matches(request.expected_revision or "", original_revision):
                return Result[SaveSemanticModelData](
                    success=False,
                    data=SaveSemanticModelData(
                        status="conflict",
                        yaml_saved=False,
                        kb_synced=False,
                        semantic_model_name=request.semantic_model_name,
                        semantic_model_file=display_path,
                        revision=original_revision,
                        retryable=False,
                        failed_stage="revision",
                    ),
                    errorCode=ErrorCode.SEMANTIC_MODEL_REVISION_CONFLICT,
                    errorMessage="Semantic model has changed since it was loaded",
                )

            try:
                is_valid, validation_errors, model_name, has_metrics = self._validate_semantic_content(
                    request,
                    semantic_model_path,
                )
            except Exception as exc:
                logger.error("Failed to validate semantic model candidate: %s", exc, exc_info=True)
                return Result[SaveSemanticModelData](
                    success=False,
                    data=SaveSemanticModelData(
                        status="validation_failed",
                        yaml_saved=False,
                        kb_synced=False,
                        semantic_model_name=request.semantic_model_name,
                        semantic_model_file=display_path,
                        revision=original_revision,
                        retryable=True,
                        failed_stage="validation",
                        validation={"valid": False, "issues": [str(exc)]},
                    ),
                    errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                    errorMessage=f"Semantic validation could not be completed: {exc}",
                )
            if not is_valid:
                return Result[SaveSemanticModelData](
                    success=False,
                    data=SaveSemanticModelData(
                        status="validation_failed",
                        yaml_saved=False,
                        kb_synced=False,
                        semantic_model_name=model_name or request.semantic_model_name,
                        semantic_model_file=display_path,
                        revision=original_revision,
                        failed_stage="validation",
                        validation={"valid": False, "issues": validation_errors},
                    ),
                    errorCode=ErrorCode.SEMANTIC_MODEL_INVALID,
                    errorMessage="; ".join(validation_errors),
                )

            candidate_revision = artifact_revision(candidate_content)
            content_changed = candidate_content != original_content
            try:
                if content_changed:
                    atomic_write_bytes(semantic_model_path, candidate_content, mode=original_mode)
            except OSError as exc:
                return Result[SaveSemanticModelData](
                    success=False,
                    errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                    errorMessage=f"Failed to write semantic model file: {exc}",
                )

            validation_payload: Dict[str, Any] = {"valid": True, "issues": []}
            validation_retryable = False
            assert model_name is not None
            try:
                valid, validation_payload, validation_error = self._full_osi_validation(
                    self.agent_config,
                    semantic_model_name=model_name,
                    has_metrics=has_metrics,
                )
            except Exception as exc:
                logger.error("Full semantic validation could not be completed: %s", exc, exc_info=True)
                valid = False
                validation_retryable = True
                validation_error = f"Semantic validation could not be completed: {exc}"
                validation_payload = {"valid": False, "issues": [str(exc)]}
            if not valid:
                if content_changed:
                    try:
                        if original_exists:
                            atomic_write_bytes(semantic_model_path, original_content, mode=original_mode)
                        else:
                            semantic_model_path.unlink(missing_ok=True)
                    except OSError as restore_exc:
                        logger.error(
                            "Failed to restore semantic model %s after validation failure: %s",
                            semantic_model_path,
                            restore_exc,
                            exc_info=True,
                        )
                        return Result[SaveSemanticModelData](
                            success=False,
                            errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                            errorMessage=(
                                f"Semantic validation failed and the original YAML could not be restored: {restore_exc}"
                            ),
                        )
                issues = validation_payload.get("issues") or []
                error_message = validation_error or "; ".join(
                    str(issue.get("message") or issue) if isinstance(issue, dict) else str(issue) for issue in issues
                )
                return Result[SaveSemanticModelData](
                    success=False,
                    data=SaveSemanticModelData(
                        status="validation_failed",
                        yaml_saved=False,
                        kb_synced=False,
                        semantic_model_name=model_name,
                        semantic_model_file=display_path,
                        revision=original_revision,
                        retryable=validation_retryable,
                        failed_stage="validation",
                        validation=validation_payload,
                    ),
                    errorCode=(
                        ErrorCode.INTERNAL_COMMAND_ERROR if validation_retryable else ErrorCode.SEMANTIC_MODEL_INVALID
                    ),
                    errorMessage=error_message or "Semantic validation failed",
                )

            try:
                from datus.tools.func_tool.generation_tools import GenerationTools

                sync_result = GenerationTools(
                    agent_config=self.agent_config,
                    authoring_format="osi",
                ).sync_osi_to_db(
                    str(semantic_model_path),
                    include_semantic_objects=True,
                    include_metrics=True,
                )
            except Exception as exc:
                logger.error("Failed to sync semantic model %s: %s", semantic_model_path, exc, exc_info=True)
                sync_result = {"success": False, "error": str(exc)}

            if not sync_result.get("success", False):
                return Result[SaveSemanticModelData](
                    success=False,
                    data=SaveSemanticModelData(
                        status="saved_not_synced",
                        yaml_saved=True,
                        kb_synced=False,
                        semantic_model_name=model_name or request.semantic_model_name,
                        semantic_model_file=display_path,
                        revision=candidate_revision,
                        retryable=True,
                        failed_stage="knowledge_base",
                        validation=validation_payload,
                        sync=sync_result,
                    ),
                    errorCode=ErrorCode.SEMANTIC_MODEL_SYNC_FAILED,
                    errorMessage=(
                        "YAML was saved, but Knowledge Base synchronization failed: "
                        f"{sync_result.get('error', 'Unknown error')}"
                    ),
                )

            return Result[SaveSemanticModelData](
                success=True,
                data=SaveSemanticModelData(
                    status="synced",
                    yaml_saved=True,
                    kb_synced=True,
                    semantic_model_name=model_name or request.semantic_model_name,
                    semantic_model_file=display_path,
                    revision=candidate_revision,
                    validation=validation_payload,
                    sync=sync_result,
                ),
            )

    async def validate_semantic_model(self, request: ValidateSemanticModelInput) -> Result[ValidateSemanticModelData]:
        """Validate submitted SemanticModel YAML without changing the live artifact."""
        logger.info("Validating semantic model YAML")
        if not self._is_osi_semantic_layer():
            from datus.agent.node.semantic_authoring import QUERY_ONLY_MIGRATION_MESSAGE

            return Result[ValidateSemanticModelData](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=QUERY_ONLY_MIGRATION_MESSAGE,
            )
        try:
            # Same addressing rule as save: validating a file you would not be
            # allowed to save is a confusing asymmetry.
            semantic_model_path = self._resolve_writable_semantic_model_file(request.semantic_model_file)
            is_valid, error_messages, _model_name, _has_metrics = self._validate_semantic_content(
                request,
                semantic_model_path,
            )

            if not is_valid:
                return Result[ValidateSemanticModelData](
                    success=True,
                    data=ValidateSemanticModelData(valid=False, invalid_message=error_messages),
                )

            return Result[ValidateSemanticModelData](
                success=True,
                data=ValidateSemanticModelData(valid=True, invalid_message=None),
            )
        except ValueError as e:
            return Result[ValidateSemanticModelData](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=str(e),
            )
        except Exception as e:
            logger.error(f"Failed to validate semantic model: {e}")
            return Result[ValidateSemanticModelData](
                success=False,
                errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                errorMessage=str(e),
            )


def _get_uri(connector: BaseSqlConnector) -> str:
    if not connector:
        return ""
    connection_string = getattr(connector, "connection_string", "")
    if connection_string:
        return redact_uri(connection_string)
    return f"{connector.dialect}://"
