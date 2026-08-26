# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import csv
import io
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

from agents import Tool
from datus_db_core import BaseSqlConnector

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.schemas.node_models import ExecuteSQLResult
from datus.storage.kb_retrieval import metadata_fts_enabled
from datus.storage.schema_metadata import create_metadata_rag
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.storage.semantic_dataset.store import SemanticDatasetRAG
from datus.tools.db_tools.capabilities import (
    get_dialect_operations,
    get_effective_capabilities,
    supports_namespace,
)
from datus.tools.db_tools.data_file_loader import (
    LOCAL_FILES_DATASOURCE,
    DataFileError,
    default_conversion_cache_dir,
    find_file_reading_functions,
    inspect_file,
    load_file,
    quote_identifier,
    registered_objects,
    unresolved_table_references,
)
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.compress_utils import DataCompressor
from datus.utils.config_utils import coerce_bool
from datus.utils.constants import DBType, SQLType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.mcp_decorators import mcp_tool, mcp_tool_class
from datus.utils.sql_utils import parse_dialect, parse_table_name_parts

logger = get_logger(__name__)

# One warning per project is enough: DBFuncTool is rebuilt per session and per
# sub-agent, and repeating this on every construction would bury it.
_STALE_PROJECTION_WARNED: Set[str] = set()


def _warn_once_if_projection_is_stale(agent_config: Any) -> None:
    """Tell the user their semantic YAML has not been projected yet."""
    project = str(getattr(agent_config, "project_name", "") or "")
    if project in _STALE_PROJECTION_WARNED:
        return
    try:
        from datus.storage.semantic_dataset.store import SYNC_YAML_HINT, semantic_projection_is_stale

        if semantic_projection_is_stale(agent_config):
            _STALE_PROJECTION_WARNED.add(project)
            logger.warning(SYNC_YAML_HINT)
    except Exception as exc:  # noqa: BLE001 - a hint must never break tool setup
        logger.debug(f"Unable to check the semantic dataset projection: {exc}")


@dataclass
class TableCoordinate:
    catalog: str = ""
    database: str = ""
    schema: str = ""
    table: str = ""


@dataclass(frozen=True)
class ScopedTablePattern:
    raw: str
    catalog: str = ""
    database: str = ""
    schema: str = ""
    table: str = ""

    def matches(self, coordinate: TableCoordinate) -> bool:
        return all(
            _pattern_matches(getattr(self, field), getattr(coordinate, field))
            for field in ("catalog", "database", "schema", "table")
        )


def _pattern_matches(pattern: str, value: str) -> bool:
    if not pattern or pattern in ("*", "%"):
        return True
    if not value:
        # Empty value means the field could not be resolved from either the SQL
        # or connector defaults (e.g. catalog_name not set).  Treat as a wildcard
        # so that scope checking only enforces fields we can actually verify.
        return True
    normalized_pattern = pattern.replace("%", "*")
    return fnmatchcase(value, normalized_pattern)


@mcp_tool_class(
    name="db_tool",
    availability_property="has_db_tools",
)
class DBFuncTool:
    """
    Database function tool that supports dynamic connector switching.

    This class can work in two modes:
    1. Single connector mode (legacy): Pass a single BaseSqlConnector
    2. Multi-connector mode: Pass a DBManager with datasource for dynamic connector lookup

    In multi-connector mode, connectors are cached with LRU eviction to avoid
    repeated lookups while limiting memory usage.
    """

    permission_category: str = "db_tools"

    DEFAULT_CONNECTOR_CACHE_SIZE = 8

    @classmethod
    def create_dynamic(cls, agent_config: AgentConfig, sub_agent_name: Optional[str] = None) -> "DBFuncTool":
        """Create DBFuncTool instance (required by mcp_tool_class contract)."""
        return cls(agent_config=agent_config, sub_agent_name=sub_agent_name)

    @classmethod
    def create_static(
        cls,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        database_name: Optional[str] = None,
    ) -> "DBFuncTool":
        """Create DBFuncTool instance with optional physical database (required by mcp_tool_class contract)."""
        return cls(agent_config=agent_config, default_database=database_name or None, sub_agent_name=sub_agent_name)

    def __init__(
        self,
        connector_or_manager: Union[BaseSqlConnector, DBManager, None] = None,
        agent_config: Optional[AgentConfig] = None,
        *,
        default_datasource: Optional[str] = None,
        default_database: Optional[str] = None,
        sub_agent_name: Optional[str] = None,
        scoped_tables: Optional[Iterable[str]] = None,
        connector_cache_size: int = DEFAULT_CONNECTOR_CACHE_SIZE,
        read_only: bool = False,
        filesystem_root: Optional[str] = None,
    ):
        """
        Initialize DBFuncTool.

        Args:
            connector_or_manager: A single BaseSqlConnector (legacy mode), a DBManager (multi-connector mode),
                                  or None to auto-create a DBManager from agent_config.
            agent_config: Agent configuration (required when connector_or_manager is None or DBManager)
            default_datasource: Datasource key (top-level ``services.datasources`` entry). Overrides
                                ``agent_config.current_datasource`` for connector routing.
            default_database: Physical database name. Metadata only (the connector targets the database
                              configured for its datasource); defaults to the datasource config's ``database``.
            sub_agent_name: Optional sub-agent name for scoped context
            scoped_tables: Optional explicit table scope patterns
            connector_cache_size: Max connectors to cache (LRU eviction), default 8
            filesystem_root: Root that ``load_file_as_table`` resolves relative paths
                       against. Pass the node's own workspace root: a node configured
                       with ``node_config["workspace_root"]`` anchors its filesystem
                       tools there, so leaving this unset would make a path the model
                       just got back from ``glob`` resolve against a different root
                       here. Defaults to ``agent_config.project_root``, which is the
                       same value whenever no override is configured.
            read_only: When True, the write paths (``execute_sql``,
                       ``execute_write``, ``execute_ddl``,
                       ``transfer_query_result``) hard-reject any non-read
                       statement at the tool layer, independent of
                       ``PermissionHooks``. Use for agents whose contract is
                       read-only (Explore, ask_report/dashboard, LLM validators)
                       so the unified write-capable entry point cannot mutate the
                       datasource even when hooks are bypassed (e.g. validators
                       run with ``hooks=None``) or under a permissive profile.
                       This is the per-instance floor only; the ``read_only``
                       property ORs it with ``AgentConfig.sql_read_only``, so
                       passing False does not make the tool writable on a
                       hardened deployment.
        """
        if connector_or_manager is None:
            if not agent_config:
                raise ValueError("agent_config is required when connector_or_manager is not provided")
            connector_or_manager = db_manager_instance(agent_config.datasource_configs)

        # Determine mode based on input type
        if isinstance(connector_or_manager, DBManager):
            if not agent_config:
                raise ValueError("agent_config is required when using DBManager mode")
            self._db_manager = connector_or_manager
            self._default_datasource = default_datasource or (agent_config.current_datasource if agent_config else "")
            self._default_database = default_database or ""
            self._datasources = list(agent_config.current_db_configs().keys()) if agent_config else []
            self._connector_cache: OrderedDict[tuple, BaseSqlConnector] = OrderedDict()
            self._connector_cache_size = connector_cache_size
            # Bind the primary connector to (default datasource, default database).
            self._primary_connector = self._db_manager.get_conn(self._default_datasource, self._default_database)
            self._is_multi_connector = True
        else:
            self._init_single_db_connector(connector_or_manager)

        model_name = agent_config.active_model().model if agent_config else "gpt-3.5-turbo"
        self.compressor = DataCompressor(model_name=model_name)
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        # Backing field for the ``read_only`` property: this instance's own
        # posture, before the deployment-wide switch is ORed in.
        self._read_only = read_only
        self._filesystem_root = filesystem_root
        if agent_config and metadata_fts_enabled(agent_config):
            self.schema_rag = create_metadata_rag(agent_config, sub_agent_name)
        else:
            self.schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name) if agent_config else None
        self._field_order = self._determine_field_order()
        self._scoped_patterns = self._load_scoped_patterns(scoped_tables)

        self._semantic_datasets = None
        if agent_config and isinstance(getattr(agent_config, "project_name", ""), str):
            try:
                self._semantic_datasets = SemanticDatasetRAG(agent_config, sub_agent_name)
            except Exception as exc:
                logger.debug(f"Failed to initialize semantic dataset storage: {exc}")
        self.has_schema = self._has_schema_storage()

        try:
            self.has_semantic_datasets = self._semantic_datasets is not None and self._semantic_datasets.get_size() > 0
        except Exception:
            self._semantic_datasets = None
            self.has_semantic_datasets = False
        if not self.has_semantic_datasets and agent_config is not None:
            _warn_once_if_projection_is_stale(agent_config)

    @property
    def policy_context(self) -> Dict[str, Any]:
        """Request-scoped policy inputs, read fresh from the config."""
        policy_context = getattr(self.agent_config, "policy_context", None)
        return dict(policy_context) if isinstance(policy_context, dict) else {}

    @property
    def read_only(self) -> bool:
        """Whether this tool refuses non-read SQL. Resolved on every access.

        ORs the per-instance flag — set by read-only agents (Explore,
        ask_report/dashboard) and by the validator's shallow copy in
        ``datus.validation.llm_runner`` — with the deployment-wide
        ``AgentConfig.sql_read_only``. Tighten-only in both directions: neither
        source can relax the other.

        This is the effective posture, not the constructor argument, so that
        anything asking "is this tool read-only?" gets the answer the write
        paths will actually enforce. The MCP server's
        ``create_dynamic``/``create_static`` factories pass a config but no
        ``read_only`` flag, so on a hardened deployment their instances read
        ``True`` here despite nothing having passed it.

        Resolved per access rather than snapshotted in ``__init__`` for the same
        reason as ``principal`` above: the API hands nodes a per-request config
        clone, and a tool built before that clone was hardened must still honour
        it.

        ``coerce_bool`` rather than ``bool``: the ``getattr`` guard exists for
        duck-typed / host-supplied config objects, which are exactly the ones
        likely to carry a raw YAML value — and ``bool("false")`` is ``True``.
        """
        if self._read_only:
            return True
        return coerce_bool(getattr(self.agent_config, "sql_read_only", False), False)

    @read_only.setter
    def read_only(self, value: bool) -> None:
        # Tighten-only, matching ``AgentConfig.sql_read_only``: a caller may
        # harden an instance (``llm_runner`` does this to a shallow copy before
        # binding it to a hooks-free validator) but may not hand a write-capable
        # view of a read-only deployment to anything downstream.
        self._read_only = self._read_only or coerce_bool(value, False)

    def _refuse_write_if_read_only(
        self,
        operation: str,
        *,
        datasource: Optional[str] = "",
        sql_type: Optional[SQLType] = None,
        statement_kind: str = "",
        error: str = "",
    ) -> Optional[FuncToolResult]:
        """Gate every write entry point on the effective read-only posture.

        Returns ``None`` when the caller may proceed, or the ``FuncToolResult``
        to hand straight back when it may not. Defense-in-depth for read-only
        agents and read-only deployments alike: it is independent of
        ``PermissionHooks``, which several callers bypass entirely (validators
        run with ``hooks=None``, and the MCP server's tool instances never see
        hooks at all).

        Shared by all four write paths so a new one cannot be added with a
        subtly different rule, and so ``AgentConfig.sql_read_only`` means the
        same thing at each of them.

        Refusals are logged because a refusal is the event an operator actually
        wants to see: on a deployment running third-party-authored content it
        means that content just tried to write. Successful reads are already
        logged in ``read_query``, so staying silent here would record the benign
        path and drop the notable one. ``source`` separates a deployment-wide
        refusal from a read-only agent doing its job — the difference between
        "investigate this" and "working as intended".

        ``statement_kind`` is the finer-grained classification from
        ``parse_sql_statement_kind`` (``create`` / ``drop`` / ``alter`` /
        ``truncate``) when the caller has the SQL to derive it. It is preferred
        over ``sql_type`` in the log because ``ddl`` alone cannot tell an
        operator whether third-party content tried to create a table or drop
        one, and those warrant very different responses.
        """
        if not self.read_only:
            return None
        logger.warning(
            f"{operation} rejected by read-only policy",
            sql_type=statement_kind or (sql_type.value if sql_type else ""),
            datasource=self._resolve_effective_datasource(datasource),
            sub_agent=self.sub_agent_name or "",
            source="agent" if self._read_only else "deployment",
        )
        return FuncToolResult(
            success=0,
            error=error or f"This agent is read-only: {operation} is not available.",
        )

    def _has_schema_storage(self) -> bool:
        if not self.schema_rag:
            return False
        get_schema_size = getattr(self.schema_rag, "get_schema_size", None)
        if callable(get_schema_size):
            try:
                return get_schema_size() > 0
            except Exception:
                return False
        schema_store = getattr(self.schema_rag, "schema_store", None)
        table_size = getattr(schema_store, "table_size", None)
        if callable(table_size):
            try:
                return table_size() > 0
            except Exception:
                return False
        return False

    @staticmethod
    def _metadata_search_rows(metadata: Any) -> List[Dict[str, Any]]:
        rows = metadata.to_pylist()
        result: List[Dict[str, Any]] = []
        metadata_fields = [
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
            "identifier",
        ]
        for row in rows:
            payload: Dict[str, Any] = {}
            payload_json = row.get("payload_json")
            if payload_json:
                try:
                    decoded = json.loads(payload_json)
                    if isinstance(decoded, dict):
                        description = decoded.get("description")
                        if description:
                            payload["description"] = description
                except (TypeError, ValueError):
                    pass
            for field in metadata_fields:
                if row.get(field) not in (None, ""):
                    payload[field] = row[field]
                else:
                    payload.setdefault(field, "")
            result.append(payload)
        return result

    @staticmethod
    def _qualified_table_name(row: Dict[str, Any]) -> str:
        parts = [
            str(row.get("catalog_name") or "").strip(),
            str(row.get("database_name") or "").strip(),
            str(row.get("schema_name") or "").strip(),
            str(row.get("table_name") or "").strip(),
        ]
        qualified_name = ".".join(part for part in parts if part)
        return qualified_name or str(row.get("identifier") or row.get("table_name") or "").strip()

    @staticmethod
    def _format_sample_rows(value: Any) -> list[Any]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except (TypeError, ValueError):
                pass
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
            if rows:
                return rows
        except csv.Error:
            pass
        return [text]

    @classmethod
    def _sample_rows_by_identifier(cls, sample_values: Any) -> Dict[str, list[Any]]:
        if sample_values is None:
            return {}
        if getattr(sample_values, "num_rows", None) == 0:
            return {}
        selected_fields = ["identifier", "sample_rows"]
        available_fields = getattr(sample_values, "column_names", None)
        if available_fields is not None:
            selected_fields = [field for field in selected_fields if field in available_fields]
        if not selected_fields:
            return {}
        rows = sample_values.select(selected_fields).to_pylist()
        result: Dict[str, list[Any]] = {}
        for row in rows:
            identifier = str(row.get("identifier") or "").strip()
            if not identifier:
                continue
            sample_rows = cls._format_sample_rows(row.get("sample_rows"))
            if sample_rows:
                result[identifier] = sample_rows
        return result

    @classmethod
    def _search_table_result_row(
        cls,
        metadata_row: Dict[str, Any],
        sample_rows_by_identifier: Dict[str, list[Any]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"table_name": cls._qualified_table_name(metadata_row)}
        description = str(metadata_row.get("description") or "").strip()
        if description:
            result["description"] = description
        sample_rows = sample_rows_by_identifier.get(str(metadata_row.get("identifier") or "").strip())
        if sample_rows:
            result["sample_rows"] = sample_rows
        return result

    def _init_single_db_connector(self, connector: BaseSqlConnector):
        # Legacy single connector mode
        self._db_manager = None
        self._default_datasource = ""
        self._default_database = ""
        # Empty rather than absent: the attribute is read unconditionally when
        # deciding which tools to mount, so leaving it unset makes this mode
        # raise AttributeError instead of simply having no named datasources.
        self._datasources = []
        self._connector_cache = OrderedDict()
        self._connector_cache_size = 0
        self._primary_connector = connector
        self._is_multi_connector = False

    @property
    def connector(self) -> BaseSqlConnector:
        """Get the primary/default connector (for backward compatibility)."""
        return self._primary_connector

    def _get_connector(self, datasource: Optional[str] = None, database: str = "") -> BaseSqlConnector:
        """
        Get connector for the specified (datasource, database).

        In single connector mode, always returns the primary connector.
        In multi-connector mode, returns cached connector or fetches from db_manager.

        Args:
            datasource: Datasource name. If None/empty, uses default datasource.
            database: Physical database within the datasource. Routes the connector to it
                (required for multi-database datasources, e.g. a sqlite/duckdb glob). If empty,
                uses this tool's default database (unless a per-call datasource override is given).

        Returns:
            BaseSqlConnector for the specified (datasource, database)
        """
        if self._db_manager is None:
            # Single connector mode
            return self._primary_connector

        # Multi-connector mode: route by (datasource, database). DBManager.get_conn binds
        # the connector to the database (selects the file for a glob datasource).
        ds = datasource or self._default_datasource
        db = database or ("" if datasource else self._default_database)
        key = (ds, db)

        # Check cache
        if key in self._connector_cache:
            # Move to end (most recently used)
            self._connector_cache.move_to_end(key)
            return self._connector_cache[key]

        try:
            connector = self._db_manager.get_conn(ds, db)
        except DatusException:
            # Preserve database-level routing errors (e.g. invalid database name with the
            # list of available databases) so ``/database`` failures stay diagnosable.
            raise
        except (KeyError, ValueError) as e:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Datasource '{ds}' is not configured. Available datasources: {', '.join(self._datasources)}.",
            ) from e

        # Ensure connector is connected
        if hasattr(connector, "connect"):
            connector.connect()

        # Add to cache with LRU eviction
        if self._connector_cache_size > 0 and len(self._connector_cache) >= self._connector_cache_size:
            # Evict least recently used (first item)
            evicted_name, _ = self._connector_cache.popitem(last=False)
            logger.debug(f"LRU evicting connector: {evicted_name}")

        self._connector_cache[key] = connector
        return connector

    def _reset_database_for_rag(self, datasource: Optional[str] = "") -> str:
        connector = self._get_connector(datasource)
        return connector.database_name

    @staticmethod
    def _active_database_of(connector: Any) -> str:
        """Return the connector's active physical database as a plain string.

        Some test fixtures use ``MagicMock`` connectors — attribute access
        returns a ``Mock`` instance that is truthy, so a naive
        ``getattr(c, "database_name", "") or ""`` leaks a Mock into the
        ``TableTarget.database`` slot. Production connectors expose this as
        a ``str``; this helper enforces that contract.
        """
        val = getattr(connector, "database_name", None)
        return val if isinstance(val, str) else ""

    def _determine_field_order(self) -> Sequence[str]:
        dialect = getattr(self._primary_connector, "dialect", "") or ""
        capabilities = get_effective_capabilities(self._primary_connector, dialect)
        fields: List[str] = []
        if "catalog" in capabilities:
            fields.append("catalog")
        if "database" in capabilities or dialect == DBType.SQLITE:
            fields.append("database")
        if "schema" in capabilities:
            fields.append("schema")
        fields.append("table")
        return fields

    def _load_scoped_patterns(self, explicit_tokens: Optional[Iterable[str]]) -> List[ScopedTablePattern]:
        tokens: List[str] = []
        if explicit_tokens:
            tokens.extend(explicit_tokens)
        else:
            tokens.extend(self._resolve_scoped_context_tables())

        patterns: List[ScopedTablePattern] = []
        for token in tokens:
            scoped_pattern = self._parse_scope_token(token)
            if scoped_pattern:
                patterns.append(scoped_pattern)
        return patterns

    def _resolve_scoped_context_tables(self) -> Sequence[str]:
        if not self.agent_config:
            return []
        scoped_entries: List[str] = []

        if self.sub_agent_name:
            sub_agent_config = self._load_sub_agent_config(self.sub_agent_name)
            if sub_agent_config and sub_agent_config.scoped_context and sub_agent_config.scoped_context.tables:
                scoped_entries.extend(sub_agent_config.scoped_context.as_lists().tables)

        return scoped_entries

    def _load_sub_agent_config(self, sub_agent_name: str) -> Optional[SubAgentConfig]:
        if not self.agent_config:
            return None
        try:
            config = self.agent_config.sub_agent_config(sub_agent_name)
        except Exception:
            return None

        if not config:
            return None
        if isinstance(config, SubAgentConfig):
            return config

        try:
            return SubAgentConfig.model_validate(config)
        except Exception:
            return None

    def _parse_scope_token(self, token: str) -> Optional[ScopedTablePattern]:
        token = (token or "").strip()
        if not token:
            return None
        dialect = getattr(self._primary_connector, "dialect", "") or ""
        parsed = parse_table_name_parts(token, dialect)
        if not parsed.get("table_name"):
            return None
        values = {
            "catalog": parsed.get("catalog_name", ""),
            "database": parsed.get("database_name", ""),
            "schema": parsed.get("schema_name", ""),
            "table": parsed.get("table_name", ""),
        }
        return ScopedTablePattern(raw=token, **values)

    def _list_table_semantic_datasets(
        self,
        catalog: str = "",
        database: str = "",
        schema: str = "",
        table_name: str = "",
        semantic_model: str = "",
    ) -> List[Dict[str, Any]]:
        """Every semantic dataset bound to one physical table, primary first."""
        if self._semantic_datasets is None:
            return []
        rows = self._semantic_datasets.list_datasets(
            catalog_name=catalog,
            database_name=database,
            schema_name=schema,
            table_name=table_name,
            semantic_model=semantic_model,
            select_fields=[
                "semantic_model_name",
                "dataset_name",
                "description",
                "yaml_path",
            ],
        )
        logger.info(f"list_table_semantic_datasets for {table_name!r}: {len(rows)} dataset(s)")
        return rows

    def _table_semantic_model_names(self, coordinate: "TableCoordinate") -> List[str]:
        """Names of every semantic model describing one table, in lookup order."""
        rows = self._list_table_semantic_datasets(
            coordinate.catalog, coordinate.database, coordinate.schema, coordinate.table
        )
        return [str(row.get("semantic_model_name") or "") for row in rows if row.get("semantic_model_name")]

    def _get_table_semantic_projection(
        self,
        catalog: str = "",
        database: str = "",
        schema: str = "",
        table_name: str = "",
        semantic_model: str = "",
    ) -> Dict[str, Any]:
        """The primary dataset for one table, with its fields and relationships."""
        if self._semantic_datasets is None:
            return {}
        projection = self._semantic_datasets.get_table_projection(
            catalog_name=catalog,
            database_name=database,
            schema_name=schema,
            table_name=table_name,
            semantic_model=semantic_model,
        )
        return projection or {}

    def _semantic_description_for_row(self, metadata_row: Dict[str, Any]) -> str:
        """Business description for one search hit, or "" when it has none.

        Reads the primary semantic dataset, so search_table and describe_table
        agree on which model speaks for a table. A table modelled by several
        semantic models is legitimate, so this lookup must never fail the
        search — it only ever contributes a description.
        """
        try:
            datasets = self._list_table_semantic_datasets(
                metadata_row.get("catalog_name", ""),
                metadata_row.get("database_name", ""),
                metadata_row.get("schema_name", ""),
                metadata_row.get("table_name", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to read semantic datasets for {metadata_row.get('table_name')!r}: {e}")
            return ""
        return str(datasets[0].get("description") or "") if datasets else ""

    @staticmethod
    def _decode_profile_json(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return default
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _semantic_field_role(field: Dict[str, Any]) -> str:
        """Name the role a field plays, from the flags stored on its row."""
        if field.get("is_primary_key"):
            return "primary_key"
        if field.get("is_time"):
            return "time_dimension"
        if field.get("is_dimension"):
            return "dimension"
        return "field"

    def _apply_table_semantic_profile(self, result_data: Dict[str, Any], projection: Dict[str, Any]) -> None:
        """Attach one semantic dataset to describe_table output.

        Descriptions, column enrichment and relationships all come from the
        primary dataset alone. OSI relationships reference dataset names, which
        are local to their semantic model, so merging two models would assert a
        join graph that exists in neither. Any further datasets are surfaced as
        navigation under ``table.alternatives`` instead, and the caller can pull
        one up in full by passing ``semantic_model``.
        """

        columns = result_data.get("columns", [])
        semantic_fields = projection.get("fields") or []
        alternatives = projection.get("alternatives") or []
        ai_context = self._decode_profile_json(projection.get("ai_context_json"), None)

        table = {
            "name": (
                projection.get("dataset_name") or projection.get("semantic_model_name") or projection.get("name") or ""
            ),
            "description": projection.get("description", ""),
        }
        if ai_context not in (None, "", [], {}):
            table["ai_context"] = ai_context
        if alternatives:
            table["semantic_model"] = projection.get("semantic_model_name", "")
            table["alternatives"] = [
                {
                    "semantic_model": alternative.get("semantic_model_name", ""),
                    "dataset": alternative.get("dataset_name", ""),
                    "description": alternative.get("description", ""),
                    "yaml_path": alternative.get("yaml_path", ""),
                }
                for alternative in alternatives
            ]
        result_data["table"] = table
        result_data["semantic"] = {
            "relationships": [
                {
                    key: value
                    for key, value in {
                        "name": relationship.get("name", ""),
                        "type": relationship.get("rel_type", ""),
                        "join_type": relationship.get("join_type", ""),
                        "from_dataset": relationship.get("from_dataset", ""),
                        "to_dataset": relationship.get("to_dataset", ""),
                        "from_columns": self._decode_profile_json(relationship.get("from_columns_json"), []),
                        "to_columns": self._decode_profile_json(relationship.get("to_columns_json"), []),
                        "ai_context": self._decode_profile_json(relationship.get("ai_context_json"), None),
                    }.items()
                    if value not in (None, "", [], {})
                }
                for relationship in projection.get("relationships") or []
            ],
        }

        semantic_lookup: Dict[str, Dict[str, Any]] = {}
        for field in semantic_fields:
            if not isinstance(field, dict):
                continue
            for key in ("expr", "name"):
                value = field.get(key)
                if value:
                    semantic_lookup.setdefault(str(value).strip("`").lower(), field)

        for col in columns:
            col_name = str(col.get("name", "")).lower()
            field = semantic_lookup.get(col_name)
            if not field:
                continue
            description = field.get("description") or ""
            field_ai_context = self._decode_profile_json(field.get("ai_context_json"), None)
            col["semantic_role"] = self._semantic_field_role(field)
            col["is_dimension"] = bool(field.get("is_dimension"))
            if field_ai_context not in (None, "", [], {}):
                col["ai_context"] = field_ai_context
            if description:
                col["semantic_description"] = description
                col["comment"] = description

    def _enrich_fields_with_descriptions(
        self, field_list_json: str, ddl_columns: List[Dict[str, Any]], field_type: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich field list with descriptions from YAML (priority) and DDL (fallback).

        Args:
            field_list_json: JSON string of field definitions from semantic model
            ddl_columns: Column metadata from DDL
            field_type: Type of fields ("dimensions", "measures", "identifiers")

        Returns:
            List of enriched field dictionaries with name and description
        """
        import json

        try:
            # Parse field list from JSON string
            if not field_list_json:
                return []

            field_list = json.loads(field_list_json) if isinstance(field_list_json, str) else field_list_json

            # Handle simple list of field names
            if isinstance(field_list, list) and all(isinstance(f, str) for f in field_list):
                field_list = [{"name": f} for f in field_list]
            elif not isinstance(field_list, list):
                return []

            # Build DDL column lookup by name
            ddl_lookup = {col.get("name", "").lower(): col for col in ddl_columns if "name" in col}

            # Enrich each field
            enriched_fields = []
            for field in field_list:
                if isinstance(field, str):
                    field = {"name": field}
                elif not isinstance(field, dict):
                    continue

                field_name = field.get("name", "")
                if not field_name:
                    continue

                enriched_field = {"name": field_name}

                # Priority 1: Use description from YAML if exists
                if "description" in field and field["description"]:
                    enriched_field["description"] = field["description"]
                else:
                    # Priority 2: Fallback to DDL column comment
                    ddl_col = ddl_lookup.get(field_name.lower())
                    if ddl_col and ddl_col.get("comment"):
                        enriched_field["description"] = ddl_col["comment"]

                # Preserve other field attributes (type, expr, entity, etc.)
                for key, value in field.items():
                    if key not in ("name", "description"):
                        enriched_field[key] = value

                enriched_fields.append(enriched_field)

            return enriched_fields

        except Exception as e:
            logger.warning(f"Failed to enrich {field_type} with descriptions: {e}")
            return []

    def _resolve_workspace_root(self) -> str:
        """Resolve workspace_root from ``agent_config.project_root``; fall back to cwd."""
        if self.agent_config and hasattr(self.agent_config, "project_root"):
            workspace_root = self.agent_config.project_root
        else:
            workspace_root = "."
        return os.path.expanduser(workspace_root)

    def _read_sql_from_file(self, file_path: str) -> str:
        """Read SQL content from a file path relative to workspace root.

        Delegates the path-safety checks and read to the shared
        :func:`read_workspace_sql_file` so the execution path and the
        permission gate resolve a ``.sql`` reference identically.
        """
        from datus.utils.sql_utils import read_workspace_sql_file

        try:
            return read_workspace_sql_file(file_path, self._resolve_workspace_root())
        except FileNotFoundError:
            raise DatusException(
                ErrorCode.COMMON_FILE_NOT_FOUND,
                message_args={"config_name": "SQL", "file_name": file_path},
            )
        except ValueError as e:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message_args={"error_message": str(e)},
            )

    @staticmethod
    def _normalize_identifier_part(value: Optional[str]) -> str:
        if value is None:
            return ""
        normalized = str(value).strip()
        if not normalized:
            return ""
        # Strip common quoting characters
        return normalized.strip("`\"'[]")

    def _default_field_value(
        self,
        field: str,
        explicit: Optional[str],
        connector: Optional[BaseSqlConnector] = None,
    ) -> str:
        if field not in self._field_order:
            return ""
        if explicit:
            return self._normalize_identifier_part(explicit)

        fallback_attr_map = {
            "catalog": "catalog_name",
            "database": "database_name",
            "schema": "schema_name",
        }
        fallback_attr = fallback_attr_map.get(field)
        source_connector = connector or self.connector
        if fallback_attr and hasattr(source_connector, fallback_attr):
            return self._normalize_identifier_part(getattr(source_connector, fallback_attr))
        return ""

    def _dialect_for_datasource(self, datasource: Optional[str] = "") -> str:
        try:
            connector = self._get_connector(datasource)
        except Exception:
            connector = self.connector
        return getattr(connector, "dialect", "") or ""

    def _normalize_namespace_args(
        self,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema: Optional[str] = "",
        datasource: Optional[str] = "",
    ) -> tuple[str, str, str]:
        catalog_value = self._normalize_identifier_part(catalog)
        database_value = self._normalize_identifier_part(database)
        schema_value = self._normalize_identifier_part(schema)

        try:
            connector = self._get_connector(datasource)
        except Exception:
            connector = self.connector
        dialect = getattr(connector, "dialect", "") or ""
        if not supports_namespace("catalog", connector=connector, dialect=dialect):
            if (
                catalog_value
                and not database_value
                and supports_namespace("database", connector=connector, dialect=dialect)
            ):
                database_value = catalog_value
            catalog_value = ""

        return catalog_value, database_value, schema_value

    def _build_table_coordinate(
        self,
        raw_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema: Optional[str] = "",
        connector: Optional[BaseSqlConnector] = None,
    ) -> TableCoordinate:
        routed_connector = connector or self.connector
        coordinate = TableCoordinate(
            catalog=self._default_field_value("catalog", catalog, routed_connector),
            database=self._default_field_value("database", database, routed_connector),
            schema=self._default_field_value("schema", schema, routed_connector),
            table=self._normalize_identifier_part(raw_name),
        )
        dialect = getattr(routed_connector, "dialect", "") or ""
        parsed = parse_table_name_parts(raw_name, dialect)
        for field, parsed_field in (
            ("catalog", "catalog_name"),
            ("database", "database_name"),
            ("schema", "schema_name"),
            ("table", "table_name"),
        ):
            if parsed.get(parsed_field):
                setattr(coordinate, field, self._normalize_identifier_part(parsed[parsed_field]))
        return coordinate

    def _table_matches_scope(self, coordinate: TableCoordinate) -> bool:
        if not self._scoped_patterns:
            return True
        return any(pattern.matches(coordinate) for pattern in self._scoped_patterns)

    def _filter_table_entries(
        self,
        entries: Sequence[Dict[str, Any]],
        catalog: Optional[str],
        database: Optional[str],
        schema: Optional[str],
        connector: BaseSqlConnector,
    ) -> List[Dict[str, Any]]:
        if not self._scoped_patterns:
            return list(entries)

        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            coordinate = self._build_table_coordinate(
                raw_name=str(entry.get("qualified_name", "")),
                catalog=catalog,
                database=database,
                schema=schema,
                connector=connector,
            )
            if self._table_matches_scope(coordinate):
                filtered.append(entry)
        return filtered

    def _matches_catalog_database(self, pattern: ScopedTablePattern, catalog: str, database: str) -> bool:
        if pattern.catalog and not _pattern_matches(pattern.catalog, catalog):
            return False
        if pattern.database and not _pattern_matches(pattern.database, database):
            return False
        return True

    def _database_matches_scope(self, catalog: Optional[str], database: str) -> bool:
        if not self._scoped_patterns:
            return True
        catalog_value = self._default_field_value("catalog", catalog or "")
        database_value = self._default_field_value("database", database or "")

        wildcard_allowed = False
        for pattern in self._scoped_patterns:
            if not self._matches_catalog_database(pattern, catalog_value, database_value):
                continue
            if pattern.database:
                if _pattern_matches(pattern.database, database_value):
                    return True
                continue
            wildcard_allowed = True
        return wildcard_allowed

    def _schema_matches_scope(self, catalog: Optional[str], database: Optional[str], schema: str) -> bool:
        if not self._scoped_patterns:
            return True
        catalog_value = self._default_field_value("catalog", catalog or "")
        database_value = self._default_field_value("database", database or "")
        schema_value = self._default_field_value("schema", schema or "")

        wildcard_allowed = False
        for pattern in self._scoped_patterns:
            if not self._matches_catalog_database(pattern, catalog_value, database_value):
                continue
            if pattern.schema:
                if _pattern_matches(pattern.schema, schema_value):
                    return True
                continue
            wildcard_allowed = True
        return wildcard_allowed

    def _check_sql_table_scope(self, sql: str, connector: Optional[BaseSqlConnector] = None) -> List[str]:
        """Return table names from *sql* that fall outside the scoped context."""
        if not self._scoped_patterns:
            return []
        from datus.utils.sql_utils import extract_table_names

        routed_connector = connector or self._primary_connector
        dialect = getattr(routed_connector, "dialect", "") or ""
        table_names = extract_table_names(sql, dialect=dialect, ignore_empty=True)
        if not table_names:
            return []  # can't parse → allow (SHOW/DESCRIBE/EXPLAIN have no tables)
        out_of_scope: List[str] = []
        for name in table_names:
            coordinate = self._build_table_coordinate(raw_name=name, connector=routed_connector)
            if not self._table_matches_scope(coordinate):
                out_of_scope.append(name)
        return out_of_scope

    # Public methods that belong to the tool-plumbing framework rather than the
    # agent-facing tool surface. ``to_function_tool`` converts a bound method into
    # a Tool; ``available_tools`` assembles the runtime list. Neither is a tool
    # itself. Mirrors ``FilesystemFuncTool._BASE_TOOL_FRAMEWORK_METHODS``.
    _FRAMEWORK_METHODS: frozenset = frozenset({"available_tools", "to_function_tool"})

    # Internal statement-dispatch targets of the unified ``execute_sql`` entry
    # point. ``execute_sql`` detects the statement type and routes to these by
    # hand (SELECT -> read_query, DML -> execute_write, DDL/other -> execute_ddl),
    # so they are never @mcp_tool()-decorated and never mounted as tools. They
    # stay public because callers across modules (semantic_discovery_tools,
    # reference_template_tools) and the connectors share the vocabulary, but they
    # must not leak into the agent tool surface (VALID_TOOL_METHODS / the saas
    # editor catalog) or the permission registry.
    _INTERNAL_SQL_METHODS: frozenset = frozenset(
        {
            "read_query",
            "execute_read_enforced",
            "guard_estimated_rows",
            "execute_write",
            "execute_ddl",
            "get_table_ddl",
        }
    )

    @staticmethod
    def all_tools_name() -> List[str]:
        # Agent-facing tool surface: every public tool method, including the ones
        # gen_job mounts directly (``transfer_query_result``, migration wrappers)
        # that never carry an @mcp_tool() decorator. Framework plumbing and the
        # internal execute_sql dispatch helpers are filtered out. Feeds both
        # VALID_TOOL_METHODS and the permission registry
        # (AgenticNode._populate_tool_registry).
        from datus.utils.class_utils import get_public_instance_methods

        excluded = DBFuncTool._FRAMEWORK_METHODS | DBFuncTool._INTERNAL_SQL_METHODS
        return [name for name in get_public_instance_methods(DBFuncTool).keys() if name not in excluded]

    @staticmethod
    def _dialect_name(value: Any) -> str:
        raw_value = getattr(value, "value", value)
        if not isinstance(raw_value, str):
            return ""
        return raw_value.strip().lower()

    def _configured_tool_dialects(self) -> set[str]:
        dialects: set[str] = set()
        if self._is_multi_connector and self.agent_config:
            try:
                db_configs = self.agent_config.current_db_configs()
            except Exception:
                db_configs = {}
            if isinstance(db_configs, dict):
                for db_config in db_configs.values():
                    if isinstance(db_config, dict):
                        dialect = db_config.get("type", "")
                    else:
                        dialect = getattr(db_config, "type", "")
                    normalized = self._dialect_name(dialect)
                    if normalized:
                        dialects.add(normalized)

        if not dialects:
            normalized = self._dialect_name(getattr(self.connector, "dialect", ""))
            if normalized:
                dialects.add(normalized)
        return dialects

    def _configured_supports(self, namespace: str) -> bool:
        primary_dialect = self._dialect_name(getattr(self.connector, "dialect", ""))
        if namespace in get_effective_capabilities(self.connector, primary_dialect):
            return True
        return any(
            namespace in get_effective_capabilities(dialect=dialect)
            for dialect in self._configured_tool_dialects()
            if dialect != primary_dialect
        )

    def _excluded_tool_params(self) -> set[str]:
        excluded: set[str] = set()
        if not self._configured_supports("catalog"):
            excluded.add("catalog")
        return excluded

    def to_function_tool(self, bound_method: Callable) -> Tool:
        return trans_to_function_tool(bound_method, excluded_params=self._excluded_tool_params())

    def available_tools(self) -> List[Tool]:
        bound_tools = []
        methods_to_convert: List[Callable] = [self.list_tables, self.describe_table]

        if self.has_schema:
            methods_to_convert.append(self.search_table)

        methods_to_convert.append(self.execute_sql)

        # Only mount the uploads loader where there is a catalog to load into.
        # Without the datasource every call would fail with "not configured", so
        # on a CLI install this would be a tool that exists only to error.
        if LOCAL_FILES_DATASOURCE in self._datasources:
            methods_to_convert.append(self.load_file_as_table)

        if self._configured_supports("database"):
            bound_tools.append(self.to_function_tool(self.list_databases))

        if self._configured_supports("schema"):
            bound_tools.append(self.to_function_tool(self.list_schemas))

        for bound_method in methods_to_convert:
            bound_tools.append(self.to_function_tool(bound_method))
        return bound_tools

    @mcp_tool(availability_check="has_schema")
    def search_table(
        self,
        query_text: str,
        catalog: str = "",
        database: str = "",
        schema_name: str = "",
        datasource: Optional[str] = "",
        top_n: int = 5,
        simple_sample_data: bool = True,
    ) -> FuncToolResult:
        """
        Retrieve table candidates from indexed metadata and optional semantic profile text.
        Use this tool when the agent needs tables matching a natural-language description.
        This tool helps find relevant tables by searching through table names, schemas (DDL),
        and sample data using configured metadata search.

        Use this tool when you need to:
        - Find tables related to a specific business concept or domain
        - Discover tables containing certain types of data
        - Locate tables for SQL query development
        - Understand what tables are available in a datasource

        **Application Guidance**:
        1. If table matches (via description/sample_rows), inspect it with describe_table before writing SQL
        2. If partitioned (e.g., date-based in definition), explore correct partition via describe_table
        3. If no match, use list_tables for broader exploration

        Args:
            query_text: Description of the table you want (e.g. "daily active users per country").
            catalog: Catalog filter. Only use for databases that support catalogs (StarRocks, Databricks).
                Leave empty for PostgreSQL, MySQL, Snowflake, SQLite, DuckDB.
            database: Database filter. Use for PostgreSQL, MySQL, Snowflake, StarRocks, DuckDB.
                Leave empty for SQLite (uses file path instead).
            schema_name: Schema filter. Use for PostgreSQL, Snowflake, DuckDB (e.g., "public").
                Leave empty for MySQL (database = schema), StarRocks, SQLite.
            datasource: Optional datasource to route the search to. Defaults to the current datasource.
            top_n: Maximum number of rows to return after scoping filters.
            simple_sample_data: Deprecated compatibility argument; sample rows are returned inline.

        Returns:
            FuncToolResult where:
                - success=1 with result={"metadata": [...]} (empty list when no matches).
                  Each metadata item contains table_name, optional description, and optional sample_rows.
                - success=0 with error text if schema storage is unavailable or lookup fails.
        """
        if not self.has_schema:
            return FuncToolResult(success=0, error="Table search is unavailable because schema storage is not ready.")

        try:
            catalog, database, schema_name = self._normalize_namespace_args(
                catalog,
                database,
                schema_name,
                datasource,
            )
            result_dict: Dict[str, Any] = {"metadata": []}
            rag_database = database or self._reset_database_for_rag(datasource)

            if metadata_fts_enabled(self.agent_config) and hasattr(self.schema_rag, "search_table"):
                metadata = self.schema_rag.search_table(
                    query_text,
                    catalog_name=catalog,
                    database_name=rag_database,
                    schema_name=schema_name,
                    table_type="full",
                    top_n=top_n,
                )
                sample_rows_for_search_results = getattr(self.schema_rag, "sample_rows_for_search_results", None)
                sample_values = (
                    sample_rows_for_search_results(metadata) if callable(sample_rows_for_search_results) else None
                )
            else:
                metadata, sample_values = self.schema_rag.search_similar(
                    query_text,
                    catalog_name=catalog,
                    database_name=rag_database,
                    schema_name=schema_name,
                    table_type="full",
                    top_n=top_n,
                )

            metadata_rows: List[Dict[str, Any]] = []
            if metadata is not None and getattr(metadata, "num_rows", 1) != 0:
                metadata_rows = self._metadata_search_rows(metadata)
            if not metadata_rows:
                return FuncToolResult(success=1, result=result_dict)

            for metadata_row in metadata_rows:
                description = self._semantic_description_for_row(metadata_row)
                if description:
                    metadata_row["description"] = description

            sample_rows_by_identifier = self._sample_rows_by_identifier(sample_values)
            result_dict["metadata"] = [
                self._search_table_result_row(metadata_row, sample_rows_by_identifier) for metadata_row in metadata_rows
            ]
            return FuncToolResult(result=result_dict)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool()
    def list_databases(
        self, catalog: Optional[str] = "", datasource: Optional[str] = "", include_sys: Optional[bool] = False
    ) -> FuncToolResult:
        """
        Enumerate databases accessible through the current connection.
        Use this when you need to discover what databases are available before querying.
        For finding specific tables by description, use search_table instead.

        Args:
            catalog: Optional catalog to scope the lookup (dialect dependent).
            datasource: Optional datasource to route the query to. Defaults to the current datasource.
            include_sys: Set True to include system databases; defaults to False.

        Returns:
            FuncToolResult with result as a list of database names ordered by the connector. On failure success=0 with
            an explanatory error message.
        """
        catalog, _, _ = self._normalize_namespace_args(catalog, "", "", datasource)
        if self._is_multi_connector and datasource and datasource not in self._datasources:
            return FuncToolResult(
                success=0, error=f"Datasource '{datasource}' not found. Available: {list(self._datasources)}"
            )
        source = datasource or self._default_datasource
        # A glob/multi-database file datasource: enumerate its configured databases (one file per db),
        # since each connector only sees its own single file.
        if self.agent_config:
            try:
                cfg = self.agent_config.current_db_config(source)
            except Exception:
                cfg = None
            if cfg is not None and getattr(cfg, "path_pattern", ""):
                databases = self.agent_config.list_databases(source)
                filtered = [db for db in databases if self._database_matches_scope(catalog, db)]
                return FuncToolResult(result=filtered)
        try:
            connector = self._get_connector(source)
            databases = connector.get_databases(catalog, include_sys=include_sys)
            filtered = [db for db in databases if self._database_matches_scope(catalog, db)]
            return FuncToolResult(result=filtered)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool()
    def list_schemas(
        self,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        datasource: Optional[str] = "",
        include_sys: bool = False,
    ) -> FuncToolResult:
        """
        List schema names under the supplied catalog/database coordinate.
        Use this to explore schema structure when working with databases that have multiple schemas
        (e.g., PostgreSQL, Snowflake).

        Args:
            catalog: Optional catalog filter. Leave blank to rely on connector defaults.
            database: Optional database filter. Leave blank to rely on connector defaults.
            datasource: Optional datasource to route the query to. Defaults to the current datasource.
            include_sys: Set True to include system schemas; defaults to False.

        Returns:
            FuncToolResult with result holding the schema name list. On failure success=0 with an explanatory message.
        """
        try:
            catalog, database, _ = self._normalize_namespace_args(catalog, database, "", datasource)
            if database and not self._database_matches_scope(catalog, database):
                return FuncToolResult(result=[])
            connector = self._get_connector(datasource, database)
            schemas = connector.get_schemas(catalog, database, include_sys=include_sys)
            filtered = [schema for schema in schemas if self._schema_matches_scope(catalog, database, schema)]
            return FuncToolResult(result=filtered)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool()
    def list_tables(
        self,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        datasource: Optional[str] = "",
        include_views: Optional[bool] = True,
    ) -> FuncToolResult:
        """
        Return table-like objects (tables, views, materialized views) visible to the connector.
        Args:
            catalog: Optional catalog filter.
            database: Optional database filter.
            schema_name: Optional schema filter.
            datasource: Optional datasource to route the query to. Defaults to the current datasource.
            include_views: When True (default) also include views and materialized views.

        Returns:
            FuncToolResult with result=[{"type": "table|view|materialized_view", "qualified_name": str}, ...].
            ``qualified_name`` is ``[db.][schema.]table``, prefixing only the levels the caller did not pass
            (e.g. pass ``database`` but not ``schema`` and each entry carries its resolved ``schema.table``).
            On failure success=0 with an explanatory error message.
        """
        try:
            catalog, database, schema_name = self._normalize_namespace_args(
                catalog,
                database,
                schema_name,
                datasource,
            )
            connector = self._get_connector(datasource, database)
            result = []
            for tb in connector.get_tables(catalog, database, schema_name):
                result.append({"type": "table", "qualified_name": tb})

            if include_views:
                # Add views. We deliberately swallow any exception — some connectors
                # don't support views (NotImplementedError/AttributeError), and others
                # raise real SQL errors when the system view the adapter targets is
                # missing on that DB version. Failing list_tables entirely for a
                # subordinate listing would hide the tables we already fetched.
                try:
                    views = connector.get_views(catalog, database, schema_name)
                    for view in views:
                        result.append({"type": "view", "qualified_name": view})
                except Exception as e:
                    logger.debug(f"get_views unavailable on {connector.dialect}: {e}")

                # Add materialized views (same reasoning as views above).
                try:
                    materialized_views = connector.get_materialized_views(catalog, database, schema_name)
                    for mv in materialized_views:
                        result.append({"type": "materialized_view", "qualified_name": mv})
                except Exception as e:
                    logger.debug(f"get_materialized_views unavailable on {connector.dialect}: {e}")

            filtered_result = self._filter_table_entries(result, catalog, database, schema_name, connector)
            return FuncToolResult(result=filtered_result)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool()
    def describe_table(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        datasource: Optional[str] = "",
        semantic_model: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Fetch detailed column metadata, enriched with Semantic Model information.
        Use this tool to understand the table schema and business meanings.

        Args:
            table_name: Table identifier to describe.
            catalog: Optional catalog override.
            database: Optional database override.
            schema_name: Optional schema override.
            datasource: Optional datasource to route the query to. Defaults to the current datasource.
            semantic_model: Optional semantic model to read the table's meaning from. Use it when
                `table.alternatives` shows the table is modelled by more than one semantic model
                and you want another one's view; defaults to the primary dataset. Naming a model
                that does not describe this table fails with the list of models that do.

        Returns:
            FuncToolResult with a dictionary containing:
            - columns (list): List of column dictionaries, each containing:
              - name (str): Column name (required)
              - type (str): Column data type (required)
              - comment (str): Column description/comment, enriched with semantic model description if available
              - pk (bool, optional): present and true when the database reports the column
                as part of the table's primary key; absent when not a key or unknown
              - nullable (bool, optional): present and false when the column is NOT NULL;
                absent when nullable or unknown
              - default_value (str, optional): column default expression, when defined
              - is_dimension (bool): Whether this column is a dimension in semantic model
                (semantic fields only present if semantic model exists)
            - table (dict, optional): Table-level metadata from semantic model (only if model exists):
              - name (str): Name of the dataset modelling this table
              - description (str): Table description from semantic model
              - ai_context (dict/list/str, optional): Extra LLM-facing business guidance
              - semantic_model (str, optional): Which semantic model the meaning above came from.
                Only present when the table is modelled more than once.
              - alternatives (list, optional): Other semantic models describing this same table, each
                with semantic_model, dataset, description and yaml_path. Only present when the table
                is modelled more than once. Re-call with `semantic_model` to read one of them; the
                views are never merged, since each model's relationships are only valid within it.
            - semantic (dict, optional): LLM-facing semantic hints:
              - relationships (list): Relationships of the dataset above, each with name, type,
                join_type, from_dataset, to_dataset, from_columns and to_columns. Endpoints are
                dataset names local to that one semantic model.
        """
        try:
            catalog, database, schema_name = self._normalize_namespace_args(
                catalog,
                database,
                schema_name,
                datasource,
            )
            connector = self._get_connector(datasource, database)
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
                connector=connector,
            )

            if not self._table_matches_scope(coordinate):
                error_msg = f"Table '{table_name}' is outside the scoped context."
                logger.warning(error_msg)
                return FuncToolResult(
                    success=0,
                    error=error_msg,
                )

            # 1. Get Physical Schema
            # Use parsed coordinate fields so that dotted names like "raw.stage"
            # are correctly split into schema="raw", table="stage" before passing
            # to the connector (avoids DuckDB treating "raw" as a catalog).
            connector = self._get_connector(datasource, coordinate.database)
            column_result = connector.get_schema(
                catalog_name=coordinate.catalog,
                database_name=coordinate.database,
                schema_name=coordinate.schema,
                table_name=coordinate.table,
            )
            logger.debug(f"Got {len(column_result)} columns from connector")

            if not column_result:
                error_msg = f"Table '{table_name}' does not exist or has no columns."
                logger.warning(error_msg)
                return FuncToolResult(success=0, error=error_msg)

            # 2. Normalize columns to ensure required fields
            columns = []
            for col in column_result:
                normalized_col = {
                    "name": col.get("name", ""),
                    "type": col.get("type", ""),
                    "comment": col.get("comment", "") or "",  # Ensure empty string if None
                }
                # Constraint facts are emitted only when informative: several
                # connectors hardcode pk=False / nullable=True when the engine
                # exposes no constraint metadata, so those values mean
                # "unknown", not "verified absent".
                pk_flag = col.get("pk")
                # bool is an int subclass; SQLite reports the 1-based position
                # within a composite key instead of a bool.
                if isinstance(pk_flag, int) and pk_flag:
                    normalized_col["pk"] = True
                if col.get("nullable") is False:
                    normalized_col["nullable"] = False
                if col.get("default_value") not in (None, ""):
                    normalized_col["default_value"] = str(col["default_value"])
                columns.append(normalized_col)

            # 3. Enrich with Semantic Model Info if available
            result_data = {"columns": columns}

            requested_model = str(semantic_model or "").strip()
            try:
                projection = self._get_table_semantic_projection(
                    coordinate.catalog,
                    coordinate.database,
                    coordinate.schema,
                    coordinate.table,
                    semantic_model=requested_model,
                )
                if projection:
                    logger.debug(
                        "Found semantic dataset %s (%d alternative(s))",
                        projection.get("dataset_name") or "unknown",
                        len(projection.get("alternatives") or []),
                    )
                    self._apply_table_semantic_profile(result_data, projection)
                elif requested_model:
                    # Silently dropping to physical columns would read as "this
                    # table has no semantic model", sending the caller to guess
                    # at raw columns when the name is merely misspelled.
                    modelled_by = self._table_semantic_model_names(coordinate)
                    if modelled_by:
                        return FuncToolResult(
                            success=0,
                            error=(
                                f"Semantic model '{requested_model}' does not describe table '{table_name}'. "
                                f"It is modelled by: {', '.join(modelled_by)}."
                            ),
                        )
            except Exception as e:
                logger.warning(f"Failed to get table semantic profile for {table_name}: {e}")

            logger.info(f"describe_table succeeded for {table_name}, returning {len(columns)} columns")
            return FuncToolResult(result=result_data)

        except Exception as e:
            import traceback

            error_msg = f"Error describing table {table_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return FuncToolResult(success=0, error=error_msg)

    # ------------------------------------------------- uploaded data files

    def _reject_unsafe_uploads_sql(self, sql: str, datasource: Optional[str]) -> Optional[FuncToolResult]:
        """Confine model-authored SQL on the uploads catalog to reads of its own tables.

        The catalog is a DuckDB datasource with external file access enabled —
        that is what makes a lazy VIEW over a spreadsheet possible. The same
        capability in hand-written SQL reaches every file the process can see,
        including the shared tenant volume, routing around the filesystem path
        policy that is the agent's only containment boundary.

        Three gates. The last is a whitelist, because a blacklist of
        file-reading syntax cannot be completed:

        * statement class — everything except SELECT / SHOW / DESCRIBE / EXPLAIN
          is refused. ``ATTACH '/data/tenants/<other>/x.duckdb'`` and
          ``COPY (...) TO '/path'`` are not function calls and would sail past a
          name check while reading or writing outside the project entirely.
          Nothing legitimate is lost: every write to this catalog comes from
          ``load_file_as_table``, which uses its own connection.
        * named file-reading functions, which also covers a reader called where
          a *value* goes rather than where a table goes — that leaves no table
          reference for the next gate to inspect.
        * every table reference must resolve to an object the catalog already
          holds. DuckDB's replacement scan reads a bare path as a table —
          ``SELECT * FROM '/data/tenants/other/x.parquet'``, globs included —
          with no function call for a name check to see, and it parses as a
          plain SELECT. Requiring resolution closes that, closes every
          file-reading function at once, and closes whatever DuckDB adds next.

        Scoped to this datasource on purpose: a project that configures its own
        DuckDB datasource already had this reach before uploads existed, and
        silently narrowing it here would be an unrelated behaviour change.
        """
        resolved = datasource or self._default_datasource
        if resolved != LOCAL_FILES_DATASOURCE:
            return None

        from datus.utils.sql_utils import parse_sql_type

        hint = (
            f"Only read queries are allowed on the '{LOCAL_FILES_DATASOURCE}' "
            f"datasource. Register a file with load_file_as_table(path=...) and "
            f"query the table it returns; call list_tables(datasource="
            f"'{LOCAL_FILES_DATASOURCE}') to see what is registered."
        )

        try:
            sql_type = parse_sql_type(sql, "duckdb")
        except Exception:
            # Unparseable SQL fails closed: this gate is the boundary, so an
            # unknown statement is refused rather than waved through.
            sql_type = None
        if sql_type not in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN):
            return FuncToolResult(
                success=0,
                error=f"{'Unparseable statement' if sql_type is None else 'Write and DDL statements'} rejected. {hint}",
            )

        # Cheap, and it covers a syntactic position the resolution check below
        # cannot see: a reader called where a *value* goes rather than where a
        # table goes leaves no table reference behind. It also gives a better
        # message for the common ``FROM read_csv_auto(...)`` shape.
        offenders = find_file_reading_functions(sql)
        if offenders:
            return FuncToolResult(
                success=0,
                error=f"{', '.join(offenders)} cannot be called directly. {hint}",
            )

        try:
            registered = list(self._registered_uploads())
        except Exception as exc:
            # No catalog reading means no way to authorise a reference, and this
            # gate is the boundary — so refuse rather than skip it.
            logger.warning("Could not read the uploads catalog to authorise SQL: %s", exc)
            return FuncToolResult(success=0, error=f"Uploads catalog is unavailable. {hint}")

        offenders = unresolved_table_references(sql, registered)
        if offenders:
            return FuncToolResult(
                success=0,
                error=(
                    f"{', '.join(offenders)} is not a table registered on the "
                    f"'{LOCAL_FILES_DATASOURCE}' datasource. {hint}"
                ),
            )
        return None

    def _registered_uploads(self) -> Dict[str, Optional[str]]:
        """Names the uploads catalog holds, read while holding the connector lock.

        The read has to happen *inside* ``exclusive_connection``. Handing the
        connection out and querying it afterwards races every other caller on
        the same connector, and an LLM emitting parallel tool calls makes that
        the normal case, not a corner: ``DuckDBPyConnection`` is not thread-safe,
        so concurrent ``execute`` either returns another statement's rows or
        segfaults the process. Returning the wrong rows is the worse outcome —
        a short read looks like an empty catalog, which this gate then reports
        as "table is not registered" about a table that is registered.
        """
        connector = self._get_connector(LOCAL_FILES_DATASOURCE, "")
        exclusive = getattr(connector, "exclusive_connection", None)
        if exclusive is None:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Datasource '{LOCAL_FILES_DATASOURCE}' is not a DuckDB datasource",
            )
        with exclusive() as connection:
            return registered_objects(connection)

    def _resolve_data_file(self, path: str):
        """Classify a user-supplied data-file path against the filesystem policy.

        ``load_file_as_table`` embeds the resolved path into a VIEW definition,
        so this is the only point where the path is checked — everything after it
        is DuckDB reading whatever it was handed. Only INTERNAL (inside the
        project) and WHITELIST (operator-granted) paths pass; HIDDEN and EXTERNAL
        are refused, with HIDDEN reported as "not found" to preserve the
        invisibility of ``.datus`` internals.
        """
        from datus.tools.func_tool.fs_path_policy import PathZone, classify_path

        agent_config = self.agent_config
        root_path = self._filesystem_root or ""
        datus_home = None
        allowlist = None
        if agent_config is not None:
            if not root_path:
                root_path = getattr(agent_config, "project_root", "") or getattr(agent_config, "home", "") or ""
            allowlist = getattr(agent_config, "filesystem_allowlist", None) or None
            path_manager = getattr(agent_config, "path_manager", None)
            if path_manager is not None:
                try:
                    datus_home = str(path_manager.datus_home)
                except Exception:
                    datus_home = None
        root = Path(root_path).expanduser().resolve(strict=False) if root_path else Path.cwd()

        resolved = classify_path(path, root_path=root, datus_home=datus_home, allowlist=allowlist)
        if resolved.zone in (PathZone.HIDDEN, PathZone.EXTERNAL):
            if resolved.zone == PathZone.HIDDEN:
                raise DataFileError(f"File not found: {resolved.display}")
            raise DataFileError(
                f"{resolved.display} is outside the project workspace. Upload the file "
                f"into the project's files before loading it."
            )
        return resolved

    @mcp_tool()
    def load_file_as_table(
        self,
        path: str,
        sheet: Optional[str] = "",
        header_row: Optional[int] = None,
        encoding: Optional[str] = "",
        materialize: bool = False,
        inspect_only: bool = False,
    ) -> FuncToolResult:
        """
        Register an uploaded data file as queryable SQL table(s).

        Use this for any spreadsheet or columnar data file the user refers to —
        ``read_file`` cannot read them, and for anything beyond a handful of rows
        you want SQL anyway (aggregation, joins against the project's own
        database, charting).

        Supported: ``.xlsx``, ``.xlsm``, ``.xls``, ``.csv``, ``.tsv``,
        ``.parquet``, ``.json``, ``.jsonl``. Each sheet of a spreadsheet becomes
        its own table; other formats produce one table.

        The registration is a lazy view over the file, so it is cheap and
        idempotent. Call it again at the start of any analysis rather than
        reasoning about staleness: for CSV/Parquet/JSON every query already re-reads
        the file, and for spreadsheets a reload is what picks up rows appended since
        the last load. Query the result with
        ``execute_sql(sql=..., datasource='local_files')``; the tables are also
        visible to ``list_tables`` / ``describe_table`` on that datasource. Note
        those tables live in a *different* datasource than the project's own
        database, so a query cannot join across the two in one statement.

        Args:
            path: Path to the data file, relative to the project workspace.
            sheet: Spreadsheets only — load just this sheet instead of all of them.
            encoding: CSV/TSV only — override the detected text encoding. Detection
                handles the common cases (UTF-8, a BOM, GB18030 from Excel on a
                Chinese Windows); pass one of ``utf-8``, ``utf-16``, ``gb18030``,
                ``big5``, ``shift_jis``, ``cp1252``, ``latin-1`` when the reported
                ``encoding`` is wrong — the symptom is garbled text in the preview.
            header_row: Spreadsheets only — 1-based row holding the column names.
                Omit to auto-detect. Pass it when the detected header (reported
                back as ``header_row``) picked up a title or a blank row: the
                symptom is column names that read like a sentence, or a row count
                far larger than the real data.
            materialize: Copy the data into a real table instead of a view. Only
                worth it for a large file queried repeatedly; the copy is a
                snapshot and stops tracking edits to the file.
            inspect_only: Register nothing; return the sheet list and the raw
                un-parsed top-left cells. Use it to work out the right
                ``header_row`` for an awkward layout.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[dict]): On success, ``datasource``,
                    ``dialect``, ``tables`` (each with its name, source sheet,
                    detected ``header_row``, row count, per-column profile and a
                    row preview), any ``skipped_sheets``, and ``example_sql``.
        """
        try:
            # A vscode session's files live on the client; ``_resolve_workspace_root``
            # deliberately reports "." there rather than leaking the daemon CWD, so
            # any path resolved here would point at the wrong machine. Say that,
            # instead of returning a "File not found" nobody can explain.
            if getattr(self.agent_config, "_client_source", None) == "vscode":
                return FuncToolResult(
                    success=0,
                    error=(
                        "load_file_as_table needs server-side access to the file, which a "
                        "vscode session does not have — its workspace lives on the client. "
                        "Ask the user to upload the file into the project on the web IDE."
                    ),
                )

            # Registering a file creates objects in the catalog, so it belongs
            # with the other write entry points under the read-only contract.
            # ``inspect_only`` creates nothing and stays available.
            if not inspect_only:
                refusal = self._refuse_write_if_read_only("load_file_as_table")
                if refusal is not None:
                    return refusal

            resolved = self._resolve_data_file(path)
            target = resolved.resolved
            if not target.exists():
                return FuncToolResult(success=0, error=f"File not found: {resolved.display}")
            if not target.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {resolved.display}")

            # Legacy single-connector mode returns the primary connector for ANY
            # requested datasource, so without this the CREATE VIEW / COMMENT ON
            # would land in the user's real database. The tool is not mounted in
            # that mode, but a Python caller can still reach it.
            if LOCAL_FILES_DATASOURCE not in self._datasources:
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Datasource '{LOCAL_FILES_DATASOURCE}' is not configured, so uploaded "
                        f"files cannot be registered in this deployment."
                    ),
                )
            connector = self._get_connector(LOCAL_FILES_DATASOURCE, "")
            exclusive = getattr(connector, "exclusive_connection", None)
            if exclusive is None:
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Datasource '{LOCAL_FILES_DATASOURCE}' is not a DuckDB datasource, "
                        f"so uploaded files cannot be registered in this deployment."
                    ),
                )

            # The tool schema types this as an integer, so a model with nothing to
            # say sends 0 rather than omitting the key — the same way it sends ""
            # for sheet and encoding, and observed in real traffic. Rows are
            # 1-based, so 0 can only mean "unspecified"; taken literally it fails
            # every sheet with "header_row must be 1 or greater" and the file
            # reads as unloadable. A negative value is a real mistake and still
            # reports as one.
            if header_row == 0:
                header_row = None

            with exclusive() as connection:
                if inspect_only:
                    details = inspect_file(target, connection=connection, sheet=sheet or None)
                    return FuncToolResult(result=details)

                loaded, skipped = load_file(
                    target,
                    resolved.display,
                    connection=connection,
                    conversion_cache_dir=default_conversion_cache_dir(getattr(connector, "db_path", "")),
                    sheet=sheet or None,
                    header_row=header_row,
                    encoding=encoding or None,
                    materialize=materialize,
                    existing_objects=registered_objects(connection),
                )

            result: Dict[str, Any] = {
                "datasource": LOCAL_FILES_DATASOURCE,
                "dialect": "duckdb",
                "tables": [item.to_dict() for item in loaded],
            }
            if skipped:
                result["skipped_sheets"] = [item.to_dict() for item in skipped]
            if loaded:
                first = loaded[0]
                result["example_sql"] = (
                    f"SELECT * FROM {quote_identifier(first.table)} LIMIT 10"
                    if not first.preview_columns
                    else f"SELECT {quote_identifier(first.preview_columns[0])}, count(*) AS n "
                    f"FROM {quote_identifier(first.table)} GROUP BY 1 ORDER BY 2 DESC"
                )
                result["usage"] = (
                    f"Query these with execute_sql(sql=..., datasource='{LOCAL_FILES_DATASOURCE}'). "
                    f"Quote any identifier that is not plain ASCII, e.g. "
                    f'SELECT "金额" FROM {first.table}.'
                )
            return FuncToolResult(result=result)

        except DataFileError as e:
            return FuncToolResult(success=0, error=str(e))
        except Exception as e:
            logger.error(f"load_file_as_table failed for {path}: {e}", exc_info=True)
            return FuncToolResult(success=0, error=f"Failed to load {path}: {e}")

    @mcp_tool()
    def execute_sql(
        self,
        sql: str,
        datasource: Optional[str] = "",
        database: Optional[str] = "",
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> FuncToolResult:
        """
        Execute a single SQL statement against the current database connection.

        This is the unified entry point for running SQL. The statement type is
        detected automatically and routed accordingly:

        * Read-only (SELECT, SHOW/DESCRIBE, EXPLAIN) — returns result rows; runs
          without confirmation.
        * DML (INSERT, UPDATE, DELETE) — modifies data and returns write metadata.
        * Any other statement — DDL (CREATE/ALTER/DROP TABLE/VIEW, CREATE/DROP
          SCHEMA or DATABASE, CTAS), plus TRUNCATE, MERGE, GRANT, etc. — runs
          generically and returns execution metadata.

        CAUTION: Everything except a read-only query modifies the database and
        requires user confirmation. Prefer a read-only SELECT for inspection, and
        only run a write/DDL statement when the task explicitly requires it.
        Multi-statement scripts are rejected — submit one statement per call.

        Args:
            sql: A single SQL statement, or a ``.sql`` file path
                (e.g. "sql/session_1/query.sql") to read and execute from the workspace.
            datasource: Optional datasource name for multi-datasource scenarios.
            database: Optional physical database to run against. Required to target a
                specific database of a multi-database datasource (e.g. one file of a
                sqlite/duckdb glob).
            min_rows: Optional minimum acceptable affected row count (DML only).
            max_rows: Optional maximum acceptable affected row count (DML only).

        Returns:
            FuncToolResult: compressed rows for read-only queries, or execution
            metadata for writes/DDL. On failure success=0 with an error message.
        """
        from datus.utils.sql_utils import (
            looks_like_sql_file_ref,
            parse_sql_statement_kind,
            parse_sql_type,
            write_statement_reads_data,
        )

        try:
            # Resolve a ``.sql`` file path up front so type detection inspects the
            # real statement, not the path. The inner methods re-detect the path
            # too, but on resolved SQL the check is a no-op. The permission gate
            # resolves the same file via the shared helper so a read-only .sql
            # file auto-allows instead of prompting. A .sql file must contain a
            # single statement; the downstream read/write/DDL paths each reject
            # multi-statement input.
            sql_stripped = sql.strip()
            if looks_like_sql_file_ref(sql_stripped):
                sql = self._read_sql_from_file(sql_stripped)

            guard_error = self._reject_unsafe_uploads_sql(sql, datasource)
            if guard_error is not None:
                return guard_error

            connector = self._get_connector(datasource, database)
            sql_type = parse_sql_type(sql, connector.dialect)

            if sql_type in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN):
                return self.read_query(sql, datasource=datasource, database=database)
            refusal = self._refuse_write_if_read_only(
                "execute_sql",
                datasource=datasource,
                sql_type=sql_type,
                # The permission layer's finer classification, so the audit line
                # says `drop` rather than a `ddl` that also covers CREATE.
                statement_kind=parse_sql_statement_kind(sql, connector.dialect),
                error=(
                    "This agent is read-only: only SELECT/SHOW/DESCRIBE/EXPLAIN "
                    "statements are allowed through execute_sql."
                ),
            )
            if refusal:
                return refusal

            # A write can carry a read. `CREATE TABLE mine AS SELECT * FROM
            # orders` is approved as a write, runs on the raw connector, and
            # lands every row the policy just withheld in a table no policy
            # covers — the person who may only SELECT two stores now owns all
            # four. The plugin cannot see it: it hooks reads, and this is a
            # write. So on a project that has a policy context at all, a write
            # that embeds a query is refused before it is dispatched.
            #
            # The permission prompt is not this check. It asks whether the
            # *user* consents to a write; consenting to a write they are
            # allowed to make is not consent to read rows they are not.
            #
            # Ordered after the read-only gate above: a hardened deployment
            # refuses every write outright, so it never needs to reason about
            # what the write reads.
            if self.policy_context and write_statement_reads_data(sql, connector.dialect):
                return FuncToolResult(
                    success=0,
                    error=(
                        "This project has row-level policies, so a write statement that "
                        "reads from a query is not allowed — it would copy filtered rows "
                        "into a table no policy covers. Create views and derived tables "
                        "on the database side instead."
                    ),
                )

            if sql_type in (SQLType.INSERT, SQLType.UPDATE, SQLType.DELETE):
                return self.execute_write(
                    sql,
                    datasource=datasource,
                    database=database,
                    min_rows=min_rows,
                    max_rows=max_rows,
                )
            # Any other statement — DDL (CREATE/ALTER/DROP, CREATE DATABASE, ...),
            # MERGE, or engine-specific commands. The permission layer has already
            # gated non-read SQL behind confirmation, so execute it generically
            # rather than rejecting it by sub-type. Only multi-statement scripts
            # are refused (one statement per call).
            return self.execute_ddl(sql, datasource=datasource, database=database)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def read_query(self, sql: str, datasource: Optional[str] = "", database: Optional[str] = "") -> FuncToolResult:
        """
        Execute a read-only SQL query and return the result rows (optionally compressed).

        Internal read path used by :meth:`execute_sql` and by Python callers
        (e.g. reference-template execution). Not exposed to the LLM as its own tool.

        Only SELECT, SHOW/DESCRIBE, and EXPLAIN statements are allowed.
        DML (INSERT/UPDATE/DELETE) and DDL (CREATE/ALTER/DROP) are rejected.

        Args:
            sql: Read-only SQL text (SELECT, SHOW, DESCRIBE, EXPLAIN), or a .sql file path
                 (e.g. "sql/session_1/query.sql") to read and execute from the workspace.
            datasource: Optional datasource name for multi-datasource scenarios.
            database: Optional physical database to run against. Required to target a specific
                database of a multi-database datasource (e.g. one file of a sqlite/duckdb glob).

        Returns:
            FuncToolResult with result=self.compressor.compress(rows) when successful. On failure success=0 with the
            underlying error message from the connector.
        """
        from datus.utils.sql_utils import looks_like_sql_file_ref

        try:
            # Support SQL file path: if sql is a simple path ending with .sql, read from file
            sql_stripped = sql.strip()
            if looks_like_sql_file_ref(sql_stripped):
                sql = self._read_sql_from_file(sql_stripped)

            connector = self._get_connector(datasource, database)
            validation_error, sql_type = self._validate_read_sql(sql, connector)
            if validation_error:
                return validation_error

            # Resolved rather than the raw argument: the refusal logs resolve it
            # too, and an operator correlating a session on ``datasource`` cannot
            # do it if the same source appears as "default" on one line and by
            # name on the next.
            logger.info(
                "read_query",
                sql_type=sql_type.value,
                datasource=self._resolve_effective_datasource(datasource),
            )
            result_format = "arrow" if connector.dialect == "snowflake" else "list"
            result = self.execute_read_enforced(
                sql,
                connector,
                datasource=datasource,
                result_format=result_format,
            )
            if result.success:
                return FuncToolResult(result=self.compressor.compress(result.sql_return))
            return FuncToolResult(success=0, error=result.error)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def execute_read_enforced(
        self,
        sql: str,
        connector: BaseSqlConnector,
        *,
        datasource: Optional[str] = "",
        result_format: str = "list",
        policy_context: Optional[Dict[str, Any]] = None,
    ) -> ExecuteSQLResult:
        """Run a read-only query through the shared read guardrails.

        Single enforcement path for every read that hits the DB directly: the
        LLM ``read_query`` path plus the report/dashboard artifact save paths
        and dashboard view-time re-execution. Rejects multi-statement input and
        applies configured policy runtimes (rewrites / denials) before
        the statement reaches the engine, then returns the connector's raw
        ``ExecuteSQLResult`` after result policies have run. Without this,
        artifact query execution bypassed policy enforcement
        and could hand the engine an unbounded statement (e.g. a cross-join
        cartesian product that OOM-killed the DB backend).
        """
        validation_error, _ = self._validate_read_sql(sql, connector)
        if validation_error:
            return ExecuteSQLResult(success=False, error=validation_error.error, sql_query=sql)
        effective_datasource = self._resolve_effective_datasource(datasource)
        effective_policy_context = self.policy_context if policy_context is None else policy_context
        try:
            from datus.tools.policy_runtime import PolicyRuntime

            runtime = PolicyRuntime(self.agent_config)
            decision = runtime.before_sql_read(
                sql,
                datasource=effective_datasource,
                dialect=connector.dialect,
                policy_context=effective_policy_context,
            )
            if not decision.allowed:
                raise DatusException(
                    ErrorCode.POLICY_DENIED,
                    message=decision.reason or "Policy denied the query",
                )
            enforced_sql = decision.sql if decision.sql is not None else sql
            if decision.applied_policies:
                logger.info(
                    "Applied pre-read policies",
                    policies=decision.applied_policies,
                    datasource=effective_datasource,
                )
        except DatusException as exc:
            return ExecuteSQLResult(
                success=False,
                error=self._policy_error_text(exc),
                error_code=exc.code.code,
                sql_query=sql,
            )
        if enforced_sql != sql:
            # A policy rewrite (e.g. an injected LIMIT) must still be a single
            # read-only statement — re-validate so it can't smuggle in DML/DDL.
            validation_error, _ = self._validate_read_sql(enforced_sql, connector)
            if validation_error:
                return ExecuteSQLResult(success=False, error=validation_error.error, sql_query=enforced_sql)
        result = connector.execute_query(enforced_sql, result_format=result_format)
        if not result.success:
            return result
        try:
            result_decision = runtime.after_read_result(
                result.sql_return,
                sql=enforced_sql,
                datasource=effective_datasource,
                dialect=connector.dialect,
                policy_context=effective_policy_context,
            )
            if not result_decision.allowed:
                raise DatusException(
                    ErrorCode.POLICY_DENIED,
                    message=result_decision.reason or "Policy denied the query result",
                )
            result.sql_return = result_decision.result
            if result_decision.applied_policies:
                logger.info(
                    "Applied result policies",
                    policies=result_decision.applied_policies,
                    datasource=effective_datasource,
                )
            return result
        except DatusException as exc:
            return ExecuteSQLResult(
                success=False,
                error=self._policy_error_text(exc),
                error_code=exc.code.code,
                sql_query=enforced_sql,
            )

    @staticmethod
    def _policy_error_text(exc: DatusException) -> str:
        """What to put in ``error`` for a failure that reached the caller.

        A policy refusal is already a sentence written for whoever is reading
        it — SaaS composes it from the attribute they are missing — so it goes
        out without the ``error_code=`` prefix that would otherwise be the
        first thing they see. Everything else keeps the prefixed form the logs
        and the existing callers expect.
        """
        return exc.detail if exc.code is ErrorCode.POLICY_DENIED else str(exc)

    def guard_estimated_rows(
        self,
        sql: str,
        connector: BaseSqlConnector,
    ) -> Optional[FuncToolResult]:
        """Reject a query whose EXPLAIN row estimate exceeds ``MAX_ESTIMATED_ROWS``.

        Runs ``EXPLAIN <sql>`` (planning only — the statement never executes),
        parses the optimizer's cardinality estimate, and returns a failure
        ``FuncToolResult`` (with an actionable rewrite message for the LLM) when
        it blows past the ceiling. Returns ``None`` to let the query proceed.

        Fail-open: EXPLAIN being unsupported / erroring / unparseable all yield
        ``None`` — the guard only ever blocks on a *confident* oversize estimate,
        never on its own inability to measure.
        """
        from datus.tools.sql_guard import MAX_ESTIMATED_ROWS, build_oversize_message, estimate_rows_from_explain

        # Reject multi-statement / non-read input before it reaches the engine
        # inside ``EXPLAIN <sql>``. Callers only prefix-check with
        # ``_looks_like_select``, which passes ``SELECT 1; DROP TABLE t`` — and a
        # driver that splits statements would run the DROP as part of the EXPLAIN.
        validation_error, _ = self._validate_read_sql(sql, connector)
        if validation_error:
            return validation_error

        try:
            explain_result = connector.execute_query(f"EXPLAIN {sql}", result_format="list")
        except Exception as exc:
            logger.debug("sql_guard EXPLAIN failed; allowing query", error=str(exc), dialect=connector.dialect)
            return None
        if not getattr(explain_result, "success", False):
            return None

        estimated = estimate_rows_from_explain(connector.dialect, explain_result.sql_return or [])
        if estimated is None or estimated <= MAX_ESTIMATED_ROWS:
            return None

        logger.warning(
            "sql_guard rejected oversized query",
            estimated_rows=estimated,
            threshold=MAX_ESTIMATED_ROWS,
            dialect=connector.dialect,
        )
        return FuncToolResult(success=0, error=build_oversize_message(estimated, MAX_ESTIMATED_ROWS))

    def _resolve_effective_datasource(self, datasource: Optional[str]) -> str:
        effective_datasource = datasource or self._default_datasource
        if not effective_datasource and self.agent_config:
            services = getattr(self.agent_config, "services", None)
            effective_datasource = getattr(services, "default_datasource", "") or ""
        return effective_datasource or "default"

    def _validate_read_sql(self, sql: str, connector: BaseSqlConnector) -> tuple[Optional[FuncToolResult], SQLType]:
        from datus.utils.sql_utils import (
            READ_ONLY_MULTI_STATEMENT,
            READ_ONLY_NON_READ,
            READ_ONLY_WRITABLE_PRAGMA,
            validate_read_only_sql,
        )

        violation, sql_type = validate_read_only_sql(sql, connector.dialect)
        if violation:
            error = {
                READ_ONLY_MULTI_STATEMENT: "Multi-statement SQL is not allowed. Please submit one query at a time.",
                READ_ONLY_NON_READ: (
                    f"Only read-only queries (SELECT, SHOW, DESCRIBE, EXPLAIN) are allowed. "
                    f"Detected SQL type: {sql_type.value}"
                ),
                READ_ONLY_WRITABLE_PRAGMA: "Writable PRAGMA statements are not allowed in read-only mode.",
            }[violation]
            # Logged for the same reason as the write-path refusals, and this
            # branch matters more than it looks: ``parse_sql_type`` classifies
            # only the FIRST statement, so ``SELECT 1; DROP TABLE t`` reaches
            # here as a SELECT and is refused by the multi-statement rule rather
            # than by the read-only gate. Without this line the sneakiest input
            # in the set would be the one that left no audit trail, while a
            # plain DROP TABLE was recorded.
            #
            # ``rule`` rather than ``source``: these refusals hold regardless of
            # ``sql_read_only``, so labelling them "deployment" would overstate
            # what the switch is doing. The violation code is the value, so an
            # operator can aggregate on it without parsing prose.
            logger.warning(
                "read_query rejected by statement-shape rules",
                sql_type=sql_type.value,
                datasource=self._resolve_effective_datasource(None),
                sub_agent=self.sub_agent_name or "",
                rule=violation,
            )
            return FuncToolResult(success=0, error=error), sql_type

        out_of_scope = self._check_sql_table_scope(sql, connector)
        if out_of_scope:
            return (
                FuncToolResult(
                    success=0,
                    error=f"Query references tables outside scoped context: {', '.join(out_of_scope)}",
                ),
                sql_type,
            )
        return None, sql_type

    def get_table_ddl(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        datasource: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Return the connector's DDL definition for the requested table.

        Use this when the agent needs a full CREATE statement (e.g. for semantic modelling or schema verification).

        Args:
            table_name: Target table identifier (supports partial qualification).
            catalog: Optional catalog override.
            database: Optional database override.
            schema_name: Optional schema override.
            datasource: Optional datasource to route the query to. Defaults to the current datasource.

        Returns:
            FuncToolResult with result dict containing keys:
                identifier, catalog_name, database_name, schema_name, table_name, table_type, definition.
            Scoped-context mismatches or connector failures surface as success=0 with an explanatory message.
        """
        try:
            catalog, database, schema_name = self._normalize_namespace_args(
                catalog,
                database,
                schema_name,
                datasource,
            )
            connector = self._get_connector(datasource, database)
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
                connector=connector,
            )
            if not self._table_matches_scope(coordinate):
                return FuncToolResult(
                    success=0,
                    error=f"Table '{table_name}' is outside the scoped context.",
                )
            # Get tables with DDL
            connector = self._get_connector(datasource, coordinate.database)
            tables_with_ddl = connector.get_tables_with_ddl(
                catalog_name=coordinate.catalog,
                database_name=coordinate.database,
                schema_name=coordinate.schema,
                tables=[coordinate.table],
            )

            if not tables_with_ddl:
                return FuncToolResult(success=0, error=f"Table '{table_name}' not found or no DDL available")

            # Return the first (and only) table's DDL
            table_info = tables_with_ddl[0]
            return FuncToolResult(result=table_info)

        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    # Regex matching allowed DDL statement prefixes
    def execute_ddl(self, sql: str, datasource: Optional[str] = "", database: Optional[str] = "") -> FuncToolResult:
        """
        Execute a single non-read, non-DML SQL statement (the generic write path).

        CAUTION: This modifies the database. Only use when explicitly instructed.
        Handles DDL (CREATE/ALTER/DROP TABLE/VIEW, CREATE/DROP SCHEMA or DATABASE,
        CTAS), as well as other non-query statements (TRUNCATE, MERGE, GRANT,
        CREATE INDEX, engine-specific commands). Statement-type permission gating
        lives in ``PermissionHooks._handle_sql_permission``; this method does not
        re-gate by sub-type. Read-only and INSERT/UPDATE/DELETE statements have
        dedicated paths and are rejected here.

        Args:
            sql: DDL SQL statement to execute
            datasource: Optional datasource name for multi-datasource scenarios.

        Returns:
            Execution result with success status
        """
        from datus.utils.sql_utils import _first_statement, parse_sql_type, strip_sql_comments

        # Reachable directly, not only via the ``execute_sql`` dispatch that
        # already gated: gen_job-style callers and host code hold the tool
        # itself. Gate before parsing so a hardened deployment refuses on
        # posture, never on statement shape.
        refusal = self._refuse_write_if_read_only("execute_ddl", datasource=datasource)
        if refusal:
            return refusal

        # Validate: strip comments, reject multi-statement SQL
        cleaned = strip_sql_comments(sql).strip().rstrip(";").strip()
        if not cleaned:
            return FuncToolResult(success=0, error="Empty SQL statement")

        # Use the quote-aware parser, not a raw ``";" in cleaned`` check, so a
        # single statement with a semicolon inside a string literal or quoted
        # identifier (e.g. ``COMMENT ON ... IS 'a;b'``) is not falsely rejected.
        if _first_statement(cleaned) != cleaned:
            return FuncToolResult(
                success=0,
                error="Multi-statement SQL is not allowed. Please submit one statement at a time.",
            )

        connector = self._get_connector(datasource, database)

        # Generic non-query execution path. There is NO sub-type allow-list:
        # once the permission layer has approved a non-read statement, run it
        # (CREATE/ALTER/DROP, CREATE DATABASE, TRUNCATE, MERGE, GRANT, ...). The
        # only guard is defense-in-depth: read-only and DML statements have
        # dedicated paths (read_query / execute_write) and must not land here.
        stmt_type = parse_sql_type(cleaned, connector.dialect)
        if stmt_type in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN):
            return FuncToolResult(
                success=0,
                error="Read-only statements (SELECT/SHOW/DESCRIBE/EXPLAIN) must run through the read path.",
            )
        if stmt_type in (SQLType.INSERT, SQLType.UPDATE, SQLType.DELETE):
            return FuncToolResult(
                success=0,
                error="DML statements (INSERT/UPDATE/DELETE) must run through the write path.",
            )

        out_of_scope = self._check_sql_table_scope(cleaned, connector)
        if out_of_scope:
            return FuncToolResult(
                success=0,
                error=f"Statement references tables outside scoped context: {', '.join(out_of_scope)}",
            )

        if not hasattr(connector, "execute_ddl"):
            return FuncToolResult(success=0, error="Current database connector does not support DDL operations")
        try:
            result = connector.execute_ddl(cleaned)
            if result.success:
                # Commit to release locks (critical for SQLAlchemy-based connectors)
                if hasattr(connector, "connection") and hasattr(connector.connection, "commit"):
                    connector.connection.commit()
                from datus.validation.target_extractor import extract_ddl_target

                effective_ds = datasource or self._default_datasource
                target = extract_ddl_target(
                    cleaned,
                    effective_ds,
                    active_database=self._active_database_of(connector),
                    dialect=getattr(connector, "dialect", ""),
                )
                result_payload: Dict[str, Any] = {
                    "message": "DDL executed successfully",
                    "sql": cleaned,
                    "datasource": effective_ds,
                }
                if target is not None:
                    result_payload["deliverable_target"] = target.model_dump(by_alias=True, exclude_none=True)
                return FuncToolResult(result=result_payload)
            else:
                return FuncToolResult(success=0, error=result.error)
        except Exception as e:
            return FuncToolResult(success=0, error=f"DDL execution failed: {str(e)}")

    def execute_write(
        self,
        sql: str,
        datasource: Optional[str] = "",
        database: Optional[str] = "",
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        dry_run: bool = False,
    ) -> FuncToolResult:
        """
        Execute a single write statement against the current database connection.

        Supported statements: INSERT, UPDATE, DELETE.
        Multi-statement SQL, read-only queries, DDL, and MERGE are rejected.

        Args:
            sql: Write SQL statement to execute, or a .sql file path.
            datasource: Optional datasource name for multi-datasource scenarios.
            min_rows: Optional minimum acceptable affected row count.
                Checked after the write is committed; violation returns success=0
                but the write is NOT rolled back.
            max_rows: Optional maximum acceptable affected row count.
                Checked after the write is committed; violation returns success=0
                but the write is NOT rolled back.
            dry_run: Reserved for future transactional preview support. Currently unsupported.

        Returns:
            FuncToolResult with execution metadata when successful.
        """
        from datus.utils.sql_utils import (
            _first_statement,
            looks_like_sql_file_ref,
            parse_sql_type,
            strip_sql_comments,
        )

        if dry_run:
            return FuncToolResult(
                success=0,
                error="dry_run is not supported yet for execute_write. Use dry_run=False.",
            )

        # Same reasoning as ``execute_ddl``: ``execute_sql`` has already gated by
        # the time it dispatches here, but this method is also callable directly.
        refusal = self._refuse_write_if_read_only("execute_write", datasource=datasource)
        if refusal:
            return refusal

        try:
            sql_stripped = sql.strip()
            if looks_like_sql_file_ref(sql_stripped):
                sql = self._read_sql_from_file(sql_stripped)

            cleaned = strip_sql_comments(sql).strip()
            normalized_sql = cleaned.rstrip(";").strip()
            if not normalized_sql:
                return FuncToolResult(success=0, error="Empty SQL statement")

            if _first_statement(normalized_sql) != normalized_sql:
                return FuncToolResult(
                    success=0,
                    error="Multi-statement SQL is not allowed. Please submit one write statement at a time.",
                )

            connector = self._get_connector(datasource, database)
            sql_type = parse_sql_type(normalized_sql, connector.dialect)
            if sql_type == SQLType.MERGE:
                return FuncToolResult(
                    success=0,
                    error="MERGE statements are not supported by execute_write yet.",
                )

            allowed_sql_types = {SQLType.INSERT, SQLType.UPDATE, SQLType.DELETE}
            if sql_type not in allowed_sql_types:
                return FuncToolResult(
                    success=0,
                    error=(
                        "Only single-statement writes (INSERT, UPDATE, DELETE) are allowed. "
                        f"Detected SQL type: {sql_type.value}"
                    ),
                )

            out_of_scope = self._check_sql_table_scope(normalized_sql, connector)
            if out_of_scope:
                return FuncToolResult(
                    success=0,
                    error=f"Write statement references tables outside scoped context: {', '.join(out_of_scope)}",
                )

            method_name = {
                SQLType.INSERT: "execute_insert",
                SQLType.UPDATE: "execute_update",
                SQLType.DELETE: "execute_delete",
            }[sql_type]

            if not hasattr(connector, method_name):
                return FuncToolResult(
                    success=0,
                    error=f"Current database connector does not support {sql_type.value.upper()} operations",
                )

            result = getattr(connector, method_name)(normalized_sql)
            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            # Commit to release locks (critical for SQLAlchemy-based connectors)
            if hasattr(connector, "connection") and hasattr(connector.connection, "commit"):
                connector.connection.commit()

            row_count = getattr(result, "row_count", None)
            if (min_rows is not None or max_rows is not None) and row_count is None:
                return FuncToolResult(
                    success=0,
                    error="Connector did not report row_count but min_rows/max_rows was requested. "
                    "Cannot verify the safety bound. Note: the write has already been committed.",
                )
            if min_rows is not None and row_count is not None and row_count < min_rows:
                return FuncToolResult(
                    success=0,
                    error=f"Write affected {row_count} rows, below min_rows={min_rows}. "
                    "Note: the write has already been committed.",
                )
            if max_rows is not None and row_count is not None and row_count > max_rows:
                return FuncToolResult(
                    success=0,
                    error=f"Write affected {row_count} rows, above max_rows={max_rows}. "
                    "Note: the write has already been committed.",
                )

            from datus.validation.target_extractor import extract_dml_target

            effective_ds = datasource or self._default_datasource
            target = extract_dml_target(
                normalized_sql,
                effective_ds,
                active_database=self._active_database_of(connector),
                dialect=getattr(connector, "dialect", ""),
            )
            result_payload: Dict[str, Any] = {
                "message": "Write executed successfully",
                "sql": normalized_sql,
                "sql_type": sql_type.value,
                "row_count": row_count,
                "datasource": effective_ds,
                "dry_run": dry_run,
            }
            if target is not None:
                if row_count is not None:
                    target = target.model_copy(update={"rows_affected": row_count})
                result_payload["deliverable_target"] = target.model_dump(by_alias=True, exclude_none=True)
            return FuncToolResult(result=result_payload)
        except Exception as e:
            return FuncToolResult(success=0, error=f"Write execution failed: {str(e)}")

    # Maximum rows allowed in a single transfer (v1 memory constraint)
    _TRANSFER_MAX_ROWS = 1_000_000

    @staticmethod
    def _identifier_quote_char(dialect: str) -> str:
        backtick_dialects = ("mysql", "starrocks", "doris", "hive", "spark", "bigquery", "clickhouse")
        return "`" if dialect in backtick_dialects else '"'

    @classmethod
    def _quote_column_identifier(cls, name: Any, dialect: str) -> str:
        text = str(name)
        if not text or "\x00" in text:
            raise ValueError(f"Invalid column name: {text!r}")
        quote_char = cls._identifier_quote_char(dialect)
        escaped = text.replace(quote_char, quote_char * 2)
        return f"{quote_char}{escaped}{quote_char}"

    @staticmethod
    def _is_missing_target_table_error(error: Any) -> bool:
        text = str(error or "").lower()
        missing_markers = (
            "does not exist",
            "doesn't exist",
            "no such table",
            "not found",
            "undefinedtable",
            "unknown table",
            "table_not_exists",
        )
        object_markers = ("table", "relation", "object", "catalog", "schema")
        return any(marker in text for marker in missing_markers) and any(marker in text for marker in object_markers)

    @classmethod
    def _infer_transfer_column_type(cls, series: Any, dialect: str) -> str:
        from datetime import date, datetime, time
        from decimal import Decimal

        from pandas.api import types as pd_types

        dialect = parse_dialect(str(dialect or "")).lower()

        def choose(default: str, *, sqlite: str = "", postgres: str = "", duckdb: str = "") -> str:
            if dialect == DBType.SQLITE:
                return sqlite or default
            if dialect in ("postgresql", "postgres"):
                return postgres or default
            if dialect == DBType.DUCKDB:
                return duckdb or default
            return default

        if pd_types.is_bool_dtype(series):
            return choose("BOOLEAN", sqlite="INTEGER")
        if pd_types.is_integer_dtype(series):
            return choose("BIGINT", sqlite="INTEGER")
        if pd_types.is_float_dtype(series):
            return choose("DOUBLE", sqlite="REAL", postgres="DOUBLE PRECISION")
        if pd_types.is_datetime64_any_dtype(series):
            return choose("TIMESTAMP", sqlite="TEXT")
        if pd_types.is_timedelta64_dtype(series):
            return choose("TEXT", duckdb="INTERVAL")

        non_null = series.dropna()
        if not non_null.empty:
            value = non_null.iloc[0]
            if isinstance(value, bool):
                return choose("BOOLEAN", sqlite="INTEGER")
            if isinstance(value, int):
                return choose("BIGINT", sqlite="INTEGER")
            if isinstance(value, float):
                return choose("DOUBLE", sqlite="REAL", postgres="DOUBLE PRECISION")
            if isinstance(value, Decimal):
                return "NUMERIC"
            if isinstance(value, datetime):
                return choose("TIMESTAMP", sqlite="TEXT")
            if isinstance(value, date):
                return choose("DATE", sqlite="TEXT")
            if isinstance(value, time):
                return choose("TIME", sqlite="TEXT")
            if isinstance(value, (bytes, bytearray, memoryview)):
                return choose("TEXT", duckdb="VARCHAR")
            if isinstance(value, (dict, list, tuple)):
                return choose("TEXT", duckdb="VARCHAR")

        return choose("TEXT", duckdb="VARCHAR")

    def _create_transfer_target_table(self, target_conn: Any, target_table: str, df: Any) -> FuncToolResult:
        if not hasattr(target_conn, "execute_ddl"):
            return FuncToolResult(success=0, error="Target datasource connector does not support DDL operations")

        columns = list(df.columns)
        if not columns:
            return FuncToolResult(
                success=0, error="Cannot create target table because the source query returned no columns"
            )

        dialect = str(getattr(target_conn, "dialect", "") or "").lower()
        operations = get_dialect_operations(connector=target_conn)
        seen_columns = set()
        column_defs = []
        try:
            for column in columns:
                column_key = str(column).casefold()
                if column_key in seen_columns:
                    return FuncToolResult(
                        success=0,
                        error=f"Cannot create target table because source query returned duplicate column '{column}'",
                    )
                seen_columns.add(column_key)
                quoted_column = (
                    operations.quote_identifier(column)
                    if operations is not None
                    else self._quote_column_identifier(column, dialect)
                )
                column_type = (
                    operations.infer_transfer_type(df[column])
                    if operations is not None
                    else self._infer_transfer_column_type(df[column], dialect)
                )
                column_defs.append(f"  {quoted_column} {column_type}")
        except Exception as e:
            return FuncToolResult(success=0, error=f"Failed to infer target table schema: {str(e)}")

        create_sql = f"CREATE TABLE {target_table} (\n" + ",\n".join(column_defs) + "\n)"
        try:
            create_result = target_conn.execute_ddl(create_sql)
            if not create_result.success:
                return FuncToolResult(success=0, error=f"Failed to create target table: {create_result.error}")
            if hasattr(target_conn, "connection") and hasattr(target_conn.connection, "commit"):
                target_conn.connection.commit()
        except Exception as e:
            return FuncToolResult(success=0, error=f"Failed to create target table: {str(e)}")

        return FuncToolResult(result={"sql": create_sql})

    def transfer_query_result(
        self,
        source_sql: str,
        source_datasource: Optional[str] = "",
        target_table: str = "",
        target_datasource: Optional[str] = "",
        mode: str = "replace",
        batch_size: int = 5000,
    ) -> FuncToolResult:
        """
        Transfer query results from a source datasource to a target table in another datasource.

        Executes source_sql on source_datasource, fetches the result as a DataFrame,
        and batch-inserts into target_table on target_datasource.

        Args:
            source_sql: SQL query to execute on the source datasource.
            source_datasource: Source datasource name. Uses default datasource if empty.
            target_table: Fully qualified target table name.
            target_datasource: Target datasource name. Uses default datasource if empty.
            mode: Transfer mode - 'replace' (TRUNCATE + INSERT, creating the target table if missing)
                  or 'append' (INSERT only).
            batch_size: Number of rows per INSERT batch.

        Returns:
            FuncToolResult with transfer metadata on success.
        """
        # ``source_sql`` is validated as read-only below, but the transfer WRITES
        # to ``target_datasource`` — CREATE TABLE / TRUNCATE / INSERT — without
        # ever going through ``execute_sql``. gen_job mounts this method as a
        # tool directly, so without this gate a hardened deployment would still
        # expose a cross-datasource write.
        refusal = self._refuse_write_if_read_only("transfer_query_result", datasource=target_datasource)
        if refusal:
            return refusal

        # Validate batch_size
        if batch_size <= 0:
            return FuncToolResult(success=0, error="batch_size must be a positive integer.")

        # Validate target_table identifier
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", target_table):
            return FuncToolResult(
                success=0,
                error=f"Invalid target_table identifier: '{target_table}'. "
                "Only alphanumeric characters, underscores, and dots are allowed.",
            )

        # Validate mode
        if mode not in ("replace", "append"):
            return FuncToolResult(
                success=0,
                error=f"Invalid mode '{mode}'. Supported modes: 'replace', 'append'.",
            )

        # Validate source_sql: must be a single read-only statement
        from datus.utils.sql_utils import _first_statement, parse_sql_type, strip_sql_comments

        cleaned_sql = strip_sql_comments(source_sql).strip().rstrip(";").strip()
        if not cleaned_sql:
            return FuncToolResult(success=0, error="source_sql is empty.")
        if _first_statement(cleaned_sql) != cleaned_sql:
            return FuncToolResult(
                success=0,
                error="Multi-statement source_sql is not allowed. Please submit one SELECT query.",
            )
        sql_type = parse_sql_type(cleaned_sql, "")
        if sql_type not in (SQLType.SELECT, SQLType.METADATA_SHOW):
            return FuncToolResult(
                success=0,
                error=f"source_sql must be a SELECT query, got {sql_type.value.upper()}. "
                "Only read-only queries are allowed as transfer source.",
            )

        # Get connectors — both must be available; do NOT fall back to a different datasource
        try:
            source_conn = self._get_connector(source_datasource)
        except Exception as e:
            return FuncToolResult(
                success=0,
                error=f"Source datasource '{source_datasource}' is not available: {str(e)}. "
                "Check that the adapter is installed and the connection config is correct. "
                "Do NOT fall back to a different source datasource.",
            )
        try:
            target_conn = self._get_connector(target_datasource)
        except Exception as e:
            return FuncToolResult(
                success=0,
                error=f"Target datasource '{target_datasource}' is not available: {str(e)}. "
                "Check that the adapter is installed and the connection config is correct. "
                "Do NOT fall back to a different target datasource — STOP and report this error to the user.",
            )

        source_operations = get_dialect_operations(connector=source_conn)
        target_operations = get_dialect_operations(connector=target_conn)

        # Authoritative source row count — wrap the user's source_sql in a COUNT
        # subquery so reconciliation does not need to re-run anything later.
        # One extra query is cheap on OLTP engines and still acceptable on
        # warehouse engines; see ValidationHook design doc §5.4.
        source_row_count: Optional[int] = None
        try:
            if hasattr(source_conn, "execute_query"):
                count_sql = (
                    source_operations.render_count(cleaned_sql, "__datus_src")
                    if source_operations is not None
                    else f"SELECT COUNT(*) AS __datus_count FROM ({cleaned_sql}) AS __datus_src"
                )
                count_result = source_conn.execute_query(count_sql)
                if count_result.success and count_result.sql_return:
                    # execute_query returns a list of rows; first row, first col is the count
                    first_row = count_result.sql_return[0]
                    if isinstance(first_row, dict):
                        source_row_count = int(next(iter(first_row.values())))
                    else:
                        source_row_count = int(first_row[0])
        except Exception as e:
            logger.debug("Source row count pre-check failed (non-fatal): %s", e)

        # Execute source query
        try:
            if not hasattr(source_conn, "execute_pandas"):
                return FuncToolResult(
                    success=0,
                    error="Source datasource connector does not support pandas execution.",
                )
            source_result = source_conn.execute_pandas(source_sql)
            if not source_result.success:
                return FuncToolResult(success=0, error=f"Source query failed: {source_result.error}")
            df = source_result.sql_return
        except Exception as e:
            return FuncToolResult(success=0, error=f"Source query execution failed: {str(e)}")

        # Check row limit
        row_count = len(df)
        # If the wrapped COUNT(*) pre-check could not run (unsupported
        # subquery on some engines, connector shape mismatch), the full
        # source result is still materialized in ``df`` — use its row
        # count as the authoritative ``source_row_count`` so Layer A's
        # parity check remains meaningful instead of being skipped.
        if source_row_count is None:
            source_row_count = row_count
        if row_count > self._TRANSFER_MAX_ROWS:
            return FuncToolResult(
                success=0,
                error=f"Result set has {row_count:,} rows, exceeding the {self._TRANSFER_MAX_ROWS:,} row limit. "
                "Please add WHERE conditions to transfer in smaller batches.",
            )

        # TRUNCATE for replace mode BEFORE empty check - mode="replace" must clear old data.
        # If the target table does not exist yet, create it from the source result schema so
        # first-time transfers do not require a separate hand-written DDL step.
        target_table_created = False
        target_table_create_sql = None
        if mode == "replace":
            try:
                truncate_result = target_conn.execute_ddl(f"TRUNCATE TABLE {target_table}")
                if not truncate_result.success:
                    if self._is_missing_target_table_error(truncate_result.error):
                        create_result = self._create_transfer_target_table(target_conn, target_table, df)
                        if not create_result.success:
                            return create_result
                        target_table_created = True
                        target_table_create_sql = create_result.result["sql"]
                    else:
                        return FuncToolResult(
                            success=0,
                            error=f"Failed to truncate target table: {truncate_result.error}",
                        )
            except Exception as e:
                if self._is_missing_target_table_error(e):
                    create_result = self._create_transfer_target_table(target_conn, target_table, df)
                    if not create_result.success:
                        return create_result
                    target_table_created = True
                    target_table_create_sql = create_result.result["sql"]
                else:
                    return FuncToolResult(success=0, error=f"Failed to truncate target table: {str(e)}")

        # Handle empty result (after truncate so replace mode still clears old data)
        if row_count == 0:
            logger.info(f"Source query returned 0 rows, nothing to transfer to {target_table}")
            if target_table_created:
                message = "Transfer completed (empty result set - target table created)"
            elif mode == "replace":
                message = "Transfer completed (empty result set - target table truncated)"
            else:
                message = "Transfer completed (empty result set)"
            return FuncToolResult(
                result={
                    "message": message,
                    "source_sql": source_sql,
                    "source_datasource": source_datasource,
                    "target_table": target_table,
                    "target_datasource": target_datasource or self._default_datasource,
                    "mode": mode,
                    "rows_transferred": 0,
                    "target_table_created": target_table_created,
                    "target_table_create_sql": target_table_create_sql,
                    # Leave as None when the pre-count failed; 0 is a legitimate
                    # verified value (empty source). See _build_transfer_target.
                    "source_row_count": source_row_count,
                    "source_row_count_verified": source_row_count is not None,
                    "transferred_row_count": 0,
                    "batch_size": batch_size,
                    "deliverable_target": self._build_transfer_target(
                        source_datasource=source_datasource,
                        target_datasource=target_datasource or self._default_datasource,
                        target_table=target_table,
                        source_row_count=source_row_count,
                        transferred_row_count=0,
                        target_active_database=self._active_database_of(target_conn),
                    ),
                }
            )

        # Convert pandas NaT/NaN to Python None for DBAPI2 compatibility
        df = df.where(df.notna(), other=None)
        # Also convert numpy types to native Python types
        df = df.astype(object).where(df.notna(), other=None)

        rows_written = 0
        try:
            if target_operations is not None:
                rows_written = int(
                    target_operations.write_dataframe(
                        target_conn,
                        target_table,
                        df,
                        batch_size,
                    )
                )
            else:
                # Legacy adapters keep the existing inline multi-row INSERT path.
                columns = list(df.columns)
                dialect = str(getattr(target_conn, "dialect", "") or "").lower()
                col_names = ", ".join(self._quote_column_identifier(c, dialect) for c in columns)
                for batch_start in range(0, row_count, batch_size):
                    batch_end = min(batch_start + batch_size, row_count)
                    batch_df = df.iloc[batch_start:batch_end]

                    value_rows = []
                    for _, row in batch_df.iterrows():
                        values = []
                        for val in row:
                            if val is None:
                                values.append("NULL")
                            elif isinstance(val, bool):
                                values.append("TRUE" if val else "FALSE")
                            elif isinstance(val, (int, float)):
                                values.append(str(val))
                            else:
                                escaped = str(val).replace("'", "''")
                                values.append(f"'{escaped}'")
                        value_rows.append(f"({', '.join(values)})")

                    insert_sql = f"INSERT INTO {target_table} ({col_names}) VALUES {', '.join(value_rows)}"
                    result = target_conn.execute_insert(insert_sql)
                    if not result.success:
                        return FuncToolResult(
                            success=0,
                            error=f"Transfer failed after writing {rows_written} rows: {result.error}",
                        )
                    rows_written += len(batch_df)

                # Commit the transaction to release locks (critical for SQLAlchemy-based connectors)
                if hasattr(target_conn, "connection") and hasattr(target_conn.connection, "commit"):
                    target_conn.connection.commit()

        except Exception as e:
            return FuncToolResult(
                success=0,
                error=f"Transfer failed after writing {rows_written} rows: {str(e)}",
            )

        logger.info(f"Transferred {rows_written} rows to {target_table} (mode={mode})")
        if source_row_count is None:
            # Pre-count failed silently (logged at debug above). Do NOT
            # backfill with rows_written — that would make Layer A's
            # transfer-parity invariant trivially pass and defeat the point
            # of verifying source vs target row counts. Leave as None so
            # ``_run_row_count_parity`` skips instead of faking equality.
            logger.warning(
                "Transfer parity check will be skipped — source row pre-count was unavailable for transfer to %s",
                target_table,
            )
        return FuncToolResult(
            result={
                "message": "Transfer completed successfully",
                "source_sql": source_sql,
                "source_datasource": source_datasource,
                "target_table": target_table,
                "target_datasource": target_datasource or self._default_datasource,
                "mode": mode,
                "rows_transferred": rows_written,
                "target_table_created": target_table_created,
                "target_table_create_sql": target_table_create_sql,
                "source_row_count": source_row_count,
                "source_row_count_verified": source_row_count is not None,
                "transferred_row_count": rows_written,
                "batch_size": batch_size,
                "deliverable_target": self._build_transfer_target(
                    source_datasource=source_datasource,
                    target_datasource=target_datasource or self._default_datasource,
                    target_table=target_table,
                    source_row_count=source_row_count,
                    transferred_row_count=rows_written,
                    target_active_database=self._active_database_of(target_conn),
                ),
            }
        )

    @staticmethod
    def _build_transfer_target(
        source_datasource: str,
        target_datasource: str,
        target_table: str,
        source_row_count: Optional[int],
        transferred_row_count: int,
        target_active_database: str = "",
    ) -> Dict[str, Any]:
        """Construct the ``deliverable_target`` payload for a transfer call.

        ``source_row_count=None`` signals "could not verify" (pre-count SQL
        failed). ``model_dump(exclude_none=True)`` drops it from the payload
        so ``_run_row_count_parity`` treats the check as skipped instead of
        trivially equal to ``transferred_row_count``.

        ``TableTarget.database`` gets the *physical* database the transfer
        wrote into — taken from the parsed ``target_table`` identifier when
        it carries a ``db.schema.table`` qualifier, otherwise from the
        target connector's active namespace (``target_active_database``).
        It is left empty when neither is available: the datasource key is a
        connection profile, not a database, and must not stand in for one
        (connector routing already uses ``target_datasource``).
        """
        from datus.utils.sql_utils import parse_table_name_parts
        from datus.validation.report import DBRef, TableTarget, TransferTarget

        parts = parse_table_name_parts(target_table)
        parsed_db = parts.get("database_name") or parts.get("catalog_name") or None
        schema = parts.get("schema_name") or None
        table = parts.get("table_name") or target_table
        effective_database = parsed_db or target_active_database or ""

        tgt = TransferTarget(
            source=DBRef(name=source_datasource),
            target=TableTarget(
                datasource=target_datasource,
                database=effective_database,
                db_schema=schema,
                table=table,
            ),
            source_row_count=source_row_count,
            transferred_row_count=transferred_row_count,
        )
        return tgt.model_dump(by_alias=True, exclude_none=True)

    # ==================== Migration Target Wrappers ====================
    #
    # Thin wrappers over ``MigrationTargetMixin`` methods on the underlying
    # connector. Uses duck typing so any datus-db-core >= the version that
    # introduced the Mixin is supported. When the connector does not expose
    # these methods, we return safe fallback values so the migration agent
    # can continue in pure-LLM mode.

    def get_migration_capabilities(self, datasource: Optional[str] = "") -> FuncToolResult:
        """
        Get migration target hints (dialect_family, requires, forbids, type_hints,
        example_ddl) for the specified target datasource.

        Args:
            datasource: Target datasource name. Uses the default datasource if empty.

        Returns:
            When the adapter implements ``MigrationTargetMixin``:
              success=1, result = the capability dict.
            Otherwise:
              success=1, result = {"supported": False, "warning": "..."}.
        """
        try:
            connector = self._get_connector(datasource)
        except DatusException as e:
            return FuncToolResult(success=0, error=str(e))

        if not hasattr(connector, "describe_migration_capabilities"):
            return FuncToolResult(
                result={
                    "supported": False,
                    "dialect_family": getattr(connector, "dialect", "unknown"),
                    "warning": (
                        "Adapter does not expose migration hints (MigrationTargetMixin not implemented); "
                        "falling back to pure LLM mode. DDL generation will rely on the LLM's own "
                        "knowledge of this dialect."
                    ),
                }
            )

        try:
            capabilities = connector.describe_migration_capabilities()
        except Exception as e:
            logger.warning(f"describe_migration_capabilities failed on {datasource}: {e}")
            return FuncToolResult(
                result={
                    "supported": False,
                    "warning": f"Adapter raised while describing capabilities: {e}",
                }
            )
        return FuncToolResult(result=capabilities)

    def suggest_table_layout(self, datasource: Optional[str] = "", columns_json: str = "[]") -> FuncToolResult:
        """
        Suggest dialect-specific table layout (distribution/partition/order) for
        the target datasource, given the source columns.

        Args:
            datasource: Target datasource name. Uses the default datasource if empty.
            columns_json: JSON array of source column defs. Each element must
                be an object with keys ``name`` (str), ``type`` (str), and
                ``nullable`` (bool). Example::

                    [{"name": "id", "type": "BIGINT", "nullable": false}]

        Returns:
            When the adapter implements the Mixin: result = suggestion dict
            (possibly empty for OLTP). Otherwise: result = {}.
        """
        try:
            columns = json.loads(columns_json) if columns_json else []
        except json.JSONDecodeError as e:
            return FuncToolResult(success=0, error=f"Invalid columns_json: {e}")
        if not isinstance(columns, list):
            return FuncToolResult(success=0, error="columns_json must be a JSON array")

        try:
            connector = self._get_connector(datasource)
        except DatusException as e:
            return FuncToolResult(success=0, error=str(e))

        if not hasattr(connector, "suggest_table_layout"):
            return FuncToolResult(result={})

        try:
            suggestion = connector.suggest_table_layout(columns)
        except Exception as e:
            logger.warning(f"suggest_table_layout failed on {datasource}: {e}")
            return FuncToolResult(result={})
        return FuncToolResult(result=suggestion)

    def validate_ddl(
        self,
        datasource: Optional[str] = "",
        database: Optional[str] = "",
        ddl: str = "",
        target_table: Optional[str] = None,
    ) -> FuncToolResult:
        """
        Statically validate a CREATE TABLE DDL against the target dialect's rules.
        Optionally runs ``dry_run_ddl`` (actual CREATE + DROP to a temp table)
        when ``target_table`` is provided and the adapter supports it.

        Args:
            datasource: Target datasource name. Uses the default datasource if empty.
            ddl: The CREATE TABLE DDL to validate.
            target_table: If provided, attempt dry-run using this table name.

        Returns:
            result = {"errors": [...], "validated": true|false}. Empty errors
            with validated=True means static checks passed.
            When the adapter has no Mixin, returns validated=False with no errors
            (the LLM is solely responsible for correctness).
        """
        if not ddl or not ddl.strip():
            return FuncToolResult(success=0, error="Empty DDL statement")

        try:
            connector = self._get_connector(datasource, database)
        except DatusException as e:
            return FuncToolResult(success=0, error=str(e))

        if not hasattr(connector, "validate_ddl"):
            return FuncToolResult(result={"errors": [], "validated": False})

        errors: List[str] = []
        try:
            static_errors = connector.validate_ddl(ddl)
            if static_errors:
                errors.extend(static_errors)
        except Exception as e:
            logger.warning(f"validate_ddl static check failed on {datasource}: {e}")
            errors.append(f"Static check raised unexpectedly: {e}")

        # If static errors were found, skip dry_run — DDL is already invalid.
        if target_table and not errors and hasattr(connector, "dry_run_ddl"):
            try:
                dry_errors = connector.dry_run_ddl(ddl, target_table)
                if dry_errors:
                    errors.extend(dry_errors)
            except NotImplementedError:
                # Adapter chose not to implement dry-run — static check is the ceiling.
                pass
            except Exception as e:
                logger.warning(f"dry_run_ddl failed on {datasource}: {e}")
                errors.append(f"Dry-run raised unexpectedly: {e}")

        return FuncToolResult(result={"errors": errors, "validated": True})


def db_function_tool_instance(
    agent_config: AgentConfig,
    database_name: str = "",
    sub_agent_name: Optional[str] = None,
    *,
    datasource: str = "",
) -> DBFuncTool:
    """Create a DBFuncTool instance. Auto-creates DBManager from agent_config.

    ``datasource`` is the datasource key (routing); ``database_name`` is the physical database (metadata).
    """
    return DBFuncTool(
        agent_config=agent_config,
        default_datasource=datasource or None,
        default_database=database_name or None,
        sub_agent_name=sub_agent_name,
    )


def db_function_tool_instance_multi(
    agent_config: AgentConfig,
    sub_agent_name: Optional[str] = None,
    connector_cache_size: int = DBFuncTool.DEFAULT_CONNECTOR_CACHE_SIZE,
) -> DBFuncTool:
    """Create a DBFuncTool instance (kept for backward compatibility)."""
    return DBFuncTool(
        agent_config=agent_config,
        sub_agent_name=sub_agent_name,
        connector_cache_size=connector_cache_size,
    )


def db_function_tools(
    agent_config: AgentConfig,
    database_name: str = "",
    sub_agent_name: Optional[str] = None,
    *,
    datasource: str = "",
) -> List[Tool]:
    return db_function_tool_instance(
        agent_config, database_name, sub_agent_name, datasource=datasource
    ).available_tools()
