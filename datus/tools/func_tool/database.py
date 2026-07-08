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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from agents import Tool
from datus_db_core import BaseSqlConnector, connector_registry

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.kb_retrieval import metadata_fts_enabled
from datus.storage.schema_metadata import create_metadata_rag
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.storage.table_semantic_profile.store import TableSemanticProfileRAG
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.compress_utils import DataCompressor
from datus.utils.constants import DBType, SQLType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.mcp_decorators import mcp_tool, mcp_tool_class

logger = get_logger(__name__)


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
        principal: Optional[Dict[str, Any]] = None,
        connector_cache_size: int = DEFAULT_CONNECTOR_CACHE_SIZE,
        read_only: bool = False,
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
            principal: Request-scoped SQL policy attributes. When omitted,
                       falls back to ``agent_config.principal`` if present.
            connector_cache_size: Max connectors to cache (LRU eviction), default 8
            read_only: When True, ``execute_sql`` hard-rejects any non-read
                       statement (INSERT/UPDATE/DELETE/DDL/MERGE/...) at the tool
                       layer, independent of ``PermissionHooks``. Use for agents
                       whose contract is read-only (Explore, ask_report/dashboard,
                       LLM validators) so the unified write-capable entry point
                       cannot mutate the datasource even when hooks are bypassed
                       (e.g. validators run with ``hooks=None``) or under a
                       permissive profile.
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
        principal_source = principal if principal is not None else getattr(agent_config, "principal", {})
        self.principal: Dict[str, Any] = dict(principal_source) if isinstance(principal_source, dict) else {}
        self.sub_agent_name = sub_agent_name
        self.read_only = read_only
        if agent_config and metadata_fts_enabled(agent_config):
            self.schema_rag = create_metadata_rag(agent_config, sub_agent_name)
        else:
            self.schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name) if agent_config else None
        self._field_order = self._determine_field_order()
        self._scoped_patterns = self._load_scoped_patterns(scoped_tables)

        self._semantic_storage = SemanticModelRAG(agent_config, sub_agent_name) if agent_config else None
        self._table_semantic_profiles = None
        if agent_config and isinstance(getattr(agent_config, "project_name", ""), str):
            try:
                self._table_semantic_profiles = TableSemanticProfileRAG(agent_config, sub_agent_name)
            except Exception as exc:
                logger.debug(f"Failed to initialize table semantic profile storage: {exc}")
        self.has_schema = self._has_schema_storage()

        self.has_semantic_models = self._semantic_storage and self._semantic_storage.get_size() > 0
        try:
            self.has_table_semantic_profiles = (
                self._table_semantic_profiles is not None and self._table_semantic_profiles.get_size() > 0
            )
        except Exception:
            self._table_semantic_profiles = None
            self.has_table_semantic_profiles = False

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
        fields: List[str] = []
        if connector_registry.support_catalog(dialect):
            fields.append("catalog")
        if connector_registry.support_database(dialect) or dialect == DBType.SQLITE:
            fields.append("database")
        if connector_registry.support_schema(dialect):
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
        parts = [self._normalize_identifier_part(part) for part in token.split(".") if part.strip()]
        if not parts:
            return None
        # Align parts from right to left (table is always rightmost)
        # e.g., for "public.wb_health_population" with field_order ["database", "schema", "table"]:
        #   - parts = ["public", "wb_health_population"]
        #   - align from right: schema="public", table="wb_health_population"
        # When parts > fields, keep only the rightmost num_fields parts
        values: Dict[str, str] = {field: "" for field in self._field_order}
        num_fields = len(self._field_order)
        trimmed_parts = parts[-num_fields:]
        start_field_idx = max(0, num_fields - len(trimmed_parts))
        for i, part in enumerate(trimmed_parts):
            field_idx = start_field_idx + i
            if field_idx < num_fields:
                values[self._field_order[field_idx]] = part
        return ScopedTablePattern(raw=token, **values)

    def _get_semantic_model(
        self, catalog: str = "", database: str = "", schema: str = "", table_name: str = ""
    ) -> Dict[str, Any]:
        if not self.has_semantic_models:
            return {}
        result = self._semantic_storage.get_semantic_model(
            catalog_name=catalog,
            database_name=database,
            schema_name=schema,
            table_name=table_name,
            select_fields=[
                "semantic_model_name",
                "dimensions",
                "measures",
                "description",
                "identifiers",
            ],
        )
        logger.info(f"get_semantic_model result: {result}")
        return result if result is not None else {}

    def _get_table_semantic_profile(
        self, catalog: str = "", database: str = "", schema: str = "", table_name: str = ""
    ) -> Dict[str, Any]:
        if self._table_semantic_profiles is None:
            return {}
        result = self._table_semantic_profiles.get_profile(
            catalog_name=catalog,
            database_name=database,
            schema_name=schema,
            table_name=table_name,
            select_fields=[
                "table_name",
                "semantic_model_name",
                "dataset_name",
                "data_source_name",
                "description",
                "ai_context_json",
                "columns_json",
                "relationships_json",
            ],
        )
        logger.info(f"get_table_semantic_profile result: {result}")
        return result if result is not None else {}

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

    def _apply_table_semantic_profile(self, result_data: Dict[str, Any], profile: Dict[str, Any]) -> None:
        """Attach a table semantic profile to describe_table output."""

        columns = result_data.get("columns", [])
        semantic_columns = self._decode_profile_json(profile.get("columns_json"), [])
        relationships = self._decode_profile_json(profile.get("relationships_json"), [])
        ai_context = self._decode_profile_json(profile.get("ai_context_json"), None)

        table = {
            "name": (
                profile.get("dataset_name")
                or profile.get("data_source_name")
                or profile.get("table_name")
                or profile.get("semantic_model_name")
                or ""
            ),
            "description": profile.get("description", ""),
        }
        if ai_context not in (None, "", [], {}):
            table["ai_context"] = ai_context
        result_data["table"] = table
        result_data["semantic"] = {
            "relationships": relationships,
        }

        if not isinstance(semantic_columns, list):
            return
        semantic_lookup = {}
        for semantic_col in semantic_columns:
            if not isinstance(semantic_col, dict):
                continue
            for key in ("expr", "name"):
                value = semantic_col.get(key)
                if value:
                    semantic_lookup.setdefault(str(value).strip("`").lower(), semantic_col)

        for col in columns:
            col_name = str(col.get("name", "")).lower()
            semantic_col = semantic_lookup.get(col_name)
            if not semantic_col:
                continue
            role = semantic_col.get("role") or ""
            description = semantic_col.get("description") or ""
            col["semantic_role"] = role
            col["is_dimension"] = role in ("dimension", "time_dimension")
            if semantic_col.get("ai_context") not in (None, "", [], {}):
                col["ai_context"] = semantic_col.get("ai_context")
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

    def _default_field_value(self, field: str, explicit: Optional[str]) -> str:
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
        if fallback_attr and hasattr(self.connector, fallback_attr):
            return self._normalize_identifier_part(getattr(self.connector, fallback_attr))
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

        dialect = self._dialect_for_datasource(datasource)
        if not connector_registry.support_catalog(dialect):
            if catalog_value and not database_value and connector_registry.support_database(dialect):
                database_value = catalog_value
            catalog_value = ""

        return catalog_value, database_value, schema_value

    def _build_table_coordinate(
        self,
        raw_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema: Optional[str] = "",
    ) -> TableCoordinate:
        coordinate = TableCoordinate(
            catalog=self._default_field_value("catalog", catalog),
            database=self._default_field_value("database", database),
            schema=self._default_field_value("schema", schema),
            table=self._normalize_identifier_part(raw_name),
        )
        parts = [self._normalize_identifier_part(part) for part in raw_name.split(".") if part.strip()]
        if parts:
            coordinate.table = parts[-1]
            idx = len(parts) - 2
            for field in reversed(self._field_order[:-1]):
                if idx < 0:
                    break
                setattr(coordinate, field, parts[idx])
                idx -= 1
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

    def _check_sql_table_scope(self, sql: str) -> List[str]:
        """Return table names from *sql* that fall outside the scoped context."""
        if not self._scoped_patterns:
            return []
        from datus.utils.sql_utils import extract_table_names

        dialect = getattr(self._primary_connector, "dialect", "") or ""
        table_names = extract_table_names(sql, dialect=dialect, ignore_empty=True)
        if not table_names:
            return []  # can't parse → allow (SHOW/DESCRIBE/EXPLAIN have no tables)
        out_of_scope: List[str] = []
        for name in table_names:
            coordinate = self._build_table_coordinate(raw_name=name)
            if not self._table_matches_scope(coordinate):
                out_of_scope.append(name)
        return out_of_scope

    @staticmethod
    def all_tools_name() -> List[str]:
        from datus.utils.class_utils import get_public_instance_methods

        result = []
        for name in get_public_instance_methods(DBFuncTool).keys():
            if name == "available_tools":
                continue
            result.append(name)
        return result

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

    def _excluded_tool_params(self) -> set[str]:
        excluded: set[str] = set()
        if not any(connector_registry.support_catalog(dialect) for dialect in self._configured_tool_dialects()):
            excluded.add("catalog")
        return excluded

    def to_function_tool(self, bound_method: Callable) -> Tool:
        return trans_to_function_tool(bound_method, excluded_params=self._excluded_tool_params())

    def available_tools(self) -> List[Tool]:
        bound_tools = []
        methods_to_convert: List[Callable] = [self.list_tables, self.describe_table]
        configured_dialects = self._configured_tool_dialects()

        if self.has_schema:
            methods_to_convert.append(self.search_table)

        methods_to_convert.append(self.execute_sql)

        if any(connector_registry.support_database(dialect) for dialect in configured_dialects):
            bound_tools.append(self.to_function_tool(self.list_databases))

        if any(connector_registry.support_schema(dialect) for dialect in configured_dialects):
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

            if self.has_semantic_models:
                for metadata_row in metadata_rows:
                    semantic_model = self._get_semantic_model(
                        metadata_row["catalog_name"],
                        metadata_row["database_name"],
                        metadata_row["schema_name"],
                        metadata_row["table_name"],
                    )
                    if semantic_model:
                        metadata_row["description"] = semantic_model.get("description", "")

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

            filtered_result = self._filter_table_entries(result, catalog, database, schema_name)
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

        Returns:
            FuncToolResult with a dictionary containing:
            - columns (list): List of column dictionaries, each containing:
              - name (str): Column name (required)
              - type (str): Column data type (required)
              - comment (str): Column description/comment, enriched with semantic model description if available
              - is_dimension (bool): Whether this column is a dimension in semantic model
                (semantic fields only present if semantic model exists)
            - table (dict, optional): Table-level metadata from semantic model (only if model exists):
              - name (str): Name of the table
              - description (str): Table description from semantic model
              - ai_context (dict/list/str, optional): Extra LLM-facing business guidance
            - semantic (dict, optional): LLM-facing semantic hints:
              - relationships (list): Relevant dataset/data-source relationships
        """
        try:
            catalog, database, schema_name = self._normalize_namespace_args(
                catalog,
                database,
                schema_name,
                datasource,
            )
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
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
                columns.append(normalized_col)

            # 3. Enrich with Semantic Model Info if available
            result_data = {"columns": columns}
            profile_applied = False

            try:
                profile = self._get_table_semantic_profile(
                    coordinate.catalog,
                    coordinate.database,
                    coordinate.schema,
                    coordinate.table,
                )
                if profile:
                    logger.debug(
                        "Found table semantic profile: %s",
                        profile.get("dataset_name") or profile.get("data_source_name") or "unknown",
                    )
                    self._apply_table_semantic_profile(result_data, profile)
                    profile_applied = True
            except Exception as e:
                logger.warning(f"Failed to get table semantic profile for {table_name}: {e}")

            if self.has_semantic_models and not profile_applied:
                try:
                    logger.debug("Checking for semantic models")
                    # Use coordinate values (resolved and stripped) for lookup
                    model = self._get_semantic_model(
                        coordinate.catalog,
                        coordinate.database,
                        coordinate.schema,
                        coordinate.table,
                    )

                    if model:
                        logger.debug(f"Found semantic model: {model.get('semantic_model_name', 'unknown')}")

                        # Add table-level metadata unless the authoring-format
                        # profile already provided a richer table projection.
                        result_data.setdefault(
                            "table",
                            {
                                "name": model.get("semantic_model_name", ""),
                                "description": model.get("description", ""),
                            },
                        )

                        # Create lookup map using expr (physical column) as key, fallback to name
                        # expr is the actual column name/expression, name is the semantic name
                        dimensions = model.get("dimensions", [])

                        # Build map: physical_col_name -> dimension_data
                        dim_map = {(d.get("expr") or d.get("name", "")).lower(): d for d in dimensions}

                        logger.debug(f"Semantic map: {len(dim_map)} dimensions")

                        # Enrich columns with dimension info
                        if dim_map:
                            for col in columns:
                                col_name = col["name"].lower()

                                if col_name in dim_map:
                                    dim_data = dim_map[col_name]
                                    col["is_dimension"] = True
                                    if dim_data.get("description"):
                                        col.setdefault("semantic_description", dim_data.get("description"))
                                        col["comment"] = dim_data.get("description")
                                else:
                                    col.setdefault("is_dimension", False)
                        else:
                            logger.debug("No dimensions defined in model")
                    else:
                        logger.debug("No semantic model found for this table")
                except Exception as e:
                    # If semantic model lookup fails, just log and continue with physical schema only
                    logger.warning(f"Failed to get semantic model for {table_name}: {e}")

            logger.info(f"describe_table succeeded for {table_name}, returning {len(columns)} columns")
            return FuncToolResult(result=result_data)

        except Exception as e:
            import traceback

            error_msg = f"Error describing table {table_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return FuncToolResult(success=0, error=error_msg)

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
        from datus.utils.sql_utils import looks_like_sql_file_ref, parse_sql_type

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

            connector = self._get_connector(datasource, database)
            sql_type = parse_sql_type(sql, connector.dialect)

            if sql_type in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN):
                return self.read_query(sql, datasource=datasource, database=database)
            if self.read_only:
                # Defense-in-depth for read-only agents: reject any non-read
                # statement at the tool layer, independent of PermissionHooks
                # (which may be bypassed, e.g. validators run with hooks=None).
                return FuncToolResult(
                    success=0,
                    error=(
                        "This agent is read-only: only SELECT/SHOW/DESCRIBE/EXPLAIN "
                        "statements are allowed through execute_sql."
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

            logger.info("read_query", sql_type=sql_type.value, datasource=datasource or "default")
            effective_datasource = self._resolve_effective_datasource(datasource)
            sql = self._enforce_sql_policy(
                sql,
                datasource=effective_datasource,
                dialect=connector.dialect,
            )
            validation_error, _ = self._validate_read_sql(sql, connector)
            if validation_error:
                return validation_error
            result = connector.execute_query(sql, result_format="arrow" if connector.dialect == "snowflake" else "list")
            if result.success:
                data = result.sql_return
                return FuncToolResult(result=self.compressor.compress(data))
            else:
                return FuncToolResult(success=0, error=result.error)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def _resolve_effective_datasource(self, datasource: Optional[str]) -> str:
        effective_datasource = datasource or self._default_datasource
        if not effective_datasource and self.agent_config:
            services = getattr(self.agent_config, "services", None)
            effective_datasource = getattr(services, "default_datasource", "") or ""
        return effective_datasource or "default"

    def _validate_read_sql(self, sql: str, connector: BaseSqlConnector) -> tuple[Optional[FuncToolResult], SQLType]:
        from datus.utils.sql_utils import _first_statement, parse_sql_type, strip_sql_comments

        cleaned = strip_sql_comments(sql).strip()
        normalized_sql = cleaned.rstrip(";").strip()
        if normalized_sql and _first_statement(normalized_sql) != normalized_sql:
            return (
                FuncToolResult(
                    success=0,
                    error="Multi-statement SQL is not allowed. Please submit one query at a time.",
                ),
                SQLType.UNKNOWN,
            )

        sql_type = parse_sql_type(sql, connector.dialect)
        readonly_sql_types = {SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN}
        if sql_type not in readonly_sql_types:
            return (
                FuncToolResult(
                    success=0,
                    error=f"Only read-only queries (SELECT, SHOW, DESCRIBE, EXPLAIN) are allowed. "
                    f"Detected SQL type: {sql_type.value}",
                ),
                sql_type,
            )

        if sql_type == SQLType.METADATA_SHOW:
            first_word = cleaned.split()[0].upper() if cleaned else ""
            if first_word == "PRAGMA" and "=" in cleaned:
                return (
                    FuncToolResult(
                        success=0,
                        error="Writable PRAGMA statements are not allowed in read-only mode.",
                    ),
                    sql_type,
                )

        out_of_scope = self._check_sql_table_scope(sql)
        if out_of_scope:
            return (
                FuncToolResult(
                    success=0,
                    error=f"Query references tables outside scoped context: {', '.join(out_of_scope)}",
                ),
                sql_type,
            )
        return None, sql_type

    def _enforce_sql_policy(self, sql: str, datasource: str, dialect: str) -> str:
        if not self.agent_config:
            return sql
        sql_policy_config = getattr(self.agent_config, "sql_policy_config", None)
        from datus.tools.sql_policy import SqlPolicyConfig, load_sql_policy_enforcer

        if not isinstance(sql_policy_config, SqlPolicyConfig) or not sql_policy_config.enabled:
            return sql

        enforced = load_sql_policy_enforcer(sql_policy_config).enforce_read(
            sql,
            datasource=datasource,
            dialect=dialect,
            principal=self.principal,
        )
        if not enforced.allowed:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message=enforced.reason or "SQL policy denied the query",
            )
        if enforced.applied_policies:
            logger.info(
                "Applied SQL policies",
                policies=enforced.applied_policies,
                datasource=datasource,
            )
        return sql if enforced.sql is None else enforced.sql

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
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
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

        out_of_scope = self._check_sql_table_scope(cleaned)
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

            out_of_scope = self._check_sql_table_scope(normalized_sql)
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
        backtick_dialects = ("mysql", "starrocks", "hive", "spark", "bigquery", "clickhouse")
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

        dialect = str(dialect or "").lower()

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
                column_defs.append(
                    f"  {self._quote_column_identifier(column, dialect)} "
                    f"{self._infer_transfer_column_type(df[column], dialect)}"
                )
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

        # Authoritative source row count — wrap the user's source_sql in a COUNT
        # subquery so reconciliation does not need to re-run anything later.
        # One extra query is cheap on OLTP engines and still acceptable on
        # warehouse engines; see ValidationHook design doc §5.4.
        source_row_count: Optional[int] = None
        try:
            if hasattr(source_conn, "execute_query"):
                count_sql = f"SELECT COUNT(*) AS __datus_count FROM ({cleaned_sql}) AS __datus_src"
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

        # Batch INSERT using connector's execute_insert (adapter-agnostic)
        # Quote column names to handle reserved words (e.g., status, order, select).
        # Use dialect-appropriate quoting: backticks for MySQL/StarRocks, double quotes for others.
        columns = list(df.columns)
        dialect = str(getattr(target_conn, "dialect", "") or "").lower()
        col_names = ", ".join(self._quote_column_identifier(c, dialect) for c in columns)

        rows_written = 0
        try:
            for batch_start in range(0, row_count, batch_size):
                batch_end = min(batch_start + batch_size, row_count)
                batch_df = df.iloc[batch_start:batch_end]

                # Build batch INSERT statement with inline values
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
