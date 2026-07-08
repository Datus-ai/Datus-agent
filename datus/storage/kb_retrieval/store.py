# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import pandas as pd
import pyarrow as pa
from datus_storage_base.conditions import And, Node, WhereExpr, and_, eq, or_

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import TableSchema, TableValue
from datus.storage.base import BaseEmbeddingStore, StorageBase
from datus.storage.datasource_scope import DATASOURCE_ID_COLUMN, datasource_condition, resolve_datasource_id
from datus.storage.embedding_models import EmbeddingModel
from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import json2csv
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)


class KbSearchMode(StrEnum):
    FTS = "fts"
    HYBRID = "hybrid"


def metadata_fts_enabled(agent_config: Any) -> bool:
    """Return True when public KB search config enables the metadata retrieval path."""

    kb_search = getattr(agent_config, "kb_search", None)
    if kb_search is None:
        return False
    enabled = getattr(kb_search, "enabled", None)
    mode = getattr(kb_search, "mode", None)
    return (
        isinstance(enabled, bool)
        and enabled
        and isinstance(mode, str)
        and mode.lower()
        in {
            KbSearchMode.FTS.value,
            KbSearchMode.HYBRID.value,
        }
    )


def _search_mode(agent_config: Any) -> KbSearchMode:
    mode = str(getattr(agent_config, "kb_search_mode", KbSearchMode.FTS.value) or KbSearchMode.FTS.value).lower()
    return KbSearchMode.HYBRID if mode == KbSearchMode.HYBRID.value else KbSearchMode.FTS


def _utc_now() -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(microsecond=(now.microsecond // 1000) * 1000)


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _normalize_row_text(value: Any) -> str:
    text = _as_text(value)
    return " ".join(text.split())


def _qualified_name(row: Dict[str, Any]) -> str:
    parts = [
        str(row.get("catalog_name") or "").strip(),
        str(row.get("database_name") or "").strip(),
        str(row.get("schema_name") or "").strip(),
        str(row.get("table_name") or "").strip(),
    ]
    return ".".join(part for part in parts if part)


def _build_where_clause(
    *,
    datasource_id: str = "",
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    table_type: TABLE_TYPE = "table",
    extra: Optional[Node] = None,
) -> Optional[Node]:
    conditions = []
    if datasource_id:
        conditions.append(datasource_condition(datasource_id))
    if catalog_name:
        conditions.append(eq("catalog_name", catalog_name))
    if database_name:
        conditions.append(eq("database_name", database_name))
    if schema_name:
        conditions.append(eq("schema_name", schema_name))
    if table_name:
        conditions.append(eq("table_name", table_name))
    if table_type and table_type != "full":
        conditions.append(eq("table_type", table_type))
    if extra is not None:
        conditions.append(extra)
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else And(conditions)


class PlainLanceStore(StorageBase):
    """Small no-vector LanceDB-backed table store used for KB facts and FTS docs."""

    def __init__(self, *, project: str, table_name: str, schema: pa.Schema, unique_column: str):
        from datus.storage.backend_holder import create_vector_connection

        super().__init__(db=create_vector_connection(project))
        self.table_name = table_name
        self._schema = schema
        self._unique_column = unique_column
        self._table_lock = Lock()
        self._write_lock = Lock()
        self.table = None

    def _ensure_table_ready(self) -> None:
        if self.table is not None:
            return
        with self._table_lock:
            if self.table is not None:
                return
            try:
                if self.db.table_exists(self.table_name):
                    self.table = self.db.open_table(self.table_name)
                else:
                    self.table = self.db.create_table(self.table_name, schema=self._schema, exist_ok=True)
            except Exception as exc:
                raise DatusException(
                    ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                    message_args={
                        "operation": "ensure_table",
                        "table_name": self.table_name,
                        "error_message": str(exc),
                    },
                ) from exc

    def _open_existing_table_for_read(self):
        if self.table is not None:
            return self.table
        if not self.db.table_exists(self.table_name):
            return None
        self.table = self.db.open_table(self.table_name)
        return self.table

    def _empty_result(self, select_fields: Optional[List[str]] = None) -> pa.Table:
        if select_fields is None:
            fields = list(self._schema)
        else:
            schema_names = set(self._schema.names)
            fields = [
                self._schema.field(name) if name in schema_names else pa.field(name, pa.null())
                for name in select_fields
            ]
        return pa.Table.from_arrays([pa.array([], type=field.type) for field in fields], schema=pa.schema(fields))

    def upsert_batch(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        self._ensure_table_ready()
        frame = pd.DataFrame(rows).drop_duplicates(subset=[self._unique_column], keep="last")
        with self._write_lock:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    self.table.merge_insert(frame, self._unique_column)
                    return
                except Exception as exc:
                    if "Commit conflict" not in str(exc):
                        raise
                    last_error = exc
                    self.table = self.db.refresh_table(self.table_name)
                    time.sleep(0.05 * (attempt + 1))
            assert last_error is not None
            raise last_error

    def delete_datasource_rows(self, datasource_id: str) -> None:
        self._delete_rows(datasource_condition(datasource_id))

    def _delete_rows(self, where: WhereExpr) -> None:
        self._ensure_table_ready()
        if where:
            self.table.delete(where)

    def _search_all(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        table = self._open_existing_table_for_read()
        if table is None:
            return self._empty_result(select_fields)
        row_limit = limit if limit is not None else table.count_rows(where)
        if row_limit == 0:
            return self._empty_result(select_fields)
        return table.search_all(where=where, select_fields=select_fields, limit=row_limit)

    def _count_rows(self, where: WhereExpr = None) -> int:
        table = self._open_existing_table_for_read()
        if table is None:
            return 0
        return table.count_rows(where)

    def create_indices(self, scalar_columns: Sequence[str], fts_columns: Optional[Sequence[str]] = None) -> None:
        self._ensure_table_ready()
        for column in scalar_columns:
            try:
                self.table.create_scalar_index(column)
            except Exception as exc:
                logger.warning("Failed to create scalar index on %s.%s: %s", self.table_name, column, exc)
        if fts_columns:
            try:
                self.table.create_fts_index(list(fts_columns))
            except Exception as exc:
                logger.warning("Failed to create FTS index on %s: %s", self.table_name, exc)

    def search_fts(
        self,
        query_text: str,
        *,
        fts_columns: Sequence[str],
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        table = self._open_existing_table_for_read()
        if table is None:
            return self._empty_result(select_fields)
        if table.count_rows(where) == 0:
            return self._empty_result(select_fields)
        if not hasattr(table, "search_fts"):
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": f"Backend for '{self.table_name}' does not support FTS search",
                    "query": query_text,
                    "where_clause": str(where) if where else "(none)",
                    "top_n": str(top_n),
                },
            )
        return table.search_fts(query_text, list(fts_columns), top_n, where=where, select_fields=select_fields)


class MetadataFactsStore(PlainLanceStore):
    def __init__(self, *, project: str):
        super().__init__(
            project=project,
            table_name="schema_metadata_facts",
            unique_column="fact_id",
            schema=pa.schema(
                [
                    pa.field("fact_id", pa.string()),
                    pa.field(DATASOURCE_ID_COLUMN, pa.string()),
                    pa.field("identifier", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("table_type", pa.string()),
                    pa.field("definition", pa.string()),
                    pa.field("sample_rows", pa.string()),
                    pa.field("content_hash", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
                ]
            ),
        )

    def create_indices(self, scalar_columns: Sequence[str] = (), fts_columns: Optional[Sequence[str]] = None) -> None:
        super().create_indices(
            scalar_columns=scalar_columns
            or [
                DATASOURCE_ID_COLUMN,
                "identifier",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
            ],
        )


class KbRetrievalDocumentStore(PlainLanceStore):
    FTS_COLUMNS = ["title", "search_text", "identifier", "table_name"]

    def __init__(self, *, project: str):
        super().__init__(
            project=project,
            table_name="kb_retrieval_document",
            unique_column="doc_id",
            schema=pa.schema(
                [
                    pa.field("doc_id", pa.string()),
                    pa.field(DATASOURCE_ID_COLUMN, pa.string()),
                    pa.field("component_type", pa.string()),
                    pa.field("entity_type", pa.string()),
                    pa.field("entity_key", pa.string()),
                    pa.field("identifier", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("table_type", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("payload_json", pa.string()),
                    pa.field("content_hash", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
                ]
            ),
        )

    def create_indices(self, scalar_columns: Sequence[str] = (), fts_columns: Optional[Sequence[str]] = None) -> None:
        super().create_indices(
            scalar_columns=scalar_columns
            or [
                DATASOURCE_ID_COLUMN,
                "component_type",
                "entity_type",
                "entity_key",
                "identifier",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
            ],
            fts_columns=fts_columns or self.FTS_COLUMNS,
        )


class MetadataRetrievalVectorStorage(BaseEmbeddingStore):
    """Optional hybrid projection over the same table-level retrieval documents."""

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
            table_name="kb_retrieval_vector",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("doc_id", pa.string()),
                    pa.field("component_type", pa.string()),
                    pa.field("entity_type", pa.string()),
                    pa.field("entity_key", pa.string()),
                    pa.field("identifier", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("table_type", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("payload_json", pa.string()),
                    pa.field("content_hash", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="search_text",
            vector_column_name="vector",
            unique_columns=["storage_key"],
            datasource_scoped=True,
            **kwargs,
        )

    def create_indices(self) -> None:
        self._ensure_table_ready()
        for column in (
            "doc_id",
            "component_type",
            "entity_type",
            "entity_key",
            "identifier",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
        ):
            self._create_scalar_index(column)
        self.create_fts_index(KbRetrievalDocumentStore.FTS_COLUMNS)


class MetadataFtsRAG:
    """No-vector metadata RAG compatible with the schema bootstrap interface."""

    SCHEMA_SELECT_FIELDS = [
        "identifier",
        "catalog_name",
        "database_name",
        "schema_name",
        "table_name",
        "table_type",
        "definition",
    ]
    VALUE_SELECT_FIELDS = [
        "identifier",
        "catalog_name",
        "database_name",
        "schema_name",
        "table_name",
        "table_type",
        "sample_rows",
    ]
    SCORE_FIELDS = ["_score", "_relevance_score", "_distance"]
    SEARCH_SELECT_FIELDS = [
        "doc_id",
        "entity_key",
        "identifier",
        "catalog_name",
        "database_name",
        "schema_name",
        "table_name",
        "table_type",
        "title",
        "payload_json",
        "_score",
        "_relevance_score",
        "_distance",
    ]

    def __init__(
        self,
        agent_config: "AgentConfig",
        sub_agent_name: Optional[str] = None,
        datasource_id: Optional[str] = None,
    ):
        from datus.storage.rag_scope import _build_sub_agent_filter

        self.agent_config = agent_config
        self.datasource_id = resolve_datasource_id(agent_config, datasource_id)
        self.search_mode = _search_mode(agent_config)
        self.facts_store = MetadataFactsStore(project=agent_config.project_name)
        self.document_store = KbRetrievalDocumentStore(project=agent_config.project_name)
        self.vector_store: Optional[MetadataRetrievalVectorStorage] = None
        if self.search_mode == KbSearchMode.HYBRID:
            from datus.storage.registry import get_storage

            self.vector_store = get_storage(
                MetadataRetrievalVectorStorage,
                "database",
                project=agent_config.project_name,
                datasource_id=self.datasource_id,
            )
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.document_store, "tables")
        self._table_semantic_profiles = None
        self.last_search_info: Dict[str, Any] = self._search_info(
            requested_mode=self.search_mode.value,
            actual_mode="",
            index_ready=False,
        )

    @staticmethod
    def _search_info(
        *,
        requested_mode: str,
        actual_mode: str,
        index_ready: bool,
        fallback: bool = False,
        vector_row_count: Optional[int] = None,
        error_reason: str = "",
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "index_ready": index_ready,
            "fallback": fallback,
        }
        if vector_row_count is not None:
            info["vector_row_count"] = vector_row_count
        if error_reason:
            info["error_reason"] = error_reason
        return info

    def _record_search_info(self, **kwargs: Any) -> None:
        self.last_search_info = self._search_info(**kwargs)

    def _sub_agent_conditions(self) -> list:
        conditions = [datasource_condition(self.datasource_id)]
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def _where(
        self,
        *,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> Optional[Node]:
        base = _build_where_clause(
            datasource_id=self.datasource_id,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table_type,
        )
        if self._sub_agent_filter is None:
            return base
        return self._sub_agent_filter if base is None else and_(base, self._sub_agent_filter)

    def truncate(self) -> None:
        self.facts_store.delete_datasource_rows(self.datasource_id)
        self.document_store.delete_datasource_rows(self.datasource_id)
        if self.vector_store is not None:
            self.vector_store.delete_datasource_rows(self.datasource_id)

    def store_batch(self, schemas: List[Dict[str, Any]], values: List[Dict[str, Any]]) -> None:
        sample_by_identifier = self._sample_rows_by_identifier(values)
        fact_rows = [
            self._fact_row(schema, sample_by_identifier.get(schema.get("identifier", ""), "")) for schema in schemas
        ]
        fact_rows = [row for row in fact_rows if row.get("identifier") and row.get("table_name")]
        if not fact_rows:
            return
        self.facts_store.upsert_batch(fact_rows)
        docs = [self._document_row(row) for row in fact_rows]
        self.document_store.upsert_batch(docs)
        if self.vector_store is not None:
            vector_rows = [dict(row, id=row["doc_id"]) for row in docs]
            from datus.storage.datasource_scope import add_datasource_scope_to_rows

            self.vector_store.upsert_batch(
                add_datasource_scope_to_rows(vector_rows, self.datasource_id), on_column="storage_key"
            )

    def after_init(self) -> None:
        self.facts_store.create_indices()
        self.document_store.create_indices()
        if self.vector_store is not None:
            self.vector_store.create_indices()

    def get_schema_size(self) -> int:
        return self.facts_store._count_rows(where=self._where(table_type="full"))

    def get_value_size(self) -> int:
        rows = self.facts_store._search_all(
            where=self._where(table_type="full"),
            select_fields=["sample_rows"],
        ).to_pylist()
        return sum(1 for row in rows if row.get("sample_rows"))

    def search_table(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
        query_type: str = "",
    ) -> pa.Table:
        mode = KbSearchMode.HYBRID if query_type == KbSearchMode.HYBRID.value else self.search_mode
        where = self._where(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        if mode == KbSearchMode.HYBRID:
            if self.vector_store is None:
                reason = "hybrid_vector_store_unavailable"
                self._record_search_info(
                    requested_mode=mode.value,
                    actual_mode="",
                    index_ready=False,
                    fallback=False,
                    error_reason=reason,
                )
                raise DatusException(
                    ErrorCode.STORAGE_SEARCH_FAILED,
                    message_args={
                        "error_message": "Hybrid metadata retrieval requested but vector projection is unavailable",
                        "query": query_text,
                        "where_clause": str(where) if where else "(none)",
                        "top_n": str(top_n),
                    },
                )

            try:
                vector_row_count = self.vector_store._count_rows(where)
            except Exception as exc:
                self._record_search_info(
                    requested_mode=mode.value,
                    actual_mode="",
                    index_ready=False,
                    fallback=False,
                    error_reason=str(exc),
                )
                raise DatusException(
                    ErrorCode.STORAGE_SEARCH_FAILED,
                    message_args={
                        "error_message": str(exc),
                        "query": query_text,
                        "where_clause": str(where) if where else "(none)",
                        "top_n": str(top_n),
                    },
                ) from exc
            if vector_row_count == 0:
                reason = "hybrid_vector_projection_empty"
                self._record_search_info(
                    requested_mode=mode.value,
                    actual_mode="",
                    index_ready=False,
                    fallback=False,
                    vector_row_count=0,
                    error_reason=reason,
                )
                raise DatusException(
                    ErrorCode.STORAGE_SEARCH_FAILED,
                    message_args={
                        "error_message": "Hybrid metadata retrieval requested but vector projection has no rows",
                        "query": query_text,
                        "where_clause": str(where) if where else "(none)",
                        "top_n": str(top_n),
                    },
                )

            try:
                result = self.vector_store.search(
                    query_text,
                    top_n=top_n,
                    where=where,
                    query_type=KbSearchMode.HYBRID.value,
                    select_fields=None,
                )
            except DatusException as exc:
                self._record_search_info(
                    requested_mode=mode.value,
                    actual_mode="",
                    index_ready=True,
                    fallback=False,
                    vector_row_count=vector_row_count,
                    error_reason=str(exc),
                )
                raise
            except Exception as exc:
                self._record_search_info(
                    requested_mode=mode.value,
                    actual_mode="",
                    index_ready=True,
                    fallback=False,
                    vector_row_count=vector_row_count,
                    error_reason=str(exc),
                )
                raise DatusException(
                    ErrorCode.STORAGE_SEARCH_FAILED,
                    message_args={
                        "error_message": str(exc),
                        "query": query_text,
                        "where_clause": str(where) if where else "(none)",
                        "top_n": str(top_n),
                    },
                ) from exc
            self._record_search_info(
                requested_mode=mode.value,
                actual_mode=KbSearchMode.HYBRID.value,
                index_ready=True,
                fallback=False,
                vector_row_count=vector_row_count,
            )
            return result

        try:
            result = self.document_store.search_fts(
                query_text,
                fts_columns=KbRetrievalDocumentStore.FTS_COLUMNS,
                top_n=top_n,
                where=where,
                select_fields=None,
            )
        except Exception as exc:
            self._record_search_info(
                requested_mode=mode.value,
                actual_mode="",
                index_ready=False,
                fallback=False,
                error_reason=str(exc),
            )
            raise
        self._record_search_info(
            requested_mode=mode.value,
            actual_mode=KbSearchMode.FTS.value,
            index_ready=True,
            fallback=False,
        )
        return result

    def sample_rows_for_search_results(self, metadata: pa.Table) -> pa.Table:
        return self._sample_rows_for_metadata(metadata)

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        query_type: str = "fts",
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
    ) -> tuple[pa.Table, pa.Table]:
        metadata = self.search_table(
            query_text,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
            top_n=top_n,
            query_type=query_type,
        )
        schemas = self._schema_rows_for_metadata(metadata)
        sample_values = self._sample_rows_for_metadata(schemas)
        return schemas, sample_values

    def get_schema(
        self, table_name: str, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> pa.Table:
        return self.facts_store._search_all(
            where=self._where(
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
                table_type="full",
            ),
            select_fields=[
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
                "definition",
            ],
        )

    def search_all_schemas(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        return self.facts_store._search_all(
            where=self._where(
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_type=table_type,
            ),
            select_fields=select_fields,
        )

    def search_all_value(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_type: TABLE_TYPE = "full"
    ) -> pa.Table:
        return self.facts_store._search_all(
            where=self._where(
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_type=table_type,
            ),
            select_fields=[
                "identifier",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
                "sample_rows",
            ],
        )

    def search_tables(
        self,
        tables: list[str],
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        dialect: str = DBType.SQLITE,
    ) -> tuple[List[TableSchema], List[TableValue]]:
        conditions = []
        for full_table in tables:
            cat, db, sch, table = self._parse_table_name(
                full_table,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                dialect=dialect,
            )
            condition = self._where(
                catalog_name=cat,
                database_name=db,
                schema_name=sch,
                table_name=table,
                table_type="full",
            )
            if condition is not None:
                conditions.append(condition)
        where = None if not conditions else conditions[0] if len(conditions) == 1 else or_(*conditions)
        rows = self.facts_store._search_all(where=where).to_pylist()
        schema_rows = [
            {
                "identifier": row.get("identifier", ""),
                "catalog_name": row.get("catalog_name", ""),
                "database_name": row.get("database_name", ""),
                "schema_name": row.get("schema_name", ""),
                "table_name": row.get("table_name", ""),
                "table_type": row.get("table_type", ""),
                "definition": row.get("definition", ""),
            }
            for row in rows
        ]
        value_rows = [
            {
                "identifier": row.get("identifier", ""),
                "catalog_name": row.get("catalog_name", ""),
                "database_name": row.get("database_name", ""),
                "schema_name": row.get("schema_name", ""),
                "table_name": row.get("table_name", ""),
                "table_type": row.get("table_type", ""),
                "sample_rows": row.get("sample_rows", ""),
            }
            for row in rows
            if row.get("sample_rows")
        ]
        schema_table = (
            pa.Table.from_pylist(schema_rows)
            if schema_rows
            else self.facts_store._empty_result(
                [
                    "identifier",
                    "catalog_name",
                    "database_name",
                    "schema_name",
                    "table_name",
                    "table_type",
                    "definition",
                ]
            )
        )
        value_table = (
            pa.Table.from_pylist(value_rows)
            if value_rows
            else self.facts_store._empty_result(
                [
                    "identifier",
                    "catalog_name",
                    "database_name",
                    "schema_name",
                    "table_name",
                    "table_type",
                    "sample_rows",
                ]
            )
        )
        return TableSchema.from_arrow(schema_table), TableValue.from_arrow(value_table)

    def remove_data(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> None:
        where = self._where(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table_type,
        )
        if where is not None:
            self.facts_store._delete_rows(where)
            self.document_store._delete_rows(where)
            if self.vector_store is not None:
                self.vector_store._delete_rows(where)

    def refresh_profiles(self, profiles: List[Dict[str, Any]]) -> int:
        updated_docs: List[Dict[str, Any]] = []
        for profile in profiles:
            table_name = str(profile.get("table_name") or "").strip()
            if not table_name:
                continue
            rows = self.facts_store._search_all(
                where=self._where(
                    catalog_name=str(profile.get("catalog_name") or ""),
                    database_name=str(profile.get("database_name") or ""),
                    schema_name=str(profile.get("schema_name") or ""),
                    table_name=table_name,
                    table_type="full",
                )
            ).to_pylist()
            for fact in rows:
                updated_docs.append(self._document_row(fact, profile=profile))
        return self._upsert_documents(updated_docs)

    def refresh_tables(self, table_refs: List[Dict[str, Any]]) -> int:
        """Rebuild metadata retrieval documents for tables using currently stored profiles."""

        updated_docs: List[Dict[str, Any]] = []
        seen_fact_ids: set[str] = set()
        for table_ref in table_refs:
            table_name = str(table_ref.get("table_name") or "").strip()
            if not table_name:
                continue
            rows = self.facts_store._search_all(
                where=self._where(
                    catalog_name=str(table_ref.get("catalog_name") or ""),
                    database_name=str(table_ref.get("database_name") or ""),
                    schema_name=str(table_ref.get("schema_name") or ""),
                    table_name=table_name,
                    table_type="full",
                )
            ).to_pylist()
            for fact in rows:
                fact_id = self._fact_id(fact)
                if fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                updated_docs.append(self._document_row(fact))
        return self._upsert_documents(updated_docs)

    def refresh_all_tables(self) -> int:
        """Rebuild every metadata retrieval document for this datasource."""

        rows = self.facts_store._search_all(
            where=self._where(table_type="full"),
        ).to_pylist()
        return self._upsert_documents([self._document_row(fact) for fact in rows])

    def _upsert_documents(self, updated_docs: List[Dict[str, Any]]) -> int:
        if not updated_docs:
            return 0
        self.document_store.upsert_batch(updated_docs)
        if self.vector_store is not None:
            from datus.storage.datasource_scope import add_datasource_scope_to_rows

            vector_rows = [dict(row, id=row["doc_id"]) for row in updated_docs]
            self.vector_store.upsert_batch(
                add_datasource_scope_to_rows(vector_rows, self.datasource_id), on_column="storage_key"
            )
            self.vector_store.create_indices()
        self.document_store.create_indices()
        return len(updated_docs)

    def _schema_rows_for_metadata(self, metadata: pa.Table) -> pa.Table:
        if metadata.num_rows == 0:
            return self.facts_store._empty_result(self.SCHEMA_SELECT_FIELDS)

        rows: List[Dict[str, Any]] = []
        for hit in metadata.to_pylist():
            fact_rows = self._facts_for_hit(hit)
            if not fact_rows:
                continue
            schema_row = {field: fact_rows[0].get(field, "") for field in self.SCHEMA_SELECT_FIELDS}
            for field in self.SCORE_FIELDS:
                if hit.get(field) is not None:
                    schema_row[field] = hit[field]
            rows.append(schema_row)
        if not rows:
            return self.facts_store._empty_result(self.SCHEMA_SELECT_FIELDS)
        return pa.Table.from_pylist(rows)

    def _facts_for_hit(self, hit: Dict[str, Any]) -> List[Dict[str, Any]]:
        fact_id = str(hit.get("entity_key") or "").strip()
        if fact_id:
            rows = self.facts_store._search_all(
                where=and_(datasource_condition(self.datasource_id), eq("fact_id", fact_id)),
                select_fields=self.SCHEMA_SELECT_FIELDS,
                limit=1,
            ).to_pylist()
            if rows:
                return rows

        identifier = str(hit.get("identifier") or "").strip()
        if not identifier:
            return []
        return self.facts_store._search_all(
            where=and_(datasource_condition(self.datasource_id), eq("identifier", identifier)),
            select_fields=self.SCHEMA_SELECT_FIELDS,
            limit=1,
        ).to_pylist()

    def _sample_rows_for_metadata(self, metadata: pa.Table) -> pa.Table:
        if metadata.num_rows == 0:
            return self.facts_store._empty_result(self.VALUE_SELECT_FIELDS)
        identifiers = [row.get("identifier") for row in metadata.to_pylist() if row.get("identifier")]
        if not identifiers:
            return self.facts_store._empty_result(self.VALUE_SELECT_FIELDS)
        rows = []
        for identifier in identifiers:
            rows.extend(
                self.facts_store._search_all(
                    where=and_(datasource_condition(self.datasource_id), eq("identifier", identifier)),
                    select_fields=self.VALUE_SELECT_FIELDS,
                    limit=1,
                ).to_pylist()
            )
        rows = [row for row in rows if row.get("sample_rows")]
        return pa.Table.from_pylist(rows) if rows else self.facts_store._empty_result(self.VALUE_SELECT_FIELDS)

    def _sample_rows_by_identifier(self, values: List[Dict[str, Any]]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for item in values:
            identifier = str(item.get("identifier") or "").strip()
            if not identifier:
                continue
            sample_rows = item.get("sample_rows")
            if isinstance(sample_rows, list):
                sample_rows = json2csv(sample_rows)
            text = _normalize_row_text(sample_rows)
            if text:
                result[identifier] = text
        return result

    def _fact_id(self, row: Dict[str, Any]) -> str:
        return "|".join(
            [
                self.datasource_id,
                str(row.get("identifier") or ""),
                str(row.get("table_type") or ""),
            ]
        )

    def _doc_id(self, row: Dict[str, Any]) -> str:
        return "metadata:table:" + self._fact_id(row)

    def _fact_row(self, schema: Dict[str, Any], sample_rows: str = "") -> Dict[str, Any]:
        row = {
            "identifier": str(schema.get("identifier") or "").strip(),
            "catalog_name": str(schema.get("catalog_name") or "").strip(),
            "database_name": str(schema.get("database_name") or "").strip(),
            "schema_name": str(schema.get("schema_name") or "").strip(),
            "table_name": str(schema.get("table_name") or "").strip(),
            "table_type": str(schema.get("table_type") or "table").strip() or "table",
            "definition": str(schema.get("definition") or ""),
            "sample_rows": sample_rows,
        }
        row[DATASOURCE_ID_COLUMN] = self.datasource_id
        row["fact_id"] = self._fact_id(row)
        row["content_hash"] = _stable_hash(row)
        row["updated_at"] = _utc_now()
        return row

    def _document_row(self, fact: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if profile is None:
            profile = self._get_profile_for_fact(fact)
        payload = {
            "identifier": fact.get("identifier", ""),
            "catalog_name": fact.get("catalog_name", ""),
            "database_name": fact.get("database_name", ""),
            "schema_name": fact.get("schema_name", ""),
            "table_name": fact.get("table_name", ""),
            "table_type": fact.get("table_type", ""),
        }
        title = _qualified_name(fact) or str(fact.get("table_name") or "")
        search_parts = [
            f"table {fact.get('table_name', '')}",
            f"qualified_name {title}",
            f"identifier {fact.get('identifier', '')}",
            f"type {fact.get('table_type', '')}",
            f"definition {fact.get('definition', '')}",
        ]
        if fact.get("sample_rows"):
            search_parts.append(f"sample_rows {fact.get('sample_rows')}")
        if profile:
            search_parts.extend(
                [
                    f"semantic_model {profile.get('semantic_model_name', '')}",
                    f"dataset {profile.get('dataset_name') or profile.get('data_source_name') or ''}",
                    f"description {profile.get('description', '')}",
                    f"ai_context {profile.get('ai_context_json', '')}",
                    f"columns {profile.get('columns_json', '')}",
                    f"relationships {profile.get('relationships_json', '')}",
                ]
            )
            payload["semantic_profile_applied"] = True
            payload["semantic_model_name"] = profile.get("semantic_model_name", "")
            payload["description"] = profile.get("description", "")
        search_text = _normalize_row_text(" ".join(search_parts))
        row = {
            "doc_id": self._doc_id(fact),
            DATASOURCE_ID_COLUMN: self.datasource_id,
            "component_type": "metadata",
            "entity_type": "table",
            "entity_key": self._fact_id(fact),
            "identifier": fact.get("identifier", ""),
            "catalog_name": fact.get("catalog_name", ""),
            "database_name": fact.get("database_name", ""),
            "schema_name": fact.get("schema_name", ""),
            "table_name": fact.get("table_name", ""),
            "table_type": fact.get("table_type", ""),
            "title": title,
            "search_text": search_text,
            "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "updated_at": _utc_now(),
        }
        row["content_hash"] = _stable_hash({k: v for k, v in row.items() if k not in {"updated_at", "content_hash"}})
        return row

    def _get_profile_for_fact(self, fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._table_semantic_profiles is None:
            try:
                from datus.storage.table_semantic_profile.store import TableSemanticProfileRAG

                self._table_semantic_profiles = TableSemanticProfileRAG(self.agent_config)
            except Exception as exc:
                logger.debug("Table semantic profile storage unavailable for metadata docs: %s", exc)
                self._table_semantic_profiles = False
        if not self._table_semantic_profiles:
            return None
        try:
            return self._table_semantic_profiles.get_profile(
                catalog_name=str(fact.get("catalog_name") or ""),
                database_name=str(fact.get("database_name") or ""),
                schema_name=str(fact.get("schema_name") or ""),
                table_name=str(fact.get("table_name") or ""),
            )
        except Exception as exc:
            logger.debug("Failed to load table semantic profile for %s: %s", fact.get("table_name"), exc)
            return None

    @staticmethod
    def _parse_table_name(
        full_table: str,
        *,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        dialect: str = DBType.SQLITE,
    ) -> tuple[str, str, str, str]:
        parts = full_table.split(".")
        table_name = parts[-1]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], table_name
        if len(parts) == 3:
            if connector_registry.support_catalog(dialect) and not connector_registry.support_schema(dialect):
                return parts[0], parts[1], "", table_name
            return catalog_name, parts[0], parts[1], table_name
        if len(parts) == 2:
            if not connector_registry.support_schema(dialect):
                return catalog_name, parts[0], "", table_name
            return catalog_name, database_name, parts[0], table_name
        return catalog_name, database_name, schema_name, table_name
