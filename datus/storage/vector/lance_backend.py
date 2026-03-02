# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Default LanceDB implementation of the vector backend abstraction."""

import os
import re
from typing import Any, Dict, List, Optional, Union

import lancedb
import pandas as pd
import pyarrow as pa
from lancedb.db import DBConnection
from lancedb.embeddings import EmbeddingFunctionConfig
from lancedb.embeddings.base import EmbeddingFunction as LanceDBEmbeddingFunction
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.query import LanceQueryBuilder
from lancedb.rerankers import LinearCombinationReranker
from lancedb.table import Table as LanceTable

from datus.storage.conditions import WhereExpr, build_where
from datus.storage.vector.base import BaseVectorBackend, EmbeddingFunction, VectorDatabase, VectorTable
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LanceDB adapter classes for embedding models
# ---------------------------------------------------------------------------


@register("fastembed")
class _LanceFastEmbedAdapter(TextEmbeddingFunction):
    """LanceDB adapter for FastEmbed embedding models."""

    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256

    def ndims(self):
        return self._get_impl().ndims()

    def generate_embeddings(self, texts, *args, **kwargs):
        return self._get_impl().generate_embeddings(texts)

    def _get_impl(self):
        if not hasattr(self, "_impl") or self._impl is None:
            from datus.storage.fastembed_embeddings import FastEmbedEmbeddings

            impl = FastEmbedEmbeddings.create(name=self.name, batch_size=self.batch_size)
            object.__setattr__(self, "_impl", impl)
        return self._impl


@register("openai")
class _LanceOpenAIAdapter(TextEmbeddingFunction):
    """LanceDB adapter for OpenAI embedding models."""

    name: str = "text-embedding-ada-002"
    dim: Optional[int] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    use_azure: bool = False

    def ndims(self):
        return self._get_impl().ndims()

    def generate_embeddings(self, texts, *args, **kwargs):
        return self._get_impl().generate_embeddings(texts)

    def _get_impl(self):
        if not hasattr(self, "_impl") or self._impl is None:
            from datus.storage.embedding_openai import OpenAIEmbeddings

            impl = OpenAIEmbeddings.create(
                name=self.name, dim=self.dim, base_url=self.base_url, api_key=self.api_key, use_azure=self.use_azure
            )
            object.__setattr__(self, "_impl", impl)
        return self._impl


# ---------------------------------------------------------------------------
# Embedding wrapping helper
# ---------------------------------------------------------------------------


def _wrap_embedding(model: EmbeddingFunction) -> TextEmbeddingFunction:
    """Wrap an EmbeddingFunction in a LanceDB-compatible adapter.

    If *model* is already a LanceDB ``EmbeddingFunction``, it is returned as-is.
    Otherwise, a ``_LanceFastEmbedAdapter`` or ``_LanceOpenAIAdapter`` is
    created and the original model is injected as the backing implementation.
    """
    if isinstance(model, LanceDBEmbeddingFunction):
        return model

    from datus.storage.fastembed_embeddings import FastEmbedEmbeddings

    if isinstance(model, FastEmbedEmbeddings):
        adapter = _LanceFastEmbedAdapter(name=model.name, batch_size=model.batch_size)
    else:
        adapter = _LanceOpenAIAdapter(
            name=getattr(model, "name", ""),
            dim=getattr(model, "dim", None),
            base_url=getattr(model, "base_url", None),
            api_key=getattr(model, "api_key", None),
            use_azure=getattr(model, "use_azure", False),
        )
    object.__setattr__(adapter, "_impl", model)
    return adapter


# ---------------------------------------------------------------------------
# LanceVectorTable
# ---------------------------------------------------------------------------


class LanceVectorTable(VectorTable):
    """LanceDB implementation of VectorTable wrapping a ``lancedb.Table``."""

    def __init__(self, lance_table: LanceTable) -> None:
        self._table = lance_table

    # -- Write operations --

    def add(self, data: pd.DataFrame) -> None:
        self._table.add(data)

    def merge_insert(self, data: pd.DataFrame, on_column: str) -> None:
        self._table.merge_insert(on_column).when_matched_update_all().when_not_matched_insert_all().execute(data)

    def delete(self, where: WhereExpr) -> None:
        compiled = build_where(where)
        if compiled:
            self._table.delete(compiled)

    def update(self, where: WhereExpr, values: Dict[str, Any]) -> None:
        compiled = build_where(where)
        if not compiled:
            raise DatusException(
                ErrorCode.STORAGE_FAILED,
                message="update() requires a non-empty where clause to prevent accidental full-table updates",
            )
        self._table.update(where=compiled, values=values)

    # -- Search operations --

    def search_vector(
        self,
        query_text: str,
        vector_column: str,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        compiled = build_where(where)
        query_builder = self._table.search(query=query_text, query_type="vector", vector_column_name=vector_column)
        query_builder = self._fill_query(query_builder, select_fields, compiled)
        return query_builder.limit(top_n).to_arrow()

    def search_hybrid(
        self,
        query_text: str,
        vector_source_column: str,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        compiled = build_where(where)
        query_builder = self._table.search(
            query=query_text, query_type="hybrid", vector_column_name=vector_source_column
        )
        query_builder = self._fill_query(query_builder, select_fields, compiled)
        reranker = LinearCombinationReranker()
        return query_builder.limit(top_n * 2).rerank(reranker).to_arrow()

    def search_all(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        compiled = build_where(where)
        query_builder = self._table.search()
        if compiled:
            query_builder = query_builder.where(compiled)
        if select_fields:
            query_builder = query_builder.select(select_fields)
        if limit is None:
            limit = self.count_rows(compiled)
        return query_builder.limit(limit).to_arrow()

    def count_rows(self, where: WhereExpr = None) -> int:
        compiled = build_where(where)
        if compiled:
            return self._table.count_rows(compiled)
        return self._table.count_rows()

    # -- Index operations --

    def create_vector_index(self, column: str, metric: str = "cosine", **kwargs) -> None:
        self._table.create_index(metric=metric, vector_column_name=column, **kwargs)

    def create_fts_index(self, field_names: Union[str, List[str]]) -> None:
        self._table.create_fts_index(field_names=field_names, replace=True)

    def create_scalar_index(self, column: str) -> None:
        self._table.create_scalar_index(column, replace=True)

    # -- Maintenance --

    def compact_files(self) -> None:
        self._table.compact_files()

    def cleanup_old_versions(self) -> None:
        self._table.cleanup_old_versions()

    # -- Internal helpers --

    @staticmethod
    def _fill_query(
        query_builder: LanceQueryBuilder,
        select_fields: Optional[List[str]] = None,
        where: Optional[str] = None,
    ) -> LanceQueryBuilder:
        if where:
            query_builder = query_builder.where(where, True)
        if select_fields and len(select_fields) > 0:
            query_builder = query_builder.select(select_fields)
        return query_builder


# ---------------------------------------------------------------------------
# LanceVectorDatabase
# ---------------------------------------------------------------------------


class LanceVectorDatabase(VectorDatabase):
    """LanceDB implementation of VectorDatabase wrapping a ``lancedb.DBConnection``."""

    def __init__(self, db_connection: DBConnection) -> None:
        self._db = db_connection

    def table_exists(self, table_name: str) -> bool:
        try:
            self._db.open_table(table_name)
            return True
        except ValueError:
            return False

    def table_names(self, limit: int = 100) -> List[str]:
        return self._db.table_names(limit=limit)

    def create_table(
        self,
        table_name: str,
        schema: Optional[pa.Schema] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
        exist_ok: bool = True,
        unique_columns: Optional[List[str]] = None,
    ) -> LanceVectorTable:
        kwargs: Dict[str, Any] = {"exist_ok": exist_ok}
        if schema is not None:
            kwargs["schema"] = schema
        if embedding_function is not None:
            lance_fn = _wrap_embedding(embedding_function)
            kwargs["embedding_functions"] = [
                EmbeddingFunctionConfig(
                    vector_column=vector_column,
                    source_column=source_column,
                    function=lance_fn,
                )
            ]
        raw_table = self._db.create_table(table_name, **kwargs)
        return LanceVectorTable(raw_table)

    def open_table(
        self,
        table_name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
    ) -> LanceVectorTable:
        # LanceDB persists embedding config in Arrow schema metadata,
        # so the embedding_function parameter is intentionally ignored here.
        raw_table = self._db.open_table(table_name)
        return LanceVectorTable(raw_table)

    def drop_table(self, table_name: str, ignore_missing: bool = False) -> None:
        self._db.drop_table(table_name, ignore_missing=ignore_missing)

    def refresh_table(
        self,
        table_name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
    ) -> LanceVectorTable:
        return self.open_table(table_name)

    def close(self) -> None:
        pass  # LanceDB connections are lightweight, no explicit close needed


# ---------------------------------------------------------------------------
# LanceVectorBackend (slim lifecycle-only class)
# ---------------------------------------------------------------------------


_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _safe_path_segment(value: str, field_name: str) -> str:
    """Validate a filesystem path segment to prevent directory traversal."""
    if not value:
        return value
    if not _SEGMENT_RE.fullmatch(value):
        raise DatusException(
            ErrorCode.STORAGE_FAILED,
            message=f"Invalid {field_name}: {value!r}. Only alphanumeric, underscore, dot, and hyphen are allowed.",
        )
    return value


class LanceVectorBackend(BaseVectorBackend):
    """LanceDB implementation of the vector backend."""

    def initialize(self, config: Dict[str, Any]) -> None:
        self._data_dir = config.get("data_dir", "")

    def connect(self, namespace: str) -> LanceVectorDatabase:
        safe_ns = _safe_path_segment(namespace, "namespace")
        db_path = os.path.join(self._data_dir, f"datus_db_{safe_ns}")
        raw_db = lancedb.connect(db_path)
        return LanceVectorDatabase(raw_db)

    def close(self) -> None:
        pass  # LanceDB connections are lightweight
