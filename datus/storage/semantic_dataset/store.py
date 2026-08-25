# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Row-wise projection of authored Dosi semantic models.

The unit of meaning is the OSI ``dataset``, not the physical table: one
dataset binds either a physical table or a reusable query, and one physical
table may be modelled by several datasets living in different semantic
models. Rows are therefore keyed by ``(semantic_model, dataset)`` and the
physical table is an attribute of a dataset rather than the identity of a row.

Fields and relationships are rows too. Keeping them out of JSON blobs is what
lets a caller filter on ``source_table``, ``is_primary_key`` or
``from_dataset`` instead of decoding every row to inspect it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa
from datus_storage_base.conditions import And, eq, in_, not_

from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.datasource_scope import add_datasource_scope_to_rows, datasource_condition, resolve_datasource_id
from datus.storage.fts import FtsField, FtsSpec
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)

KIND_DATASET = "dataset"
KIND_FIELD = "field"
KIND_RELATIONSHIP = "relationship"

_TABLE_REF_FIELDS = ["catalog_name", "database_name", "schema_name", "source_table"]
_DATASET_ORDER_FIELDS = ("semantic_model_name", "dataset_name")
_DATASET_NAMESPACE_FIELDS = ("catalog_name", "database_name", "schema_name")


def _strip_identifier_quotes(value: str) -> str:
    text = str(value or "").strip()
    while len(text) >= 2 and (
        (text[0] == text[-1] and text[0] in {'"', "'", "`"}) or (text[0] == "[" and text[-1] == "]")
    ):
        text = text[1:-1].strip()
    return text


def _identifier_variants(value: str) -> List[str]:
    """Return common exact-match variants for SQL identifiers."""

    parts = [_strip_identifier_quotes(part) for part in str(value or "").split(".") if part.strip()]
    variants: List[str] = []
    for start in range(len(parts)):
        candidate = ".".join(parts[start:])
        if candidate and candidate not in variants:
            variants.append(candidate)
    if parts:
        leaf = parts[-1]
        if leaf and leaf not in variants:
            variants.append(leaf)
    raw = _strip_identifier_quotes(value)
    if raw and raw not in variants:
        variants.append(raw)
    lower_variants = []
    for item in variants:
        lowered = item.lower()
        if lowered and lowered not in variants and lowered not in lower_variants:
            lower_variants.append(lowered)
    variants.extend(lower_variants)
    return variants


def _normalized_identifier(value: str) -> str:
    parts = [_strip_identifier_quotes(part) for part in str(value or "").split(".") if part.strip()]
    return ".".join(parts).lower()


class MetricKindNotHere(ValueError):
    """Raised when a caller asks this store for metrics.

    Metrics live in their own store. Answering with an empty list instead
    reads as "no such metric" and sends the caller down the wrong path.
    """


@dataclass(frozen=True)
class ResolvedKinds:
    kinds: list[str]
    primary_key_only: bool


# Pre-Dosi callers speak of tables and columns; an ``entity`` was a key field.
_LEGACY_KIND_ALIASES = {
    "table": KIND_DATASET,
    "column": KIND_FIELD,
    "entity": KIND_FIELD,
}


def resolve_object_kinds(kinds: Optional[List[str] | str]) -> ResolvedKinds:
    """Normalize a requested kind filter onto this store's vocabulary."""
    requested = [kinds] if isinstance(kinds, str) else list(kinds or [])
    names = [str(kind).strip().lower() for kind in requested]
    names = [name for name in names if name]
    if "metric" in names:
        raise MetricKindNotHere(
            "search_semantic_objects covers modelled datasets, fields and relationships only; "
            "metrics are stored separately. Use search_metrics instead."
        )

    normalized: list[str] = []
    for name in names:
        resolved = _LEGACY_KIND_ALIASES.get(name, name)
        if resolved not in normalized:
            normalized.append(resolved)

    # "entity" meant key fields specifically. Both it and "column" resolve to
    # "field", so the narrowing has to be decided on what was asked for: asking
    # for columns as well would otherwise silently drop the non-key ones.
    primary_key_only = set(names) == {"entity"}
    return ResolvedKinds(kinds=normalized, primary_key_only=primary_key_only)


def dataset_row_id(semantic_model: str, dataset_name: str) -> str:
    return f"{KIND_DATASET}:{semantic_model}:{dataset_name}"


def field_row_id(semantic_model: str, dataset_name: str, field_name: str) -> str:
    return f"{KIND_FIELD}:{semantic_model}:{dataset_name}.{field_name}"


def relationship_row_id(semantic_model: str, relationship_name: str) -> str:
    return f"rel:{semantic_model}:{relationship_name}"


class SemanticDatasetStorage(BaseEmbeddingStore):
    """One row per authored dataset, field or relationship."""

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
            table_name="semantic_dataset",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    # -- Identity --
                    pa.field("id", pa.string()),
                    pa.field("kind", pa.string()),  # dataset | field | relationship
                    pa.field("semantic_model_name", pa.string()),
                    pa.field("dataset_name", pa.string()),
                    pa.field("name", pa.string()),
                    # -- Physical binding (dataset rows) --
                    # source_table is empty for a query-backed dataset, which is
                    # what keeps it from being mistaken for a real table.
                    pa.field("source_table", pa.string()),
                    pa.field("source_query", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    # A key may span several columns, so this is a list of
                    # groups rather than a flag on each field row.
                    pa.field("unique_keys_json", pa.string()),
                    # -- Content --
                    pa.field("description", pa.string()),
                    pa.field("ai_context_json", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                    # -- Field rows --
                    pa.field("expr", pa.string()),
                    pa.field("field_type", pa.string()),
                    pa.field("is_dimension", pa.bool_()),
                    pa.field("is_time", pa.bool_()),
                    pa.field("is_primary_key", pa.bool_()),
                    pa.field("time_granularity", pa.string()),
                    # -- Relationship rows --
                    # Column tuples stay JSON: they are ordered pairs that are
                    # never filtered on, and a list column degrades to TEXT on
                    # the PostgreSQL backend.
                    pa.field("from_dataset", pa.string()),
                    pa.field("to_dataset", pa.string()),
                    pa.field("from_columns_json", pa.string()),
                    pa.field("to_columns_json", pa.string()),
                    pa.field("rel_type", pa.string()),
                    pa.field("join_type", pa.string()),
                    # -- Lineage --
                    pa.field("yaml_path", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
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
        self._create_scalar_index("id")
        self._create_scalar_index("kind")
        self._create_scalar_index("semantic_model_name")
        self._create_scalar_index("dataset_name")
        self._create_scalar_index("source_table")
        self.create_fts_index(
            FtsSpec((FtsField("name", boost=3.0), FtsField("dataset_name", boost=2.0), FtsField("search_text")))
        )


class SemanticDatasetRAG:
    """Read and replace the dataset projection for one datasource."""

    def __init__(
        self,
        agent_config: "AgentConfig",
        sub_agent_name: Optional[str] = None,
        datasource_id: Optional[str] = None,
    ):
        from datus.storage.rag_scope import _build_sub_agent_filter
        from datus.storage.registry import get_storage

        self.agent_config = agent_config
        self.datasource_id = resolve_datasource_id(agent_config, datasource_id)
        self.storage: SemanticDatasetStorage = get_storage(
            SemanticDatasetStorage,
            "semantic_model",
            project=agent_config.project_name,
            datasource_id=self.datasource_id,
        )
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.storage, "tables")

    def _sub_agent_conditions(self) -> list:
        conditions = [datasource_condition(self.datasource_id)]
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_datasets(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        semantic_model: str = "",
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return every dataset bound to one physical table, primary first.

        Query-backed datasets are excluded by construction: they carry no
        ``source_table``, so they can never shadow a real table that happens
        to share their name.

        Ordering is a lexicographic sort on ``(semantic_model_name,
        dataset_name)``. It deliberately uses no relevance score: this is a
        scalar point lookup, so neither backend produces a distance column,
        and neither guarantees row order without an explicit sort.

        ``semantic_model`` filters rather than re-ranks: a caller that already
        knows the model wants that model or nothing.
        """
        if not table_name:
            logger.warning("list_datasets called without table_name")
            return []

        query_fields = self._with_order_fields(select_fields)
        base_conds = [eq("kind", KIND_DATASET), *self._sub_agent_conditions()]
        if semantic_model:
            base_conds.append(eq("semantic_model_name", semantic_model))

        exact = [eq("source_table", table_name), *base_conds]
        if catalog_name:
            exact.append(eq("catalog_name", catalog_name))
        if database_name:
            exact.append(eq("database_name", database_name))
        if schema_name:
            exact.append(eq("schema_name", schema_name))
        rows = self.storage._search_all(where=And(exact), select_fields=query_fields).to_pylist()

        if not rows and (catalog_name or database_name or schema_name):
            broad = [eq("source_table", table_name), *base_conds]
            broad_rows = self.storage._search_all(where=And(broad), select_fields=query_fields).to_pylist()
            compatible = [
                row
                for row in broad_rows
                if self._namespace_compatible(
                    row,
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                )
            ]
            # Namespace-compatible hits that agree on their own coordinates are
            # all the same physical table under an under-qualified reference --
            # several of them is a table modelled more than once, not an
            # ambiguity. Hits that disagree name different tables, so nothing is
            # returned rather than a guess.
            rows = compatible if self._share_one_namespace(compatible) else []

        if not rows and table_name.lower() != table_name:
            lower = [eq("source_table", table_name.lower()), *base_conds]
            if catalog_name:
                lower.append(eq("catalog_name", catalog_name))
            if database_name:
                lower.append(eq("database_name", database_name))
            if schema_name:
                lower.append(eq("schema_name", schema_name))
            rows = self.storage._search_all(where=And(lower), select_fields=query_fields).to_pylist()

        rows.sort(key=self._dataset_sort_key)
        return self._project(rows, select_fields)

    def list_fields(self, semantic_model: str, dataset_name: str) -> List[Dict[str, Any]]:
        """Return one dataset's fields, keys first, then time, then the rest.

        Storage returns no guaranteed order, so this restores the shape the
        model was authored in instead of letting the display drift between
        reads.
        """
        if not dataset_name:
            return []
        conditions = [
            eq("kind", KIND_FIELD),
            eq("semantic_model_name", semantic_model),
            eq("dataset_name", dataset_name),
            *self._sub_agent_conditions(),
        ]
        rows = self.storage._search_all(where=And(conditions)).to_pylist()
        rows.sort(
            key=lambda row: (
                not row.get("is_primary_key"),
                not row.get("is_time"),
                not row.get("is_dimension"),
                str(row.get("name") or ""),
            )
        )
        return rows

    def list_relationships(self, semantic_model: str, dataset_name: str) -> List[Dict[str, Any]]:
        """Return the relationships that touch one dataset, either end."""
        if not dataset_name:
            return []
        conditions = [
            eq("kind", KIND_RELATIONSHIP),
            eq("semantic_model_name", semantic_model),
            *self._sub_agent_conditions(),
        ]
        rows = self.storage._search_all(where=And(conditions)).to_pylist()
        touching = [row for row in rows if dataset_name in (row.get("from_dataset", ""), row.get("to_dataset", ""))]
        touching.sort(key=lambda row: str(row.get("name") or ""))
        return touching

    def list_objects(
        self,
        query_text: str,
        kinds: Optional[List[str]] = None,
        table_name: str = "",
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search datasets, fields and relationships by description.

        ``kinds`` accepts the pre-Dosi vocabulary as well, so a caller that
        still asks for ``table`` or ``column`` keeps working.

        ``table_name`` narrows to one physical table. Relationships span two
        datasets and bind no single table, so they never match it.
        """
        resolved = resolve_object_kinds(kinds)
        conditions = list(self._sub_agent_conditions())
        if resolved.kinds:
            conditions.append(in_("kind", resolved.kinds))
        if resolved.primary_key_only:
            conditions.append(eq("is_primary_key", True))
        if table_name:
            conditions.append(eq("source_table", table_name))
        return self.storage.search(query_txt=query_text, top_n=top_n, where=And(conditions)).to_pylist()

    def table_exists(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> bool:
        """Whether any dataset in scope is bound to this physical table."""
        if not table_name:
            return False
        return bool(
            self.list_datasets(
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
                select_fields=["id"],
            )
        )

    def get_table_projection(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        semantic_model: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Assemble the primary dataset plus its fields and relationships.

        Consumers that render or index a whole table at once want it in one
        piece; they should not have to know the row layout.
        """
        datasets = self.list_datasets(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            semantic_model=semantic_model,
        )
        if not datasets:
            return None
        primary = datasets[0]
        model = str(primary.get("semantic_model_name") or "")
        dataset = str(primary.get("dataset_name") or "")
        return {
            **primary,
            "fields": self.list_fields(model, dataset),
            "relationships": self.list_relationships(model, dataset),
            "alternatives": datasets[1:],
        }

    def get_size(self) -> int:
        try:
            return self.storage._count_rows(where=And([eq("kind", KIND_DATASET), *self._sub_agent_conditions()]))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Artifact replacement
    # ------------------------------------------------------------------

    def truncate(self) -> None:
        rows = self._table_refs(where=And(self._sub_agent_conditions()))
        self.storage.delete_datasource_rows(self.datasource_id)
        self._refresh_metadata_documents_for_tables(rows)

    def list_artifact_paths(self) -> List[str]:
        """Every distinct ``yaml_path`` this store currently holds rows for.

        A tree-wide sync needs this to notice files that disappeared: the
        per-artifact deletes are scoped to a path, so a model whose file was
        removed is never visited and its rows would outlive it.
        """
        rows = self.storage._search_all(
            where=And(self._sub_agent_conditions()), select_fields=["yaml_path"]
        ).to_pylist()
        return sorted({str(row.get("yaml_path") or "") for row in rows} - {""})

    def delete_artifact_rows(self, yaml_path: str) -> None:
        """Delete rows projected from a single YAML artifact."""
        if not yaml_path:
            return
        where = And([eq("yaml_path", yaml_path)] + self._sub_agent_conditions())
        rows = self._table_refs(where=where)
        self.storage._delete_rows(where)
        self._refresh_metadata_documents_for_tables(rows)

    def delete_artifact_rows_except(self, yaml_path: str, keep_ids: List[str]) -> None:
        """Delete stale rows for one YAML artifact after replacement succeeds."""
        if not yaml_path:
            return
        normalized_keep_ids = [row_id for row_id in keep_ids if row_id]
        if not normalized_keep_ids:
            self.delete_artifact_rows(yaml_path)
            return
        where = And([eq("yaml_path", yaml_path), not_(in_("id", normalized_keep_ids))] + self._sub_agent_conditions())
        rows = self._table_refs(where=where)
        self.storage._delete_rows(where)
        self._refresh_metadata_documents_for_tables(rows)

    def list_artifact_rows(self, yaml_path: str) -> List[Dict[str, Any]]:
        """Return rows projected from a single YAML artifact."""
        if not yaml_path:
            return []
        return self.storage._search_all(
            where=And([eq("yaml_path", yaml_path)] + self._sub_agent_conditions())
        ).to_pylist()

    def restore_artifact_rows(self, yaml_path: str, rows: List[Dict[str, Any]]) -> None:
        """Restore one YAML artifact to a previously captured row snapshot."""
        if not yaml_path:
            return
        self.delete_artifact_rows(yaml_path)
        if rows:
            self.upsert_batch(rows)
        self.create_indices()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def store_batch(self, rows: List[Dict[str, Any]]) -> None:
        self.storage.store_batch(add_datasource_scope_to_rows(rows, self.datasource_id))
        self._refresh_metadata_documents_with_rows(rows)

    def upsert_batch(self, rows: List[Dict[str, Any]]) -> None:
        self.storage.upsert_batch(add_datasource_scope_to_rows(rows, self.datasource_id), on_column="storage_key")
        self._refresh_metadata_documents_with_rows(rows)

    def create_indices(self) -> None:
        self.storage.create_indices()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dataset_sort_key(row: Dict[str, Any]) -> tuple:
        return tuple(str(row.get(name) or "") for name in _DATASET_ORDER_FIELDS)

    @staticmethod
    def _with_order_fields(select_fields: Optional[List[str]]) -> Optional[List[str]]:
        """Widen a projection so the sort and namespace keys are always readable.

        A narrow projection would otherwise read the namespace columns back as
        None and make every row look namespace-compatible.
        """
        if not select_fields:
            return None
        required = (*_DATASET_ORDER_FIELDS, *_DATASET_NAMESPACE_FIELDS)
        missing = [name for name in required if name not in select_fields]
        return [*select_fields, *missing] if missing else list(select_fields)

    @staticmethod
    def _project(rows: List[Dict[str, Any]], select_fields: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Narrow rows back to what the caller asked for."""
        if not select_fields:
            return rows
        return [{name: row.get(name) for name in select_fields} for row in rows]

    @staticmethod
    def _namespace_compatible(
        row: Dict[str, Any],
        *,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> bool:
        for field, requested in (
            ("catalog_name", catalog_name),
            ("database_name", database_name),
            ("schema_name", schema_name),
        ):
            if requested and row.get(field) not in ("", None, requested):
                return False
        return True

    @staticmethod
    def _share_one_namespace(rows: List[Dict[str, Any]]) -> bool:
        """Whether every row names the same physical table."""
        if not rows:
            return False
        coordinates = {tuple(str(row.get(name) or "") for name in _DATASET_NAMESPACE_FIELDS) for row in rows}
        return len(coordinates) == 1

    def _table_refs(self, where) -> List[Dict[str, Any]]:
        try:
            return self.storage._search_all(where=where, select_fields=_TABLE_REF_FIELDS).to_pylist()
        except Exception as exc:
            logger.debug("Failed to load semantic dataset rows before deletion: %s", exc)
            return []

    def _refresh_metadata_documents_with_rows(self, rows: List[Dict[str, Any]]) -> None:
        table_refs = [
            {
                "catalog_name": row.get("catalog_name", ""),
                "database_name": row.get("database_name", ""),
                "schema_name": row.get("schema_name", ""),
                "source_table": row.get("source_table", ""),
            }
            for row in rows
            if row.get("kind") == KIND_DATASET and row.get("source_table")
        ]
        self._refresh_metadata_documents_for_tables(table_refs)

    def _refresh_metadata_documents_for_tables(self, table_refs: List[Dict[str, Any]]) -> None:
        if not table_refs:
            return
        try:
            from datus.storage.kb_retrieval import MetadataFtsRAG, metadata_fts_enabled

            if metadata_fts_enabled(self.agent_config):
                MetadataFtsRAG(self.agent_config, datasource_id=self.datasource_id).refresh_tables(
                    [{**ref, "table_name": ref.get("source_table", "")} for ref in table_refs]
                )
        except Exception as exc:
            logger.debug("Failed to refresh metadata retrieval documents from semantic datasets: %s", exc)


SYNC_YAML_HINT = (
    "Authored semantic YAML exists but its knowledge-base projection is empty. "
    "Run: datus-agent bootstrap-kb --components semantic_model --kb_update_strategy sync-yaml"
)


def semantic_projection_is_stale(agent_config: "AgentConfig") -> bool:
    """Whether authored YAML exists with nothing projected from it.

    True right after an upgrade that changed the projection layout, and after
    any manual wipe of the store. Callers surface the hint rather than syncing
    on the spot: a full re-projection is too slow to run from an interactive
    lookup, and stalling silently is worse than saying what to run.
    """
    try:
        if SemanticDatasetRAG(agent_config).get_size() > 0:
            return False
    except Exception as exc:
        logger.debug("Unable to size the semantic dataset projection: %s", exc)
        return False

    try:
        from datus.agent.node.semantic_authoring import _osi_semantic_model_dir

        # Reuse what sync-yaml itself would pick up, so the hint is never shown
        # for files that a sync would then skip.
        from datus.storage.semantic_model.semantic_model_init import semantic_yaml_files

        model_dir = _osi_semantic_model_dir(agent_config)
        return bool(model_dir is not None and semantic_yaml_files(model_dir))
    except Exception as exc:
        logger.debug("Unable to resolve the semantic model directory: %s", exc)
        return False
