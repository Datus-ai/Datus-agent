import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest

from datus.storage.semantic_dataset.store import (
    MetricKindNotHere,
    SemanticDatasetRAG,
    resolve_object_kinds,
)


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return self._rows


def _rag_with_rows(*row_sets):
    rag = SemanticDatasetRAG.__new__(SemanticDatasetRAG)
    rag.storage = Mock()
    rag._sub_agent_conditions = lambda: []
    rag.storage._search_all.side_effect = [_Rows(rows) for rows in row_sets]
    return rag


def test_list_datasets_uses_unique_namespace_fallback_only():
    expected = {"source_table": "orders", "database_name": "shop"}
    rag = _rag_with_rows([], [expected])

    result = rag.list_datasets(database_name="shop", table_name="orders")

    assert result == [expected]


def test_list_datasets_ignores_broad_hits_from_another_namespace():
    """A hit in a different database must not veto the one that matches."""
    expected = {"source_table": "orders", "database_name": "shop"}
    rag = _rag_with_rows(
        [],
        [expected, {"source_table": "orders", "database_name": "archive"}],
    )

    result = rag.list_datasets(database_name="shop", table_name="orders")

    assert result == [expected]


def test_list_datasets_keeps_every_model_of_an_under_qualified_table():
    """Authored YAML often leaves catalog blank while the connector fills in a
    default, so the broad lookup is the normal path for a table modelled more
    than once -- all of its models must survive it."""
    rows = [
        {"source_table": "orders", "database_name": "shop", "semantic_model_name": "sales"},
        {"source_table": "orders", "database_name": "shop", "semantic_model_name": "fulfillment"},
    ]
    rag = _rag_with_rows([], rows)

    result = rag.list_datasets(catalog_name="default_catalog", database_name="shop", table_name="orders")

    assert [row["semantic_model_name"] for row in result] == ["fulfillment", "sales"]


def test_list_datasets_rejects_broad_hits_that_name_different_tables():
    """Both rows are namespace-compatible with the request but sit in different
    catalogs, so the request cannot tell which table it meant."""
    rag = _rag_with_rows(
        [],
        [
            {"source_table": "orders", "catalog_name": "", "database_name": "shop"},
            {"source_table": "orders", "catalog_name": "warehouse", "database_name": "shop"},
        ],
    )

    result = rag.list_datasets(database_name="shop", table_name="orders")

    assert result == []


def test_list_datasets_rejects_unique_fallback_with_conflicting_namespace():
    rag = _rag_with_rows([], [{"source_table": "orders", "database_name": "archive"}])

    result = rag.list_datasets(database_name="shop", table_name="orders")

    assert result == []


def test_list_datasets_lowercase_fallback_runs_after_ambiguous_broad_lookup():
    expected = {"source_table": "orders", "database_name": "shop"}
    rag = _rag_with_rows(
        [],
        [
            {"source_table": "Orders", "catalog_name": "", "database_name": "shop"},
            {"source_table": "Orders", "catalog_name": "warehouse", "database_name": "shop"},
        ],
        [expected],
    )

    result = rag.list_datasets(database_name="shop", table_name="Orders")

    assert result == [expected]


def test_list_datasets_returns_every_model_for_one_table():
    rows = [
        {"source_table": "orders", "semantic_model_name": "sales", "dataset_name": "orders"},
        {"source_table": "orders", "semantic_model_name": "fulfillment", "dataset_name": "orders"},
    ]
    rag = _rag_with_rows(rows)

    result = rag.list_datasets(table_name="orders")

    assert [row["semantic_model_name"] for row in result] == ["fulfillment", "sales"]


def test_list_datasets_order_does_not_depend_on_storage_row_order():
    """Neither backend guarantees row order for a scalar scan, so the sort has
    to come from the data: LanceDB scans without a query vector and PostgreSQL
    returns heap order, which shifts after UPDATE/VACUUM."""
    forward = [
        {"source_table": "orders", "semantic_model_name": "sales", "dataset_name": "orders"},
        {"source_table": "orders", "semantic_model_name": "fulfillment", "dataset_name": "orders"},
    ]
    reversed_rows = list(reversed(forward))

    first = _rag_with_rows(forward).list_datasets(table_name="orders")
    second = _rag_with_rows(reversed_rows).list_datasets(table_name="orders")

    assert first == second
    assert [row["semantic_model_name"] for row in first] == ["fulfillment", "sales"]


def test_list_datasets_breaks_ties_on_dataset_name():
    rows = [
        {"source_table": "orders", "semantic_model_name": "sales", "dataset_name": "orders_returned"},
        {"source_table": "orders", "semantic_model_name": "sales", "dataset_name": "orders_all"},
    ]
    rag = _rag_with_rows(rows)

    result = rag.list_datasets(table_name="orders")

    assert [row["dataset_name"] for row in result] == ["orders_all", "orders_returned"]


def test_list_datasets_filters_on_semantic_model_instead_of_reranking():
    rag = _rag_with_rows([{"source_table": "orders", "semantic_model_name": "fulfillment", "dataset_name": "orders"}])

    result = rag.list_datasets(table_name="orders", semantic_model="fulfillment")

    assert [row["semantic_model_name"] for row in result] == ["fulfillment"]
    where = rag.storage._search_all.call_args.kwargs["where"]
    assert "fulfillment" in str(where)


def test_list_datasets_without_table_name_returns_empty():
    rag = _rag_with_rows([{"source_table": "orders"}])

    assert rag.list_datasets(table_name="") == []
    rag.storage._search_all.assert_not_called()


def test_list_datasets_projects_back_to_requested_fields():
    """Sort and namespace keys are read even when the caller did not ask for
    them: without the namespace columns every row reads back as None and the
    ambiguity check cannot tell two tables apart."""
    rows = [
        {"source_table": "orders", "semantic_model_name": "sales", "dataset_name": "orders"},
        {"source_table": "orders", "semantic_model_name": "fulfillment", "dataset_name": "orders"},
    ]
    rag = _rag_with_rows(rows)

    result = rag.list_datasets(table_name="orders", select_fields=["source_table"])

    assert result == [{"source_table": "orders"}, {"source_table": "orders"}]
    queried = rag.storage._search_all.call_args.kwargs["select_fields"]
    assert set(queried) == {
        "source_table",
        "semantic_model_name",
        "dataset_name",
        "catalog_name",
        "database_name",
        "schema_name",
    }


def test_list_datasets_puts_the_primary_dataset_first():
    rows = [
        {"source_table": "orders", "semantic_model_name": "sales", "dataset_name": "orders"},
        {"source_table": "orders", "semantic_model_name": "fulfillment", "dataset_name": "orders"},
    ]
    rag = _rag_with_rows(rows)

    assert rag.list_datasets(table_name="orders")[0]["semantic_model_name"] == "fulfillment"


def _artifact_rag():
    rag = SemanticDatasetRAG.__new__(SemanticDatasetRAG)
    rag.agent_config = SimpleNamespace(kb_search=SimpleNamespace(mode="vector"), kb_search_mode="vector")
    rag.datasource_id = "test_datasource"
    rag.storage = Mock()
    rag.storage._search_all.return_value = _Rows([])
    rag._sub_agent_conditions = Mock(return_value=[])
    return rag


def test_delete_artifact_rows_ignores_empty_yaml_path():
    rag = _artifact_rag()

    rag.delete_artifact_rows("")

    rag._sub_agent_conditions.assert_not_called()
    rag.storage._delete_rows.assert_not_called()


def test_delete_artifact_rows_uses_sub_agent_scope():
    rag = _artifact_rag()

    rag.delete_artifact_rows("semantic/orders.yml")

    rag._sub_agent_conditions.assert_called_once_with()
    rag.storage._search_all.assert_called_once()
    rag.storage._delete_rows.assert_called_once()


def test_delete_artifact_rows_except_deletes_all_when_keep_ids_empty():
    rag = SemanticDatasetRAG.__new__(SemanticDatasetRAG)
    rag.delete_artifact_rows = Mock()

    rag.delete_artifact_rows_except("semantic/orders.yml", ["", None])

    rag.delete_artifact_rows.assert_called_once_with("semantic/orders.yml")


def test_delete_artifact_rows_except_keeps_current_ids():
    rag = _artifact_rag()

    rag.delete_artifact_rows_except("semantic/orders.yml", ["profile:orders"])

    rag._sub_agent_conditions.assert_called_once_with()
    rag.storage._search_all.assert_called_once()
    rag.storage._delete_rows.assert_called_once()


def test_delete_artifact_rows_refreshes_metadata_documents_for_deleted_tables():
    rag = _artifact_rag()
    rag.agent_config = SimpleNamespace(kb_search=SimpleNamespace(mode="fts"), kb_search_mode="fts")
    deleted_rows = [{"catalog_name": "", "database_name": "db", "schema_name": "public", "source_table": "orders"}]
    rag.storage._search_all.return_value = _Rows(deleted_rows)

    with patch("datus.storage.kb_retrieval.MetadataFtsRAG") as metadata_cls:
        rag.delete_artifact_rows("semantic/orders.yml")

    metadata_cls.assert_called_once_with(rag.agent_config, datasource_id=rag.datasource_id)
    metadata_cls.return_value.refresh_tables.assert_called_once_with([{**deleted_rows[0], "table_name": "orders"}])


def test_truncate_refreshes_metadata_documents_for_deleted_tables():
    rag = _artifact_rag()
    rag.agent_config = SimpleNamespace(kb_search=SimpleNamespace(mode="fts"), kb_search_mode="fts")
    deleted_rows = [{"catalog_name": "", "database_name": "db", "schema_name": "public", "source_table": "orders"}]
    rag.storage._search_all.return_value = _Rows(deleted_rows)

    with patch("datus.storage.kb_retrieval.MetadataFtsRAG") as metadata_cls:
        rag.truncate()

    rag.storage.delete_datasource_rows.assert_called_once_with("test_datasource")
    metadata_cls.return_value.refresh_tables.assert_called_once_with([{**deleted_rows[0], "table_name": "orders"}])


def test_list_artifact_rows_handles_empty_and_non_empty_paths():
    rag = _artifact_rag()
    rag.storage._search_all.return_value = _Rows([{"id": "profile:orders"}])

    assert rag.list_artifact_rows("") == []
    assert rag.list_artifact_rows("semantic/orders.yml") == [{"id": "profile:orders"}]

    rag._sub_agent_conditions.assert_called_once_with()
    rag.storage._search_all.assert_called_once()


def test_restore_artifact_rows_handles_empty_and_non_empty_paths():
    rag = SemanticDatasetRAG.__new__(SemanticDatasetRAG)
    rag.delete_artifact_rows = Mock()
    rag.upsert_batch = Mock()
    rag.create_indices = Mock()
    rows = [{"id": "profile:orders"}]

    rag.restore_artifact_rows("", rows)
    rag.restore_artifact_rows("semantic/orders.yml", rows)

    rag.delete_artifact_rows.assert_called_once_with("semantic/orders.yml")
    rag.upsert_batch.assert_called_once_with(rows)
    rag.create_indices.assert_called_once_with()


def _kind_rag(rows):
    rag = SemanticDatasetRAG.__new__(SemanticDatasetRAG)
    rag.storage = Mock()
    rag._sub_agent_conditions = lambda: []
    rag.storage._search_all.return_value = _Rows(rows)
    return rag


def test_list_fields_orders_keys_then_time_then_the_rest():
    """Storage returns no guaranteed order, so the authored shape is restored
    here rather than letting the display drift between reads."""
    rag = _kind_rag(
        [
            {"name": "amount"},
            {"name": "order_date", "is_dimension": True, "is_time": True},
            {"name": "order_id", "is_primary_key": True},
            {"name": "segment", "is_dimension": True},
        ]
    )

    names = [row["name"] for row in rag.list_fields("sales", "orders")]

    assert names == ["order_id", "order_date", "segment", "amount"]


def test_list_fields_without_dataset_returns_empty():
    rag = _kind_rag([{"name": "amount"}])

    assert rag.list_fields("sales", "") == []
    rag.storage._search_all.assert_not_called()


def test_list_relationships_keeps_only_the_ones_touching_the_dataset():
    rag = _kind_rag(
        [
            {"name": "b_rel", "from_dataset": "orders", "to_dataset": "customers"},
            {"name": "a_rel", "from_dataset": "shipments", "to_dataset": "orders"},
            {"name": "other", "from_dataset": "invoices", "to_dataset": "customers"},
        ]
    )

    names = [row["name"] for row in rag.list_relationships("sales", "orders")]

    assert names == ["a_rel", "b_rel"]


def test_resolve_object_kinds_maps_the_pre_dosi_vocabulary():
    assert resolve_object_kinds(["table", "column"]).kinds == ["dataset", "field"]
    assert resolve_object_kinds("column").kinds == ["field"]
    assert resolve_object_kinds(None).kinds == []


def test_resolve_object_kinds_narrows_entity_to_key_fields():
    resolved = resolve_object_kinds(["entity"])

    assert resolved.kinds == ["field"]
    assert resolved.primary_key_only is True


def test_resolve_object_kinds_drops_the_narrowing_when_widened():
    """`entity` alone means key fields; asking for plain fields too would make
    the narrowing silently drop rows the caller did ask for."""
    resolved = resolve_object_kinds(["entity", "column"])

    assert resolved.kinds == ["field"]
    assert resolved.primary_key_only is False


def test_resolve_object_kinds_rejects_metrics():
    """An empty result would read as "no such metric" instead of "wrong tool"."""
    with pytest.raises(MetricKindNotHere, match="search_metrics"):
        resolve_object_kinds(["metric"])


# ---------------------------------------------------------------------------
# Sub-agent table scope, against real storage
# ---------------------------------------------------------------------------


def _dataset_row(source_table: str, *, database_name: str = "") -> dict:
    """A minimal ``kind=dataset`` row bound to one physical table."""
    return {
        "id": f"dataset:sales:{source_table}",
        "kind": "dataset",
        "semantic_model_name": "sales",
        "dataset_name": source_table,
        "name": source_table,
        "source_table": source_table,
        "source_query": "",
        "catalog_name": "",
        "database_name": database_name,
        "schema_name": "",
        "description": f"{source_table} dataset",
        "ai_context_json": "",
        "search_text": f"{source_table} dataset",
        "expr": "",
        "field_type": "",
        "is_dimension": False,
        "is_time": False,
        "is_primary_key": False,
        "time_granularity": "",
        "from_dataset": "",
        "to_dataset": "",
        "from_columns_json": "",
        "to_columns_json": "",
        "rel_type": "",
        "join_type": "",
        "yaml_path": "",
        "updated_at": pa.scalar(0, type=pa.timestamp("ms")),
    }


class TestSubAgentTableScope:
    """A sub-agent scoped to tables must still see the datasets bound to them.

    These run against real storage on purpose: the scope filter names a column,
    and a filter naming a column the table does not have is exactly the failure
    a mocked ``_sub_agent_conditions`` cannot reproduce.
    """

    @staticmethod
    def _scoped_rag(real_agent_config, tables: str):
        real_agent_config.agentic_nodes["scoped_reader"] = {"scoped_context": {"tables": tables}}
        return SemanticDatasetRAG(real_agent_config, sub_agent_name="scoped_reader")

    def test_scoped_sub_agent_counts_datasets_of_its_tables(self, real_agent_config):
        SemanticDatasetRAG(real_agent_config).upsert_batch([_dataset_row("orders")])

        assert self._scoped_rag(real_agent_config, "orders").get_size() == 1

    def test_scoped_sub_agent_reads_datasets_of_its_tables(self, real_agent_config):
        SemanticDatasetRAG(real_agent_config).upsert_batch([_dataset_row("orders")])

        rows = self._scoped_rag(real_agent_config, "orders").list_datasets(table_name="orders")

        assert [row["source_table"] for row in rows] == ["orders"]

    def test_scoped_sub_agent_resolves_a_qualified_token(self, real_agent_config):
        """The namespace parts of a scope token keep their own column names --
        only the table leaf lives under a different one here.

        The qualifier is a database because the fixture's datasource is SQLite,
        whose identifier parser fills ``database_name`` for a two-part name.
        """
        SemanticDatasetRAG(real_agent_config).upsert_batch([_dataset_row("orders", database_name="shop")])

        assert self._scoped_rag(real_agent_config, "shop.orders").get_size() == 1

    def test_scoped_sub_agent_excludes_datasets_outside_its_tables(self, real_agent_config):
        rag = SemanticDatasetRAG(real_agent_config)
        rag.upsert_batch([_dataset_row("orders"), _dataset_row("customers")])

        assert self._scoped_rag(real_agent_config, "orders").get_size() == 1


def test_get_size_reports_a_failed_count_instead_of_swallowing_it(caplog):
    """0 means "nothing projected" to every caller, so a failed count that
    returns 0 silently is indistinguishable from an empty projection -- which
    is how a filter naming a missing column stayed invisible."""
    rag = SemanticDatasetRAG.__new__(SemanticDatasetRAG)
    rag.storage = Mock()
    rag.datasource_id = "ds_main"
    rag._sub_agent_conditions = lambda: []
    rag.storage._count_rows.side_effect = RuntimeError("Schema error: No field named table_name")

    with caplog.at_level(logging.WARNING):
        size = rag.get_size()

    assert size == 0
    assert "ds_main" in caplog.text
    assert "No field named table_name" in caplog.text
