from types import SimpleNamespace
from unittest.mock import Mock, patch

from datus.storage.table_semantic_profile.store import TableSemanticProfileRAG


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return self._rows


def _rag_with_rows(*row_sets):
    rag = TableSemanticProfileRAG.__new__(TableSemanticProfileRAG)
    rag.storage = Mock()
    rag._sub_agent_conditions = lambda: []
    rag.storage._search_all.side_effect = [_Rows(rows) for rows in row_sets]
    return rag


def test_get_profile_uses_unique_namespace_fallback_only():
    expected = {"table_name": "orders", "database_name": "shop"}
    rag = _rag_with_rows([], [expected])

    result = rag.get_profile(database_name="shop", table_name="orders")

    assert result == expected


def test_get_profile_rejects_ambiguous_namespace_fallback():
    rag = _rag_with_rows(
        [],
        [
            {"table_name": "orders", "database_name": "shop"},
            {"table_name": "orders", "database_name": "archive"},
        ],
    )

    result = rag.get_profile(database_name="shop", table_name="orders")

    assert result is None


def test_get_profile_rejects_unique_fallback_with_conflicting_namespace():
    rag = _rag_with_rows([], [{"table_name": "orders", "database_name": "archive"}])

    result = rag.get_profile(database_name="shop", table_name="orders")

    assert result is None


def test_get_profile_lowercase_fallback_runs_after_ambiguous_broad_lookup():
    expected = {"table_name": "orders", "database_name": "shop"}
    rag = _rag_with_rows(
        [],
        [
            {"table_name": "Orders", "database_name": "shop"},
            {"table_name": "Orders", "database_name": "archive"},
        ],
        [expected],
    )

    result = rag.get_profile(database_name="shop", table_name="Orders")

    assert result == expected


def _artifact_rag():
    rag = TableSemanticProfileRAG.__new__(TableSemanticProfileRAG)
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
    rag = TableSemanticProfileRAG.__new__(TableSemanticProfileRAG)
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
    deleted_rows = [{"catalog_name": "", "database_name": "db", "schema_name": "public", "table_name": "orders"}]
    rag.storage._search_all.return_value = _Rows(deleted_rows)

    with patch("datus.storage.kb_retrieval.MetadataFtsRAG") as metadata_cls:
        rag.delete_artifact_rows("semantic/orders.yml")

    metadata_cls.assert_called_once_with(rag.agent_config, datasource_id=rag.datasource_id)
    metadata_cls.return_value.refresh_tables.assert_called_once_with(deleted_rows)


def test_truncate_refreshes_metadata_documents_for_deleted_tables():
    rag = _artifact_rag()
    rag.agent_config = SimpleNamespace(kb_search=SimpleNamespace(mode="fts"), kb_search_mode="fts")
    deleted_rows = [{"catalog_name": "", "database_name": "db", "schema_name": "public", "table_name": "orders"}]
    rag.storage._search_all.return_value = _Rows(deleted_rows)

    with patch("datus.storage.kb_retrieval.MetadataFtsRAG") as metadata_cls:
        rag.truncate()

    rag.storage.delete_datasource_rows.assert_called_once_with("test_datasource")
    metadata_cls.return_value.refresh_tables.assert_called_once_with(deleted_rows)


def test_list_artifact_rows_handles_empty_and_non_empty_paths():
    rag = _artifact_rag()
    rag.storage._search_all.return_value = _Rows([{"id": "profile:orders"}])

    assert rag.list_artifact_rows("") == []
    assert rag.list_artifact_rows("semantic/orders.yml") == [{"id": "profile:orders"}]

    rag._sub_agent_conditions.assert_called_once_with()
    rag.storage._search_all.assert_called_once()


def test_restore_artifact_rows_handles_empty_and_non_empty_paths():
    rag = TableSemanticProfileRAG.__new__(TableSemanticProfileRAG)
    rag.delete_artifact_rows = Mock()
    rag.upsert_batch = Mock()
    rag.create_indices = Mock()
    rows = [{"id": "profile:orders"}]

    rag.restore_artifact_rows("", rows)
    rag.restore_artifact_rows("semantic/orders.yml", rows)

    rag.delete_artifact_rows.assert_called_once_with("semantic/orders.yml")
    rag.upsert_batch.assert_called_once_with(rows)
    rag.create_indices.assert_called_once_with()
