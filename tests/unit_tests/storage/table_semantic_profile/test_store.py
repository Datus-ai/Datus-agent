from unittest.mock import Mock

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
