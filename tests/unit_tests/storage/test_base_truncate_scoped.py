# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from datus.storage.base import BaseEmbeddingStore
from datus.utils.exceptions import DatusException


def _store_with_table(table):
    store = object.__new__(BaseEmbeddingStore)
    store._shared = SimpleNamespace(table=table, initialized=True)
    store._table_lock = MagicMock()
    store.db = Mock()
    store.table_name = "metrics"
    return store


def test_truncate_scoped_physical_delegates_to_truncate():
    store = _store_with_table(Mock())
    store.truncate = Mock()

    with patch("datus.storage.backend_holder.get_isolation_type", return_value="physical"):
        store.truncate_scoped()

    store.truncate.assert_called_once_with()


def test_truncate_scoped_logical_deletes_visible_rows_without_drop():
    table = Mock()
    table.supports_logical_scoped_delete_all = True
    store = _store_with_table(table)
    store.truncate = Mock()

    with patch("datus.storage.backend_holder.get_isolation_type", return_value="logical"):
        store.truncate_scoped()

    table.delete.assert_called_once_with(None)
    store.db.drop_table.assert_not_called()
    store.truncate.assert_not_called()
    assert store._shared.table is None
    assert store._shared.initialized is False


def test_truncate_scoped_logical_requires_scoped_delete_support():
    table = Mock()
    table.supports_logical_scoped_delete_all = False
    store = _store_with_table(table)
    store.truncate = Mock()

    with patch("datus.storage.backend_holder.get_isolation_type", return_value="logical"):
        with pytest.raises(DatusException, match="supports_logical_scoped_delete_all"):
            store.truncate_scoped()

    table.delete.assert_not_called()
    store.db.drop_table.assert_not_called()
    store.truncate.assert_not_called()
