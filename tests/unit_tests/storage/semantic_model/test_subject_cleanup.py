"""Cleanup work survives a projection that already replaced the old KB rows."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from datus.storage.semantic_model.subject_cleanup import finish_cleanup, remember_cleanup


def test_retry_retains_old_nodes_after_cleanup_failure(tmp_path):
    config = SimpleNamespace(path_manager=SimpleNamespace(project_data_dir=tmp_path))
    nodes = remember_cleanup(config, "warehouse", "model.yml", {11, 12})
    with patch("datus.storage.semantic_model.reconcile._DatasourceStores") as stores:
        stores.return_value.remove_emptied_nodes.side_effect = RuntimeError("database unavailable")
        with pytest.raises(RuntimeError, match="database unavailable"):
            finish_cleanup(config, "warehouse", "model.yml", nodes)
    retry = remember_cleanup(config, "warehouse", "model.yml", {15})
    assert retry == {11, 12, 15}
    assert remember_cleanup(config, "other", "model.yml", []) == set()
    with patch("datus.storage.semantic_model.reconcile._DatasourceStores") as stores:
        finish_cleanup(config, "warehouse", "model.yml", retry)
        stores.return_value.remove_emptied_nodes.assert_called_once_with({11, 12, 15})
    assert remember_cleanup(config, "warehouse", "model.yml", []) == set()
