"""Stable datasource references survive renames without rewriting saved data."""

import json
from types import SimpleNamespace

import pytest

from datus.configuration.agent_config import DbConfig
from datus.configuration.datasource_reference import resolve_datasource, resolve_execution_context
from datus.schemas.database_context import DatabaseExecutionContext
from datus.storage.datasource_scope import add_datasource_scope_to_rows, resolve_datasource_id
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.func_tool.database import DBFuncTool
from datus.utils.exceptions import DatusException


def config(sources=None):
    return SimpleNamespace(
        services=SimpleNamespace(datasources=sources or {"renamed": DbConfig(datasource_id="ds-stable")}),
        current_datasource="renamed",
    )


def test_id_survives_rename_and_conflicting_legacy_name():
    cfg = config({"renamed": DbConfig(datasource_id="ds-stable"), "old": DbConfig(datasource_id="different")})
    saved = {"datasource_id": "ds-stable", "datasource": "old", "database": "business", "catalog": "cat"}
    before = json.dumps(saved)
    restored = resolve_execution_context(cfg, saved)
    assert restored == {**saved, "datasource": "renamed"}
    assert json.dumps(saved) == before


@pytest.mark.parametrize("source_id", ["missing", "", "old"])
def test_explicit_unknown_id_never_uses_name_or_default(source_id):
    with pytest.raises(DatusException, match="not bound"):
        resolve_datasource(config(), datasource_id=source_id, datasource="renamed")


def test_legacy_name_still_resolves_without_data_migration():
    saved = {"datasource": "renamed", "database": "business"}
    assert resolve_execution_context(config(), saved)["datasource_id"] == "ds-stable"
    assert "datasource_id" not in saved


def test_unknown_legacy_name_is_not_rebound_to_only_source():
    with pytest.raises(DatusException, match="not bound"):
        resolve_datasource(config(), datasource="deleted")


def test_context_identity_does_not_change_when_display_name_changes():
    old = DatabaseExecutionContext(datasource="old", datasource_id="stable", database="business")
    new = DatabaseExecutionContext(datasource="new", datasource_id="stable", database="business")
    assert old.identity == new.identity
    assert DatabaseExecutionContext.from_mapping(old.to_dict()) == old


def test_kb_writes_use_stable_identity_with_legacy_read_scope():
    scope = resolve_datasource_id(config())
    assert scope == "ds-stable"
    assert scope.legacy_name == "renamed"
    rows = add_datasource_scope_to_rows([{"id": "metric"}], scope)
    assert rows[0]["datasource_id"] == "ds-stable"
    assert rows[0]["storage_key"] == "ds-stable:metric"


@pytest.mark.parametrize("saved_name", ["renamed", "old"])
def test_real_sqlite_artifact_connector_is_restored_by_id(tmp_path, saved_name):
    import sqlite3

    file = tmp_path / "business.db"
    with sqlite3.connect(file) as conn:
        conn.execute("CREATE TABLE marker (value TEXT)")
        conn.execute("INSERT INTO marker VALUES ('correct-source')")
    cfg = config({"renamed": DbConfig(datasource_id="stable", type="sqlite", uri=f"sqlite:///{file}")})
    tool = DBFuncTool.__new__(DBFuncTool)
    tool.agent_config = cfg
    tool._db_manager = DBManager(cfg.services.datasources)
    tool._tables_by_datasource = None
    try:
        connector = tool.connector_for_execution_context(
            {"datasource_id": "stable", "datasource": saved_name, "database": "business"}
        )
        assert connector.execute_query("SELECT value FROM marker", result_format="list").success
        assert tool._connector_datasources[id(connector)] == "renamed"
    finally:
        tool._db_manager.close()


def test_adapter_receives_connection_parameters_without_identity_metadata():
    manager = DBManager({"source": DbConfig(datasource_id="stable", type="postgresql", username="user", database="db")})
    parameters = manager._db_config_to_connection_config(manager._db_configs["source"])
    assert parameters["username"] == "user"
    assert parameters["database"] == "db"
    assert "datasource_id" not in parameters


def test_legacy_aliases_survive_future_renames_and_ambiguity_is_rejected():
    cfg = config({"renamed": DbConfig(datasource_id="stable", datasource_aliases=["__SR"])})
    assert resolve_datasource(cfg, datasource="__SR") == "renamed"
    cfg.services.datasources["__SR"] = DbConfig(datasource_id="other")
    with pytest.raises(DatusException, match="not bound"):
        resolve_datasource(cfg, datasource="__SR")
    assert resolve_datasource(cfg, datasource_id="stable", datasource="__SR") == "renamed"


def test_model_id_directory_and_legacy_reads_require_no_file_migration(tmp_path):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.engine import resolve_model_files
    from datus.utils.path_manager import DatusPathManager

    manager = DatusPathManager(tmp_path / "home", project_name="project", project_root=tmp_path)
    manager.datasource_ids = {"renamed": "stable"}
    manager.datasource_aliases = {"renamed": ["old"]}
    old = manager.semantic_models_dir / "old"
    old.mkdir(parents=True)
    legacy = old / "legacy.yml"
    legacy.write_text("legacy model remains untouched")
    primary = manager.semantic_model_path("renamed")
    assert primary.name == "stable"
    current = primary / "new.yml"
    current.write_text("new model")
    paths = resolve_model_files(
        DosiConfig(
            semantic_models_path=str(primary),
            legacy_semantic_models_paths=manager.legacy_semantic_model_paths("renamed"),
        )
    )
    assert set(paths) == {str(current), str(legacy)}
    assert legacy.read_text() == "legacy model remains untouched"
