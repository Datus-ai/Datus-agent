"""Stable datasource references survive renames without rewriting saved data."""

import json
from types import SimpleNamespace

import pytest

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.configuration.datasource_reference import resolve_datasource, resolve_execution_context
from datus.schemas.database_context import DatabaseExecutionContext
from datus.storage.datasource_scope import add_datasource_scope_to_rows, resolve_datasource_id
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


def test_legacy_aliases_survive_future_renames_and_ambiguity_is_rejected():
    cfg = config({"renamed": DbConfig(datasource_id="stable", datasource_aliases=["__SR"])})
    assert resolve_datasource(cfg, datasource="__SR") == "renamed"
    cfg.services.datasources["__SR"] = DbConfig(datasource_id="other")
    with pytest.raises(DatusException, match="not bound"):
        resolve_datasource(cfg, datasource="__SR")
    assert resolve_datasource(cfg, datasource_id="stable", datasource="__SR") == "renamed"


def test_kb_legacy_scope_excludes_reused_names_and_other_canonical_ids():
    cfg = config(
        {
            "renamed": DbConfig(datasource_id="stable", datasource_aliases=["old", "other-id"]),
            "old": DbConfig(datasource_id="other-id"),
        }
    )
    scope = resolve_datasource_id(cfg, "stable")
    assert scope == "stable"
    assert scope.legacy_names == ("renamed",)


def test_explicit_kb_id_wins_over_another_sources_identical_display_key():
    cfg = config({"stable": DbConfig(datasource_id="other"), "renamed": DbConfig(datasource_id="stable")})
    assert resolve_datasource_id(cfg, "stable") == "stable"
    assert resolve_datasource_id(cfg, "stable").legacy_names == ("renamed",)


def test_runtime_setter_does_not_overwrite_an_explicit_camel_case_id(tmp_path):
    cfg = AgentConfig(nodes={}, home=str(tmp_path), skip_init_dirs=True)
    cfg.services.datasources["renamed"] = DbConfig(datasource_id="stable")
    with pytest.raises(DatusException, match="not bound"):
        cfg.set_runtime_db_context({"datasourceId": "unbound", "datasource": "renamed"})
    cfg.set_runtime_db_context({"datasourceId": "stable", "datasource": "old"})
    assert cfg.runtime_db_context()["datasource_id"] == "stable"
    assert cfg.runtime_db_context()["datasource"] == "renamed"


def test_unknown_display_label_cannot_acquire_the_primary_identity():
    from datus.configuration.datasource_reference import configured_datasource_id

    cfg = config()
    cfg.current_db_config = lambda _: cfg.services.datasources["renamed"]
    assert configured_datasource_id(cfg, "old") is None
    assert configured_datasource_id(cfg, "renamed") == "ds-stable"


def test_frozen_subagent_id_scopes_resolve_only_on_a_runtime_copy():
    from datus.configuration.datasource_reference import resolve_subagent_config

    cfg = config()
    node = {
        "scoped_context": {
            "datasource_id": "ds-stable",
            "datasource": "old",
            "tables_by_datasource_id": {"ds-stable": ["demo.orders"]},
        }
    }
    saved = json.dumps(node)
    resolved = resolve_subagent_config(cfg, node)
    assert resolved["scoped_context"] == {"datasource": "renamed", "tables_by_datasource": {"renamed": ["demo.orders"]}}
    assert json.dumps(node) == saved


@pytest.mark.parametrize("aliases", [None, "old", 1, {"old": True}, ["old", "../bad", None, "中文", 12]])
def test_file_and_server_aliases_use_the_same_safe_list_contract(tmp_path, aliases):
    from datus.configuration.agent_config import _parse_single_file_db

    expected = ["old"] if isinstance(aliases, list) else []
    raw = {"type": "sqlite", "uri": str(tmp_path / "db.sqlite"), "datasource_aliases": aliases}
    assert _parse_single_file_db(raw, "sqlite").datasource_aliases == expected
    assert DbConfig.filter_kwargs(DbConfig, raw).datasource_aliases == expected
    cfg = AgentConfig(nodes={}, home=str(tmp_path), skip_init_dirs=True)
    cfg._init_services_config({"files": {**raw, "uri": str(tmp_path / "*.sqlite")}})
    assert cfg.services.datasources["files"].datasource_aliases == expected


@pytest.mark.parametrize(
    "sources",
    [
        {"a": DbConfig(datasource_id="shared"), "b": DbConfig(datasource_id="shared")},
        {"shared": DbConfig(), "b": DbConfig(datasource_id="shared")},
    ],
)
def test_canonical_identity_collisions_fail_before_building_path_maps(tmp_path, sources):
    cfg = AgentConfig(nodes={}, home=str(tmp_path), skip_init_dirs=True)
    cfg.services.datasources = sources
    with pytest.raises(DatusException, match="shared") as exc:
        cfg._sync_path_manager_datasources()
    assert all(name in str(exc.value) for name in sources)


def test_configured_id_resolves_unique_alias_but_never_an_ambiguous_or_missing_name():
    from datus.configuration.datasource_reference import configured_datasource_id

    cfg = config({"renamed": DbConfig(datasource_id="stable", datasource_aliases=["old"])})
    assert configured_datasource_id(cfg, "old") == "stable"
    assert configured_datasource_id(cfg, "missing") is None
    cfg.services.datasources["other"] = DbConfig(datasource_id="other-id", datasource_aliases=["old"])
    assert configured_datasource_id(cfg, "old") is None
