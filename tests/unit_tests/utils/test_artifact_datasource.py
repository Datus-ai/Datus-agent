# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Stable identities do not alter runtime keys or reinterpret missing IDs as names."""

import json
from types import SimpleNamespace

import pytest

from datus.configuration.agent_config import DbConfig
from datus.schemas.gen_visual_report_models import QueryResultFile
from datus.utils.artifact_datasource import artifact_identity, resolve_artifact_datasource, runtime_artifact_query
from datus.utils.artifact_metadata import collect_query_datasources


def config(**entries):
    return SimpleNamespace(services=SimpleNamespace(datasources=entries))


@pytest.mark.parametrize("reference", [{"datasource_id": "stable"}, {"datasource": "旧名字"}, {"datasource": "___"}])
def test_rename_resolves_only_the_original_identity(reference):
    cfg = config(new=DbConfig(artifact_id="stable", artifact_aliases=["旧名字", "___"]))
    assert resolve_artifact_datasource(cfg, reference) == "new"
    assert artifact_identity(cfg, "new") == {"datasource_id": "stable"}
    assert list(cfg.services.datasources) == ["new"]


@pytest.mark.parametrize("identity", ["missing", "", None, " "])
def test_explicit_invalid_id_never_falls_back_to_a_valid_name(identity):
    cfg = config(valid=DbConfig(artifact_id="stable"))
    with pytest.raises(ValueError, match="DATASOURCE_UNAVAILABLE"):
        resolve_artifact_datasource(cfg, {"datasource_id": identity, "datasource": "valid"})


def test_name_reuse_and_sanitization_collisions_are_ambiguous():
    cfg = config(old=DbConfig(artifact_id="other"), renamed=DbConfig(artifact_id="original", artifact_aliases=["old"]))
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_artifact_datasource(cfg, {"datasource": "old"})
    assert resolve_artifact_datasource(cfg, {"datasource_id": "original"}) == "renamed"


def test_cli_uses_configuration_key_as_identity():
    cfg = config(local=DbConfig())
    assert artifact_identity(cfg, "local") == {"datasource_id": "local"}
    assert resolve_artifact_datasource(cfg, {"datasource_id": "local"}) == "local"


def test_runtime_copy_preserves_saved_bytes_and_database_coordinates():
    query = QueryResultFile(
        datasource_id="stable",
        executed_at="now",
        row_count=1,
        execution_context={"datasource_id": "stable", "database": "historical", "schema": ""},
        columns=[{"name": "n", "type": "number"}],
        rows=[{"n": 1}],
    )
    before = query.model_dump_json()
    runtime = runtime_artifact_query(config(new=DbConfig(artifact_id="stable", database="different")), query)
    assert runtime.datasource == "new"
    assert runtime.execution_context == {"datasource": "new", "database": "historical", "schema": ""}
    assert query.model_dump_json() == before
    assert "datasource" not in json.loads(before)


@pytest.mark.parametrize("identity", ["", None, " "])
def test_schema_rejects_present_invalid_id_even_with_legacy_name(identity):
    with pytest.raises(ValueError):
        QueryResultFile(
            datasource_id=identity,
            datasource="valid",
            executed_at="now",
            row_count=0,
            columns=[{"name": "n", "type": "number"}],
            rows=[],
        )


def test_mixed_bundle_indexes_ids_and_only_legacy_names():
    files = {
        "queries/new.sql": "select 1",
        "queries/new.json": '{"datasource_id":"id","datasource":"stale"}',
        "queries/old.sql": "select 2",
        "queries/old.json": '{"datasource":"legacy"}',
    }
    assert collect_query_datasources(files, files.__getitem__, field="datasource_id") == ["id"]
    assert collect_query_datasources(files, files.__getitem__) == ["legacy"]
