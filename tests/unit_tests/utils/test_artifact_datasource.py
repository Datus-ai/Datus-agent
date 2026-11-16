# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Stable identities do not alter runtime keys or reinterpret missing IDs as names."""

import json
from types import SimpleNamespace

import pytest

from datus.configuration.agent_config import DbConfig
from datus.schemas.gen_visual_report_models import QueryResultFile
from datus.utils.artifact_datasource import artifact_identity, resolve_artifact_datasource, runtime_artifact_query


def config(**entries):
    return SimpleNamespace(services=SimpleNamespace(datasources=entries))


@pytest.mark.parametrize("reference", [{"datasource_id": "stable"}, {"datasource": "new"}])
def test_rename_resolves_only_the_original_identity(reference):
    cfg = config(new=DbConfig(datasource_id="stable"))
    assert resolve_artifact_datasource(cfg, reference) == "new"
    assert artifact_identity(cfg, "new") == {"datasource_id": "stable"}
    assert list(cfg.services.datasources) == ["new"]


@pytest.mark.parametrize("identity", ["missing", "", None, " "])
def test_explicit_invalid_id_never_falls_back_to_a_valid_name(identity):
    cfg = config(valid=DbConfig(datasource_id="stable"))
    with pytest.raises(ValueError, match="DATASOURCE_UNAVAILABLE"):
        resolve_artifact_datasource(cfg, {"datasource_id": identity, "datasource": "valid"})


def test_name_reuse_does_not_redirect_explicit_identity():
    cfg = config(old=DbConfig(datasource_id="other"), renamed=DbConfig(datasource_id="original"))
    assert resolve_artifact_datasource(cfg, {"datasource": "old"}) == "old"
    assert resolve_artifact_datasource(cfg, {"datasource_id": "original"}) == "renamed"
    with pytest.raises(ValueError, match="DATASOURCE_UNAVAILABLE"):
        resolve_artifact_datasource(cfg, {"datasource": "removed"})


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
    runtime = runtime_artifact_query(config(new=DbConfig(datasource_id="stable", database="different")), query)
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


def test_legacy_runtime_copy_keeps_resolved_identity_after_name_reuse():
    """Repeated conversion cannot redirect a previously resolved legacy query."""
    query = QueryResultFile(
        datasource="old", executed_at="now", row_count=0, columns=[{"name": "n", "type": "number"}], rows=[]
    )
    before = query.model_dump_json()
    runtime = runtime_artifact_query(config(old=DbConfig(datasource_id="original")), query)
    renamed = config(old=DbConfig(datasource_id="other"), new=DbConfig(datasource_id="original"))
    repeated = runtime_artifact_query(renamed, runtime)
    assert repeated.datasource == "new"
    assert repeated.datasource_id == "original"
    assert query.model_dump_json() == before
