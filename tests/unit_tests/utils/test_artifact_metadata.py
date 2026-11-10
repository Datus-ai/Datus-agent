# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""One datasource aggregation contract for finalize and publication snapshots."""

import json
from unittest.mock import Mock

import pytest

from datus.utils.artifact_metadata import collect_query_datasources


@pytest.mark.ci
@pytest.mark.parametrize(
    ("metadata_suffix", "query_suffix"),
    [(".json", ".sql"), (".params.json", ".sql.j2"), (".json", ".metric.json"), (".params.json", ".metric.json")],
)
def test_collects_current_sources_from_each_saved_query_kind(metadata_suffix, query_suffix):
    """Derive distinct sources only from paired queries and reflect source edits or deletion."""
    files = {}
    for name, source in [("a", " sr "), ("b", "mysql"), ("c", "sr")]:
        files[f"queries/{name}{query_suffix}"] = "query content"
        files[f"queries/{name}{metadata_suffix}"] = json.dumps({"datasource": source})
    assert collect_query_datasources(files, files.__getitem__) == ["sr", "mysql"]
    files.pop(f"queries/b{query_suffix}")
    assert collect_query_datasources(files, files.__getitem__) == ["sr"]
    files["queries/a" + metadata_suffix] = '{"datasource":"pg"}'
    assert collect_query_datasources(files, files.__getitem__) == ["pg", "sr"]


@pytest.mark.ci
@pytest.mark.parametrize("payload", ["{broken", "[]", "null", '{"datasource": 123}', "{}", '{"datasource":"  "}'])
def test_invalid_metadata_never_invents_a_source(payload):
    """Ignore malformed or source-free metadata without inventing datasource bindings."""
    files = {"queries/a.sql": "SELECT 1", "queries/a.json": payload}
    assert collect_query_datasources(files, files.__getitem__) == []


@pytest.mark.ci
def test_only_paired_query_metadata_is_read():
    """Read only direct query metadata with a matching executable query file."""
    files = {
        "queries/a.params.json": '{"datasource":"sr"}',
        "queries/a.sql.j2": "SELECT 1",
        "queries/orphan.json": '{"datasource":"retired"}',
        "queries/a.brief.json": '{"datasource":"brief"}',
        "queries/sub/a.json": '{"datasource":"nested"}',
        "queries/sub/a.sql": "SELECT 1",
        "render/data.json": '{"datasource":"render"}',
    }
    reader = Mock(side_effect=files.__getitem__)
    assert collect_query_datasources(files, reader) == ["sr"]
    reader.assert_called_once_with("queries/a.params.json")


@pytest.mark.ci
def test_disk_read_failure_is_not_reported_as_a_complete_empty_index():
    """Propagate metadata read failures so a partial scan cannot replace a complete index."""
    reader = Mock(side_effect=OSError("unreadable query metadata"))
    with pytest.raises(OSError, match="unreadable"):
        collect_query_datasources(["queries/a.sql", "queries/a.json"], reader)
