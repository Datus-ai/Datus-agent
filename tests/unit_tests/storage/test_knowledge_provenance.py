# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace

from datus.storage.knowledge_provenance import (
    METRIC_ARTIFACT_TYPE,
    REFERENCE_SQL_ARTIFACT_TYPE,
    KnowledgeProvenanceStore,
    build_metric_provenance_rows,
    build_reference_sql_provenance_rows,
    enrich_metric_results,
    enrich_reference_sql_results,
    is_knowledge_provenance_enabled,
    reference_sql_artifact_ids_for_items,
)
from datus.storage.reference_sql.init_utils import gen_reference_sql_id


def _config(tmp_path, enabled=True):
    return SimpleNamespace(
        knowledge_base={"provenance": {"enabled": enabled}},
        path_manager=SimpleNamespace(project_data_dir=tmp_path),
    )


def test_knowledge_provenance_disabled_by_default(tmp_path):
    config = SimpleNamespace(path_manager=SimpleNamespace(project_data_dir=tmp_path))

    assert is_knowledge_provenance_enabled(config) is False
    result = enrich_reference_sql_results(config, [{"id": "sql-1", "name": "q"}])
    assert result == [{"id": "sql-1", "name": "q"}]


def test_knowledge_provenance_enabled_coerces_config_values(tmp_path):
    assert is_knowledge_provenance_enabled(_config(tmp_path, enabled="true")) is True
    assert is_knowledge_provenance_enabled(_config(tmp_path, enabled="false")) is False
    assert is_knowledge_provenance_enabled(_config(tmp_path, enabled=1)) is True

    assert (
        is_knowledge_provenance_enabled(
            SimpleNamespace(knowledge_base="bad", path_manager=SimpleNamespace(project_data_dir=tmp_path))
        )
        is False
    )
    assert (
        is_knowledge_provenance_enabled(
            SimpleNamespace(
                knowledge_base={"provenance": "bad"}, path_manager=SimpleNamespace(project_data_dir=tmp_path)
            )
        )
        is False
    )


def test_reference_sql_provenance_sidecar_enriches_results(tmp_path):
    config = _config(tmp_path, enabled=True)
    artifact_id = gen_reference_sql_id("SELECT 1")
    rows = build_reference_sql_provenance_rows(
        [
            {
                "sql": "SELECT 1",
                "source_id": "seed_context:0",
                "source_context_ids": ["refsql:task:0", "refsql:task:1"],
                "source_type": "seed_context",
                "source_metadata": {"task_id": "0"},
            }
        ]
    )

    written = KnowledgeProvenanceStore(config).upsert_many(rows)
    enriched = enrich_reference_sql_results(config, [{"id": artifact_id, "name": "q"}])

    assert written == 2
    assert enriched[0]["source_ids"] == ["seed_context:0"]
    assert enriched[0]["source_context_ids"] == ["refsql:task:0", "refsql:task:1"]
    assert enriched[0]["source_metadata"] == [{"task_id": "0"}]


def test_reference_sql_artifact_ids_are_stable_and_deduped():
    sql = "SELECT * FROM orders"

    assert reference_sql_artifact_ids_for_items([{"sql": sql}, {"sql": sql}, {"id": "explicit"}]) == [
        gen_reference_sql_id(sql),
        "explicit",
    ]


def test_build_reference_sql_provenance_rows_accepts_string_contexts_and_skips_empty_rows():
    rows = build_reference_sql_provenance_rows(
        [
            {
                "sql": "SELECT 1",
                "source_context_id": "refsql:task:0, refsql:task:1",
                "source_metadata": {"source_id": "seed_context:0", "source_type": "seed_context"},
            },
            {"sql": "SELECT 2"},
        ]
    )

    assert [row["source_context_id"] for row in rows] == ["refsql:task:0", "refsql:task:1"]
    assert {row["source_id"] for row in rows} == {"seed_context:0"}
    assert {row["artifact_type"] for row in rows} == {REFERENCE_SQL_ARTIFACT_TYPE}


def test_provenance_store_ignores_corrupt_or_non_list_sidecar(tmp_path):
    config = _config(tmp_path, enabled=True)
    store = KnowledgeProvenanceStore(config)

    store.file_path.write_text("{not-json", encoding="utf-8")
    assert store.find_by_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, ["missing"]) == {}

    store.file_path.write_text("{}", encoding="utf-8")
    assert store.find_by_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, ["missing"]) == {}


def test_replace_for_artifact_ids_removes_stale_rows(tmp_path):
    config = _config(tmp_path, enabled=True)
    artifact_id = gen_reference_sql_id("SELECT 1")
    stale_rows = build_reference_sql_provenance_rows(
        [
            {
                "sql": "SELECT 1",
                "source_id": "seed_context:0",
                "source_context_ids": ["refsql:task:old", "refsql:task:stale"],
            }
        ]
    )
    replacement_rows = build_reference_sql_provenance_rows(
        [
            {
                "sql": "SELECT 1",
                "source_id": "seed_context:0",
                "source_context_ids": ["refsql:task:new"],
            }
        ]
    )

    store = KnowledgeProvenanceStore(config)
    assert store.upsert_many(stale_rows) == 2
    assert store.replace_for_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, [artifact_id], replacement_rows) == 1

    found = store.find_by_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, [artifact_id])
    assert found[artifact_id]["source_context_ids"] == ["refsql:task:new"]


def test_replace_for_artifact_ids_deletes_when_current_rows_are_empty(tmp_path):
    config = _config(tmp_path, enabled=True)
    artifact_id = gen_reference_sql_id("SELECT 1")
    store = KnowledgeProvenanceStore(config)
    store.upsert_many(
        build_reference_sql_provenance_rows(
            [{"sql": "SELECT 1", "source_id": "seed_context:0", "source_context_id": "refsql:task:old"}]
        )
    )

    assert store.replace_for_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, [artifact_id], []) == 0
    assert store.find_by_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, [artifact_id]) == {}


def test_metric_provenance_sidecar_enriches_results(tmp_path):
    config = _config(tmp_path, enabled=True)
    rows = build_metric_provenance_rows(
        [
            {
                "id": "metric:Sales.activity_count",
                "source_id": "seed_context.csv:7",
                "source_context_ids": ["metric:seed:7", "metric:task:21"],
                "source_type": "success_story",
                "source_metadata": {"row_index": 7},
            }
        ]
    )

    written = KnowledgeProvenanceStore(config).upsert_many(rows)
    enriched = enrich_metric_results(config, [{"id": "metric:Sales.activity_count", "name": "activity_count"}])

    assert written == 2
    assert enriched[0]["source_ids"] == ["seed_context.csv:7"]
    assert enriched[0]["source_context_ids"] == ["metric:seed:7", "metric:task:21"]
    assert enriched[0]["source_metadata"] == [{"row_index": 7}]


def test_delete_for_artifact_type_only_removes_matching_type(tmp_path):
    config = _config(tmp_path, enabled=True)
    store = KnowledgeProvenanceStore(config)
    store.upsert_many(
        build_reference_sql_provenance_rows([{"sql": "SELECT 1", "source_context_id": "refsql:task:0"}])
        + build_metric_provenance_rows([{"id": "metric:Sales.activity_count", "source_context_id": "metric:seed:0"}])
    )

    assert store.delete_for_artifact_type(METRIC_ARTIFACT_TYPE) == 1
    assert store.find_by_artifact_ids(METRIC_ARTIFACT_TYPE, ["metric:Sales.activity_count"]) == {}
    assert store.find_by_artifact_ids(REFERENCE_SQL_ARTIFACT_TYPE, [gen_reference_sql_id("SELECT 1")])
