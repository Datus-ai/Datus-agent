# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace

import pytest

from datus.storage.knowledge_provenance import (
    KnowledgeProvenanceStore,
    build_reference_sql_provenance_rows,
    enrich_reference_sql_results,
    is_knowledge_provenance_enabled,
)
from datus.storage.reference_sql.init_utils import gen_reference_sql_id


def _config(tmp_path, enabled=True):
    return SimpleNamespace(
        knowledge_base={"provenance": {"enabled": enabled}},
        path_manager=SimpleNamespace(project_data_dir=tmp_path),
    )


@pytest.mark.ci
def test_knowledge_provenance_disabled_by_default(tmp_path):
    config = SimpleNamespace(path_manager=SimpleNamespace(project_data_dir=tmp_path))

    assert is_knowledge_provenance_enabled(config) is False
    result = enrich_reference_sql_results(config, [{"id": "sql-1", "name": "q"}])
    assert result == [{"id": "sql-1", "name": "q"}]


@pytest.mark.ci
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
