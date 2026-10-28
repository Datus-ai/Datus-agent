# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import MagicMock

from datus.storage.artifact_replacement import restore_artifact_replacements


def test_restore_artifact_replacements_reports_restore_failures():
    ok_rag = MagicMock()
    failing_rag = MagicMock()
    failing_rag.restore_artifact_rows.side_effect = RuntimeError("storage unavailable")

    failures = restore_artifact_replacements(
        [
            (ok_rag, "semantic/orders.yml", [{"id": "old-semantic"}]),
            (failing_rag, "metrics/orders.yml", [{"id": "old-metric"}]),
        ]
    )

    assert len(failures) == 1
    assert "metrics/orders.yml" in failures[0]
    ok_rag.restore_artifact_rows.assert_called_once_with("semantic/orders.yml", [{"id": "old-semantic"}])
    failing_rag.restore_artifact_rows.assert_called_once_with("metrics/orders.yml", [{"id": "old-metric"}])
