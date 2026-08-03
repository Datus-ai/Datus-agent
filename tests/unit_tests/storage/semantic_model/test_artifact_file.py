# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.storage.semantic_model.artifact_file import semantic_artifact_lock


def test_semantic_artifact_lock_does_not_create_files(tmp_path):
    target = tmp_path / "semantic_models" / "orders.yml"

    with semantic_artifact_lock(target):
        assert not target.parent.exists()

    assert not target.parent.exists()
