# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import multiprocessing

from datus.storage.semantic_model.artifact_file import semantic_artifact_lock


def test_semantic_artifact_lock_does_not_create_files(tmp_path):
    target = tmp_path / "semantic_models" / "orders.yml"

    with semantic_artifact_lock(target):
        assert not target.parent.exists()

    assert not target.parent.exists()


def _ontology_writer(root, started, finished):
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    started.set()
    result = FilesystemFuncTool(root_path=root).write_file(
        "subject/semantic_models/ontology/flights/ontology.yml", "from another worker"
    )
    if result.success:
        finished.set()


def test_ontology_review_and_native_writer_share_a_reentrant_process_lock(tmp_path):
    import os

    import pytest

    if os.name == "nt":
        pytest.skip("POSIX API worker lock")
    target = tmp_path / "subject/semantic_models/ontology/flights/ontology.yml"
    target.parent.mkdir(parents=True)
    target.write_text("baseline")
    ctx = multiprocessing.get_context("spawn")
    started, finished = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_ontology_writer, args=(str(tmp_path), started, finished))
    try:
        with semantic_artifact_lock(target):
            with semantic_artifact_lock(target):
                process.start()
                assert started.wait(4)
                assert not finished.wait(0.1)
                assert target.read_text() == "baseline"
        assert finished.wait(4)
        process.join(4)
        assert process.exitcode == 0
        assert target.read_text() == "from another worker"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
