# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for the per-file projection digests reconcile compares against."""

import multiprocessing
import os
from types import SimpleNamespace

import pytest

from datus.storage.semantic_model.sync_state import forget_digests, load_digests, record_digests


def _config(tmp_path):
    return SimpleNamespace(path_manager=SimpleNamespace(project_data_dir=str(tmp_path)))


def _writer(data_dir, worker, rounds):
    config = _config(data_dir)
    for i in range(rounds):
        record_digests(config, "ds", {f"/models/w{worker}-{i}.yml": "digest"})


@pytest.mark.skipif(os.name != "posix", reason="fork start method and flock are POSIX-only")
def test_concurrent_processes_lose_no_update(tmp_path):
    """Each writer reads, changes and rewrites the whole file; without a shared
    lock one of them overwrites the others with an older snapshot."""
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_writer, args=(tmp_path, n, 25)) for n in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0

    assert len(load_digests(_config(tmp_path), "ds")) == 4 * 25


def test_forget_removes_only_the_named_files(tmp_path):
    config = _config(tmp_path)
    record_digests(config, "ds", {"/models/a.yml": "1", "/models/b.yml": "2"})

    forget_digests(config, "ds", ["/models/a.yml"])

    assert load_digests(config, "ds") == {"/models/b.yml": "2"}


def test_a_datasource_never_recorded_has_no_digests(tmp_path):
    assert load_digests(_config(tmp_path), "ds") is None
