# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Contracts of the single-file lineage store: replace-by-source, determinism and safe writes."""

import json
import os
import threading

import pytest

from datus.storage.lineage.models import Evidence, LineageDocument, SourceContribution, SourceRecord
from datus.storage.lineage.store import LineageStore, remove_source, replace_source, serialize
from datus.utils.exceptions import DatusException


def contribution(source, edges, labels=None):
    record = SourceRecord(kind="file", sha256=source, analyzer_version=1)
    return SourceContribution(
        record=record,
        edges=[(up, down, Evidence(source=source, line=line, operation=op)) for up, down, line, op in edges],
        query_labels=labels or {},
    )


def test_replacing_a_source_withdraws_its_previous_edges_and_orphan_nodes():
    document = LineageDocument()
    replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert"), ("x", "y", 2, "insert")]))

    replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))

    assert set(document.nodes) == {"s", "t"}
    assert [(e.upstream, e.downstream) for e in document.edges] == [("s", "t")]


def test_node_kind_follows_the_remaining_evidence():
    document = LineageDocument()
    replace_source(document, "v.sql", contribution("v.sql", [("t", "v", 1, "create_view")]))
    replace_source(document, "w.sql", contribution("w.sql", [("t", "v", 1, "insert")]))
    assert document.nodes["v"].kind == "view"

    remove_source(document, "v.sql")

    assert document.nodes["v"].kind == "table"


def test_query_label_is_kept_while_the_query_survives():
    document = LineageDocument()
    replace_source(document, "a.sql", contribution("a.sql", [("t", "query:1", 1, None)], {"query:1": "GMV"}))
    replace_source(document, "b.sql", contribution("b.sql", [("t", "query:1", 3, None)]))
    assert document.nodes["query:1"].label == "GMV"

    remove_source(document, "b.sql")
    assert document.nodes["query:1"].label == "GMV"
    remove_source(document, "a.sql")
    assert "query:1" not in document.nodes


def test_the_same_graph_built_separately_serializes_identically(tmp_path):
    """Two checkouts that analyze the same sources must not conflict: nothing counts or timestamps writes."""
    first, second = LineageStore(tmp_path / "1.json"), LineageStore(tmp_path / "2.json")
    with first.edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))
    for _ in range(3):
        with second.edit() as document:
            replace_source(document, "a.sql", contribution("a.sql", [("s", "x", 1, "insert")]))
        with second.edit() as document:
            replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))

    assert first.path.read_text() == second.path.read_text()


def test_discarding_rebuilds_a_file_that_no_longer_parses(tmp_path):
    store = LineageStore(tmp_path / "lineage.json")
    store.path.write_text("<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> branch\n")
    with pytest.raises(DatusException):
        store.load()

    with store.edit(discard=True) as document:
        assert document.sources == {}
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))

    assert list(store.load().sources) == ["a.sql"]


def test_writes_without_upstream_keep_their_target_until_the_source_goes():
    document = LineageDocument()
    replace_source(document, "seed.sql", contribution("seed.sql", [(None, "flag", 1, "insert")]))
    replace_source(document, "a.sql", contribution("a.sql", [("flag", "t", 1, "insert")]))
    assert set(document.nodes) == {"flag", "t"}
    assert json.loads(serialize(document))["edges"][0] == {
        "to": "flag",
        "evidence": [{"source": "seed.sql", "line": 1, "operation": "insert"}],
    }

    remove_source(document, "a.sql")
    assert set(document.nodes) == {"flag"}
    remove_source(document, "seed.sql")
    assert document.nodes == {}


def test_serialization_does_not_depend_on_insertion_order(tmp_path):
    edges = [("s1", "t", 1, "insert"), ("s2", "t", 1, "insert"), ("t", "u", 2, "overwrite")]
    first, second = LineageStore(tmp_path / "1.json"), LineageStore(tmp_path / "2.json")
    with first.edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", edges))
        replace_source(document, "b.sql", contribution("b.sql", edges[:1]))
    with second.edit() as document:
        replace_source(document, "b.sql", contribution("b.sql", edges[:1]))
        replace_source(document, "a.sql", contribution("a.sql", list(reversed(edges))))

    assert first.path.read_text() == second.path.read_text()
    saved = json.loads(first.path.read_text())
    assert [(e["from"], e["to"]) for e in saved["edges"]] == [("s1", "t"), ("s2", "t"), ("t", "u")]
    assert [ev["source"] for ev in saved["edges"][0]["evidence"]] == ["a.sql", "b.sql"]


def test_edit_writes_only_changes(tmp_path):
    store = LineageStore(tmp_path / "lineage" / "lineage.json")
    with store.edit() as document:
        pass
    assert not store.path.exists()

    with store.edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))
    written = store.path.stat().st_mtime_ns
    with store.edit() as document:
        pass

    assert store.path.stat().st_mtime_ns == written
    assert store.path.read_text() == serialize(store.load())
    # Nothing but the graph is left in its directory: no lock or temporary file to commit.
    assert [p.name for p in store.path.parent.iterdir()] == ["lineage.json"]


def test_a_failed_edit_leaves_the_file_untouched(tmp_path):
    store = LineageStore(tmp_path / "lineage.json")
    with store.edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))
    before = store.path.read_text()

    with pytest.raises(RuntimeError):
        with store.edit() as document:
            remove_source(document, "a.sql")
            raise RuntimeError("analysis failed")

    assert store.path.read_text() == before


def test_concurrent_edits_from_separate_stores_lose_no_update(tmp_path):
    path = tmp_path / "lineage.json"

    def upsert(i):
        # A store per thread, as separate tool instances or processes would have.
        with LineageStore(path).edit() as document:
            replace_source(document, f"{i}.sql", contribution(f"{i}.sql", [("s", f"t{i}", 1, "insert")]))

    threads = [threading.Thread(target=upsert, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    document = LineageStore(path).load()
    assert len(document.sources) == 16


def test_load_reflects_writes_by_another_store(tmp_path):
    path = tmp_path / "lineage.json"
    reader = LineageStore(path)
    assert reader.load().sources == {}

    with LineageStore(path).edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t", 1, "insert")]))

    assert list(reader.load().sources) == ["a.sql"]


def test_load_notices_a_rewrite_with_the_same_mtime_and_size(tmp_path):
    # Linux mtimes advance in coarse ticks, so two writes can share both the mtime and the size.
    path = tmp_path / "lineage.json"
    with LineageStore(path).edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t1", 1, "insert")]))
    reader = LineageStore(path)
    reader.load()
    before = path.stat()

    with LineageStore(path).edit() as document:
        replace_source(document, "a.sql", contribution("a.sql", [("s", "t2", 1, "insert")]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert path.stat().st_size == before.st_size
    assert set(reader.load().nodes) == {"s", "t2"}
