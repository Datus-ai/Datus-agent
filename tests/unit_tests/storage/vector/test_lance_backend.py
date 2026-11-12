# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for LanceDB vector backend covering table, database, and backend operations."""

import os
import threading
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock

import lancedb
import pandas as pd
import pyarrow as pa
import pytest
from datus_storage_base.conditions import and_, eq, in_

from datus.storage.fts import FtsField, FtsIndexStatus, FtsSpec
from datus.storage.vector.lance_backend import LanceVectorBackend, LanceVectorDatabase, LanceVectorTable
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# LanceVectorTable tests
# ---------------------------------------------------------------------------


class TestLanceVectorTableWriteOps:
    """Tests for table-level write operations."""

    def test_add(self):
        """add() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        data = MagicMock()
        table.add(data)
        raw_table.add.assert_called_once_with(data)

    def test_delete_with_node_eq(self):
        """delete() compiles eq() Node where and delegates."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.delete(eq("id", 1))
        raw_table.delete.assert_called_once_with("id = 1")

    def test_delete_with_node(self):
        """delete() compiles Node where and delegates."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.delete(eq("status", "inactive"))
        raw_table.delete.assert_called_once_with("status = 'inactive'")

    def test_delete_with_none(self):
        """delete() with None where does not call table.delete()."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.delete(None)
        raw_table.delete.assert_not_called()

    def test_delete_with_in_node(self):
        """delete() keeps membership native instead of expanding an OR chain."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.delete(in_("chunk_id", ["a", "b", "c"]))
        raw_table.delete.assert_called_once_with("(chunk_id IN ('a', 'b', 'c'))")

    def test_update_with_node_eq(self):
        """update() compiles eq() Node where and delegates."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.update(eq("id", 1), {"name": "new"})
        raw_table.update.assert_called_once_with(where="id = 1", values={"name": "new"})

    def test_update_with_node(self):
        """update() compiles Node where and delegates."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.update(eq("id", 1), {"name": "new"})
        raw_table.update.assert_called_once_with(where="id = 1", values={"name": "new"})

    def test_merge_insert(self):
        """merge_insert() chains the correct builder methods."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        data = MagicMock()
        table.merge_insert(data, "id")
        raw_table.merge_insert.assert_called_once_with("id")


class TestLanceVectorTableSearchOps:
    """Tests for table-level search operations."""

    def test_count_rows_no_where(self):
        """count_rows() without where calls count_rows()."""
        raw_table = MagicMock()
        raw_table.count_rows.return_value = 42
        table = LanceVectorTable(raw_table)
        assert table.count_rows() == 42
        raw_table.count_rows.assert_called_once_with()

    def test_count_rows_with_eq_node_where(self):
        """count_rows() with eq() Node where compiles and passes the filter."""
        raw_table = MagicMock()
        raw_table.count_rows.return_value = 5
        table = LanceVectorTable(raw_table)
        assert table.count_rows(where=eq("status", "active")) == 5
        raw_table.count_rows.assert_called_once_with("status = 'active'")

    def test_count_rows_with_node_where(self):
        """count_rows() with Node where compiles and passes the filter."""
        raw_table = MagicMock()
        raw_table.count_rows.return_value = 3
        table = LanceVectorTable(raw_table)
        assert table.count_rows(where=eq("status", "active")) == 3
        raw_table.count_rows.assert_called_once_with("status = 'active'")

    def test_count_rows_with_compound_where(self):
        """count_rows() with compound Node where compiles correctly."""
        raw_table = MagicMock()
        raw_table.count_rows.return_value = 2
        table = LanceVectorTable(raw_table)
        where = and_(eq("status", "active"), eq("role", "admin"))
        assert table.count_rows(where=where) == 2
        raw_table.count_rows.assert_called_once_with("(status = 'active' AND role = 'admin')")

    def test_multimatch_searches_independently_indexed_fields(self, tmp_path):
        raw_table = lancedb.connect(tmp_path).create_table(
            "multi_fts",
            data=pa.table(
                {
                    "id": [1, 2],
                    "title": ["orders handbook", "customer handbook"],
                    "body": ["monthly revenue", "retention cohorts"],
                }
            ),
        )
        table = LanceVectorTable(raw_table)
        spec = FtsSpec((FtsField("title", boost=3.0), FtsField("body")))
        for field in spec.fields:
            table.create_fts_index(field)

        assert table.fts_index_status(spec) == FtsIndexStatus.READY
        assert table.search_fts("orders", spec, 5).to_pylist()[0]["id"] == 1
        assert table.search_fts("retention", spec, 5).to_pylist()[0]["id"] == 2

    def test_ngram_fts_matches_chinese_without_whitespace(self, tmp_path):
        raw_table = lancedb.connect(tmp_path).create_table(
            "ngram_fts",
            data=pa.table({"id": [1, 2], "search_text": ["订单购买分析", "客户流失分析"]}),
        )
        table = LanceVectorTable(raw_table)
        spec = FtsSpec((FtsField("search_text", tokenizer="ngram"),))
        table.create_fts_index(spec.fields[0])

        assert table.search_fts("购买", spec, 5).to_pylist()[0]["id"] == 1

    def test_fts_status_detects_tokenizer_mismatch(self, tmp_path):
        raw_table = lancedb.connect(tmp_path).create_table(
            "mismatched_fts",
            data=pa.table({"text": ["orders"]}),
        )
        table = LanceVectorTable(raw_table)
        table.create_fts_index(FtsField("text"))

        expected = FtsSpec((FtsField("text", tokenizer="ngram"),))
        assert table.fts_index_status(expected) == FtsIndexStatus.VERSION_MISMATCH

    def test_legacy_fts_directory_is_detected_and_removed_for_rebuild(self, tmp_path):
        raw_table = lancedb.connect(tmp_path).create_table(
            "legacy_fts",
            data=pa.table({"text": ["orders"]}),
        )
        table = LanceVectorTable(raw_table)
        spec = FtsSpec((FtsField("text"),))
        legacy_path = tmp_path / "legacy_fts.lance" / "_indices" / "fts"
        legacy_path.mkdir(parents=True)
        (legacy_path / "marker").write_text("legacy", encoding="utf-8")

        assert table.fts_index_status(spec) == FtsIndexStatus.LEGACY
        assert table.remove_legacy_fts_index() is True
        assert not legacy_path.exists()


class TestLanceVectorTableIndexOps:
    """Tests for table-level index operations."""

    def test_create_vector_index(self):
        """create_vector_index() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.create_vector_index("vec_col", metric="l2", replace=True)
        raw_table.create_index.assert_called_once_with(metric="l2", vector_column_name="vec_col", replace=True)

    def test_create_fts_index(self):
        """create_fts_index() uses the unified single-field API."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.create_fts_index(FtsField("body", tokenizer="ngram"))
        raw_table.create_index.assert_called_once()
        args, kwargs = raw_table.create_index.call_args
        assert args == ("body",)
        assert kwargs["replace"] is True
        assert kwargs["config"].base_tokenizer == "ngram"
        assert kwargs["config"].ngram_min_length == 2
        assert kwargs["config"].ngram_max_length == 2

    def test_create_scalar_index(self):
        """create_scalar_index() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.create_scalar_index("category")
        raw_table.create_scalar_index.assert_called_once_with("category", replace=True)

    def test_optimize(self):
        """optimize() incrementally updates the underlying Lance indices."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.optimize()
        raw_table.optimize.assert_called_once_with()

    def test_compact_files(self):
        """compact_files() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.compact_files()
        raw_table.compact_files.assert_called_once()

    def test_cleanup_old_versions(self):
        """cleanup_old_versions() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.cleanup_old_versions()
        raw_table.cleanup_old_versions.assert_called_once()


# ---------------------------------------------------------------------------
# LanceVectorDatabase tests
# ---------------------------------------------------------------------------


class TestLanceVectorDatabase:
    """Tests for database-level operations."""

    def test_table_names(self):
        """table_names() delegates to db connection."""
        raw_db = MagicMock()
        raw_db.table_names.return_value = ["t1", "t2"]
        db = LanceVectorDatabase(raw_db)
        assert db.table_names(limit=50) == ["t1", "t2"]
        raw_db.table_names.assert_called_once_with(limit=50)

    def test_table_exists_true(self):
        """table_exists() returns True when open_table succeeds."""
        raw_db = MagicMock()
        raw_db.open_table.return_value = MagicMock()
        db = LanceVectorDatabase(raw_db)
        assert db.table_exists("my_table") is True
        raw_db.open_table.assert_called_once_with("my_table")

    def test_table_exists_false(self):
        """table_exists() returns False when open_table raises ValueError."""
        raw_db = MagicMock()
        raw_db.open_table.side_effect = ValueError("Table not found")
        db = LanceVectorDatabase(raw_db)
        assert db.table_exists("missing") is False

    def test_table_exists_false_for_corrupt_lance_table_metadata(self):
        """table_exists() treats incomplete Lance table directories as missing."""
        raw_db = MagicMock()
        raw_db.open_table.side_effect = RuntimeError('FileDoesNotExist("meta.json")')
        db = LanceVectorDatabase(raw_db)
        assert db.table_exists("missing") is False

    def test_table_exists_reraises_non_missing_errors(self):
        raw_db = MagicMock()
        raw_db.open_table.side_effect = RuntimeError("connection lost")
        db = LanceVectorDatabase(raw_db)

        with pytest.raises(RuntimeError, match="connection lost"):
            db.table_exists("my_table")

    def test_table_exists_reraises_meta_json_errors_without_missing_table_signature(self):
        raw_db = MagicMock()
        raw_db.open_table.side_effect = RuntimeError("failed to parse meta.json")
        db = LanceVectorDatabase(raw_db)

        with pytest.raises(RuntimeError, match="failed to parse meta.json"):
            db.table_exists("my_table")

    def test_create_table_passes_kwargs(self):
        """create_table() passes schema to db connection without embedding."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        schema = pa.schema([pa.field("id", pa.string())])

        result = db.create_table("new_table", schema=schema, exist_ok=False)
        raw_db.create_table.assert_called_once_with("new_table", exist_ok=False, schema=schema)
        assert isinstance(result, LanceVectorTable)

    def test_create_table_with_embedding_function(self):
        """create_table() wraps embedding function and builds EmbeddingFunctionConfig."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        schema = pa.schema([pa.field("id", pa.string())])

        mock_embed = MagicMock()
        mock_embed.name = "test-model"
        mock_embed.batch_size = 64
        mock_embed.ndims.return_value = 128
        mock_embed.generate_embeddings.return_value = [[0.1] * 128]

        with (
            unittest.mock.patch("datus.storage.vector.lance_backend._wrap_embedding") as mock_wrap,
            unittest.mock.patch("datus.storage.vector.lance_backend.EmbeddingFunctionConfig") as mock_config_cls,
        ):
            mock_lance_fn = MagicMock()
            mock_wrap.return_value = mock_lance_fn
            mock_config = MagicMock()
            mock_config_cls.return_value = mock_config

            result = db.create_table(
                "new_table",
                schema=schema,
                embedding_function=mock_embed,
                vector_column="vec",
                source_column="text",
                exist_ok=False,
            )

            mock_wrap.assert_called_once_with(mock_embed)
            mock_config_cls.assert_called_once_with(
                vector_column="vec",
                source_column="text",
                function=mock_lance_fn,
            )
            call_kwargs = raw_db.create_table.call_args[1]
            assert call_kwargs["exist_ok"] is False
            assert call_kwargs["schema"] is schema
            assert call_kwargs["embedding_functions"] == [mock_config]
            assert isinstance(result, LanceVectorTable)

    def test_open_table(self):
        """open_table() delegates to db connection and returns LanceVectorTable."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        result = db.open_table("t1")
        raw_db.open_table.assert_called_once_with("t1")
        assert isinstance(result, LanceVectorTable)

    def test_open_table_ignores_embedding_function(self):
        """open_table() accepts but ignores embedding_function parameter."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        mock_embed = MagicMock()
        result = db.open_table("t1", embedding_function=mock_embed, vector_column="vec", source_column="text")
        raw_db.open_table.assert_called_once_with("t1")
        assert isinstance(result, LanceVectorTable)

    def test_drop_table(self):
        """drop_table() delegates to db connection."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        db.drop_table("t1", ignore_missing=True)
        raw_db.drop_table.assert_called_once_with("t1", ignore_missing=True)

    def test_refresh_table(self):
        """refresh_table() re-opens the table."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        result = db.refresh_table("t1")
        raw_db.open_table.assert_called_once_with("t1")
        assert isinstance(result, LanceVectorTable)

    def test_refresh_table_ignores_embedding_function(self):
        """refresh_table() accepts but ignores embedding_function parameter."""
        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db)
        mock_embed = MagicMock()
        result = db.refresh_table("t1", embedding_function=mock_embed, vector_column="vec", source_column="text")
        raw_db.open_table.assert_called_once_with("t1")
        assert isinstance(result, LanceVectorTable)


# ---------------------------------------------------------------------------
# LanceVectorBackend tests
# ---------------------------------------------------------------------------


class TestLanceVectorBackend:
    """Tests for backend lifecycle."""

    def test_initialize_noop(self):
        """initialize() stores config and doesn't raise."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/datus"})
        assert backend._data_dir == "/tmp/datus"

    def test_close_noop(self):
        """close() is a no-op for LanceDB."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/datus"})
        backend.close()
        assert backend._data_dir == "/tmp/datus"


# ---------------------------------------------------------------------------
# LanceVectorBackend connect() tests
# ---------------------------------------------------------------------------


class TestLanceVectorBackendConnect:
    """Connect-path tests: project is required, directory layout is per-project."""

    def test_initialize_reads_data_dir(self):
        """initialize() only records data_dir; no isolation knobs."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test"})
        assert backend._data_dir == "/tmp/test"

    def test_connect_builds_project_scoped_dir(self):
        """Each project gets ``{data_dir}/{project}/datus_db``."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test"})
        with unittest.mock.patch("datus.storage.vector.lance_backend.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = MagicMock()
            backend.connect("proj_a")
            mock_lancedb.connect.assert_called_once_with("/tmp/test/proj_a/datus_db")

    def test_connect_empty_project_raises(self):
        """connect('') raises DatusException instead of silently un-sharding."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test"})
        with pytest.raises(DatusException):
            backend.connect("")

    def test_connect_rejects_unsafe_project(self):
        """connect() rejects project identifiers with path separators."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test"})
        with pytest.raises(DatusException):
            backend.connect("../escape")

    def test_single_instance_reused_across_projects(self):
        """A single LanceVectorBackend instance serves many projects via separate connect() calls."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test"})
        with unittest.mock.patch("datus.storage.vector.lance_backend.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = MagicMock()
            backend.connect("proj_a")
            backend.connect("proj_b")
            paths = [c.args[0] for c in mock_lancedb.connect.call_args_list]
            assert "/tmp/test/proj_a/datus_db" in paths
            assert "/tmp/test/proj_b/datus_db" in paths


# ---------------------------------------------------------------------------
# _wrap_embedding tests
# ---------------------------------------------------------------------------


class TestWrapEmbedding:
    """Tests for module-level _wrap_embedding function."""

    def test_wrap_fastembed(self):
        """_wrap_embedding wraps FastEmbedEmbeddings into _LanceFastEmbedAdapter."""
        from datus.storage.fastembed_embeddings import FastEmbedEmbeddings
        from datus.storage.vector.lance_backend import _LanceFastEmbedAdapter, _wrap_embedding

        mock_model = MagicMock(spec=FastEmbedEmbeddings)
        mock_model.name = "all-MiniLM-L6-v2"
        mock_model.batch_size = 128
        mock_model.ndims.return_value = 384
        mock_model.generate_embeddings.return_value = [[0.1] * 384]

        adapter = _wrap_embedding(mock_model)

        assert isinstance(adapter, _LanceFastEmbedAdapter)
        assert adapter.name == "all-MiniLM-L6-v2"
        assert adapter.batch_size == 128

    def test_wrap_openai(self):
        """_wrap_embedding wraps non-FastEmbed models into _LanceOpenAIAdapter."""
        from datus.storage.vector.lance_backend import _LanceOpenAIAdapter, _wrap_embedding

        mock_model = MagicMock()
        mock_model.name = "text-embedding-3-small"
        mock_model.dim = 1536
        mock_model.base_url = None
        mock_model.api_key = "test-key"
        mock_model.use_azure = False

        adapter = _wrap_embedding(mock_model)

        assert isinstance(adapter, _LanceOpenAIAdapter)
        assert adapter.name == "text-embedding-3-small"

    def test_wrap_passthrough_lance_embedding(self):
        """_wrap_embedding returns LanceDB EmbeddingFunction as-is."""
        from lancedb.embeddings.base import TextEmbeddingFunction

        from datus.storage.vector.lance_backend import _wrap_embedding

        mock_lance = MagicMock(spec=TextEmbeddingFunction)
        result = _wrap_embedding(mock_lance)
        assert result is mock_lance


# ---------------------------------------------------------------------------
# Cross-process write serialization
# ---------------------------------------------------------------------------


def _hold_directory_lock(directory):
    """Take the same advisory lock a second Datus process would hold on ``directory``."""
    import fcntl

    fd = os.open(directory, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_directory_lock(fd):
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _run_in_thread(fn):
    done = threading.Event()
    errors = []

    def runner():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — surfaced to the test thread
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=runner, daemon=True).start()
    return done, errors


@pytest.mark.skipif(os.name != "posix", reason="directory flock is the POSIX coordination primitive")
class TestLanceWriteSerialization:
    """Writes to one LanceDB directory wait for a lock another process may hold.

    Two Datus processes (``datus web`` and ``datus-mcp``) commonly share one
    ``datus_db``. LanceDB coordinates concurrent local commits only
    optimistically, so every mutating call must block on the database
    directory's advisory lock instead of racing a foreign writer.
    """

    @pytest.fixture
    def db_dir(self, tmp_path):
        return tmp_path / "datus_db"

    @pytest.fixture
    def raw_table(self, db_dir):
        db = lancedb.connect(str(db_dir))
        return db.create_table("items", data=[{"id": 1, "text": "a"}])

    @pytest.mark.parametrize(
        "write",
        [
            pytest.param(lambda t: t.add(pd.DataFrame([{"id": 2, "text": "b"}])), id="add"),
            pytest.param(lambda t: t.merge_insert(pd.DataFrame([{"id": 1, "text": "z"}]), "id"), id="merge_insert"),
            pytest.param(lambda t: t.update(eq("id", 1), {"text": "y"}), id="update"),
            pytest.param(lambda t: t.delete(eq("id", 1)), id="delete"),
        ],
    )
    def test_table_write_waits_for_foreign_directory_lock(self, db_dir, raw_table, write):
        table = LanceVectorTable(raw_table)
        fd = _hold_directory_lock(db_dir)
        try:
            done, errors = _run_in_thread(lambda: write(table))
            assert not done.wait(0.5), "write committed while another holder owned the directory lock"
        finally:
            _release_directory_lock(fd)
        assert done.wait(10), "write never resumed after the directory lock was released"
        assert not errors

    def test_database_create_waits_for_foreign_directory_lock(self, db_dir):
        db = LanceVectorDatabase(lancedb.connect(str(db_dir)))
        schema = pa.schema([pa.field("id", pa.int64())])
        fd = _hold_directory_lock(db_dir)
        try:
            done, errors = _run_in_thread(lambda: db.create_table("second", schema=schema))
            assert not done.wait(0.5), "DDL ran while another holder owned the directory lock"
        finally:
            _release_directory_lock(fd)
        assert done.wait(10)
        assert not errors
        assert "second" in db.table_names()

    def test_database_drop_waits_for_foreign_directory_lock(self, db_dir):
        db = LanceVectorDatabase(lancedb.connect(str(db_dir)))
        db.create_table("second", schema=pa.schema([pa.field("id", pa.int64())]))
        fd = _hold_directory_lock(db_dir)
        try:
            done, errors = _run_in_thread(lambda: db.drop_table("second"))
            assert not done.wait(0.5), "drop ran while another holder owned the directory lock"
        finally:
            _release_directory_lock(fd)
        assert done.wait(10)
        assert not errors
        assert "second" not in db.table_names()

    def test_write_lock_is_reentrant_within_one_thread(self, db_dir, raw_table):
        from datus.storage.vector.lance_backend import lance_write_lock

        with lance_write_lock(db_dir):
            with lance_write_lock(db_dir):
                LanceVectorTable(raw_table).add(pd.DataFrame([{"id": 2, "text": "b"}]))
        assert raw_table.count_rows() == 2

    def test_write_lock_released_after_failed_write(self, db_dir, raw_table):
        table = LanceVectorTable(raw_table)
        with pytest.raises(ValueError):
            table.add(pd.DataFrame([{"not_a_column": 1}]))
        done, errors = _run_in_thread(lambda: table.add(pd.DataFrame([{"id": 2, "text": "b"}])))
        assert done.wait(10), "lock leaked after a failed write"
        assert not errors

    def test_remote_or_unknown_uri_does_not_touch_filesystem(self, tmp_path):
        raw = MagicMock()
        raw.uri = "s3://bucket/datus_db/items.lance"
        LanceVectorTable(raw).add(pd.DataFrame([{"id": 1}]))
        raw.add.assert_called_once()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "uri,expected",
    [
        pytest.param("/data/proj/datus_db", Path("/data/proj/datus_db"), id="posix-absolute"),
        pytest.param("file:///data/proj/datus_db", Path("/data/proj/datus_db"), id="file-uri"),
        pytest.param("s3://bucket/proj/datus_db", None, id="remote"),
        pytest.param("gs://bucket/proj/datus_db", None, id="remote-gcs"),
        pytest.param("", None, id="empty"),
        pytest.param(None, None, id="missing"),
    ],
)
def test_local_path_resolves_supported_uris(uri, expected):
    from datus.storage.vector.lance_backend import _local_path

    assert _local_path(uri) == expected


@pytest.mark.parametrize("uri", ["C:\\data\\datus_db", "c:\\data\\datus_db", "D:/data/datus_db"])
def test_local_path_treats_windows_drive_letters_as_local(uri):
    """A drive letter reads as a URI scheme to urlparse; it must still resolve.

    A ``None`` result disables the cross-process write lock entirely, so this
    pins the class of path rather than the platform-specific normalisation.
    """
    from datus.storage.vector.lance_backend import _local_path

    # A ``None`` would stringify as ``"None"`` and fail the suffix check too;
    # the suffix pins the drive letter was kept, not stripped as a scheme.
    assert str(_local_path(uri)).replace("\\", "/").endswith("/data/datus_db")
