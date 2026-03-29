# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for LanceDB vector backend covering table, database, and backend operations."""

import unittest.mock
from unittest.mock import MagicMock

import pyarrow as pa
from datus_storage_base.conditions import and_, eq, in_

from datus.storage.vector.lance_backend import LanceVectorBackend, LanceVectorDatabase, LanceVectorTable

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
        """delete() with in_() Node compiles to OR chain."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.delete(in_("chunk_id", ["a", "b", "c"]))
        raw_table.delete.assert_called_once_with("(chunk_id = 'a' OR chunk_id = 'b' OR chunk_id = 'c')")

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


class TestLanceVectorTableIndexOps:
    """Tests for table-level index operations."""

    def test_create_vector_index(self):
        """create_vector_index() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.create_vector_index("vec_col", metric="l2", replace=True)
        raw_table.create_index.assert_called_once_with(metric="l2", vector_column_name="vec_col", replace=True)

    def test_create_fts_index(self):
        """create_fts_index() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.create_fts_index(["title", "body"])
        raw_table.create_fts_index.assert_called_once_with(field_names=["title", "body"], replace=True)

    def test_create_scalar_index(self):
        """create_scalar_index() delegates to underlying table."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        table.create_scalar_index("category")
        raw_table.create_scalar_index.assert_called_once_with("category", replace=True)

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
        backend.initialize({})

    def test_close_noop(self):
        """close() is a no-op for LanceDB."""
        backend = LanceVectorBackend()
        backend.close()


# ---------------------------------------------------------------------------
# _wrap_embedding tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logical isolation tests
# ---------------------------------------------------------------------------


class TestLanceVectorTableLogicalIsolation:
    """Tests for IsolationType.LOGICAL in LanceVectorTable."""

    def _logical_table(self, raw_table=None, datasource_id="tenant_1"):
        from datus.storage.vector.lance_backend import IsolationType

        return LanceVectorTable(raw_table or MagicMock(), isolation=IsolationType.LOGICAL, datasource_id=datasource_id)

    def test_inject_datasource_df_adds_column(self):
        """_inject_datasource_df adds datasource_id column in LOGICAL mode."""
        import pandas as pd

        table = self._logical_table()
        df = pd.DataFrame({"id": ["a", "b"], "text": ["hello", "world"]})
        result = table._inject_datasource_df(df)
        assert "datasource_id" in result.columns
        assert list(result["datasource_id"]) == ["tenant_1", "tenant_1"]

    def test_inject_datasource_df_overwrites_existing(self):
        """_inject_datasource_df overwrites existing datasource_id value."""
        import pandas as pd

        table = self._logical_table()
        df = pd.DataFrame({"id": ["a"], "datasource_id": ["old"]})
        result = table._inject_datasource_df(df)
        assert result["datasource_id"].iloc[0] == "tenant_1"

    def test_inject_datasource_df_noop_in_physical(self):
        """_inject_datasource_df is a no-op when isolation is PHYSICAL."""
        import pandas as pd

        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)  # default = PHYSICAL
        df = pd.DataFrame({"id": ["a"]})
        result = table._inject_datasource_df(df)
        assert "datasource_id" not in result.columns

    def test_ds_where_no_existing(self):
        """_ds_where returns datasource_id condition when no existing WHERE."""
        table = self._logical_table()
        result = table._ds_where(None)
        assert result == "datasource_id = 'tenant_1'"

    def test_ds_where_with_existing(self):
        """_ds_where prepends datasource_id to existing WHERE."""
        table = self._logical_table()
        result = table._ds_where("status = 'active'")
        assert result == "datasource_id = 'tenant_1' AND (status = 'active')"

    def test_ds_where_noop_in_physical(self):
        """_ds_where is a no-op when isolation is PHYSICAL."""
        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)
        result = table._ds_where("status = 'active'")
        assert result == "status = 'active'"

    def test_ds_where_escapes_quotes(self):
        """_ds_where escapes single quotes in datasource_id to prevent SQL injection."""
        table = self._logical_table(datasource_id="tenant's_test")
        result = table._ds_where(None)
        assert result == "datasource_id = 'tenant''s_test'"

    def test_add_injects_datasource_in_logical(self):
        """add() injects datasource_id column in LOGICAL mode."""
        import pandas as pd

        raw_table = MagicMock()
        table = self._logical_table(raw_table)
        df = pd.DataFrame({"id": ["a"]})
        table.add(df)
        call_df = raw_table.add.call_args[0][0]
        assert "datasource_id" in call_df.columns

    def test_merge_insert_uses_composite_key_in_logical(self):
        """merge_insert() uses [datasource_id, on_column] key in LOGICAL mode."""
        import pandas as pd

        raw_table = MagicMock()
        table = self._logical_table(raw_table)
        df = pd.DataFrame({"id": ["a"], "text": ["hello"]})
        table.merge_insert(df, "id")
        raw_table.merge_insert.assert_called_once_with(["datasource_id", "id"])

    def test_merge_insert_uses_single_key_in_physical(self):
        """merge_insert() uses single on_column key in PHYSICAL mode."""
        import pandas as pd

        raw_table = MagicMock()
        table = LanceVectorTable(raw_table)  # PHYSICAL
        df = pd.DataFrame({"id": ["a"]})
        table.merge_insert(df, "id")
        raw_table.merge_insert.assert_called_once_with("id")

    def test_delete_prepends_datasource_filter(self):
        """delete() prepends datasource_id condition in LOGICAL mode."""
        raw_table = MagicMock()
        table = self._logical_table(raw_table)
        table.delete(eq("status", "old"))
        raw_table.delete.assert_called_once_with("datasource_id = 'tenant_1' AND (status = 'old')")

    def test_delete_none_with_logical_uses_ds_filter(self):
        """delete(None) in LOGICAL mode uses datasource_id filter only."""
        raw_table = MagicMock()
        table = self._logical_table(raw_table)
        table.delete(None)
        raw_table.delete.assert_called_once_with("datasource_id = 'tenant_1'")

    def test_count_rows_prepends_datasource_filter(self):
        """count_rows() prepends datasource_id condition in LOGICAL mode."""
        raw_table = MagicMock()
        raw_table.count_rows.return_value = 10
        table = self._logical_table(raw_table)
        result = table.count_rows()
        raw_table.count_rows.assert_called_once_with("datasource_id = 'tenant_1'")
        assert result == 10

    def test_count_rows_with_where_in_logical(self):
        """count_rows(where) prepends datasource_id in LOGICAL mode."""
        raw_table = MagicMock()
        raw_table.count_rows.return_value = 3
        table = self._logical_table(raw_table)
        result = table.count_rows(where=eq("kind", "table"))
        raw_table.count_rows.assert_called_once_with("datasource_id = 'tenant_1' AND (kind = 'table')")
        assert result == 3


class TestLanceVectorDatabaseLogicalIsolation:
    """Tests for IsolationType.LOGICAL in LanceVectorDatabase."""

    def test_create_table_injects_datasource_field(self):
        """create_table() adds datasource_id field to schema in LOGICAL mode."""
        from datus.storage.vector.lance_backend import IsolationType

        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db, isolation=IsolationType.LOGICAL, datasource_id="ds_1")
        schema = pa.schema([pa.field("id", pa.string())])
        db.create_table("test_table", schema=schema)

        call_kwargs = raw_db.create_table.call_args[1]
        created_schema = call_kwargs["schema"]
        field_names = [f.name for f in created_schema]
        assert "datasource_id" in field_names

    def test_create_table_no_duplicate_datasource_field(self):
        """create_table() doesn't duplicate datasource_id if already in schema."""
        from datus.storage.vector.lance_backend import IsolationType

        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db, isolation=IsolationType.LOGICAL, datasource_id="ds_1")
        schema = pa.schema([pa.field("id", pa.string()), pa.field("datasource_id", pa.string())])
        db.create_table("test_table", schema=schema)

        call_kwargs = raw_db.create_table.call_args[1]
        created_schema = call_kwargs["schema"]
        ds_fields = [f for f in created_schema if f.name == "datasource_id"]
        assert len(ds_fields) == 1

    def test_open_table_propagates_isolation(self):
        """open_table() propagates isolation and datasource_id to LanceVectorTable."""
        from datus.storage.vector.lance_backend import IsolationType

        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db, isolation=IsolationType.LOGICAL, datasource_id="ds_1")
        table = db.open_table("test_table")
        assert isinstance(table, LanceVectorTable)
        assert table._isolation == IsolationType.LOGICAL
        assert table._datasource_id == "ds_1"

    def test_refresh_table_propagates_isolation(self):
        """refresh_table() propagates isolation and datasource_id."""
        from datus.storage.vector.lance_backend import IsolationType

        raw_db = MagicMock()
        db = LanceVectorDatabase(raw_db, isolation=IsolationType.LOGICAL, datasource_id="ds_1")
        table = db.refresh_table("test_table")
        assert table._isolation == IsolationType.LOGICAL
        assert table._datasource_id == "ds_1"


class TestLanceVectorBackendLogicalIsolation:
    """Tests for IsolationType.LOGICAL in LanceVectorBackend."""

    def test_initialize_stores_isolation(self):
        """initialize() stores isolation type from config."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test", "isolation": "logical"})
        from datus.storage.vector.lance_backend import IsolationType

        assert backend._isolation == IsolationType.LOGICAL

    def test_initialize_defaults_to_physical(self):
        """initialize() defaults to PHYSICAL when isolation not specified."""
        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test"})
        from datus.storage.vector.lance_backend import IsolationType

        assert backend._isolation == IsolationType.PHYSICAL

    def test_connect_logical_shares_datus_db(self):
        """connect() in LOGICAL mode always uses datus_db directory."""
        import unittest.mock

        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test", "isolation": "logical"})
        with unittest.mock.patch("datus.storage.vector.lance_backend.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = MagicMock()
            db = backend.connect(namespace="tenant_a")
            mock_lancedb.connect.assert_called_once_with("/tmp/test/datus_db")
            from datus.storage.vector.lance_backend import IsolationType

            assert db._isolation == IsolationType.LOGICAL
            assert db._datasource_id == "tenant_a"

    def test_connect_physical_uses_namespace_as_dir(self):
        """connect() in PHYSICAL mode uses datus_db_{namespace} as directory name."""
        import unittest.mock

        backend = LanceVectorBackend()
        backend.initialize({"data_dir": "/tmp/test", "isolation": "physical"})
        with unittest.mock.patch("datus.storage.vector.lance_backend.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = MagicMock()
            db = backend.connect(namespace="tenant_a")
            mock_lancedb.connect.assert_called_once_with("/tmp/test/datus_db_tenant_a")
            assert db._datasource_id is None


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
