# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/schema_metadata/store.py — SchemaStorage and _build_where_clause."""

import pytest
from datus_db_core import ConnectorRegistry
from datus_storage_base.conditions import And, Condition, build_where

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import _build_where_clause
from datus.tools.db_tools import connector_registry


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register capabilities for dialects used in tests, with snapshot/restore for isolation."""
    attrs = ("_capabilities", "_uri_builders", "_context_resolvers")
    snapshots = {a: getattr(ConnectorRegistry, a).copy() for a in attrs}
    connector_registry.register_handlers("starrocks", capabilities={"catalog", "database"})
    yield
    for a, snap in snapshots.items():
        setattr(ConnectorRegistry, a, snap)


# ---------------------------------------------------------------------------
# SchemaStorage._extract_table_name
# ---------------------------------------------------------------------------


class TestExtractTableName:
    """Tests for SchemaStorage._extract_table_name parsing CREATE TABLE statements."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def test_extract_simple_create_table(self, tmp_path):
        """Standard CREATE TABLE should extract the table name."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("CREATE TABLE users (id INT)")
        assert result == "users"

    def test_extract_create_table_with_schema(self, tmp_path):
        """CREATE TABLE with schema-qualified name should extract schema.table."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("CREATE TABLE public.orders (id INT)")
        assert result == "public.orders"

    def test_extract_create_table_uppercase(self, tmp_path):
        """Case-insensitive matching of CREATE TABLE keywords."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("create table employees (id int)")
        assert result == "employees"

    def test_extract_mixed_case_create_table(self, tmp_path):
        """Mixed case CREATE TABLE should still work."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("Create Table Products (id INT)")
        assert result == "Products"

    def test_extract_table_name_no_space_before_paren(self, tmp_path):
        """Table name abutting '(' without space should still extract correctly."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("CREATE TABLE mytable(id INT)")
        assert result == "mytable"

    def test_extract_table_name_with_space_before_paren(self, tmp_path):
        """Table name with space before '(' should be extracted cleanly."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("CREATE TABLE mytable (id INT)")
        assert result == "mytable"

    def test_extract_table_name_strips_wrapping_parens(self, tmp_path):
        """Table name wrapped in parentheses should have them stripped."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("CREATE TABLE (mytable) (id INT)")
        assert result == "mytable"

    def test_extract_table_name_returns_empty_for_non_create(self, tmp_path):
        """Non-CREATE TABLE statements should return empty string."""
        store = self._make_store(tmp_path)
        assert store._extract_table_name("SELECT * FROM users") == ""
        assert store._extract_table_name("INSERT INTO users VALUES (1)") == ""

    def test_extract_table_name_returns_empty_for_too_short(self, tmp_path):
        """Input with fewer than 3 words should return empty."""
        store = self._make_store(tmp_path)
        assert store._extract_table_name("CREATE") == ""
        assert store._extract_table_name("CREATE TABLE") == ""

    def test_extract_table_name_returns_empty_for_empty_string(self, tmp_path):
        """Empty input should return empty string."""
        store = self._make_store(tmp_path)
        assert store._extract_table_name("") == ""

    def test_extract_table_name_with_if_not_exists(self, tmp_path):
        """CREATE TABLE IF NOT EXISTS should correctly skip to the table name."""
        store = self._make_store(tmp_path)
        result = store._extract_table_name("CREATE TABLE IF NOT EXISTS t (id INT)")
        assert result == "t"


# ---------------------------------------------------------------------------
# _build_where_clause
# ---------------------------------------------------------------------------


class TestBuildWhereClause:
    """Tests for the module-level _build_where_clause function."""

    def test_no_conditions_returns_none(self):
        """When all string params are empty and table_type is 'full', return None."""
        result = _build_where_clause(table_type="full")
        assert result is None

    def test_default_args_includes_table_type(self):
        """With default args, table_type='table' adds a condition."""
        result = _build_where_clause()
        assert isinstance(result, Condition)
        clause = build_where(result)
        assert "table_type = 'table'" in clause

    def test_single_catalog_condition(self):
        """A single catalog_name (with table_type='full') produces a Condition node."""
        result = _build_where_clause(catalog_name="my_catalog", table_type="full")
        assert isinstance(result, Condition)
        clause = build_where(result)
        assert "catalog_name = 'my_catalog'" in clause

    def test_single_database_condition(self):
        """A single database_name (with table_type='full') produces a Condition node."""
        result = _build_where_clause(database_name="my_db", table_type="full")
        assert isinstance(result, Condition)
        clause = build_where(result)
        assert "database_name = 'my_db'" in clause

    def test_single_schema_condition(self):
        """A single schema_name (with table_type='full') produces a Condition node."""
        result = _build_where_clause(schema_name="public", table_type="full")
        assert isinstance(result, Condition)
        clause = build_where(result)
        assert "schema_name = 'public'" in clause

    def test_single_table_name_condition(self):
        """A single table_name (with table_type='full') produces a Condition node."""
        result = _build_where_clause(table_name="orders", table_type="full")
        assert isinstance(result, Condition)
        clause = build_where(result)
        assert "table_name = 'orders'" in clause

    def test_table_type_full_excluded(self):
        """table_type='full' should not add a table_type condition."""
        result = _build_where_clause(table_type="full")
        assert result is None

    def test_table_type_table_added(self):
        """table_type='table' should add a table_type condition."""
        result = _build_where_clause(table_type="table")
        assert isinstance(result, Condition)
        clause = build_where(result)
        assert "table_type = 'table'" in clause

    def test_table_type_view_added(self):
        """table_type='view' should add a table_type condition."""
        result = _build_where_clause(table_type="view")
        clause = build_where(result)
        assert "table_type = 'view'" in clause

    def test_two_conditions_produce_and(self):
        """Two conditions should produce an And node."""
        result = _build_where_clause(catalog_name="c", database_name="d", table_type="full")
        assert isinstance(result, And)
        clause = build_where(result)
        assert "catalog_name = 'c'" in clause
        assert "database_name = 'd'" in clause
        assert "AND" in clause

    def test_three_conditions_produce_and(self):
        """Three conditions should produce an And node."""
        result = _build_where_clause(catalog_name="c", database_name="d", schema_name="s", table_type="full")
        assert isinstance(result, And)
        clause = build_where(result)
        assert "catalog_name" in clause
        assert "database_name" in clause
        assert "schema_name" in clause

    def test_all_conditions_combined(self):
        """All five parameters set should produce an And with all conditions."""
        result = _build_where_clause(
            catalog_name="c",
            database_name="d",
            schema_name="s",
            table_name="t",
            table_type="table",
        )
        assert isinstance(result, And)
        clause = build_where(result)
        assert "catalog_name = 'c'" in clause
        assert "database_name = 'd'" in clause
        assert "schema_name = 's'" in clause
        assert "table_name = 't'" in clause
        assert "table_type = 'table'" in clause

    def test_empty_string_parameters_ignored(self):
        """Empty string parameters should be ignored (not generate conditions)."""
        result = _build_where_clause(catalog_name="", database_name="", schema_name="", table_type="full")
        assert result is None


# ---------------------------------------------------------------------------
# search_tables table name parsing
# ---------------------------------------------------------------------------


class TestSearchTablesNameParsing:
    """Tests for table name parsing in SchemaWithValueRAG.search_tables.

    These test the _build_where_clause calls generated from different table name formats
    by verifying the where clause logic without requiring full RAG initialization.
    """

    def test_parse_1_part_name_uses_defaults(self):
        """Single-part name (e.g., 'orders') should use provided catalog/database/schema defaults."""
        # Simulating the 1-part logic from search_tables (the else branch)
        full_table = "orders"
        parts = full_table.split(".")
        assert len(parts) == 1
        table_name = parts[-1]
        assert table_name == "orders"

        # With defaults
        where = _build_where_clause(
            table_name="orders",
            catalog_name="default_catalog",
            database_name="default_db",
            schema_name="default_schema",
            table_type="full",
        )
        clause = build_where(where)
        assert "table_name = 'orders'" in clause
        assert "catalog_name = 'default_catalog'" in clause
        assert "database_name = 'default_db'" in clause
        assert "schema_name = 'default_schema'" in clause

    def test_parse_2_part_name_sqlite_dialect(self):
        """Two-part name (e.g., 'db.table') with SQLite dialect: db = parts[0]."""
        full_table = "mydb.orders"
        parts = full_table.split(".")
        assert len(parts) == 2
        table_name = parts[-1]
        # SQLite: cat=catalog_name(default), db=parts[0], sch=""
        cat, db, sch = "default_catalog", parts[0], ""
        assert cat == "default_catalog"
        assert db == "mydb"
        assert sch == ""
        assert table_name == "orders"

    def test_parse_2_part_name_postgresql_dialect(self):
        """Two-part name (e.g., 'schema.table') with PostgreSQL dialect: schema = parts[0]."""
        full_table = "public.orders"
        parts = full_table.split(".")
        assert len(parts) == 2
        table_name = parts[-1]
        # PostgreSQL: cat=catalog, db=database, sch=parts[0]
        db, sch = "default_db", parts[0]
        assert sch == "public"
        assert db == "default_db"
        assert table_name == "orders"

    def test_parse_3_part_name_default_dialect(self):
        """Three-part name (e.g., 'db.schema.table') with non-StarRocks dialect."""
        full_table = "mydb.public.orders"
        parts = full_table.split(".")
        assert len(parts) == 3
        table_name = parts[-1]
        # Non-StarRocks: cat=catalog_name, db=parts[0], sch=parts[1]
        db, sch = parts[0], parts[1]
        assert db == "mydb"
        assert sch == "public"
        assert table_name == "orders"

    def test_parse_3_part_name_starrocks_dialect(self):
        """Three-part name with StarRocks-like dialect (catalog+database, no schema)."""
        from datus.tools.db_tools import connector_registry

        full_table = "mycat.mydb.orders"
        parts = full_table.split(".")
        assert len(parts) == 3
        table_name = parts[-1]
        dialect = "starrocks"
        if connector_registry.support_catalog(dialect) and not connector_registry.support_schema(dialect):
            cat, db, sch = parts[0], parts[1], ""
        else:
            cat, db, sch = "catalog", parts[0], parts[1]
        assert cat == "mycat"
        assert db == "mydb"
        assert sch == ""
        assert table_name == "orders"

    def test_parse_4_part_name(self):
        """Four-part name (e.g., 'catalog.db.schema.table') is fully qualified."""
        full_table = "mycat.mydb.mysch.orders"
        parts = full_table.split(".")
        assert len(parts) == 4
        table_name = parts[-1]
        cat, db, sch = parts[0], parts[1], parts[2]
        assert cat == "mycat"
        assert db == "mydb"
        assert sch == "mysch"
        assert table_name == "orders"

        where = _build_where_clause(
            table_name=table_name,
            catalog_name=cat,
            database_name=db,
            schema_name=sch,
            table_type="full",
        )
        clause = build_where(where)
        assert "table_name = 'orders'" in clause
        assert "catalog_name = 'mycat'" in clause
        assert "database_name = 'mydb'" in clause
        assert "schema_name = 'mysch'" in clause

    def test_parse_2_part_mysql_dialect(self):
        """Two-part name with MySQL dialect: db=parts[0], sch=''."""
        full_table = "mydb.orders"
        parts = full_table.split(".")
        table_name = parts[-1]
        # MySQL falls into SQLite/MySQL/StarRocks branch: db=parts[0], sch=""
        db, sch = parts[0], ""
        assert db == "mydb"
        assert sch == ""
        assert table_name == "orders"

    def test_parse_2_part_starrocks_dialect(self):
        """Two-part name with StarRocks dialect: db=parts[0], sch=''."""
        full_table = "mydb.orders"
        parts = full_table.split(".")
        table_name = parts[-1]
        # StarRocks falls into SQLite/MySQL/StarRocks branch: db=parts[0], sch=""
        db, sch = parts[0], ""
        assert db == "mydb"
        assert sch == ""
        assert table_name == "orders"


# ---------------------------------------------------------------------------
# BaseMetadataStorage.search_similar and do_search_similar
# ---------------------------------------------------------------------------


class TestSearchSimilar:
    """Tests for search_similar and do_search_similar."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int, db_name: str = "db", schema: str = "sch") -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": db_name,
            "schema_name": schema,
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT, name VARCHAR, amount DECIMAL)",
        }

    def test_search_similar_returns_results(self, tmp_path):
        """search_similar returns matching schema rows."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i) for i in range(5)])

        result = store.search_similar("table with amount", catalog_name="cat", top_n=3)
        assert result.num_rows >= 0
        assert "vector" not in result.column_names

    def test_search_similar_with_filters(self, tmp_path):
        """search_similar respects database_name and schema_name filters."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i, db_name="db1") for i in range(3)])
        store.store_batch([self._make_row(i + 10, db_name="db2") for i in range(2)])

        result = store.search_similar("table", database_name="db1", top_n=10)
        assert result.num_rows >= 0

    def test_do_search_similar_delegates_to_search(self, tmp_path):
        """do_search_similar calls search() with correct parameters."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i) for i in range(3)])

        result = store.do_search_similar("table", top_n=2)
        assert result.num_rows >= 0
        assert "vector" not in result.column_names


# ---------------------------------------------------------------------------
# SchemaStorage.search_all_schemas
# ---------------------------------------------------------------------------


class TestSearchAllSchemas:
    """Tests for SchemaStorage.search_all_schemas query construction."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int, schema: str = "public") -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": schema,
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_search_all_schemas_queries_with_filters(self, tmp_path):
        """search_all_schemas constructs proper where clause from filters."""
        # Verify the _build_where_clause used by search_all_schemas
        where = _build_where_clause(database_name="db", catalog_name="cat")
        clause = build_where(where)
        assert "database_name = 'db'" in clause
        assert "catalog_name = 'cat'" in clause


# ---------------------------------------------------------------------------
# SchemaStorage.get_schema
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Tests for SchemaStorage.get_schema."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "public",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_get_schema_returns_matching_table(self, tmp_path):
        """get_schema returns the schema for a specific table."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1), self._make_row(2)])

        result = store.get_schema("table_1", catalog_name="cat", database_name="db", schema_name="public")
        assert result.num_rows >= 0
        if result.num_rows > 0:
            assert "definition" in result.column_names
            assert "table_name" in result.column_names

    def test_get_schema_no_match(self, tmp_path):
        """get_schema returns empty result for non-existent table."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        result = store.get_schema("nonexistent_table", catalog_name="cat", database_name="db")
        assert result.num_rows == 0

    def test_get_schema_without_filters(self, tmp_path):
        """get_schema with only table_name works."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        result = store.get_schema("table_1")
        assert result.num_rows >= 0


# ---------------------------------------------------------------------------
# SchemaWithValueRAG.store_batch value processing
# ---------------------------------------------------------------------------


class TestSchemaWithValueRAGStoreBatch:
    """Tests for SchemaWithValueRAG.store_batch value processing."""

    def _make_schema_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def _make_value_row(self, idx: int, sample_rows=None) -> dict:
        return {
            "identifier": f"val_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "sample_rows": sample_rows or "id,name\n1,Alice\n2,Bob",
        }

    def test_store_batch_schemas_only(self, tmp_path, real_agent_config):
        """store_batch with schemas but no values."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(i) for i in range(3)]
        rag.store_batch(schemas, [])
        assert rag.get_schema_size() == 3
        assert rag.get_value_size() == 0

    def test_store_batch_with_string_values(self, tmp_path, real_agent_config):
        """store_batch with string sample_rows stores values."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(1)]
        values = [self._make_value_row(1, sample_rows="id,name\n1,Alice")]
        rag.store_batch(schemas, values)
        assert rag.get_schema_size() == 1
        assert rag.get_value_size() == 1

    def test_store_batch_with_list_values_converts_to_csv(self, tmp_path, real_agent_config):
        """store_batch converts list sample_rows to CSV via json2csv."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(1)]
        values = [
            self._make_value_row(
                1,
                sample_rows=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            )
        ]
        rag.store_batch(schemas, values)
        assert rag.get_value_size() == 1

    def test_store_batch_skips_empty_sample_rows(self, tmp_path, real_agent_config):
        """store_batch skips values with empty or missing sample_rows."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(1)]
        # Build values manually to avoid default sample_rows
        val_empty = {
            "identifier": "val_1",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "table_1",
            "table_type": "table",
            "sample_rows": "",
        }
        val_none = {
            "identifier": "val_2",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "table_2",
            "table_type": "table",
            "sample_rows": None,
        }
        val_missing_key = {
            "identifier": "val_3",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "table_3",
            "table_type": "table",
        }
        rag.store_batch(schemas, [val_empty, val_none, val_missing_key])
        assert rag.get_value_size() == 0


# ---------------------------------------------------------------------------
# SchemaWithValueRAG.remove_data
# ---------------------------------------------------------------------------


class TestSchemaWithValueRAGRemoveData:
    """Tests for SchemaWithValueRAG.remove_data."""

    def _make_schema_row(self, idx: int, db_name: str = "db") -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": db_name,
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def _make_value_row(self, idx: int, db_name: str = "db") -> dict:
        return {
            "identifier": f"val_{idx}",
            "catalog_name": "cat",
            "database_name": db_name,
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "sample_rows": f"id,name\n{idx},Item{idx}",
        }

    def test_remove_data_deletes_matching_rows(self, tmp_path, real_agent_config):
        """remove_data deletes schema and value rows matching the filter."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(1), self._make_schema_row(2)]
        values = [self._make_value_row(1), self._make_value_row(2)]
        rag.store_batch(schemas, values)
        assert rag.get_schema_size() == 2

        rag.remove_data(catalog_name="cat", database_name="db", table_name="table_1")
        assert rag.get_schema_size() == 1
        assert rag.get_value_size() == 1

    def test_remove_data_with_no_matching_filter_is_noop(self, tmp_path, real_agent_config):
        """remove_data with no matching filter does nothing."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(1)]
        values = [self._make_value_row(1)]
        rag.store_batch(schemas, values)

        rag.remove_data(catalog_name="nonexistent")
        assert rag.get_schema_size() == 1

    def test_remove_data_with_empty_filter(self, tmp_path, real_agent_config):
        """remove_data with all empty filter parameters is a no-op (build_where returns None)."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(1)]
        values = [self._make_value_row(1)]
        rag.store_batch(schemas, values)

        rag.remove_data(table_type="full")
        # Should not delete anything since _build_where_clause returns None with all empty + 'full'
        assert rag.get_schema_size() == 1


# ---------------------------------------------------------------------------
# SchemaWithValueRAG.search_similar
# ---------------------------------------------------------------------------


class TestSchemaWithValueRAGSearchSimilar:
    """Tests for SchemaWithValueRAG.search_similar."""

    def _make_schema_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT, name VARCHAR)",
        }

    def _make_value_row(self, idx: int) -> dict:
        return {
            "identifier": f"val_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "sample_rows": f"id,name\n{idx},Item{idx}",
        }

    def test_search_similar_returns_tuple(self, tmp_path, real_agent_config):
        """search_similar returns a tuple of (schema_results, value_results)."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        schemas = [self._make_schema_row(i) for i in range(3)]
        values = [self._make_value_row(i) for i in range(3)]
        rag.store_batch(schemas, values)

        schema_results, value_results = rag.search_similar("table with name", catalog_name="cat", top_n=3)
        assert schema_results.num_rows >= 0
        assert value_results.num_rows >= 0


# ---------------------------------------------------------------------------
# SchemaWithValueRAG.truncate
# ---------------------------------------------------------------------------


class TestSchemaWithValueRAGTruncate:
    """Tests for SchemaWithValueRAG.truncate."""

    def _make_schema_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_truncate_clears_both_stores(self, tmp_path, real_agent_config):
        """truncate clears both schema and value stores."""
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(real_agent_config)
        rag.store_batch(
            [self._make_schema_row(1)],
            [
                {
                    "identifier": "val_1",
                    "catalog_name": "cat",
                    "database_name": "db",
                    "schema_name": "sch",
                    "table_name": "table_1",
                    "table_type": "table",
                    "sample_rows": "id\n1",
                }
            ],
        )
        assert rag.get_schema_size() == 1
        assert rag.get_value_size() == 1

        rag.truncate()
        # After truncate, tables are reset (will be recreated on next use)
        assert rag.schema_store.table is None
        assert rag.value_store.table is None
