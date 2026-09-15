# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json

import duckdb
import pytest

from datus.tools.db_tools.database_import import (
    DatabaseImportError,
    _dependency_order,
    import_duckdb_file,
    read_generator_meta,
)

SOURCE_DDL = """
CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    sku VARCHAR UNIQUE,
    product_name VARCHAR NOT NULL,
    list_price DECIMAL(18, 2)
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    product_id BIGINT REFERENCES products(product_id),
    paid_amount DECIMAL(18, 2)
);
"""


@pytest.fixture
def source_db(tmp_path):
    """A small two-table source carrying every constraint kind plus comments."""
    path = tmp_path / "source.duckdb"
    con = duckdb.connect(str(path))
    con.execute(SOURCE_DDL)
    con.execute("INSERT INTO products VALUES (1, 'SKU-1', 'Widget', 9.90), (2, 'SKU-2', 'Gadget', 19.90)")
    con.execute("INSERT INTO orders VALUES (10, 1, 9.90), (11, 2, 19.90), (12, 1, 9.90)")
    con.execute("COMMENT ON TABLE orders IS 'Order header'")
    con.execute("COMMENT ON COLUMN orders.paid_amount IS 'Amount actually paid'")
    con.close()
    return path


@pytest.fixture
def target(tmp_path):
    con = duckdb.connect(str(tmp_path / "target.duckdb"))
    yield con
    con.close()


def _constraints(con):
    return {
        (t, ctype)
        for t, ctype in con.execute(
            "SELECT table_name, constraint_type FROM duckdb_constraints() WHERE database_name = current_database()"
        ).fetchall()
    }


@pytest.mark.acceptance
def test_import_copies_rows(source_db, target):
    out = import_duckdb_file(target, source_db)

    assert out["imported"] == {"products": 2, "orders": 3}
    assert out["total_rows"] == 5
    assert out["degraded"] == []
    assert target.execute("SELECT count(*) FROM orders").fetchone()[0] == 3


@pytest.mark.acceptance
def test_import_preserves_declared_constraints(source_db, target):
    """The whole point of replaying the DDL: CREATE TABLE AS SELECT would drop all of these."""
    import_duckdb_file(target, source_db)

    found = _constraints(target)
    assert ("products", "PRIMARY KEY") in found
    assert ("products", "UNIQUE") in found
    assert ("orders", "PRIMARY KEY") in found
    assert ("orders", "FOREIGN KEY") in found

    with pytest.raises(duckdb.ConstraintException):
        target.execute("INSERT INTO products VALUES (1, 'SKU-9', 'Dup', 1.0)")


@pytest.mark.acceptance
def test_import_without_constraints_when_disabled(source_db, target):
    import_duckdb_file(target, source_db, keep_constraints=False)

    assert target.execute("SELECT count(*) FROM orders").fetchone()[0] == 3
    assert not [c for c in _constraints(target) if c[1] in ("PRIMARY KEY", "FOREIGN KEY", "UNIQUE")]


@pytest.mark.acceptance
def test_import_copies_comments(source_db, target):
    import_duckdb_file(target, source_db)

    table_note = target.execute(
        "SELECT comment FROM duckdb_tables() WHERE table_name = 'orders' AND database_name = current_database()"
    ).fetchone()[0]
    column_note = target.execute(
        "SELECT comment FROM duckdb_columns() "
        "WHERE table_name = 'orders' AND column_name = 'paid_amount' "
        "AND database_name = current_database()"
    ).fetchone()[0]

    assert table_note == "Order header"
    assert column_note == "Amount actually paid"


@pytest.mark.acceptance
def test_import_detaches_source_so_it_can_run_twice(source_db, target):
    import_duckdb_file(target, source_db)
    out = import_duckdb_file(target, source_db)

    assert out["imported"] == {"products": 2, "orders": 3}
    attached = [r[0] for r in target.execute("SELECT database_name FROM duckdb_databases()").fetchall()]
    assert not [a for a in attached if a.startswith("datus_import_")]


@pytest.mark.acceptance
def test_skip_existing_leaves_target_untouched(source_db, target):
    target.execute("CREATE TABLE products (product_id BIGINT, note VARCHAR)")
    target.execute("INSERT INTO products VALUES (99, 'mine')")

    out = import_duckdb_file(target, source_db, mode="skip_existing")

    assert out["skipped"] == ["products"]
    assert "products" not in out["imported"]
    assert target.execute("SELECT note FROM products").fetchall() == [("mine",)]


@pytest.mark.acceptance
def test_subset_import(source_db, target):
    out = import_duckdb_file(target, source_db, tables=["products"])

    assert list(out["imported"]) == ["products"]
    assert target.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'orders'").fetchone()[0] == 0


@pytest.mark.acceptance
def test_unknown_table_is_reported(source_db, target):
    with pytest.raises(DatabaseImportError, match="nope"):
        import_duckdb_file(target, source_db, tables=["nope"])


@pytest.mark.acceptance
def test_missing_file(tmp_path, target):
    with pytest.raises(DatabaseImportError, match="not found"):
        import_duckdb_file(target, tmp_path / "absent.duckdb")


@pytest.mark.acceptance
def test_unknown_mode(source_db, target):
    with pytest.raises(DatabaseImportError, match="Unknown mode"):
        import_duckdb_file(target, source_db, mode="merge")


@pytest.mark.acceptance
def test_dependency_order_puts_parents_first():
    order = _dependency_order(["order_items", "orders", "products"], {"order_items": {"orders", "products"}})

    assert order.index("orders") < order.index("order_items")
    assert order.index("products") < order.index("order_items")


@pytest.mark.acceptance
def test_dependency_order_survives_a_cycle():
    order = _dependency_order(["a", "b"], {"a": {"b"}, "b": {"a"}})

    assert sorted(order) == ["a", "b"]


@pytest.mark.acceptance
def test_read_generator_meta(tmp_path):
    db = tmp_path / "datasource.duckdb"
    db.write_bytes(b"")
    (tmp_path / ".datasource.meta.json").write_text(json.dumps({"roles": {"orders": "fact"}}), encoding="utf-8")

    assert read_generator_meta(db) == {"roles": {"orders": "fact"}}
    assert read_generator_meta(tmp_path / "other.duckdb") is None


@pytest.mark.acceptance
def test_existing_table_probe_ignores_other_catalogs(source_db, target, tmp_path):
    """A connector may ATTACH another catalog (an Iceberg REST catalog, a staging database).

    Counting its tables as "already here" made skip_existing skip a table the target does not have,
    and report success while importing nothing.
    """
    other = tmp_path / "other.duckdb"
    con = duckdb.connect(str(other))
    con.execute("CREATE TABLE products (x BIGINT)")
    con.close()
    target.execute(f"ATTACH '{other}' AS lake (READ_ONLY)")

    out = import_duckdb_file(target, source_db, mode="skip_existing")

    assert out["skipped"] == []
    assert out["imported"] == {"products": 2, "orders": 3}


@pytest.mark.acceptance
def test_dependency_order_ignores_out_of_scope_parents():
    """A subset import must not fall into the cycle branch because a parent was left out - the
    tables that do have an in-scope parent still need parents-first ordering."""
    order = _dependency_order(["order_items", "orders"], {"order_items": {"orders", "products"}})

    assert order.index("orders") < order.index("order_items")


@pytest.mark.acceptance
def test_refused_table_is_not_reported_as_degraded(target, tmp_path):
    """`degraded` means "imported, constraints lost". A table that was not imported at all is a
    different outcome and the caller has to be able to tell them apart."""
    src = tmp_path / "odd.duckdb"
    con = duckdb.connect(str(src))
    con.execute('CREATE TABLE "weird-name" (id BIGINT)')
    con.execute('INSERT INTO "weird-name" VALUES (1)')
    con.close()

    out = import_duckdb_file(target, src)

    assert out["imported"] == {}
    assert out["degraded"] == []
    assert out["refused"] and "weird-name" in out["refused"][0]


class TestImportTool:
    """The tool layer around ``import_duckdb_file``.

    The function itself is covered above; this covers the wrapper an agent actually calls - path
    resolution, the DuckDB-only gate, and the shape of what comes back.
    """

    @pytest.fixture
    def tool(self, tmp_path, monkeypatch):
        from datus.tools.db_tools.config import DuckDBConfig
        from datus.tools.db_tools.duckdb_connector import DuckdbConnector
        from datus.tools.func_tool.database import DBFuncTool

        monkeypatch.chdir(tmp_path)
        connector = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "target.duckdb")))
        return DBFuncTool(connector)

    @pytest.fixture
    def build_db(self, tmp_path):
        """A generated database sitting where the skill puts it, relative to the workspace."""
        build = tmp_path / "data" / "_build"
        build.mkdir(parents=True)
        con = duckdb.connect(str(build / "datasource.duckdb"))
        con.execute(SOURCE_DDL)
        con.execute("INSERT INTO products VALUES (1, 'SKU-1', 'Widget', 9.90)")
        con.execute("INSERT INTO orders VALUES (10, 1, 9.90)")
        con.execute("COMMENT ON TABLE orders IS 'Order header'")
        con.close()
        return "data/_build/datasource.duckdb"

    @pytest.mark.acceptance
    def test_import_reports_what_landed(self, tool, build_db):
        result = tool.import_database_file(path=build_db)

        assert result.success == 1
        assert result.result["tables"] == {"products": 1, "orders": 1}
        assert result.result["table_count"] == 2
        assert result.result["total_rows"] == 2
        assert "degraded" not in result.result
        with tool.connector.exclusive_connection() as con:
            assert con.execute("SELECT count(*) FROM orders").fetchone()[0] == 1

    @pytest.mark.acceptance
    def test_import_preserves_keys_through_the_tool(self, tool, build_db):
        tool.import_database_file(path=build_db)

        with tool.connector.exclusive_connection() as con:
            found = {
                (t, k)
                for t, k in con.execute(
                    "SELECT table_name, constraint_type FROM duckdb_constraints() "
                    "WHERE database_name = current_database()"
                ).fetchall()
            }
        assert ("products", "PRIMARY KEY") in found
        assert ("orders", "FOREIGN KEY") in found

    @pytest.mark.acceptance
    def test_subset_and_mode_are_passed_through(self, tool, build_db):
        result = tool.import_database_file(path=build_db, tables=["products"], mode="replace")

        assert list(result.result["tables"]) == ["products"]

    @pytest.mark.acceptance
    def test_missing_file_is_reported(self, tool):
        result = tool.import_database_file(path="data/_build/absent.duckdb")

        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    @pytest.mark.acceptance
    def test_unknown_table_is_reported(self, tool, build_db):
        result = tool.import_database_file(path=build_db, tables=["nope"])

        assert result.success == 0
        assert "nope" in (result.error or "")

    @pytest.mark.acceptance
    def test_refused_table_is_surfaced_to_the_caller(self, tool, tmp_path):
        """A table left out has to reach the agent, or it silently ships an incomplete datasource."""
        build = tmp_path / "data" / "_build"
        build.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(build / "odd.duckdb"))
        con.execute('CREATE TABLE "weird-name" (id BIGINT)')
        con.close()

        result = tool.import_database_file(path="data/_build/odd.duckdb")

        assert result.success == 1
        assert result.result["refused"]
        assert "weird-name" in result.result["refused"][0]
        assert "NOT imported" in result.result["note"]

    @pytest.mark.acceptance
    def test_refused_and_degraded_notes_both_survive(self, tool, tmp_path):
        """They are different problems with different fixes; one must not overwrite the other.

        `weird-name` is refused outright, and `child`'s foreign key to it then cannot be created,
        so the same import produces both outcomes.
        """
        build = tmp_path / "data" / "_build"
        build.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(build / "both.duckdb"))
        con.execute('CREATE TABLE "weird-name" (id BIGINT PRIMARY KEY)')
        con.execute('INSERT INTO "weird-name" VALUES (1)')
        con.execute('CREATE TABLE child (cid BIGINT PRIMARY KEY, pid BIGINT REFERENCES "weird-name"(id))')
        con.execute("INSERT INTO child VALUES (10, 1)")
        con.close()

        result = tool.import_database_file(path="data/_build/both.duckdb")

        assert result.result["refused"], "the unsafe name must be reported"
        assert result.result["degraded"], "the child lost its foreign key"
        assert "NOT imported" in result.result["note"]
        assert "without their constraints" in result.result["note"]
