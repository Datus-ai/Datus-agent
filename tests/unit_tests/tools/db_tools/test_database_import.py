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
