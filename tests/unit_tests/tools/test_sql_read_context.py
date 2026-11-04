"""Physical name binding uses the live session, including attached databases."""

from types import SimpleNamespace

import duckdb
import pytest
import sqlglot

from datus.tools.sql_read_context import SqlReadContext


class Connector:
    dialect = "duckdb"
    # Deliberately stale: the resolver must query the live connection.
    database_name = "configured_default"
    schema_name = "configured_schema"
    catalog_name = ""

    def __init__(self, db):
        self.db = db

    def execute_query(self, sql, **kwargs):
        return SimpleNamespace(success=True, sql_return=self.db.execute(sql).fetch_arrow_table())


def parts(reference):
    table = sqlglot.parse_one(f"SELECT * FROM {reference}", read="duckdb").find(sqlglot.exp.Table)
    return [{"name": part.name, "quoted": bool(part.args.get("quoted"))} for part in table.parts]


@pytest.fixture
def db():
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE orders(id BIGINT, phone VARCHAR)")
        yield connection


def resolve(context, reference):
    return context.resolve_source_parts(parts(reference), context.execution_context, parser=parts)


def test_sqlite_uses_live_attachments_and_native_column_metadata(tmp_path):
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    connector = SQLiteConnector(SQLiteConfig(db_path=str(tmp_path / "warehouse.sqlite"), database_name="stale"))
    try:
        connector.connect()
        connector.connection.execute('CREATE TABLE "Orders.X" ("Phone.X" TEXT, amount REAL)')
        context = SqlReadContext(connector, "warehouse")
        table = resolve(context, 'main."Orders.X"')
        assert context.execution_context["database"] == "main"
        assert table["object"] == {
            "datasource": "warehouse",
            "catalog": None,
            "database": "main",
            "schema": None,
            "table": "orders.x",
        }
        assert table["columns"] == [
            {"name": "Phone.X", "sql_type": "TEXT", "is_text": True},
            {"name": "amount", "sql_type": "REAL", "is_text": False},
        ]
        assert table["sql_reference"] == '"main"."orders.x"'
        connector.connection.execute("ATTACH DATABASE ':memory:' AS other")
        with pytest.raises(ValueError, match="namespace changed"):
            context.verify_context()
    finally:
        connector.close()


def test_reads_actual_namespace_and_returns_canonical_metadata(db):
    context = SqlReadContext(Connector(db), "warehouse")
    table = resolve(context, "ORDERS")
    assert table["object"] == {
        "datasource": "warehouse",
        "catalog": None,
        "database": "memory",
        "schema": "main",
        "table": "orders",
    }
    assert table["sql_reference"] == '"memory"."main"."orders"'
    assert table["columns"][1] == {"name": "phone", "sql_type": "VARCHAR", "is_text": True}


def test_policy_targets_fill_omitted_prefixes_from_live_namespace(db):
    context = SqlReadContext(Connector(db), "warehouse")
    expected = {
        "datasource": "warehouse",
        "catalog": None,
        "database": "memory",
        "schema": "main",
        "table": "orders",
    }
    assert context.fixed_object(parts("orders")) == expected
    assert context.fixed_object(parts("main.orders")) == expected
    assert context.fixed_object(parts("sales.orders")) == {**expected, "schema": "sales"}
    assert context.fixed_object(parts("OTHER.SALES.ORDERS"))["database"] == "other"


def test_policy_defaults_rebind_after_namespace_switch_and_full_targets_stay_fixed(db):
    context = SqlReadContext(Connector(db), "warehouse")
    original = context.fixed_object(parts("orders"))
    db.execute("ATTACH ':memory:' AS other; CREATE SCHEMA other.sales; USE other.sales")
    with pytest.raises(ValueError, match="namespace changed"):
        context.fixed_object(parts("orders"))
    rebound = SqlReadContext(Connector(db), "warehouse")
    assert rebound.fixed_object(parts("orders")) == {**original, "database": "other", "schema": "sales"}
    assert rebound.fixed_object(parts("memory.main.orders")) == original


def test_policy_defaults_require_an_available_current_namespace():
    connector = SimpleNamespace(
        dialect="postgres",
        execute_query=lambda *args, **kwargs: SimpleNamespace(
            success=True,
            sql_return=[{"database_name": "analytics", "schema_name": None, "search_path": []}],
        ),
    )
    context = SqlReadContext(connector, "warehouse")
    with pytest.raises(ValueError, match="current execution schema is unavailable"):
        context.fixed_object(parts("orders"))
    assert context.fixed_object(parts("public.orders"))["schema"] == "public"


@pytest.mark.parametrize("dialect", ["starrocks", "doris"])
def test_policy_defaults_include_current_catalog(monkeypatch, dialect):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["catalog_name", "database_name", "table_name"]
    )
    connector = SimpleNamespace(
        dialect=dialect,
        get_current_context=lambda: {"catalog_name": "prod", "database_name": "analytics"},
    )
    context = SqlReadContext(connector, "warehouse")
    expected = {
        "datasource": "warehouse",
        "catalog": "prod",
        "database": "analytics",
        "schema": None,
        "table": "orders",
    }
    assert context.fixed_object(parts("orders")) == expected
    assert context.fixed_object(parts("sales.orders")) == {**expected, "database": "sales"}
    assert context.fixed_object(parts("other.sales.orders")) == {**expected, "catalog": "other", "database": "sales"}


@pytest.mark.parametrize("dialect", ["oracle", "hana"])
def test_schema_only_objects_ignore_connection_database_and_fold_unquoted_names(monkeypatch, dialect):
    monkeypatch.setattr("datus.tools.sql_read_context.table_name_field_order", lambda _: ["schema_name", "table_name"])
    connector = SimpleNamespace(
        dialect=dialect,
        config=SimpleNamespace(service_name="FREEPDB1", database="HXE"),
        get_current_context=lambda: {"database_name": "connection_target", "schema_name": "SALES"},
        get_tables=lambda **_: ["ORDERS"],
        get_views=lambda **_: [],
        get_schema=lambda **_: [{"name": "PHONE", "type": "NVARCHAR(255)"}],
    )
    context = SqlReadContext(connector, "warehouse")
    assert context.execution_context["database"] == ("FREEPDB1" if dialect == "oracle" else "HXE")
    table = resolve(context, "orders")
    assert table["object"] == context.fixed_object(parts("orders"))
    assert table["object"]["database"] is None
    assert table["object"]["schema"] == "SALES"
    assert table["object"]["table"] == "ORDERS"
    assert table["columns"][0]["is_text"] is True
    assert context.column_name({"name": "phone"}) == "PHONE"
    assert context.column_name({"name": "Phone", "quoted": True}) == "Phone"


def test_search_path_and_quoted_dots(db):
    db.execute('CREATE SCHEMA "a.b"; CREATE TABLE "a.b"."Orders.X" (id BIGINT)')
    db.execute("SET search_path='\"a.b\",main'")
    context = SqlReadContext(Connector(db), "warehouse")
    table = resolve(context, '"Orders.X"')
    assert table["object"]["schema"] == "a.b"
    assert table["object"]["table"] == "orders.x"
    assert db.execute(f"SELECT * FROM {table['sql_reference']}").fetchall() == []


def test_temporary_table_shadows_persistent_table(db):
    db.execute("CREATE TEMP TABLE orders(secret VARCHAR)")
    context = SqlReadContext(Connector(db), "warehouse")
    assert resolve(context, "orders")["object"]["database"] == "temp"
    assert resolve(context, "memory.main.orders")["object"]["database"] == "memory"


def test_attached_database_switch_requires_fresh_binding(db):
    context = SqlReadContext(Connector(db), "warehouse")
    db.execute("ATTACH ':memory:' AS other; CREATE SCHEMA other.sales; USE other.sales")
    db.execute("CREATE TABLE orders(id BIGINT)")
    with pytest.raises(ValueError, match="namespace changed"):
        resolve(context, "orders")
    rebound = SqlReadContext(Connector(db), "warehouse")
    assert resolve(rebound, "orders")["object"]["database"] == "other"
    assert resolve(rebound, "memory.main.orders")["object"]["database"] == "memory"


def test_views_fail_closed(db):
    db.execute("CREATE VIEW hidden AS SELECT * FROM orders")
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(Connector(db), "warehouse"), "hidden")


def test_view_classification_wins_when_adapter_also_lists_it_as_a_table(monkeypatch):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    connector = SimpleNamespace(
        dialect="mysql",
        get_current_context=lambda: {"database_name": "analytics"},
        get_tables=lambda **_: ["orders_view"],
        get_views=lambda **_: ["orders_view"],
    )
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(connector, "warehouse"), "orders_view")


def test_unicode_identifiers_do_not_use_unicode_case_folding(db):
    db.execute('CREATE TABLE "Ä"(x BIGINT); CREATE TABLE "ä"(y BIGINT)')
    context = SqlReadContext(Connector(db), "warehouse")
    assert resolve(context, '"Ä"')["columns"][0]["name"] == "x"
    assert resolve(context, '"ä"')["columns"][0]["name"] == "y"


def test_mismatched_execution_context_is_rejected(db):
    context = SqlReadContext(Connector(db), "warehouse")
    with pytest.raises(ValueError, match="does not match"):
        context.resolve_source_parts(parts("orders"), {}, parser=parts)


@pytest.mark.parametrize(
    "native_type,is_text",
    [
        ("Nullable(String)", True),
        ("LowCardinality(Nullable(String))", True),
        ("String", True),
        ("Nullable(Float64)", False),
        ("FixedString(20)", False),
    ],
)
def test_clickhouse_text_metadata_unwraps_nullable_and_dictionary_types(monkeypatch, native_type, is_text):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    connector = SimpleNamespace(
        dialect="clickhouse",
        get_current_context=lambda: {"database_name": "sales"},
        get_tables=lambda **_: ["sales.orders"],
        get_views=lambda **_: [],
        get_schema=lambda **_: [{"name": "phone", "type": native_type}],
    )
    context = SqlReadContext(connector, "warehouse")
    column = resolve(context, "orders")["columns"][0]
    assert column == {"name": "phone", "sql_type": native_type, "is_text": is_text}
