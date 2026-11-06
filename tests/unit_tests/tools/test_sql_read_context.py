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


@pytest.mark.parametrize("shadow_schema", ["main", "temp", "other"])
@pytest.mark.parametrize("shadow_kind", ["TABLE", "VIEW"])
def test_sqlite_metadata_pragmas_cannot_be_shadowed_by_user_objects(tmp_path, shadow_schema, shadow_kind):
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    connector = SQLiteConnector(SQLiteConfig(db_path=str(tmp_path / "warehouse.sqlite")))
    try:
        connector.connect()
        connection = connector.connection
        connection.execute("ATTACH DATABASE ':memory:' AS other")
        for schema in ("main", "temp", "other"):
            connection.execute(f'CREATE TABLE "{schema}".orders (id INTEGER, phone TEXT)')
        for name in ("pragma_table_list", "pragma_table_xinfo", "pragma_database_list"):
            definition = "(id INTEGER)" if shadow_kind == "TABLE" else "AS SELECT 999 AS id"
            connection.execute(f'CREATE {shadow_kind} "{shadow_schema}"."{name}" {definition}')
        context = SqlReadContext(connector, "warehouse")
        for schema in ("main", "temp", "other"):
            table = resolve(context, f"{schema}.orders")
            assert table["physical"] is True
            assert table["object"]["database"] == schema
            assert table["columns"] == [
                {"name": "id", "sql_type": "INTEGER", "is_text": False},
                {"name": "phone", "sql_type": "TEXT", "is_text": True},
            ]
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


def test_postgres_unquoted_identifiers_fold_only_ascii_letters():
    connector = SimpleNamespace(
        dialect="postgres",
        execute_query=lambda *args, **kwargs: SimpleNamespace(
            success=True,
            sql_return=[{"database_name": "analytics", "schema_name": "public", "search_path": ["public"]}],
        ),
    )
    context = SqlReadContext(connector, "warehouse")
    assert context.fixed_object(parts("Keys"))["table"] == "Keys"
    assert context.fixed_object(parts("KEYS"))["table"] == "keys"
    assert context.column_name({"name": "KEY"}) == "Key"
    assert context.column_name({"name": "KEY", "quoted": True}) == "KEY"


@pytest.mark.parametrize("listing", ["tables", "views"])
@pytest.mark.parametrize(
    "unrelated", ["public.orders", "analytics.public.orders", "archive.sales.orders.history", "unclosed( identifier"]
)
def test_unrelated_raw_metadata_names_do_not_abort_resolution(listing, unrelated):
    connector = SimpleNamespace(
        dialect="postgres",
        execute_query=lambda *_, **__: SimpleNamespace(
            success=True,
            sql_return=[{"database_name": "analytics", "schema_name": "public", "search_path": ["public"]}],
        ),
        get_tables=lambda **_: [unrelated, "orders"] if listing == "tables" else ["orders"],
        get_views=lambda **_: [unrelated] if listing == "views" else [],
        get_schema=lambda **_: [{"name": "id", "type": "INTEGER"}],
    )
    assert resolve(SqlReadContext(connector, "warehouse"), "orders")["object"]["table"] == "orders"


@pytest.mark.parametrize("kind", ["table", "view"])
def test_scoped_metadata_preserves_dots_in_the_stored_object_name(kind):
    connector = SimpleNamespace(
        dialect="postgres",
        execute_query=lambda *_, **__: SimpleNamespace(
            success=True,
            sql_return=[{"database_name": "analytics", "schema_name": "public", "search_path": ["public"]}],
        ),
        get_tables=lambda **_: ["public.orders"],
        get_views=lambda **_: ["public.orders"] if kind == "view" else [],
        get_schema=lambda **_: [{"name": "id", "type": "INTEGER"}],
    )
    context = SqlReadContext(connector, "warehouse")
    with pytest.raises(ValueError, match="could not be resolved"):
        resolve(context, "orders")
    if kind == "view":
        with pytest.raises(ValueError, match="opaque database views"):
            resolve(context, '"public.orders"')
    else:
        assert resolve(context, '"public.orders"')["object"]["table"] == "public.orders"


@pytest.mark.parametrize("kind", ["table", "view"])
def test_hana_qualified_metadata_preserves_quoted_dots_and_quotes(monkeypatch, kind):
    monkeypatch.setattr("datus.tools.sql_read_context.table_name_field_order", lambda _: ["schema_name", "table_name"])
    connector = SimpleNamespace(
        dialect="hana",
        get_current_context=lambda: {"schema_name": "Sales.X"},
        get_tables=lambda **_: ['"Sales.X"."Orders.Y""Z"'],
        get_views=lambda **_: ['"Sales.X"."Orders.Y""Z"'] if kind == "view" else [],
        get_schema=lambda **_: [{"name": "Id", "type": "INTEGER"}],
    )
    context = SqlReadContext(connector, "warehouse")
    reference = '"Orders.Y""Z"'
    if kind == "view":
        with pytest.raises(ValueError, match="opaque database views"):
            resolve(context, reference)
    else:
        assert resolve(context, reference)["object"]["table"] == 'Orders.Y"Z'


@pytest.mark.parametrize("dialect", ["postgres", "oracle", "hana", "snowflake"])
@pytest.mark.parametrize("stored", ["Orders", "Orders.X"])
def test_metadata_listings_preserve_stored_identifier_case(monkeypatch, dialect, stored):
    monkeypatch.setattr("datus.tools.sql_read_context.table_name_field_order", lambda _: ["schema_name", "table_name"])
    connector = SimpleNamespace(
        dialect=dialect,
        get_current_context=lambda: {"schema_name": "sales"},
        execute_query=lambda *_, **__: SimpleNamespace(
            success=True,
            sql_return=[{"database_name": "analytics", "schema_name": "sales", "search_path": ["sales"]}],
        ),
        get_tables=lambda **_: [f'"sales"."{stored}"' if dialect == "hana" else stored],
        get_views=lambda **_: [],
        get_schema=lambda **_: [{"name": "Id", "type": "INTEGER"}],
    )
    table = resolve(SqlReadContext(connector, "warehouse"), '"sales"."' + stored + '"')
    assert table["object"]["table"] == stored
    assert table["columns"][0]["name"] == "Id"


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
        get_tables=lambda **_: ['"SALES"."ORDERS"' if dialect == "hana" else "ORDERS"],
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


@pytest.mark.parametrize("reference", ["orders", "main.orders", "MAIN.ORDERS"])
def test_temporary_table_shadows_persistent_table(db, reference):
    db.execute("INSERT INTO orders(id) VALUES (1)")
    db.execute("CREATE TEMP TABLE orders(secret VARCHAR)")
    db.execute("INSERT INTO temp.main.orders VALUES ('temporary value')")
    context = SqlReadContext(Connector(db), "warehouse")
    table = resolve(context, reference)
    assert table["object"]["database"] == "temp"
    assert [column["name"] for column in table["columns"]] == ["secret"]
    assert (
        db.execute(f"SELECT * FROM {table['sql_reference']}").fetchall()
        == db.execute(f"SELECT * FROM {reference}").fetchall()
        == [("temporary value",)]
    )
    persistent = resolve(context, "memory.main.orders")
    assert persistent["object"]["database"] == "memory"
    assert db.execute(f"SELECT id FROM {persistent['sql_reference']}").fetchall() == [(1,)]


def test_attached_database_switch_requires_fresh_binding(db):
    context = SqlReadContext(Connector(db), "warehouse")
    db.execute("ATTACH ':memory:' AS other; CREATE SCHEMA other.sales; USE other.sales")
    db.execute("CREATE TABLE orders(id BIGINT)")
    with pytest.raises(ValueError, match="namespace changed"):
        resolve(context, "orders")
    rebound = SqlReadContext(Connector(db), "warehouse")
    assert resolve(rebound, "orders")["object"]["database"] == "other"
    assert resolve(rebound, "memory.main.orders")["object"]["database"] == "memory"


@pytest.mark.parametrize("reference", ["other.orders", '"OTHER"."ORDERS"'])
def test_duckdb_two_part_catalog_source_resolves_the_attached_physical_table(db, reference):
    db.execute("ATTACH ':memory:' AS other; CREATE TABLE other.main.orders AS SELECT 42 AS tenant_id")
    context = SqlReadContext(Connector(db), "warehouse")
    table = resolve(context, reference)
    assert table["object"]["database"] == "other"
    assert table["object"]["schema"] == "main"
    assert (
        db.execute(f"SELECT * FROM {table['sql_reference']}").fetchall()
        == db.execute("SELECT * FROM " + reference).fetchall()
        == [(42,)]
    )
    # A short profile target still follows documented right alignment.
    assert context.fixed_object(parts(reference))["schema"] == "other"
    assert context.fixed_object(parts(reference))["database"] == "memory"


def test_duckdb_two_part_catalog_schema_ambiguity_is_refused(db):
    db.execute("ATTACH ':memory:' AS other; CREATE SCHEMA other.sales; USE other.sales")
    db.execute("CREATE TABLE other.main.orders AS SELECT 42 AS id")
    db.execute("CREATE SCHEMA other.other")
    with pytest.raises(duckdb.BinderException, match="Ambiguous reference to catalog or schema"):
        db.execute("SELECT * FROM other.orders")
    context = SqlReadContext(Connector(db), "warehouse")
    with pytest.raises(ValueError, match="ambiguous DuckDB catalog or schema"):
        resolve(context, "other.orders")
    assert resolve(context, "other.main.orders")["object"]["schema"] == "main"


@pytest.mark.parametrize("change", ["attach", "detach"])
def test_duckdb_catalog_topology_changes_require_fresh_binding(db, change):
    db.execute("ATTACH ':memory:' AS other")
    context = SqlReadContext(Connector(db), "warehouse")
    db.execute("ATTACH ':memory:' AS extra" if change == "attach" else "DETACH other")
    with pytest.raises(ValueError, match="namespace changed"):
        context.verify_context()
    # A fresh context can observe the new topology normally.
    SqlReadContext(Connector(db), "warehouse").verify_context()


@pytest.mark.parametrize("path", ["other.sales,memory.main", "memory.main,other.sales"])
def test_duckdb_two_part_catalog_source_follows_that_catalog_search_path(db, path):
    db.execute("ATTACH ':memory:' AS other; CREATE SCHEMA other.sales")
    db.execute("CREATE TABLE other.main.orders AS SELECT 1 AS id")
    db.execute("CREATE TABLE other.sales.orders AS SELECT 2 AS id")
    db.execute("SET search_path='" + path + "'")
    context = SqlReadContext(Connector(db), "warehouse")
    table = resolve(context, "other.orders")
    assert table["object"]["schema"] == "sales"
    assert (
        db.execute(f"SELECT * FROM {table['sql_reference']}").fetchall()
        == db.execute("SELECT * FROM other.orders").fetchall()
        == [(2,)]
    )


@pytest.mark.parametrize("dialect", ["hive", "spark", "trino"])
@pytest.mark.parametrize("reference", ["ORDERS", '"ORDERS"'])
def test_case_insensitive_sources_and_policy_targets_share_canonical_identifiers(monkeypatch, dialect, reference):
    order = ["catalog_name", "schema_name", "table_name"] if dialect == "trino" else ["database_name", "table_name"]
    monkeypatch.setattr("datus.tools.sql_read_context.table_name_field_order", lambda _: order)

    def get_schema(*, table_name, **namespace):
        # Trino's information_schema predicate is case-sensitive even though
        # identifiers in business SQL are not. Normalize before metadata IO.
        assert table_name == "orders"
        assert all(value == value.lower() for value in namespace.values() if isinstance(value, str))
        return [{"name": "phone", "type": "VARCHAR(30)"}]

    connector = SimpleNamespace(
        dialect=dialect,
        get_current_context=lambda: {"catalog_name": "Memory", "database_name": "Sales", "schema_name": "Sales"},
        get_tables=lambda **_: ["orders"],
        get_views=lambda **_: [],
        get_schema=get_schema,
        get_identifier_case_sensitive=lambda: False,
        get_default_identifier_case_sensitive=lambda: False,
        get_relation_info=lambda **_: {"name": "orders", "type": "MANAGED"},
    )
    context = SqlReadContext(connector, "warehouse")
    assert context.execution_context["source_dialect"] == dialect
    table = resolve(context, reference)
    assert table["object"] == context.fixed_object(parts(reference))
    assert table["object"]["table"] == "orders"
    assert context.column_name({"name": "PHONE", "quoted": True}) == "phone"


def test_views_fail_closed(db):
    db.execute("CREATE VIEW hidden AS SELECT * FROM orders")
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(Connector(db), "warehouse"), "hidden")


def test_duckdb_unqualified_source_uses_implicit_main_fallback(db):
    db.execute("CREATE SCHEMA sales; SET search_path='sales'")
    context = SqlReadContext(Connector(db), "warehouse")
    assert db.execute("SELECT * FROM orders").fetchall() == []
    assert resolve(context, "orders")["object"]["schema"] == "main"
    with pytest.raises(ValueError, match="could not be resolved"):
        resolve(context, "sales.orders")
    db.execute("CREATE VIEW sales.orders AS SELECT * FROM main.orders")
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(Connector(db), "warehouse"), "orders")


def test_sqlite_table_can_share_its_name_with_a_trigger(tmp_path):
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    connector = SQLiteConnector(SQLiteConfig(db_path=str(tmp_path / "triggers.sqlite")))
    try:
        connector.connect()
        connector.connection.executescript(
            "CREATE TABLE orders(id INTEGER); CREATE TRIGGER orders AFTER INSERT ON orders BEGIN SELECT 1; END;"
        )
        table = resolve(SqlReadContext(connector, "warehouse"), "orders")
        assert table["object"]["table"] == "orders"
        assert table["columns"][0]["name"] == "id"
    finally:
        connector.close()


def test_spark_live_case_mode_is_bound_and_rechecked(monkeypatch):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    state = {"value": True}
    connector = SimpleNamespace(
        dialect="spark",
        get_current_context=lambda: {"database_name": "default"},
        get_identifier_case_sensitive=lambda: state["value"],
        get_default_identifier_case_sensitive=lambda: False,
    )
    context = SqlReadContext(connector, "warehouse")
    assert context.execution_context["case_sensitive_table_names"] is True
    assert context.canonical_name("Orders") == "Orders"
    assert context.canonical_object_name("Orders") == "orders"
    assert context.fixed_object(parts("DEFAULT.ORDERS"))["database"] == "default"
    assert context.fixed_object(parts("DEFAULT.ORDERS"))["table"] == "orders"
    with pytest.raises(ValueError, match="server-owned identifier binding"):
        context.canonical_object_name("库存")
    with pytest.raises(ValueError, match="cannot reproduce Spark"):
        context.verify_execution_target("hive")
    state["value"] = False
    with pytest.raises(ValueError, match="namespace changed"):
        context.verify_context()
    assert SqlReadContext(connector, "warehouse").canonical_name("Orders") == "orders"
    for name in ["İ", "i\u0307", "Keys"]:
        with pytest.raises(ValueError, match="server-owned identifier binding"):
            SqlReadContext(connector, "warehouse").canonical_name(name)
    state["value"] = "unknown"
    with pytest.raises(ValueError, match="case mode is unavailable"):
        SqlReadContext(connector, "warehouse")


def test_spark_resolves_the_requested_relation_without_scanning_unrelated_names(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    info = Mock(return_value={"name": "orders", "type": "MANAGED"})
    connector = SimpleNamespace(
        dialect="spark",
        get_current_context=lambda: {"database_name": "default"},
        get_identifier_case_sensitive=lambda: False,
        get_default_identifier_case_sensitive=lambda: False,
        get_relation_info=info,
        get_tables=Mock(side_effect=AssertionError("must not enumerate unrelated 库存 metadata")),
        get_views=Mock(side_effect=AssertionError("must not enumerate unrelated 库存 metadata")),
        get_schema=lambda **_: [{"name": "id", "type": "INT"}],
    )
    context = SqlReadContext(connector, "warehouse")
    assert resolve(context, "ORDERS")["object"]["table"] == "orders"
    info.assert_called_once_with(table_name="orders", database_name="default", unqualified=True)
    assert resolve(context, "default.orders")["object"]["table"] == "orders"
    assert info.call_args.kwargs["unqualified"] is False
    info.return_value = {"name": "orders", "type": "UNKNOWN"}
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(connector, "warehouse"), "orders")
    info.return_value = {"name": "Keys", "type": "MANAGED"}
    with pytest.raises(ValueError, match="server-owned identifier binding"):
        resolve(SqlReadContext(connector, "warehouse"), "keys")


def test_dosi_refuses_a_spark_server_default_it_cannot_reproduce(monkeypatch):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    connector = SimpleNamespace(
        dialect="spark",
        get_current_context=lambda: {"database_name": "default"},
        get_identifier_case_sensitive=lambda: False,
        get_default_identifier_case_sensitive=lambda: True,
    )
    with pytest.raises(ValueError, match="cannot reproduce Spark"):
        SqlReadContext(connector, "warehouse").verify_execution_target("hive")


@pytest.mark.parametrize("identifier", ["orders_table_name", "table_name", "o'table_name"])
def test_duckdb_view_name_literals_cannot_fall_through_to_a_different_table(db, identifier):
    quoted = '"' + identifier.replace('"', '""') + '"'
    db.execute(f"CREATE TABLE main.{quoted}(id INTEGER)")
    db.execute(f"INSERT INTO main.{quoted} VALUES (99)")
    db.execute("CREATE SCHEMA sales; SET search_path='sales,main'")
    db.execute(f"CREATE VIEW sales.{quoted} AS SELECT 42 AS id")
    assert db.execute(f"SELECT * FROM {quoted}").fetchall() == [(42,)]
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(Connector(db), "warehouse"), quoted)


def test_duckdb_view_namespace_literals_are_preserved(db):
    db.execute("ATTACH ':memory:' AS catalog_table_name; CREATE SCHEMA catalog_table_name.schema_table_name")
    db.execute("CREATE VIEW catalog_table_name.schema_table_name.hidden AS SELECT 42 AS id")
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(Connector(db), "warehouse"), "catalog_table_name.schema_table_name.hidden")


def test_view_classification_wins_when_adapter_also_lists_it_as_a_table(monkeypatch):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    connector = SimpleNamespace(
        dialect="mysql",
        get_current_context=lambda: {"database_name": "analytics"},
        execute_query=lambda *_, **__: SimpleNamespace(success=True, sql_return=[{"lower_case_table_names": 0}]),
        get_tables=lambda **_: ["orders_view"],
        get_views=lambda **_: ["orders_view"],
    )
    with pytest.raises(ValueError, match="opaque database views"):
        resolve(SqlReadContext(connector, "warehouse"), "orders_view")


@pytest.mark.parametrize("dialect", ["mysql", "tidb"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_mysql_table_case_matching_uses_the_live_server_setting(monkeypatch, dialect, mode):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    connector = SimpleNamespace(
        dialect=dialect,
        get_current_context=lambda: {"database_name": "analytics"},
        execute_query=lambda *_, **__: SimpleNamespace(success=True, sql_return=[{"lower_case_table_names": mode}]),
        get_tables=lambda **_: ["orders"],
        get_views=lambda **_: [],
        get_schema=lambda **_: [{"name": "Phone", "type": "VARCHAR(30)"}],
    )
    context = SqlReadContext(connector, "warehouse")
    assert context.execution_context["case_sensitive_table_names"] is (mode == 0)
    assert resolve(context, "orders")["object"]["table"] == "orders"
    if mode:
        assert resolve(context, "ORDERS")["object"]["table"] == "orders"
        assert context.fixed_object(parts("ORDERS"))["table"] == "orders"
    else:
        with pytest.raises(ValueError, match="could not be resolved"):
            resolve(context, "ORDERS")
    assert context.column_name({"name": "Phone", "quoted": True}) == "Phone"


def test_mysql_non_ascii_object_case_is_normalized_by_the_server(monkeypatch):
    monkeypatch.setattr(
        "datus.tools.sql_read_context.table_name_field_order", lambda _: ["database_name", "table_name"]
    )
    queries = []

    def query(sql, **_):
        queries.append(sql)
        row = {"folded_name": "ä"} if "folded_name" in sql else {"lower_case_table_names": 1}
        return SimpleNamespace(success=True, sql_return=[row])

    connector = SimpleNamespace(
        dialect="mysql", get_current_context=lambda: {"database_name": "analytics"}, execute_query=query
    )
    context = SqlReadContext(connector, "warehouse")
    assert context.canonical_name("Ä") == "ä"
    assert context.canonical_name("Ä") == "ä"
    assert sum("CONVERT(0xc384 USING utf8mb4)" in sql for sql in queries) == 1


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
        get_tables=lambda **_: ["orders"],
        get_views=lambda **_: [],
        get_schema=lambda **_: [{"name": "phone", "type": native_type}],
    )
    context = SqlReadContext(connector, "warehouse")
    column = resolve(context, "orders")["columns"][0]
    assert column == {"name": "phone", "sql_type": native_type, "is_text": is_text}
