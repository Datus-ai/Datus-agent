from datus.utils.benchmark_utils import _default_sql_dialect, collect_sql_tables, compute_table_matches


def test_compute_table_matches_backwards_alignment():
    actual_tables = ["sales.dim_customers", "sales.fact_orders"]
    expected_tables = ["foo", ".fact_orders"]

    assert compute_table_matches(actual_tables, expected_tables) == [".fact_orders"]


def test_compute_table_matches_stops_at_first_empty_entry():
    actual_tables = ["public.orders", "", "warehouse.inventory"]
    expected_tables = ["foo", "warehouse.inventory"]

    assert compute_table_matches(actual_tables, expected_tables) == ["warehouse.inventory"]


def test_compute_table_matches_returns_empty_when_trailing_entry_blank():
    actual_tables = ["warehouse.shipments", "  "]
    expected_tables = ["warehouse.shipments"]

    assert compute_table_matches(actual_tables, expected_tables) == []


def test_compute_table_matches_handles_simple_and_qualified_equivalence():
    actual_tables = ["analytics.public.orders"]
    expected_tables = ["orders"]

    assert compute_table_matches(actual_tables, expected_tables) == ["orders"]


def test_collect_sql_tables_default_dialect():
    """Cover line 295: dialect defaults to snowflake when None."""
    tables = collect_sql_tables("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
    assert "users" in tables
    assert "orders" in tables


def test_collect_sql_tables_empty():
    assert collect_sql_tables(None) == []
    assert collect_sql_tables("") == []


def test_default_sql_dialect():
    """Cover line 1870: non-bird benchmark returns snowflake."""
    assert _default_sql_dialect("bird_dev") == "sqlite"
    assert _default_sql_dialect("spider2") == "snowflake"
    assert _default_sql_dialect("other") == "snowflake"
