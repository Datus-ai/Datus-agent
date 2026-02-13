from datus.utils.benchmark_utils import compute_table_matches


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
