import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from datus.cli.screen.catalog_screen import CatalogScreen


@pytest.mark.parametrize(
    "capabilities",
    [
        {"catalog", "schema"},
        {"catalog", "database", "schema"},
    ],
)
def test_catalog_capabilities_always_build_catalog_first_tree(capabilities):
    screen = object.__new__(CatalogScreen)
    tree = MagicMock()
    helper = MagicMock()
    screen.query_one = MagicMock(side_effect=[tree, helper])
    screen.db_connector = MagicMock()
    screen.db_type = "flexdb"
    screen.database_name = ""
    screen._supports = lambda namespace: namespace in capabilities
    screen._load_catalogs_lazy = MagicMock()
    screen._load_databases_lazy = MagicMock()

    screen._build_catalog_tree()

    screen._load_catalogs_lazy.assert_called_once_with(tree)
    screen._load_databases_lazy.assert_not_called()


@pytest.mark.parametrize(
    ("node_data", "expected_catalog", "expected_database"),
    [
        ({"type": "catalog", "name": "catalog_a"}, "catalog_a", ""),
        ({"type": "database", "name": "database_a", "catalog": "catalog_a"}, "catalog_a", "database_a"),
    ],
)
def test_schema_loading_preserves_catalog_and_database_coordinates(
    node_data,
    expected_catalog,
    expected_database,
):
    screen = object.__new__(CatalogScreen)
    screen.db_connector = MagicMock()
    screen.db_connector.get_schemas.return_value = ["analytics"]
    parent_node = MagicMock()
    parent_node.label = node_data["name"]
    parent_node.data = node_data

    screen._load_schemas_for_database(parent_node)

    screen.db_connector.switch_context.assert_called_once_with(
        catalog_name=expected_catalog,
        database_name=expected_database,
    )
    screen.db_connector.get_schemas.assert_called_once_with(
        catalog_name=expected_catalog,
        database_name=expected_database,
    )
    parent_node.add.assert_called_once_with(
        "📂 analytics",
        data={
            "type": "schema",
            "name": "analytics",
            "database": expected_database,
            "catalog": expected_catalog,
        },
    )


def test_catalog_screen_builds_generic_record_from_table_semantic_profile():
    screen = object.__new__(CatalogScreen)
    record = screen._semantic_record_from_table_profile(
        {
            "format": "osi",
            "table_name": "orders",
            "semantic_model_name": "shop",
            "dataset_name": "orders",
            "data_source_name": "",
            "description": "Orders dataset",
            "ai_context_json": '{"instructions":"Use this dataset for order analytics."}',
            "columns_json": (
                "["
                '{"name":"order_id","expr":"order_id","role":"primary_key","description":"Order key"},'
                '{"name":"order_date","expr":"order_date","role":"time_dimension","description":"Order date"},'
                '{"name":"segment","expr":"segment","role":"dimension","description":"Customer segment"},'
                '{"name":"amount","expr":"amount","role":"measure","description":"Order amount"}'
                "]"
            ),
            "relationships_json": '[{"name":"orders_to_customers","to_dataset":"customers"}]',
        }
    )

    assert record["format"] == "osi"
    assert record["dataset_name"] == "orders"
    assert record["ai_context"]["instructions"] == "Use this dataset for order analytics."
    assert [item["name"] for item in record["identifiers"]] == ["order_id"]
    assert [item["name"] for item in record["dimensions"]] == ["order_date", "segment"]
    assert record["relationships"][0]["name"] == "orders_to_customers"
    assert "filters" not in record


def test_catalog_screen_readonly_panel_shows_profile_fields_without_measures():
    screen = object.__new__(CatalogScreen)
    group = screen._render_readonly_panel(
        {
            "format": "metricflow",
            "semantic_model_name": "orders_source",
            "data_source_name": "orders_source",
            "description": "Orders data source",
            "ai_context": {"instructions": "Use this data source for sales analytics."},
            "identifiers": [{"name": "order_id"}],
            "dimensions": [{"name": "order_date"}],
            "relationships": [{"name": "orders_to_customers"}],
            "measures": [{"name": "amount"}],
        }
    )

    console = Console(record=True, width=180, file=io.StringIO())
    console.print(group)
    rendered = console.export_text()

    assert "Data Source" in rendered
    assert "AI Context" in rendered
    assert "Relationships" in rendered
    assert "Filters" not in rendered
    assert "Measures" not in rendered
    assert "amount" not in rendered


def test_catalog_screen_nested_semantic_table_uses_readable_column_order():
    screen = object.__new__(CatalogScreen)
    table = screen._create_nested_table_for_json(
        [
            {
                "description": "Activity key",
                "expr": "ac_code",
                "name": "activity",
                "role": "primary_key",
                "type": "PRIMARY",
            },
            {
                "description": "Start date",
                "expr": "start_date",
                "name": "start_date",
                "role": "dimension",
                "time_granularity": "DAY",
                "type": "TIME",
            },
        ]
    )

    headers = [column.header for column in table.columns]
    assert headers == ["name", "expr", "role", "type", "time_granularity", "description"]
