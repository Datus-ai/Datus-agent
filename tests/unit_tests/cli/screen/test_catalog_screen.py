import io

from rich.console import Console

from datus.cli.screen.catalog_screen import CatalogScreen


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
