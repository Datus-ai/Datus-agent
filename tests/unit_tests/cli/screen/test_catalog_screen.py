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
            "source_table": "orders",
            "semantic_model_name": "shop",
            "dataset_name": "orders",
            "description": "Orders dataset",
            "ai_context_json": '{"instructions":"Use this dataset for order analytics."}',
            "fields": [
                {"name": "order_id", "expr": "order_id", "is_primary_key": True, "description": "Order key"},
                {
                    "name": "order_date",
                    "expr": "order_date",
                    "is_dimension": True,
                    "is_time": True,
                    "description": "Order date",
                },
                {"name": "segment", "expr": "segment", "is_dimension": True, "description": "Customer segment"},
                {"name": "amount", "expr": "amount", "description": "Order amount"},
            ],
            "relationships": [
                {"name": "orders_to_customers", "from_dataset": "orders", "to_dataset": "customers"},
            ],
            "alternatives": [],
        }
    )

    assert record["dataset_name"] == "orders"
    assert record["table_name"] == "orders"
    assert record["ai_context"]["instructions"] == "Use this dataset for order analytics."
    # Every authored field survives, in order. The old split into identifiers
    # and dimensions silently dropped `amount`, which is neither.
    assert [item["name"] for item in record["modelled_fields"]] == [
        "order_id",
        "order_date",
        "segment",
        "amount",
    ]
    by_name = {item["name"]: item for item in record["modelled_fields"]}
    assert by_name["order_id"]["is_primary_key"] is True
    assert by_name["order_date"]["is_time"] is True
    assert by_name["amount"]["is_primary_key"] is False and by_name["amount"]["is_time"] is False
    assert record["relationships"][0]["name"] == "orders_to_customers"
    assert "filters" not in record


def test_catalog_screen_readonly_panel_shows_profile_fields_without_measures():
    screen = object.__new__(CatalogScreen)
    group = screen._render_readonly_panel(
        {
            "semantic_model_name": "sales",
            "dataset_name": "orders",
            "description": "Orders dataset",
            "ai_context": {"instructions": "Use this dataset for sales analytics."},
            "identifiers": [{"name": "order_id"}],
            "dimensions": [{"name": "order_date"}],
            "relationships": [{"name": "orders_to_customers"}],
            "measures": [{"name": "amount"}],
        }
    )

    console = Console(record=True, width=180, file=io.StringIO())
    console.print(group)
    rendered = console.export_text()

    assert "Dataset" in rendered
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


def _screen_with_columns(columns, raises=False):
    """A screen whose connector reports `columns` (or fails outright)."""
    screen = object.__new__(CatalogScreen)
    connector = MagicMock()
    if raises:
        connector.get_schema.side_effect = RuntimeError("datasource is down")
    else:
        connector.get_schema.return_value = columns
    screen.db_connector = connector
    screen.catalog_name = ""
    screen.database_name = "shop"
    return screen


def _render_columns_text(screen, record):
    table = screen._create_columns_table(record)
    console = Console(width=200, record=True)
    console.print(table)
    return console.export_text()


def test_columns_table_lists_unmodelled_physical_columns_too():
    """The gap between physical and modelled is what this view exists to show,
    so a column no field describes is dimmed, not hidden."""
    screen = _screen_with_columns(
        [
            {"name": "order_id", "type": "bigint", "comment": "DDL key"},
            {"name": "audit_ts", "type": "datetime", "comment": "DDL audit stamp"},
        ]
    )
    record = {
        "table_name": "orders",
        "modelled_fields": [
            {
                "name": "order_id",
                "expr": "order_id",
                "is_primary_key": True,
                "is_time": False,
                "description": "Order key",
            },
        ],
    }

    text = _render_columns_text(screen, record)

    assert "1 modelled / 2 total" in text
    assert "order_id" in text and "Order key" in text  # model description wins
    assert "audit_ts" in text and "DDL audit stamp" in text  # unmodelled still listed


def test_columns_table_falls_back_to_ddl_comment_when_the_model_has_none():
    screen = _screen_with_columns([{"name": "region", "type": "varchar", "comment": "DDL region"}])
    record = {
        "table_name": "orders",
        "modelled_fields": [{"name": "region", "expr": "region", "description": ""}],
    }

    assert "DDL region" in _render_columns_text(screen, record)


def test_columns_table_survives_an_unreachable_datasource():
    """The catalog browses indexed content; a datasource that is down must dim
    the physical half of the view, not fail the panel."""
    screen = _screen_with_columns([], raises=True)
    record = {
        "table_name": "orders",
        "modelled_fields": [{"name": "order_id", "expr": "order_id", "description": "Order key"}],
    }

    text = _render_columns_text(screen, record)

    assert "order_id" in text
    assert "1 modelled / 1 total" in text


def test_panel_shows_a_composite_unique_key_as_one_group():
    """Ticking each column would lose which columns form the key, so it is
    rendered below the table instead of as a per-column mark."""
    screen = _screen_with_columns([{"name": "ac_code", "type": "varchar", "comment": ""}])
    group = screen._render_readonly_panel(
        {
            "semantic_model_name": "ops",
            "dataset_name": "activity",
            "table_name": "activity",
            "unique_keys": [["ac_code", "subject_seq", "product_code"]],
            "modelled_fields": [{"name": "ac_code", "expr": "ac_code", "description": "Activity code"}],
        }
    )
    console = Console(width=200, record=True)
    console.print(group)
    text = console.export_text()

    assert "Unique keys" in text
    assert "(ac_code, subject_seq, product_code)" in text


def test_panel_omits_the_unique_keys_line_when_there_are_none():
    screen = _screen_with_columns([{"name": "ac_code", "type": "varchar", "comment": ""}])
    group = screen._render_readonly_panel(
        {
            "semantic_model_name": "ops",
            "dataset_name": "activity",
            "table_name": "activity",
            "unique_keys": [],
            "modelled_fields": [{"name": "ac_code", "expr": "ac_code", "description": ""}],
        }
    )
    console = Console(width=200, record=True)
    console.print(group)

    assert "Unique keys" not in console.export_text()
