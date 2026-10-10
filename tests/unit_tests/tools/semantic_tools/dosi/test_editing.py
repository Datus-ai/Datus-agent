"""Candidate authoring contracts against the native compiler, without a database."""

import json

import pytest
import yaml

from datus.tools.semantic_tools.dosi.editing import prepare_metric_definitions


def model():
    def metric(name, sql, formula=None):
        result = {"name": name, "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": sql}]}}
        if formula:
            result["custom_extensions"] = [
                {
                    "vendor_name": "DATUS",
                    "data": json.dumps({"v": "1.9", "derive": {"type": "compose", "expr": formula}, "unit": "ratio"}),
                }
            ]
        return result

    return {
        "version": "0.2.0.dev0",
        "semantic_model": [
            {
                "name": "sales",
                "datasets": [
                    {
                        "name": "orders",
                        "source": "orders",
                        "fields": [
                            {"name": "id", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "id"}]}},
                            {
                                "name": "amount",
                                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "amount"}]},
                            },
                        ],
                    }
                ],
                "metrics": [
                    metric("revenue", "SUM(orders.amount)"),
                    metric("orders_count", "COUNT(orders.id)"),
                    metric("aov", "SUM(orders.amount) / COUNT(orders.id)", "revenue / orders_count"),
                    metric("double_aov", "SUM(orders.amount) / COUNT(orders.id) * 2", "aov * 2"),
                ],
            }
        ],
    }


def test_updates_transitive_dependents_and_retains_metadata():
    doc = model()
    doc["semantic_model"][0]["metrics"][0]["expression"]["dialects"][0]["expression"] = "SUM(orders.amount) * 3"
    result = prepare_metric_definitions(yaml.safe_dump(doc), ["revenue"])
    assert result["valid"], result["validation"]
    assert {patch["name"] for patch in result["patches"]} == {"aov", "double_aov"}
    assert all("custom_extensions" not in patch["fields"] for patch in result["patches"])
    assert "* 3" in result["patches"][-1]["fields"]["expression"]["dialects"][0]["expression"]


@pytest.mark.parametrize(
    "formula",
    [
        "missing / orders_count",
        "double_aov / orders_count",
        "__import__('os').system('whoami')",
        "revenue ** 2",
        "revenue; orders_count",
        "revenue / (orders_count",
    ],
)
def test_invalid_formula_never_returns_partial_patches(formula):
    doc = model()
    metric = doc["semantic_model"][0]["metrics"][2]
    hints = json.loads(metric["custom_extensions"][0]["data"])
    hints["derive"]["expr"] = formula
    metric["custom_extensions"][0]["data"] = json.dumps(hints)
    result = prepare_metric_definitions(yaml.safe_dump(doc), ["aov"])
    assert result["valid"] is False
    assert result["patches"] == []
    assert result["validation"]["issues"]


def test_rename_updates_identifier_references_only():
    doc = model()
    doc["semantic_model"][0]["metrics"][0]["name"] = "net_revenue"
    result = prepare_metric_definitions(
        yaml.safe_dump(doc), ["net_revenue"], rename_from="revenue", rename_to="net_revenue"
    )
    assert result["valid"], result["validation"]
    patch = next(patch for patch in result["patches"] if patch["name"] == "aov")
    hints = json.loads(patch["fields"]["custom_extensions"][0]["data"])
    assert hints["derive"]["expr"] == "net_revenue / orders_count"
    assert hints["unit"] == "ratio"


def test_missing_dialect_is_not_substituted():
    doc = model()
    doc["semantic_model"][0]["metrics"][0]["expression"]["dialects"][0]["dialect"] = "SNOWFLAKE"
    result = prepare_metric_definitions(yaml.safe_dump(doc), ["revenue"])
    assert not result["valid"]
    assert result["patches"] == []
    assert "ANSI_SQL" in result["validation"]["issues"][0]["message"]


def test_delete_retained_reference_fails_native_validation():
    doc = model()
    doc["semantic_model"][0]["metrics"].pop(0)
    result = prepare_metric_definitions(yaml.safe_dump(doc), [])
    assert not result["valid"]
    assert result["patches"] == []


def test_empty_metric_list_keeps_dataset():
    doc = model()
    doc["semantic_model"][0]["metrics"] = []
    result = prepare_metric_definitions(yaml.safe_dump(doc), [])
    assert result["valid"], result["validation"]
