"""Unit tests for semantic authoring format resolution."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from datus.agent.node import semantic_authoring
from datus.agent.node.semantic_authoring import (
    AUTHORING_FORMAT_METRICFLOW,
    AUTHORING_FORMAT_OSI,
    default_optional_skills,
    discover_osi_semantic_models,
    plan_osi_semantic_model_target,
    required_authoring_skills,
    resolve_authoring_format,
    resolve_semantic_adapter_type,
    validate_osi_core_document,
)
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture(autouse=True)
def _stub_osi_schema_validation(monkeypatch):
    monkeypatch.setattr(semantic_authoring, "validate_osi_core_document", lambda document: None)


def _agent_config(adapter):
    return SimpleNamespace(resolve_semantic_adapter=lambda requested=None: requested or adapter)


def _osi_config(tmp_path):
    model_dir = tmp_path / "subject" / "semantic_models" / "warehouse"
    return SimpleNamespace(
        current_datasource="warehouse",
        project_root=str(tmp_path),
        path_manager=SimpleNamespace(semantic_model_path=lambda datasource: model_dir),
    )


def _write_osi_model(tmp_path, filename, model_name, datasets):
    target = tmp_path / "subject" / "semantic_models" / "warehouse" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "version": "0.2.0.dev0",
                "semantic_model": [{"name": model_name, "datasets": datasets}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return target


def test_validate_osi_core_document_uses_canonical_validator(monkeypatch):
    class FakeOSIValidationError(Exception):
        pass

    profile_module = ModuleType("datus_semantic_osi.profile")
    errors_module = ModuleType("datus_semantic_osi.errors")
    package_module = ModuleType("datus_semantic_osi")
    package_module.__path__ = []
    errors_module.OSIValidationError = FakeOSIValidationError
    profile_module.validate_osi_core_schema = lambda document: None
    monkeypatch.setitem(sys.modules, "datus_semantic_osi", package_module)
    monkeypatch.setitem(sys.modules, "datus_semantic_osi.profile", profile_module)
    monkeypatch.setitem(sys.modules, "datus_semantic_osi.errors", errors_module)

    assert validate_osi_core_document({"version": "valid"}) is None

    def reject(document):
        raise FakeOSIValidationError("schema mismatch")

    profile_module.validate_osi_core_schema = reject
    assert validate_osi_core_document({"version": "invalid"}) == "schema mismatch"


def test_legacy_node_config_fields_are_ignored():
    assert (
        resolve_authoring_format(_agent_config("metricflow"), {"authoring_format": "osi"})
        == AUTHORING_FORMAT_METRICFLOW
    )
    assert resolve_authoring_format(_agent_config("osi"), {"authoring_format": "metricflow"}) == AUTHORING_FORMAT_OSI


def test_derives_from_active_semantic_adapter():
    assert resolve_authoring_format(_agent_config("osi"), None) == AUTHORING_FORMAT_OSI
    assert resolve_authoring_format(_agent_config("metricflow"), None) == AUTHORING_FORMAT_METRICFLOW


def test_legacy_node_semantic_adapter_is_ignored():
    assert (
        resolve_authoring_format(_agent_config("metricflow"), {"semantic_adapter": "osi"})
        == AUTHORING_FORMAT_METRICFLOW
    )


def test_osi_target_explicit_name_wins_over_domain_and_existing_fact(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "legacy_sales.yml",
        "legacy_sales",
        [{"name": "orders", "source": "analytics.fact_orders"}],
    )

    target = plan_osi_semantic_model_target(
        config,
        semantic_model_name="Executive Sales",
        business_domain="commerce",
        fact_tables=["analytics.fact_orders"],
    )

    assert target["semantic_model_name"] == "executive_sales"
    assert target["semantic_model_file"] == "subject/semantic_models/warehouse/executive_sales.yml"
    assert target["matched_by"] == "explicit_name"
    assert target["exists"] is False


def test_osi_target_uses_business_domain_for_a_new_model(tmp_path):
    target = plan_osi_semantic_model_target(
        _osi_config(tmp_path),
        business_domain="Order Fulfillment",
        fact_tables=["analytics.fact_orders"],
        dimension_tables=["analytics.dim_customer"],
    )

    assert target["semantic_model_name"] == "order_fulfillment"
    assert target["matched_by"] == "business_domain"


def test_osi_target_fact_fallback_does_not_change_when_dimensions_change(tmp_path):
    config = _osi_config(tmp_path)
    first = plan_osi_semantic_model_target(
        config,
        fact_tables=["analytics.fact_order_items"],
        dimension_tables=["analytics.dim_customer"],
    )
    second = plan_osi_semantic_model_target(
        config,
        fact_tables=["analytics.fact_order_items"],
        dimension_tables=["analytics.dim_customer", "analytics.dim_product"],
    )

    assert first["semantic_model_name"] == "fact_order_items_analytics"
    assert second["semantic_model_name"] == first["semantic_model_name"]
    assert second["semantic_model_file"] == first["semantic_model_file"]


def test_osi_target_reuses_existing_model_name_when_dimensions_are_added(tmp_path):
    config = _osi_config(tmp_path)
    existing = _write_osi_model(
        tmp_path,
        "durable_revenue.yml",
        "revenue_v1",
        [{"name": "orders", "source": "analytics.fact_orders"}],
    )

    target = plan_osi_semantic_model_target(
        config,
        business_domain="new_domain_label",
        fact_tables=["analytics.fact_orders"],
        dimension_tables=["analytics.dim_customer"],
    )

    assert target["semantic_model_name"] == "revenue_v1"
    assert target["semantic_model_file"].endswith("/durable_revenue.yml")
    assert target["absolute_path"] == str(existing)
    assert target["matched_by"] == "existing_fact_table"


def test_osi_target_identity_uses_only_the_core_fact_table(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "shared_inventory.yml",
        "shared_inventory",
        [{"name": "inventory", "source": "analytics.fact_inventory"}],
    )

    target = plan_osi_semantic_model_target(
        config,
        business_domain="support",
        fact_tables=["support.fact_tickets", "analytics.fact_inventory"],
    )

    assert target["semantic_model_name"] == "support"
    assert target["matched_by"] == "business_domain"
    assert target["exists"] is False


def test_osi_target_creates_a_different_file_for_an_unrelated_fact(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "orders_analytics.yml",
        "orders_analytics",
        [{"name": "orders", "source": "analytics.fact_orders"}],
    )

    target = plan_osi_semantic_model_target(config, fact_tables=["finance.fact_payments"])

    assert target["semantic_model_name"] == "fact_payments_analytics"
    assert target["semantic_model_file"].endswith("/fact_payments_analytics.yml")
    assert target["exists"] is False
    assert len(discover_osi_semantic_models(config)) == 1


def test_osi_target_does_not_reuse_same_leaf_table_from_another_schema(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "sales_orders.yml",
        "sales_orders",
        [{"name": "orders", "source": "sales.fact_orders"}],
    )

    target = plan_osi_semantic_model_target(config, fact_tables=["finance.fact_orders"])

    assert target["semantic_model_name"] == "fact_orders_analytics"
    assert target["semantic_model_file"].endswith("/fact_orders_analytics.yml")
    assert target["exists"] is False


def test_osi_target_preserves_qualified_table_component_boundaries(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "sales_orders.yml",
        "sales_orders",
        [{"name": "orders", "source": "sales_fact.orders"}],
    )

    target = plan_osi_semantic_model_target(config, fact_tables=["sales.fact_orders"])

    assert target["semantic_model_name"] == "fact_orders_analytics"
    assert target["exists"] is False


def test_osi_target_allows_leaf_fallback_for_unqualified_fact_reference(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "sales_orders.yml",
        "sales_orders",
        [{"name": "orders", "source": "sales.fact_orders"}],
    )

    target = plan_osi_semantic_model_target(config, fact_tables=["fact_orders"])

    assert target["semantic_model_name"] == "sales_orders"
    assert target["matched_by"] == "existing_fact_table"


def test_osi_target_refuses_to_overwrite_an_unparseable_target_file(tmp_path):
    config = _osi_config(tmp_path)
    target_path = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("semantic_model: [\n", encoding="utf-8")

    target = plan_osi_semantic_model_target(config, semantic_model_name="sales")

    assert target["ambiguous"] is True
    assert "already exists" in target["reason"]
    assert target["candidates"][0]["semantic_model_file"].endswith("/sales.yml")


def test_osi_target_refuses_an_unsafe_generic_fallback(tmp_path):
    target = plan_osi_semantic_model_target(
        _osi_config(tmp_path),
        dimension_tables=["analytics.dim_customer"],
    )

    assert target["ambiguous"] is True
    assert target["matched_by"] == "missing_core_fact_table"
    assert "business domain or core fact table" in target["reason"]


def test_osi_target_refuses_to_reuse_an_occupied_filename_with_a_different_model_name(tmp_path):
    config = _osi_config(tmp_path)
    _write_osi_model(
        tmp_path,
        "sales.yml",
        "legacy_sales_model",
        [{"name": "orders", "source": "analytics.fact_orders"}],
    )

    target = plan_osi_semantic_model_target(
        config,
        semantic_model_name="sales",
        fact_tables=["analytics.fact_payments"],
    )

    assert target["ambiguous"] is True
    assert "already occupied" in target["reason"]


def test_defaults_to_metricflow_when_unknown():
    assert resolve_authoring_format(None, None) == AUTHORING_FORMAT_METRICFLOW
    assert resolve_authoring_format(_agent_config(None), {}) == AUTHORING_FORMAT_METRICFLOW


def test_resolution_propagates_agent_config_errors():
    def _boom(_requested=None):
        raise RuntimeError("no semantic layer")

    bad = SimpleNamespace(resolve_semantic_adapter=_boom)
    with pytest.raises(RuntimeError, match="no semantic layer"):
        resolve_authoring_format(bad, None)


def test_resolution_propagates_semantic_layer_config_errors():
    def _boom(_requested=None):
        raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message="multiple semantic layers")

    bad = SimpleNamespace(resolve_semantic_adapter=_boom)
    with pytest.raises(DatusException, match="multiple semantic layers"):
        resolve_authoring_format(bad, None)


def test_adapter_type_resolution_propagates_agent_config_errors():
    def _boom(_requested=None):
        raise RuntimeError("resolver unavailable")

    bad = SimpleNamespace(resolve_semantic_adapter=_boom)
    with pytest.raises(RuntimeError, match="resolver unavailable"):
        resolve_semantic_adapter_type(bad)


@pytest.mark.parametrize(
    "node_name, adapter, expected",
    [
        (
            "gen_semantic_model",
            "metricflow",
            "sql-modeling-preflight,metricflow-semantic-authoring",
        ),
        ("gen_semantic_model", "osi", "sql-modeling-preflight,osi-semantic-authoring"),
        ("gen_metrics", "metricflow", "sql-modeling-preflight,gen-metrics"),
        ("gen_metrics", "osi", "sql-modeling-preflight,osi-metrics-authoring"),
        ("unknown_node", "metricflow", ""),
    ],
)
def test_required_authoring_skills_derive_from_format(node_name, adapter, expected):
    assert required_authoring_skills(_agent_config(adapter), node_name) == expected


@pytest.mark.parametrize(
    "node_name, adapter, expected",
    [
        ("gen_semantic_model", "metricflow", "semantic-sql-history-profiler"),
        ("gen_semantic_model", "osi", "semantic-sql-history-profiler"),
        ("gen_metrics", "metricflow", "metricflow-semantic-authoring"),
        ("gen_metrics", "osi", ""),
        ("unknown_node", "osi", ""),
    ],
)
def test_default_optional_skills_derive_from_format(node_name, adapter, expected):
    assert default_optional_skills(_agent_config(adapter), node_name) == expected


@pytest.mark.parametrize("adapter", ["metricflow", "osi"])
def test_node_skill_defaults_follow_authoring_format(monkeypatch, adapter):
    """Both nodes default node_config['skills'] from the format, then defer to the base setup."""
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

    parent_calls = []
    monkeypatch.setattr(AgenticNode, "_setup_skill_func_tools", lambda self: parent_calls.append(type(self).__name__))

    metrics_node = GenMetricsAgenticNode.__new__(GenMetricsAgenticNode)
    metrics_node.agent_config = _agent_config(adapter)
    metrics_node.node_config = {}
    metrics_node._setup_skill_func_tools()

    semantic_node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
    semantic_node.agent_config = _agent_config(adapter)
    semantic_node.node_config = {}
    semantic_node._setup_skill_func_tools()

    assert parent_calls == ["GenMetricsAgenticNode", "GenSemanticModelAgenticNode"]
    assert semantic_node.node_config["skills"] == "semantic-sql-history-profiler"
    expected_metrics_optional = "metricflow-semantic-authoring" if adapter == "metricflow" else ""
    assert metrics_node.node_config["skills"] == expected_metrics_optional


def test_node_skill_defaults_respect_explicit_config(monkeypatch):
    """An explicit skills entry (including opt-out '') is never overwritten."""
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

    monkeypatch.setattr(AgenticNode, "_setup_skill_func_tools", lambda self: None)

    node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
    node.agent_config = _agent_config("osi")
    node.node_config = {"skills": ""}
    node._setup_skill_func_tools()

    assert node.node_config["skills"] == ""


@pytest.mark.parametrize(
    "adapter, expected",
    [
        ("metricflow", ["sql-modeling-preflight", "metricflow-semantic-authoring"]),
        ("osi", ["sql-modeling-preflight", "osi-semantic-authoring"]),
    ],
)
def test_gen_semantic_model_required_skills(adapter, expected):
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

    node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
    node.agent_config = _agent_config(adapter)
    assert node._get_required_skills() == expected


@pytest.mark.parametrize(
    "adapter, expected",
    [
        ("metricflow", ["sql-modeling-preflight", "gen-metrics"]),
        ("osi", ["sql-modeling-preflight", "osi-metrics-authoring"]),
    ],
)
def test_gen_metrics_required_skills(adapter, expected):
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

    node = GenMetricsAgenticNode.__new__(GenMetricsAgenticNode)
    node.agent_config = _agent_config(adapter)
    assert node._get_required_skills() == expected
