"""Unit tests for semantic authoring format resolution."""

import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from datus.agent.node import semantic_authoring
from datus.agent.node.semantic_authoring import (
    AUTHORING_FORMAT_METRICFLOW,
    AUTHORING_FORMAT_OSI,
    default_optional_skills,
    discover_osi_semantic_models,
    is_osi_semantic_adapter,
    plan_osi_semantic_model_target,
    required_authoring_skills,
    resolve_authoring_format,
    resolve_semantic_adapter_type,
    validate_osi_authoring_document,
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


def test_validate_dosi_authoring_document_uses_native_validator(monkeypatch):
    calls = []
    package_module = ModuleType("datus_semantic_dosi")
    package_module.__path__ = []
    authoring_module = ModuleType("datus_semantic_dosi.authoring")
    authoring_module.validate_dosi_document = calls.append
    monkeypatch.setitem(sys.modules, "datus_semantic_dosi", package_module)
    monkeypatch.setitem(sys.modules, "datus_semantic_dosi.authoring", authoring_module)

    document = {"version": "0.2.0.dev0"}
    assert validate_osi_authoring_document(document, semantic_adapter="dosi") is None
    assert calls == [document]


def test_dosi_prompt_rendering_reports_missing_adapter_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "datus_semantic_dosi.authoring_spec", None)

    with pytest.raises(DatusException, match="requires the datus-semantic-dosi package"):
        semantic_authoring.render_required_authoring_skill("dosi-semantic-authoring", "authoring")


def test_dosi_prompt_snapshot_reports_missing_adapter_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "datus_semantic_dosi.engine", None)

    with pytest.raises(DatusException, match="requires the datus-semantic-dosi package"):
        semantic_authoring.authoring_prompt_snapshot_meta(_agent_config("dosi"), "semantic_modeling")


def test_legacy_node_config_fields_are_ignored():
    assert (
        resolve_authoring_format(_agent_config("metricflow"), {"authoring_format": "osi"})
        == AUTHORING_FORMAT_METRICFLOW
    )
    assert resolve_authoring_format(_agent_config("dosi"), {"authoring_format": "metricflow"}) == AUTHORING_FORMAT_OSI


def test_derives_from_active_semantic_adapter():
    # Plain-OSI projects are query-only: only Dosi resolves to the OSI
    # authoring format.
    assert resolve_authoring_format(_agent_config("osi"), None) == AUTHORING_FORMAT_METRICFLOW
    assert resolve_authoring_format(_agent_config("dosi"), None) == AUTHORING_FORMAT_OSI
    assert resolve_authoring_format(_agent_config("metricflow"), None) == AUTHORING_FORMAT_METRICFLOW


def test_osi_authoring_adapter_classification():
    assert is_osi_semantic_adapter("osi") is False
    assert is_osi_semantic_adapter(" DOSI ") is True
    assert is_osi_semantic_adapter("metricflow") is False


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
        ("semantic_modeling", "metricflow", ""),
        ("semantic_modeling", "osi", ""),
        ("semantic_modeling", "dosi", "dosi-semantic-authoring"),
        ("gen_semantic_model", "dosi", ""),
        ("gen_metrics", "dosi", ""),
        ("unknown_node", "metricflow", ""),
    ],
)
def test_required_authoring_skills_derive_from_format(node_name, adapter, expected):
    assert required_authoring_skills(_agent_config(adapter), node_name) == expected


@pytest.mark.parametrize(
    "node_name, adapter, expected",
    [
        ("semantic_modeling", "metricflow", ""),
        ("semantic_modeling", "osi", ""),
        ("semantic_modeling", "dosi", ""),
        ("gen_semantic_model", "metricflow", ""),
        ("gen_metrics", "metricflow", ""),
        ("unknown_node", "osi", ""),
    ],
)
def test_default_optional_skills_derive_from_format(node_name, adapter, expected):
    assert default_optional_skills(_agent_config(adapter), node_name) == expected


def test_semantic_modeling_skill_defaults_follow_dosi_adapter(monkeypatch):
    """The unified node defers optional-skill setup to the shared runtime."""
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    parent_calls = []
    monkeypatch.setattr(AgenticNode, "_setup_skill_func_tools", lambda self: parent_calls.append(type(self).__name__))

    semantic_node = SemanticModelingAgenticNode.__new__(SemanticModelingAgenticNode)
    semantic_node.agent_config = _agent_config("dosi")
    semantic_node.node_config = {}
    semantic_node._setup_skill_func_tools()

    assert parent_calls == ["SemanticModelingAgenticNode"]
    assert semantic_node.node_config["skills"] == ""


def test_node_skill_defaults_respect_explicit_config(monkeypatch):
    """An explicit skills entry (including opt-out '') is never overwritten."""
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    monkeypatch.setattr(AgenticNode, "_setup_skill_func_tools", lambda self: None)

    node = SemanticModelingAgenticNode.__new__(SemanticModelingAgenticNode)
    node.agent_config = _agent_config("dosi")
    node.node_config = {"skills": ""}
    node._setup_skill_func_tools()

    assert node.node_config["skills"] == ""


def test_semantic_modeling_required_skills_are_dosi_native():
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    node = SemanticModelingAgenticNode.__new__(SemanticModelingAgenticNode)
    node.agent_config = _agent_config("dosi")
    assert node._get_required_skills() == ["dosi-semantic-authoring"]


def test_semantic_authoring_base_configuration_is_neutral():
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode

    assert SemanticAuthoringAgenticNode.NODE_NAME == "semantic_authoring"
    assert SemanticAuthoringAgenticNode.INCLUDE_OSI_CORE_SPEC is False
    assert SemanticAuthoringAgenticNode.COMPACT_SOURCE_INSPECTION is False


@pytest.mark.asyncio
async def test_semantic_authoring_base_resets_request_local_state(monkeypatch):
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode

    resets: list[str] = []

    async def _parent_before_stream(_self, _ctx):
        resets.append("parent")

    monkeypatch.setattr(AgenticNode, "_before_stream", _parent_before_stream)
    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node.result = object()
    node.generation_evidence = SimpleNamespace(reset=lambda: resets.append("evidence"))
    node.osi_target_state = SimpleNamespace(reset=lambda: resets.append("target"))
    node.osi_target_tools = SimpleNamespace(invalidate_inventory=lambda: resets.append("inventory"))
    node.semantic_discovery_tools = SimpleNamespace(reset_request_cache=lambda: resets.append("discovery"))

    await node._before_stream(object())

    assert node.result is None
    assert resets == ["parent", "evidence", "target", "inventory", "discovery"]


@pytest.mark.asyncio
async def test_semantic_authoring_base_rolls_back_terminal_failure(monkeypatch):
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode

    marker = object()
    rolled_back: list[bool] = []

    @asynccontextmanager
    async def _guard(_agent_config):
        yield

    async def _parent_execute_stream(_self, _manager):
        yield marker

    monkeypatch.setattr(semantic_authoring, "semantic_authoring_guard", _guard)
    monkeypatch.setattr(AgenticNode, "execute_stream", _parent_execute_stream)
    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node.agent_config = _agent_config("dosi")
    node.result = SimpleNamespace(success=False)
    node.filesystem_func_tool = SimpleNamespace(
        rollback_failed_authoring=lambda: rolled_back.append(True) or True,
    )

    actions = [action async for action in node.execute_stream()]

    assert actions == [marker]
    assert rolled_back == [True]


def test_semantic_authoring_base_reports_unavailable_warehouse_connection():
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode

    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node.db_func_tool = None

    assert node._warehouse_dry_run_compiled_sql("SELECT 1") == {
        "status": "failed",
        "error": "Database connection is unavailable.",
    }


@pytest.mark.parametrize(
    ("read_result", "expected"),
    [
        (
            SimpleNamespace(success=False, error="EXPLAIN rejected"),
            {"status": "failed", "error": "EXPLAIN rejected"},
        ),
        (
            SimpleNamespace(success=True, error=None),
            {"status": "success", "datasource": "warehouse", "database": "analytics"},
        ),
    ],
)
def test_semantic_authoring_base_validates_compiled_sql_with_warehouse_explain(read_result, expected):
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode

    calls: list[tuple[str, str, str]] = []
    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node._semantic_runtime_db_context = lambda: {"datasource": "warehouse", "database": "analytics"}
    node.db_func_tool = SimpleNamespace(
        read_query=lambda sql, *, datasource, database: calls.append((sql, datasource, database)) or read_result,
    )

    assert node._warehouse_dry_run_compiled_sql("SELECT 1;") == expected
    assert calls == [("EXPLAIN SELECT 1", "warehouse", "analytics")]
