"""Unit tests for Dosi semantic authoring."""

import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import yaml

from datus.agent.node import semantic_authoring
from datus.agent.node.semantic_authoring import (
    discover_osi_semantic_models,
    plan_osi_semantic_model_target,
    required_authoring_skills,
    validate_osi_authoring_document,
)


@pytest.fixture(autouse=True)
def _accept_any_authoring_document(monkeypatch):
    """Target selection is what these exercise; the validator is covered on its own."""
    monkeypatch.setattr(semantic_authoring, "validate_osi_authoring_document", lambda document, **kwargs: None)


def _agent_config():
    return SimpleNamespace()


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


def test_validate_dosi_authoring_document_uses_native_validator(monkeypatch):
    from datus.tools.semantic_tools.dosi import authoring

    calls = []
    monkeypatch.setattr(authoring, "validate_dosi_document", calls.append)
    document = {"version": "0.2.0.dev0"}
    assert validate_osi_authoring_document(document) is None
    assert calls == [document]


def test_dosi_prompt_uses_engine_owned_contract(monkeypatch):
    from datus.tools.semantic_tools.dosi import authoring_spec, engine

    monkeypatch.setattr(authoring_spec, "compact_authoring_spec_text", lambda dialect: f"core dialect: {dialect}")
    monkeypatch.setattr(authoring_spec, "authoring_overview_text", lambda: "engine contract")
    monkeypatch.setattr(engine, "datus_extension_version", lambda: "1.8")
    rendered = semantic_authoring.render_required_authoring_skill(
        "dosi-semantic-authoring", 'extension version: "<datus_extension_version>"', include_osi_core=True
    )
    assert 'extension version: "1.8"' in rendered
    assert "core dialect: <osi_dialect>" in rendered
    assert "engine contract" in rendered


def test_dosi_prompt_snapshot_includes_engine_and_core_contract_digests(monkeypatch):
    from datus.tools.semantic_tools.dosi import authoring_spec, engine

    monkeypatch.setattr(authoring_spec, "authoring_spec_text", lambda dialect: f"core {dialect}")
    monkeypatch.setattr(authoring_spec, "datus_extension_authoring_spec_digest", lambda: "sha256:contract")
    monkeypatch.setattr(engine, "datus_extension_version", lambda: "1.8")
    core_digest = hashlib.sha256(b"core <osi_dialect>").hexdigest()
    assert semantic_authoring.authoring_prompt_snapshot_meta(_agent_config(), "semantic_modeling") == {
        "datus_extension_version": "1.8",
        "datus_authoring_contract_digest": "sha256:contract",
        "osi_core_authoring_spec_digest": f"sha256:{core_digest}",
    }


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


def test_semantic_modeling_required_skills_are_dosi_native():
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    node = SemanticModelingAgenticNode.__new__(SemanticModelingAgenticNode)
    node.agent_config = _agent_config()
    assert node._get_required_skills() == ["dosi-semantic-authoring"]
    assert required_authoring_skills(None, "semantic_modeling") == "dosi-semantic-authoring"


def test_semantic_authoring_base_configuration_is_neutral():
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode

    assert SemanticAuthoringAgenticNode.NODE_NAME == "semantic_authoring"
    assert SemanticAuthoringAgenticNode.INCLUDE_OSI_CORE_SPEC is False


@pytest.mark.parametrize(
    "message",
    [
        "SELECT COUNT(*) n FROM tickets",
        "Please model this:\n```sql\nSELECT COUNT(*) n FROM tickets\n```",
        "Please model this:\n~~~sql\nSELECT COUNT(*) n FROM tickets\n~~~",
        "Please model this:\n````sql\nSELECT COUNT(*) n FROM tickets\n`````",
    ],
)
def test_source_evidence_retains_exact_user_sql_and_attached_references(message):
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode
    from datus.schemas.node_models import ReferenceSql
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node.input = SemanticNodeInput(
        user_message=message, reference_sql=[ReferenceSql(name="attached", sql="SELECT SUM(cost) cost FROM tickets")]
    )
    entries = node._semantic_discovery_source_sql()
    assert entries[0] == {"name": "attached", "sql": "SELECT SUM(cost) cost FROM tickets"}
    assert entries[1]["sql"].strip() == "SELECT COUNT(*) n FROM tickets"
    node.input.user_message = "Create a semantic model for repair workorders"
    assert node._semantic_discovery_source_sql() == entries[:1]
    node.input.user_message = "SELECT 'unfinished"
    assert node._semantic_discovery_source_sql() == entries[:1]


@pytest.mark.asyncio
async def test_caller_sql_files_are_pinned_before_generation(monkeypatch, tmp_path):
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools

    async def _parent_before_stream(_self, _ctx):
        pass

    monkeypatch.setattr(AgenticNode, "_before_stream", _parent_before_stream)
    original = tmp_path / "original query.sql"
    original.write_text("SELECT SUM(cost) cost FROM tickets", encoding="utf-8")
    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node.input = SemanticNodeInput(user_message="Model the calculation in `original query.sql`.")
    node.generation_evidence = GenerationEvidence()
    node.osi_target_state = SimpleNamespace(reset=lambda: None)
    node.osi_target_tools = None
    node.semantic_discovery_tools = SemanticDiscoveryTools(root_path=str(tmp_path))

    await node._before_stream(object())
    result = node.semantic_discovery_tools.inspect_semantic_sources()
    assert result.success
    assert result.result["calculation_evidence"]["statements"][0]["input_kind"] == "original_file"

    original.write_text("SELECT SUM(cost) + 1 cost FROM tickets", encoding="utf-8")
    diagnostic = tmp_path / "generated.sql"
    diagnostic.write_text("SELECT COUNT(*) n FROM tickets", encoding="utf-8")
    result = node.semantic_discovery_tools.inspect_semantic_sources(sql_files=["generated.sql"])
    assert result.success
    assert {s["input_kind"] for s in result.result["calculation_evidence"]["statements"]} == {"file"}


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Use sql/original.sql and `SQL/monthly.SQL`.", ["sql/original.sql", "SQL/monthly.SQL"]),
        (
            'Use "original query.sql" and [ETL](<sql/monthly query.sql>).',
            ["original query.sql", "sql/monthly query.sql"],
        ),
        ("Use `explicit.sql` and explicit.sql.", ["explicit.sql"]),
        ("Use reports/kpi.sql. Then model it.", ["reports/kpi.sql"]),
        ("Use reports/kpi.sql.bak, source.sql/part, source.sql?download=1 and source.sql#section.", []),
        ("SELECT 'not_a_file.sql' AS label", []),
        ("-- Source note\n/* Original query */ SELECT 'not_a_file.sql' AS label", []),
        ("Model this:\n````sql\nSELECT 'not_a_file.sql' AS label\n`````\nUse `real.sql`.", ["real.sql"]),
        ("Model this:\n~~~sql\nSELECT 'not_a_file.sql' AS label\n~~~\nUse `real.sql`.", ["real.sql"]),
        ("Use https://example.org/source.sql and `sql/*.sql`.", []),
    ],
)
def test_caller_sql_file_references(message, expected):
    from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

    node = SemanticAuthoringAgenticNode.__new__(SemanticAuthoringAgenticNode)
    node.input = SemanticNodeInput(user_message=message, source_sql_files=["explicit.sql"])
    assert node._semantic_discovery_source_files() == list(dict.fromkeys(["explicit.sql", *expected]))


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
    node.input = SimpleNamespace(source_sql_files=["original.sql"])
    node.semantic_discovery_tools = SimpleNamespace(
        reset_request_cache=lambda: resets.append("discovery"),
        pin_source_files=lambda paths: resets.append(paths),
    )

    await node._before_stream(object())

    assert node.result is None
    assert resets == ["parent", "evidence", "target", "inventory", "discovery", ["original.sql"]]


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
    node.agent_config = _agent_config()
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
