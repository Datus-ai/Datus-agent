"""Unit tests for semantic authoring format resolution."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from datus.agent.node.semantic_authoring import (
    AUTHORING_FORMAT_METRICFLOW,
    AUTHORING_FORMAT_OSI,
    default_optional_skills,
    default_osi_semantic_model_file,
    default_osi_semantic_model_name,
    required_authoring_skills,
    resolve_authoring_format,
    resolve_semantic_adapter_type,
)
from datus.utils.exceptions import DatusException, ErrorCode


@dataclass
class _DbScope:
    database: str = ""
    schema: str = ""
    catalog: str = ""


def _agent_config(adapter):
    return SimpleNamespace(resolve_semantic_adapter=lambda requested=None: requested or adapter)


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


def test_default_osi_semantic_model_name_uses_database_scope():
    config = SimpleNamespace(
        current_datasource="warehouse",
        current_db_config=lambda: _DbScope(database="Sales Domain"),
    )

    assert default_osi_semantic_model_name(config) == "sales_domain"
    assert default_osi_semantic_model_file(config) == "subject/semantic_models/warehouse/sales_domain.yml"


def test_default_osi_semantic_model_name_prefers_runtime_database_scope():
    config = SimpleNamespace(
        current_datasource="starrocks",
        current_db_config=lambda: _DbScope(),
        runtime_db_context=lambda: {"database": "ac_manage"},
    )

    assert default_osi_semantic_model_name(config) == "ac_manage"
    assert default_osi_semantic_model_file(config) == "subject/semantic_models/starrocks/ac_manage.yml"


def test_default_osi_semantic_model_name_uses_declared_db_scope_fallbacks():
    config = SimpleNamespace(
        current_datasource="warehouse",
        current_db_config=lambda: _DbScope(schema="Reporting Schema", catalog="Lake House"),
    )

    assert default_osi_semantic_model_name(config) == "reporting_schema"
    assert default_osi_semantic_model_file(config) == "subject/semantic_models/warehouse/reporting_schema.yml"


def test_default_osi_semantic_model_name_skips_undeclared_schema_method():
    class DbScopeWithSchemaMethod:
        __annotations__ = {"database": str, "catalog": str}

        database = ""
        catalog = "Lake House"

        def schema(self):
            return "method-value"

    config = SimpleNamespace(
        current_datasource="warehouse",
        current_db_config=lambda: DbScopeWithSchemaMethod(),
    )

    assert default_osi_semantic_model_name(config) == "lake_house"


def test_default_osi_semantic_model_name_uses_agent_scope_fallbacks():
    config = SimpleNamespace(
        current_datasource="",
        project_name="Project Alpha",
        current_db_config=lambda: _DbScope(),
    )

    assert default_osi_semantic_model_name(config) == "project_alpha"
    assert default_osi_semantic_model_file(config) == "subject/semantic_models/default/project_alpha.yml"


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
        ("gen_semantic_model", "metricflow", "metricflow-semantic-authoring"),
        ("gen_semantic_model", "osi", "osi-semantic-authoring"),
        ("gen_metrics", "metricflow", "gen-metrics"),
        ("gen_metrics", "osi", "osi-metrics-authoring"),
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
        ("gen_metrics", "osi", "osi-semantic-authoring"),
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
    expected_metrics_optional = "metricflow-semantic-authoring" if adapter == "metricflow" else "osi-semantic-authoring"
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
    [("metricflow", ["metricflow-semantic-authoring"]), ("osi", ["osi-semantic-authoring"])],
)
def test_gen_semantic_model_required_skills(adapter, expected):
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

    node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
    node.agent_config = _agent_config(adapter)
    assert node._get_required_skills() == expected


@pytest.mark.parametrize(
    "adapter, expected",
    [("metricflow", ["gen-metrics"]), ("osi", ["osi-metrics-authoring"])],
)
def test_gen_metrics_required_skills(adapter, expected):
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

    node = GenMetricsAgenticNode.__new__(GenMetricsAgenticNode)
    node.agent_config = _agent_config(adapter)
    assert node._get_required_skills() == expected
