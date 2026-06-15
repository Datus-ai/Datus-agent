"""Unit tests for semantic authoring format resolution."""

from types import SimpleNamespace
from unittest.mock import patch

from datus.agent.node.semantic_authoring import (
    AUTHORING_FORMAT_METRICFLOW,
    AUTHORING_FORMAT_OSI,
    osi_prompt_version,
    osi_template_name,
    resolve_authoring_format,
)
from datus.prompts.prompt_manager import get_prompt_manager


def _agent_config(adapter):
    return SimpleNamespace(resolve_semantic_adapter=lambda requested=None: requested or adapter)


def test_explicit_node_config_override_wins():
    assert resolve_authoring_format(_agent_config("metricflow"), {"authoring_format": "osi"}) == AUTHORING_FORMAT_OSI
    assert (
        resolve_authoring_format(_agent_config("osi"), {"authoring_format": "metricflow"})
        == AUTHORING_FORMAT_METRICFLOW
    )


def test_derives_from_active_semantic_adapter():
    assert resolve_authoring_format(_agent_config("osi"), None) == AUTHORING_FORMAT_OSI
    assert resolve_authoring_format(_agent_config("metricflow"), None) == AUTHORING_FORMAT_METRICFLOW


def test_derives_from_node_semantic_adapter():
    assert resolve_authoring_format(_agent_config("metricflow"), {"semantic_adapter": "osi"}) == AUTHORING_FORMAT_OSI


def test_defaults_to_metricflow_when_unknown():
    assert resolve_authoring_format(None, None) == AUTHORING_FORMAT_METRICFLOW
    assert resolve_authoring_format(_agent_config(None), {}) == AUTHORING_FORMAT_METRICFLOW


def test_resolution_is_resilient_to_agent_config_errors():
    def _boom():
        raise RuntimeError("no semantic layer")

    bad = SimpleNamespace(resolve_semantic_adapter=_boom)
    assert resolve_authoring_format(bad, None) == AUTHORING_FORMAT_METRICFLOW


def test_resolution_logs_agent_config_errors():
    def _boom(_requested=None):
        raise RuntimeError("no semantic layer")

    bad = SimpleNamespace(resolve_semantic_adapter=_boom)
    with patch("datus.agent.node.semantic_authoring.logger") as mock_logger:
        assert resolve_authoring_format(bad, {"semantic_adapter": "osi"}) == AUTHORING_FORMAT_METRICFLOW

    mock_logger.debug.assert_called_once()
    assert "Failed to resolve semantic adapter" in mock_logger.debug.call_args.args[0]


def test_osi_template_name_is_isolated_from_metricflow_template():
    # Separate template_name so the default `{node}_system` latest scan is unaffected.
    assert osi_template_name("gen_metrics") == "gen_metrics_osi_system"
    assert osi_template_name("gen_semantic_model") == "gen_semantic_model_osi_system"


def test_osi_prompt_version_ignores_injected_metricflow_version():
    # Bootstrap (init_success_story_metrics) injects the latest *metricflow*
    # template version (e.g. "1.2"), which is not a version of the OSI template.
    # It must be ignored (-> None -> latest OSI template), not break rendering.
    assert osi_prompt_version(None, "gen_metrics", "1.2") is None
    assert osi_prompt_version(None, "gen_metrics", None) is None
    # a real OSI template version is honored
    assert osi_prompt_version(None, "gen_metrics", "1.0") == "1.0"


def test_osi_prompt_version_logs_template_lookup_errors():
    with (
        patch(
            "datus.prompts.prompt_manager.get_prompt_manager",
            side_effect=RuntimeError("template registry unavailable"),
        ),
        patch("datus.agent.node.semantic_authoring.logger") as mock_logger,
    ):
        assert osi_prompt_version(None, "gen_metrics", "1.2") is None

    mock_logger.debug.assert_called_once()
    assert "Failed to list OSI prompt template versions" in mock_logger.debug.call_args.args[0]


def test_osi_metrics_template_renders_despite_injected_metricflow_version():
    # Regression: the OSI template must still render when the resolved version
    # falls back to latest (the injected "1.2" does not exist for it).
    pm = get_prompt_manager()
    version = osi_prompt_version(None, "gen_metrics", "1.2")
    text = pm.render_template(template_name="gen_metrics_osi_system", version=version)
    assert "OSI" in text


def test_osi_skill_skip_handles_missing_node_config(monkeypatch):
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

    parent_calls = []

    def _unexpected_parent_call(self):
        parent_calls.append(type(self).__name__)

    monkeypatch.setattr(AgenticNode, "_setup_skill_func_tools", _unexpected_parent_call)

    metrics_node = GenMetricsAgenticNode.__new__(GenMetricsAgenticNode)
    metrics_node.agent_config = _agent_config("osi")
    metrics_node.node_config = None
    metrics_node._setup_skill_func_tools()

    semantic_node = GenSemanticModelAgenticNode.__new__(GenSemanticModelAgenticNode)
    semantic_node.agent_config = _agent_config("osi")
    semantic_node.node_config = None
    semantic_node._setup_skill_func_tools()

    assert parent_calls == [], "MetricFlow skills should not be injected in OSI mode"
