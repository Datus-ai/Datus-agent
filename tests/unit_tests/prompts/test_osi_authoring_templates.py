"""Shared generation prompt templates render both authoring formats.

Both formats resolve to one `{node}_system` template; the format-specific
authoring specification lives in required skills (see
tests/unit_tests/tools/skill_tools and test_semantic_authoring.py), so these
tests cover the orchestration layer only: role boundary, dynamic OSI values,
profiler gate, and the final JSON contract.
"""

import pytest

from datus.prompts.prompt_manager import get_prompt_manager

COMMON_VARS = {
    "native_tools": "get_table_ddl, write_file",
    "mcp_tools": "None",
    "has_ask_user_tool": True,
    "knowledge_base_dir": "/kb",
    "semantic_model_dir": "/kb/semantic_models/duckdb",
    "kind_subdir": "subject/semantic_models/duckdb",
    "default_osi_semantic_model_name": "sales_domain",
    "default_osi_semantic_model_file": "subject/semantic_models/duckdb/sales_domain.yml",
}


def _render(template_name: str, authoring_format: str, datasource: str = "duckdb") -> str:
    pm = get_prompt_manager()
    return pm.render_template(
        template_name=template_name,
        authoring_format=authoring_format,
        current_datasource=datasource,
        **COMMON_VARS,
    )


@pytest.mark.parametrize("template_name", ["gen_semantic_model_system", "gen_metrics_system"])
def test_templates_reference_required_skill_spec(template_name):
    for authoring_format in ("metricflow", "osi"):
        text = _render(template_name, authoring_format)
        assert "<required_skill>" in text, (template_name, authoring_format)


def test_semantic_model_template_metricflow_mode():
    text = _render("gen_semantic_model_system", "metricflow")
    assert "MetricFlow expert" in text
    assert "OSI expression dialect" not in text
    assert '"semantic_model_files"' in text
    assert "validate_semantic" in text
    assert "end_semantic_model_generation" in text


def test_semantic_model_template_osi_mode():
    text = _render("gen_semantic_model_system", "osi")
    assert "OSI (Open Semantic Interchange) core schema" in text
    assert "never write backend YAML" in text
    assert "Target semantic model file: `subject/semantic_models/duckdb/sales_domain.yml`" in text
    assert '"semantic_model_files"' in text  # same publish contract as metricflow


@pytest.mark.parametrize("template_name", ["gen_semantic_model_system", "gen_metrics_system"])
@pytest.mark.parametrize(
    "datasource, expected_dialect",
    [("starrocks", "ANSI_SQL"), ("mysql", "ANSI_SQL"), ("snowflake", "SNOWFLAKE"), ("databricks", "DATABRICKS")],
)
def test_osi_mode_expression_dialect_derivation(template_name, datasource, expected_dialect):
    text = _render(template_name, "osi", datasource=datasource)
    assert f"- Active datasource: `{datasource}`" in text
    assert f"OSI expression dialect for this run: `{expected_dialect}`" in text


def test_metrics_template_metricflow_mode_contract():
    text = _render("gen_metrics_system", "metricflow")
    assert "MetricFlow metric definition expert" in text
    assert '"semantic_model_files"' in text
    assert '"metric_file"' in text
    assert "locked_metadata.tags" in text
    assert "OSI expression dialect" not in text
    assert "end_metric_generation" in text


def test_metrics_template_osi_mode_contract():
    text = _render("gen_metrics_system", "osi")
    assert "OSI (Open Semantic Interchange) core semantic model" in text
    # OSI metrics report a singular semantic_model_file in the final JSON.
    assert '"semantic_model_file"' in text
    assert "subject_path" in text
    assert "locked_metadata.tags" not in text.split("Record the classification")[1].split("\n")[0]
    assert "Covered by an existing base metric" in text


def test_semantic_model_template_includes_profiler_gate_both_formats():
    for authoring_format in ("metricflow", "osi"):
        text = _render("gen_semantic_model_system", authoring_format)
        assert "Optional SQL History & Distribution Profiling" in text, authoring_format
        assert 'load_skill("semantic-sql-history-profiler")' in text, authoring_format
        # Explicit-ask trigger: providing SQL alone must not trigger profiling.
        assert "Providing SQL alone is NOT a trigger" in text, authoring_format
        assert "still use it directly as modeling context" in text, authoring_format


def test_metrics_template_has_no_profiler_gate():
    for authoring_format in ("metricflow", "osi"):
        text = _render("gen_metrics_system", authoring_format)
        assert "semantic-sql-history-profiler" not in text, authoring_format


def test_latest_versions_resolve_to_shared_templates():
    pm = get_prompt_manager()
    assert pm.get_latest_version("gen_semantic_model_system") == "2.0"
    assert pm.get_latest_version("gen_metrics_system") == "2.0"


def test_legacy_osi_templates_are_removed():
    pm = get_prompt_manager()
    for template_name in ("gen_semantic_model_osi_system", "gen_metrics_osi_system"):
        assert not pm.list_template_versions(template_name), template_name


def test_rollback_anchor_versions_still_render():
    """Pinning prompt_version to the pre-refactor templates must keep working."""
    pm = get_prompt_manager()
    old_semantic = pm.render_template(template_name="gen_semantic_model_system", version="1.1")
    assert "MetricFlow" in old_semantic
    old_metrics = pm.render_template(template_name="gen_metrics_system", version="1.2")
    assert "MetricFlow" in old_metrics
