"""OSI-mode generation prompt templates render and stay backend-agnostic.

These also assert that adding the OSI templates does not change which template
the default (metricflow) mode resolves to.
"""

from datus.prompts.prompt_manager import get_prompt_manager


def test_osi_metrics_template_is_backend_agnostic():
    pm = get_prompt_manager()
    text = pm.render_template(template_name="gen_metrics_osi_system")
    assert "OSI" in text
    # explicit boundary: the LLM must not emit execution-engine syntax
    assert "do NOT write MetricFlow YAML" in text
    assert "version: 0.2.0.dev0" in text
    assert "semantic_model:" in text
    assert "metrics:" in text
    assert "dialects:" in text
    assert "custom_extensions" in text
    assert '"dataset"' in text
    assert "not OSI core top-level metric fields" in text
    assert "metric:" in text  # only mentioned as a forbidden MetricFlow block
    assert '"status": "skipped"' in text
    assert "No metric generated" in text
    assert "from_columns" in text
    assert "to_columns" in text
    assert "Never put `relationships` inside a dataset" in text
    assert "analyze_metric_candidates_from_history" in text
    assert "metric_generation_skips" in text
    assert "offset_window" in text
    assert "window_aggregation" in text
    assert "Allowed values are `sum`, `avg`, `min`, `max`, `count`, and `row_count`" in text
    assert "ROW_NUMBER()`, `RANK() OVER`, TopN per group" in text


def test_osi_semantic_model_template_is_backend_agnostic():
    pm = get_prompt_manager()
    text = pm.render_template(template_name="gen_semantic_model_osi_system")
    assert "OSI" in text
    assert "version: 0.2.0.dev0" in text
    assert "semantic_model:" in text
    assert "datasets:" in text
    assert "fields:" in text
    assert "dialects:" in text
    assert "custom_extensions" in text
    assert "Dataset `source` is a string" in text
    assert "Dataset description and AI context are required for every dataset" in text
    assert "ai_context" in text
    assert "row grain" in text
    assert "relationships:" in text
    assert "from_columns" in text
    assert "to_columns" in text
    assert "do NOT write MetricFlow" in text
    assert "never inside a dataset" in text


def test_default_metricflow_templates_are_unchanged():
    pm = get_prompt_manager()
    # The default metricflow mode still resolves to its existing latest versions,
    # unaffected by the new OSI templates (separate template name).
    assert pm.get_latest_version("gen_metrics_system") == "1.2"
    assert pm.get_latest_version("gen_semantic_model_system") == "1.1"
    # OSI templates have their own independent versioning.
    assert pm.get_latest_version("gen_metrics_osi_system") == "1.0"
