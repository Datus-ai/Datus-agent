"""Analytics prompts follow the available tool surface and a read-only analysis contract."""

import pytest

from datus.prompts.prompt_manager import PromptManager
from datus.utils.path_manager import DatusPathManager


@pytest.mark.parametrize(
    "kind, role_boundary",
    [
        ("analytics_explorer", "Never modify files, create or publish artifacts"),
        ("analytics_builder", "Report creation, edits and validation only from successful"),
    ],
)
@pytest.mark.parametrize("interactive", [True, False])
def test_analytics_prompt_uses_available_tools_and_business_rules(tmp_path, kind, role_boundary, interactive):
    manager = PromptManager(path_manager=DatusPathManager(tmp_path / "home"))
    tools = [
        "search_metrics",
        "get_metric",
        "get_dataset",
        "query_metrics",
        "execute_sql",
        "list_tables",
        "describe_table",
    ]
    prompt = manager.render_template(
        f"{kind}_system",
        tool_names=tools,
        has_ask_user_tool=interactive,
        rules=["Revenue excludes refunds."],
    )
    assert "Revenue excludes refunds." in prompt
    assert "read-only SQL" in prompt
    assert "`query_metrics`" in prompt
    assert "`list_tables` and `describe_table`" in prompt
    assert ("`ask_user`" in prompt) == interactive
    assert "causal" in prompt
    # Role boundary: neither role may follow the main agent's authoring exceptions.
    assert "Never modify databases, semantic definitions, shared knowledge or configuration" in prompt
    assert "dashboard-bootstrap" not in prompt
    assert "update the relevant knowledge" not in prompt
    assert "`search_reference_sql`" not in prompt
    assert "`search_semantic_objects`" not in prompt
    assert role_boundary in prompt


def test_analytics_prompt_without_query_tools_explains_capability_gap(tmp_path):
    manager = PromptManager(path_manager=DatusPathManager(tmp_path / "home"))
    prompt = manager.render_template("analytics_explorer_system", tool_names=["read_file"], rules=[])
    assert "specific missing capability" in prompt
    assert "`query_metrics`" not in prompt
    assert "`execute_sql`" not in prompt
    assert "`read_file`" in prompt
    assert "file contents never grant additional database or metric access" in prompt
