from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[4]
SKILL = ROOT / "datus/resources/skills/dashboard-bootstrap/SKILL.md"


def _parts() -> tuple[dict, str]:
    text = SKILL.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body


def test_dashboard_bootstrap_skill_is_generic_and_discoverable():
    metadata, body = _parts()

    assert metadata == {
        "name": "dashboard-bootstrap",
        "description": metadata["description"],
    }
    assert "dashboard" in metadata["description"].lower()
    assert "reference SQL" in metadata["description"]
    assert "metrics" in metadata["description"]
    assert all(vendor not in body.lower() for vendor in ("superset", "tableau", "metabase"))


def test_dashboard_bootstrap_skill_has_ordered_selection_and_generation_gate():
    _, body = _parts()

    ordered_markers = (
        "Step 1 — Select a BI plugin and profile",
        "Step 2 — Select a dashboard",
        "Step 3 — Select queries for reference SQL",
        "Step 4 — Select queries for metrics",
        "Turn boundary — Emit the Generation Manifest",
        "Step 5 — Export the confirmed SQL",
        "Step 6 — Build reference-SQL context",
        "Step 7 — Build semantic-model and metric context",
        "Step 8 — Create dashboard subagents when supported",
        "Step 9 — Report",
    )
    offsets = [body.index(marker) for marker in ordered_markers]

    assert offsets == sorted(offsets)
    assert "STOP after the manifest and end the turn" in body
    assert "auto_run=true" in body
    assert "Never call an export command to inspect, probe, preview, or validate its contract" in body
    assert "temporary output roots such as `/tmp`" in body
    assert "Do not invoke or request help from the export subcommand itself before confirmation" in body
    assert "cannot contain exported file paths or checksums" in body


def test_dashboard_bootstrap_skill_routes_to_builtin_owners():
    _, body = _parts()

    assert 'task(\n  type="gen_sql_summary"' in body
    assert 'task(\n  type="semantic_modeling"' in body
    assert "Never combine multiple SQL queries" in body
    assert "partition successful SQL" in body
    assert "Never persist a BI-profile-level datasource mapping" in body
    assert "Dataset/table/schema equality" in body
    assert "matched_datus_datasource" in body
    assert "Never hand-write their YAML or index rows" in body


def test_dashboard_bootstrap_optionally_creates_legacy_shaped_subagents_last():
    _, body = _parts()

    assert "Inspect the skills available to the current main agent for `create-subagent`" in body
    assert "load `create-subagent`" in body
    assert "mutable configuration skill unavailable" in body
    assert "<platform>_<dashboard>" in body
    assert "<base-name>_attribution" in body
    assert "node_class: gen_sql" in body
    assert "node_class: gen_report" in body
    assert "context_search_tools,db_tools.search_table,db_tools.describe_table,db_tools.execute_sql" in body
    assert "semantic_tools,context_search_tools.list_subject_tree" in body
    assert "successful active-datasource artifacts" in body
    assert "creation failure does not invalidate context already built" in body
    assert "<metric.subject_path>.<metric.name>" in body
    assert "<subject_tree>.<name>" in body
    assert "returned `semantic_models` file" in body
    assert "every metric in that selected model" in body
    assert "returned `sql_summary_file`" in body
    assert "never store metric IDs" in body
    assert "datasource-only visibility" in body

    create_offset = body.index("Step 8 — Create dashboard subagents when supported")
    report_offset = body.index("Step 9 — Report")
    assert create_offset < report_offset
