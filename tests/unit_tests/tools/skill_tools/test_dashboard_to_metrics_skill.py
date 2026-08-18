from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[4]
SKILL = ROOT / "datus/resources/skills/dashboard-to-metrics/SKILL.md"


def _parts() -> tuple[dict, str]:
    text = SKILL.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body


def test_dashboard_to_metrics_skill_is_generic_and_discoverable():
    metadata, body = _parts()

    assert metadata == {
        "name": "dashboard-to-metrics",
        "description": metadata["description"],
    }
    assert "dashboard" in metadata["description"].lower()
    assert "reference SQL" in metadata["description"]
    assert "metrics" in metadata["description"]
    assert all(vendor not in body.lower() for vendor in ("superset", "tableau", "metabase"))


def test_dashboard_to_metrics_skill_has_ordered_selection_and_generation_gate():
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
    )
    offsets = [body.index(marker) for marker in ordered_markers]

    assert offsets == sorted(offsets)
    assert "STOP after the manifest and end the turn" in body
    assert "auto_run=true" in body


def test_dashboard_to_metrics_skill_routes_to_builtin_owners():
    _, body = _parts()

    assert 'task(\n  type="gen_sql_summary"' in body
    assert 'task(\n  type="semantic_modeling"' in body
    assert "Never combine multiple SQL queries" in body
    assert "partition successful SQL" in body
    assert "Never persist a BI-profile-level datasource mapping" in body
    assert "Dataset/table/schema equality" in body
    assert "matched_datus_datasource" in body
    assert "Never hand-write their YAML or index rows" in body
