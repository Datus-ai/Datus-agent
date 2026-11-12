"""Core journeys remain mandatory while domain tests follow the changed paths."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ci.harness.pr_test_plan import SHARED_PATHS, pr_test_plan, validate_integration_consumers
from ci.harness.test_plan import ROOT, collapse_targets, load_map

pytestmark = pytest.mark.acceptance


def test_core_journeys_are_selected_for_a_leaf_change():
    plan = pr_test_plan(["datus/utils/sql_lineage.py"])
    assert all(
        any(
            node == target or node.startswith(target + "::") or (target.endswith("/") and node.startswith(target))
            for target in plan["pytest_targets"]
        )
        for node in load_map()["execution"]["core_nodeids"]
    )
    assert "tests/unit_tests/utils/test_sql_lineage.py" in plan["pytest_targets"]
    assert plan["integration_groups"] == []


def test_domain_change_and_shared_configuration_select_integration_contracts():
    data = load_map()
    groups = {rule["id"] for rule in data["execution"]["integration_rules"]}
    assert pr_test_plan(["datus/cli/chat_commands.py"])["integration_groups"] == ["cli"]
    assert set(pr_test_plan(["datus/configuration/agent_config.py"])["integration_groups"]) == groups
    assert all((ROOT / path).exists() for path in SHARED_PATHS)


@pytest.mark.parametrize(
    "source,target",
    [
        ("datus/api/services/dashboard_service.py", "tests/integration/tools/test_dosi_query_filters.py"),
        ("datus/tools/func_tool/metric_artifact_tools.py", "tests/integration/tools/test_dosi_query_filters.py"),
        ("datus/tools/func_tool/dashboard_artifact_tools.py", "tests/integration/tools/test_dosi_query_filters.py"),
        ("datus/utils/metric_artifact_models.py", "tests/integration/tools/test_dosi_query_filters.py"),
        ("datus/utils/mcp_execution.py", "tests/integration/tools/test_mcp_acceptance.py"),
        ("datus/utils/mcp_decorators.py", "tests/integration/tools/test_mcp_acceptance.py"),
        ("datus/tools/search_tools/search_tool.py", "tests/integration/storage/test_platform_doc.py"),
    ],
)
def test_cross_package_changes_keep_their_real_integration_contract(source, target):
    assert target in pr_test_plan([source])["pytest_targets"]


def test_integration_rule_audit_rejects_omitted_cross_package_consumer():
    data = load_map()
    rule = next(rule for rule in data["execution"]["integration_rules"] if rule["id"] == "semantic")
    rule["paths"].remove("datus/api/services/dashboard_service.py")
    errors = validate_integration_consumers(data)
    assert any(
        "dashboard_service.py -> tests/integration/tools/test_dosi_query_filters.py" in error for error in errors
    )


@pytest.mark.parametrize(
    "missing", ["tests/integration/tools/test_dosi_attribution.py", "tests/integration/storage/test_doc_search.py"]
)
def test_stale_integration_target_fails_before_selection_and_strict_validation(monkeypatch, missing):
    from ci.harness.validate_coverage_map import validate_coverage_map

    missing_path = ROOT / missing
    is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: False if path == missing_path else is_file(path))
    with pytest.raises(ValueError, match="integration target does not exist"):
        pr_test_plan(["datus/models/accepted_turn.py"])
    errors, _ = validate_coverage_map(ROOT, load_map())
    assert any(f"integration target does not exist: {missing}" in error for error in errors)


def test_deleted_unregistered_diff_test_is_not_a_stale_configured_target():
    deleted = "tests/integration/test_deleted_file.py"
    plan = pr_test_plan([deleted])
    assert plan["deleted_targets"] == [deleted]
    assert deleted not in plan["pytest_targets"]


def test_whole_file_selection_collects_core_tests_once():
    assert collapse_targets(["tests/a.py::test_core", "tests/a.py", "tests/b.py::test_core"]) == [
        "tests/a.py",
        "tests/b.py::test_core",
    ]


def test_missing_mandatory_core_is_an_error():
    data = load_map()
    data["execution"]["core_nodeids"].append("tests/deleted.py::test_core")
    with pytest.raises(ValueError, match="Missing mandatory core"):
        pr_test_plan([], data)


def test_pr_collection_keeps_marked_units_and_deterministic_integration_only(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    cases = {
        "tests/unit_tests/test_owner.py": """
import pytest
def test_unit(): pass
@pytest.mark.component
def test_component(): pass
@pytest.mark.llm_harness
def test_harness(): pass
@pytest.mark.acceptance
def test_acceptance(): pass
@pytest.mark.nightly
def test_remote(): raise AssertionError('nightly only')
@pytest.mark.quarantine
def test_quarantined(): raise AssertionError('quarantine')
""",
        "tests/integration/test_domain.py": """
import pytest
@pytest.mark.acceptance
def test_contract(): pass
def test_external(): raise AssertionError('unclassified integration')
@pytest.mark.acceptance
@pytest.mark.nightly
def test_live_contract(): raise AssertionError('nightly only')
""",
    }
    for name, source in cases.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    report = tmp_path / "report.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "ci.pytest_gate_plugin",
            "--pr-gate",
            "--gate-report",
            str(report),
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    execution = json.loads(report.read_text())
    assert {node.rsplit("::", 1)[-1] for node in execution["collected"]} == {
        "test_unit",
        "test_component",
        "test_harness",
        "test_acceptance",
        "test_contract",
    }
    assert set(execution["outcomes"].values()) == {"passed"}
