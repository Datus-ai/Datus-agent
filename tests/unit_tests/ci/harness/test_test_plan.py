"""Selection must include cross-module consumers and prove actual execution."""

import json
import os
import subprocess
import sys

import pytest

from ci.harness.test_plan import ROOT, check_execution, impact_plan

pytestmark = pytest.mark.acceptance


def test_model_changes_select_runtime_consumers():
    plan = impact_plan(["datus/models/base.py"])
    assert {"tests/unit_tests/models/", "tests/unit_tests/observability/", "tests/unit_tests/agent/node/"} <= set(
        plan["unit_targets"]
    )
    assert "tests/integration/agent/test_core_live.py" in plan["live_targets"]


def test_node_and_prompt_changes_select_live_product_questions():
    for path in [
        "datus/agent/node/chat_agentic_node.py",
        "datus/prompts/chat.yaml",
        "datus/resources/skills/init/SKILL.md",
    ]:
        assert "tests/integration/agent/test_core_live.py" in impact_plan([path])["live_targets"]


def test_chat_service_changes_select_the_api_lifecycle():
    plan = impact_plan(["datus/api/services/chat_task_manager.py"])
    assert "tests/unit_tests/api/" in plan["unit_targets"]
    assert "tests/integration/api/test_api_chat.py" in plan["live_targets"]


def test_documentation_changes_have_no_remote_model_requirement():
    assert impact_plan(["docs/index.md"])["live_targets"] == []


@pytest.mark.parametrize(
    "node,group",
    [
        ("semantic_modeling", "P0 Dosi Semantic Modeling E2E"),
        ("gen_dashboard", "P0 Dashboard Bootstrap Skill E2E"),
        ("scheduler", "Airflow Nightly Tests"),
    ],
)
def test_service_backed_nodes_keep_their_provisioned_group(node, group):
    plan = impact_plan([f"datus/agent/node/{node}_agentic_node.py"])
    assert group in plan["nightly_groups"]
    assert f"tests/integration/agent/test_{node}_agentic.py" in plan["live_targets"]


def test_merge_group_changes_use_the_event_base(monkeypatch, tmp_path):
    from ci.harness import test_plan

    event = tmp_path / "event.json"
    event.write_text(json.dumps({"merge_group": {"base_sha": "group-base"}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    calls = []

    def diff(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "datus/models/base.py\ndatus/api/app.py\n")

    monkeypatch.setattr(test_plan.subprocess, "run", diff)
    assert test_plan.changed_files() == ["datus/models/base.py", "datus/api/app.py"]
    assert calls[0][-1] == "group-base...HEAD"


def test_impact_targets_exist():
    from ci.harness.test_plan import load_map

    for rule in load_map()["execution"]["impact_rules"]:
        for target in rule.get("unit_targets", []) + rule.get("live_targets", []):
            assert (ROOT / target.split("::", 1)[0]).exists(), (rule["id"], target)


def _map():
    return {
        "execution": {"core_nodeids": ["test_core.py::test_query"]},
        "flow_groups": {
            "chat": {
                "flows": [
                    {"coverage": {"merge_queue": {"status": "covered", "nodeids": ["test_api.py::test_history"]}}}
                ]
            }
        },
    }


def test_execution_evidence_requires_both_product_and_api_cases(tmp_path):
    reports = []
    for name, nodeid in [("unit", "test_core.py::test_query"), ("integration", "test_api.py::test_history")]:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"collected": [nodeid], "outcomes": {nodeid: "passed"}}))
        reports.append(p)
    assert check_execution(reports, "merge_queue", _map())["status"] == "success"
    reports[1].unlink()
    assert check_execution(reports, "merge_queue", _map())["status"] == "failure"


@pytest.mark.parametrize("outcome", ["skipped", "failed", "xfailed", "not_executed"])
def test_required_cases_must_execute_successfully(tmp_path, outcome):
    report = tmp_path / "execution.json"
    report.write_text(
        json.dumps(
            {
                "collected": ["test_core.py::test_query", "test_api.py::test_history"],
                "outcomes": {"test_core.py::test_query": "passed", "test_api.py::test_history": outcome},
            }
        )
    )
    result = check_execution([report], "merge_queue", _map())
    assert result["status"] == "failure"
    assert "test_api.py::test_history" in result["errors"][0]


def test_pytest_report_distinguishes_collected_from_executed(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    source = tmp_path / "test_sample.py"
    source.write_text(
        "import pytest\ndef test_passes(): assert 1 + 1 == 2\n@pytest.mark.skip(reason='fixture unavailable')\ndef test_skipped(): pass\n"
    )
    report = tmp_path / "execution.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(source),
            "-p",
            "ci.pytest_gate_plugin",
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
    data = json.loads(report.read_text())
    assert set(data["collected"]) == {"test_sample.py::test_passes", "test_sample.py::test_skipped"}
    assert data["outcomes"] == {"test_sample.py::test_passes": "passed", "test_sample.py::test_skipped": "skipped"}
