"""Selection must include cross-module consumers and prove actual execution."""

import json
import os
import subprocess
import sys

import pytest
import yaml

from ci.harness.test_plan import ROOT, check_execution, check_product_execution, impact_plan

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


@pytest.mark.parametrize("path", ["datus/storage/feedback/store.py", "pyproject.toml", "uv.lock"])
def test_storage_changes_select_required_postgresql_acceptance(path):
    plan = impact_plan([path])
    assert "P0 PostgreSQL Agent Storage Contracts" in plan["nightly_groups"]
    assert (
        "tests/integration/storage/test_postgresql_feedback_store.py::"
        "test_postgresql_feedback_storage_write_read_update_and_delete"
    ) in plan["live_targets"]


def test_documentation_changes_have_no_remote_model_requirement():
    assert impact_plan(["docs/index.md"])["live_targets"] == []


@pytest.mark.parametrize(
    "node,group",
    [
        ("semantic_modeling", "P0 Dosi Semantic Modeling E2E"),
        ("gen_dashboard", "P0 BI Product E2E"),
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


@pytest.mark.parametrize("shallow", [True, False])
def test_nightly_base_fetch_finds_the_feature_merge_base(tmp_path, monkeypatch, shallow):
    from ci.harness import test_plan

    source = tmp_path / "source"
    source.mkdir()

    def git(*args, cwd=source):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    (source / "base.txt").write_text("common ancestor")
    git("add", ".")
    git("commit", "-m", "base")
    git("checkout", "-b", "feature")
    (source / "feature.txt").write_text("feature")
    git("add", ".")
    git("commit", "-m", "feature")
    git("checkout", "main")
    (source / "main.txt").write_text("main advanced")
    git("add", ".")
    git("commit", "-m", "main advanced")
    checkout = tmp_path / "checkout"
    depth_args = ["--depth", "1"] if shallow else []
    git("clone", *depth_args, "--branch", "feature", source.as_uri(), str(checkout))
    workflow = yaml.safe_load((ROOT / ".github/workflows/run-nightly.yml").read_text())
    step = next(s for s in workflow["jobs"]["run-nightly"]["steps"] if s.get("name") == "Fetch product validation base")
    output = tmp_path / "github-env"
    result = subprocess.run(
        ["bash", "-eu", "-c", step["run"]],
        cwd=checkout,
        env={**os.environ, "PRODUCT_BASE": "main", "GITHUB_ENV": str(output)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    base = output.read_text().strip().split("=", 1)[1]
    assert base == git("rev-parse", "main", cwd=source)
    monkeypatch.setattr(test_plan, "ROOT", checkout)
    assert test_plan.changed_files(base) == ["feature.txt"]


@pytest.mark.parametrize("product_marker", [False, True])
def test_each_planned_target_must_survive_marker_selection(tmp_path, product_marker):
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n nightly: nightly\n product_e2e: product\n")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    core = tmp_path / "test_core.py"
    ontology = tmp_path / "test_ontology.py"
    core.write_text("import pytest\npytestmark=[pytest.mark.nightly,pytest.mark.product_e2e]\ndef test_core(): pass\n")
    mark = ",pytest.mark.product_e2e" if product_marker else ""
    ontology.write_text(f"import pytest\npytestmark=[pytest.mark.nightly{mark}]\ndef test_ontology(): pass\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(core),
            str(ontology),
            "-m",
            "nightly and product_e2e",
            "-p",
            "ci.pytest_gate_plugin",
            "--gate-report",
            str(report_dir / "product.json"),
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"live_targets": ["test_core.py", "test_ontology.py"]}))
    verified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ci/plan-product-tests.py"),
            "--verify-execution",
            str(report_dir),
            "--plan",
            str(plan),
            "--output",
            str(tmp_path / "coverage.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verified.returncode == (0 if product_marker else 1), verified.stdout + verified.stderr
    data = json.loads((tmp_path / "coverage.json").read_text())
    assert data["status"] == ("success" if product_marker else "failure")


def test_dedicated_groups_provide_execution_evidence_for_their_targets(tmp_path):
    plan = {"live_targets": ["test_core.py", "test_scheduler.py::test_schedule"]}
    reports = []
    for group, nodeid in [("product", "test_core.py::test_chat"), ("airflow", "test_scheduler.py::test_schedule")]:
        path = tmp_path / f"{group}.json"
        path.write_text(json.dumps({"collected": [nodeid], "outcomes": {nodeid: "passed"}, "exit_code": 0}))
        reports.append(path)
    assert check_product_execution(plan, reports[:1])["status"] == "failure"
    assert check_product_execution(plan, reports)["status"] == "success"
