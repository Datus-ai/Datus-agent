from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "run-merge-queue-tests.py"


@pytest.fixture()
def run_merge_queue_tests():
    module_spec = importlib.util.spec_from_file_location("_test_run_merge_queue_tests", MODULE_PATH)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError(f"Unable to load run-merge-queue-tests from {MODULE_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_acceptance_integration_targets_filters_unit_targets(run_merge_queue_tests):
    targets = [
        "tests/unit_tests/agent/node/test_chat_agentic_node.py",
        "tests/integration/api/test_api.py",
        "tests/integration/cli/test_cli_commands.py",
    ]

    assert run_merge_queue_tests.acceptance_integration_targets(targets) == [
        "tests/integration/api/test_api.py",
        "tests/integration/cli/test_cli_commands.py",
    ]


def test_acceptance_unit_targets_filters_integration_targets(run_merge_queue_tests):
    targets = [
        "tests/unit_tests/agent/node/test_chat_agentic_node.py",
        "tests/integration/api/test_api.py",
        "tests/integration/cli/test_cli_commands.py",
    ]

    assert run_merge_queue_tests.acceptance_unit_targets(targets) == [
        "tests/unit_tests/agent/node/test_chat_agentic_node.py",
    ]


def test_merge_group_expands_fixed_suites_with_shared_impact_selection(run_merge_queue_tests):
    fixed = run_merge_queue_tests.suite_definitions()
    changed = ["datus/models/base.py", "datus/validation/hook.py"]
    suites = run_merge_queue_tests.suite_definitions(changed)
    for name, suite in fixed.items():
        assert suites[name] == suite
    targets = set(suites["impacted-unit"]["targets"])
    assert {"tests/unit_tests/models/", "tests/unit_tests/observability/", "tests/unit_tests/validation/"} <= targets


def test_build_pytest_command_includes_marker_junit_and_extra_args(tmp_path, run_merge_queue_tests):
    junit_xml = tmp_path / "results.xml"

    command = run_merge_queue_tests.build_pytest_command(
        ["tests/unit_tests/"],
        mark_expr="not nightly",
        junit_xml=junit_xml,
        extra_args=["--timeout=120", "-n", "auto"],
        basetemp=tmp_path / "pytest-temp" / "unit",
    )

    assert command[:4] == [run_merge_queue_tests.sys.executable, "-m", "pytest", "tests/unit_tests/"]
    assert command[4] == f"--basetemp={tmp_path / 'pytest-temp' / 'unit'}"
    assert command[5:7] == ["-m", "not nightly"]
    assert f"--junitxml={junit_xml}" in command
    assert ["--timeout=120", "-n", "auto"] == command[-3:]


def test_run_suite_fails_loudly_on_missing_target(tmp_path, monkeypatch, run_merge_queue_tests):
    monkeypatch.setattr(run_merge_queue_tests, "REPO_ROOT", tmp_path)
    suite = {
        "targets": ["tests/unit_tests/gone.py"],
        "mark_expr": "acceptance",
        "junit_xml": tmp_path / "results.xml",
        "extra_args": [],
    }

    result = run_merge_queue_tests.run_suite("acceptance-unit", suite, timeout=10)

    assert result["exit_code"] == 1
    assert result["missing"] == ["tests/unit_tests/gone.py"]
    assert result["targets"] == []


@pytest.mark.parametrize("name,expected", [("impacted-unit", 0), ("acceptance-unit", 5)])
def test_empty_collection_is_allowed_only_for_additional_tests(
    name, expected, tmp_path, monkeypatch, run_merge_queue_tests
):
    monkeypatch.setattr(run_merge_queue_tests, "run_command", lambda *args, **kwargs: 5)
    monkeypatch.setattr(run_merge_queue_tests, "PYTEST_BASETEMP", tmp_path / "pytest-temp")
    suite = {
        "targets": ["tests/unit_tests/models/test_request_capacity.py"],
        "mark_expr": "not component",
        "junit_xml": run_merge_queue_tests.OUT_DIR / "test-results-empty.xml",
        "extra_args": [],
    }
    assert run_merge_queue_tests.run_suite(name, suite, timeout=10)["exit_code"] == expected


@pytest.mark.component
def test_nested_runner_preserves_outer_pytest_temp_in_a_merge_group(tmp_path, monkeypatch, run_merge_queue_tests):
    """Run the real runner self-test with the same inherited env as merge queue."""
    outer = tmp_path / "outer-runner"
    outer.mkdir()
    sentinel = outer / "parent-owned.txt"
    sentinel.write_text("outer runner remains active")
    event = tmp_path / "merge-group.json"
    event.write_text(json.dumps({"merge_group": {"base_sha": "HEAD"}}))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "merge_group")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv(run_merge_queue_tests.PYTEST_BASETEMP_ENV, str(outer))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setattr(run_merge_queue_tests, "PYTEST_BASETEMP", outer)
    suite = "impacted-unit"
    command = [
        run_merge_queue_tests.sys.executable,
        "-m",
        "pytest",
        "tests/unit_tests/ci/test_run_merge_queue_tests.py::test_main_runs_selected_suite_and_writes_report",
        f"--basetemp={run_merge_queue_tests.suite_pytest_basetemp(suite)}",
        "-n",
        "2",
        "--dist=loadfile",
        "-q",
        "--no-showlocals",
        "-o",
        "log_cli=false",
    ]
    result = run_merge_queue_tests.run_command(command, suite_name=suite, timeout=60)
    assert result == 0
    assert sentinel.read_text() == "outer runner remains active"
    assert os.environ[run_merge_queue_tests.PYTEST_BASETEMP_ENV] == str(outer)


def test_main_runs_selected_suite_and_writes_report(tmp_path, monkeypatch, run_merge_queue_tests):
    repo_root = tmp_path
    out_dir = repo_root / "ci"
    tests_dir = repo_root / "tests" / "unit_tests"
    out_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    calls = []

    class CompletedProcess:
        returncode = 0

    def fake_run(command, **kwargs):
        calls.append({"command": command, "cwd": kwargs["cwd"], "timeout": kwargs["timeout"]})
        return CompletedProcess()

    monkeypatch.setattr(run_merge_queue_tests, "REPO_ROOT", repo_root)
    monkeypatch.setattr(run_merge_queue_tests, "OUT_DIR", out_dir)
    monkeypatch.setattr(run_merge_queue_tests, "DEFAULT_REPORT", out_dir / "merge-queue-results.json")
    monkeypatch.setattr(run_merge_queue_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_merge_queue_tests,
        "load_pr_harness_config",
        lambda: (["tests/unit_tests/"], "acceptance or component or llm_harness"),
    )

    assert run_merge_queue_tests.main(["--suite", "acceptance-unit", "--timeout", "10"]) == 0

    assert calls == [
        {
            "command": run_merge_queue_tests.build_pytest_command(
                ["tests/unit_tests/"],
                mark_expr="acceptance or component or llm_harness",
                junit_xml=out_dir / "test-results-merge-acceptance-unit.xml",
                extra_args=["--timeout=120", "--dist=loadscope", "-n", "auto"],
                basetemp=run_merge_queue_tests.suite_pytest_basetemp("acceptance-unit"),
            ),
            "cwd": repo_root,
            "timeout": 10,
        }
    ]
    assert '"status": "success"' in (out_dir / "merge-queue-results.json").read_text(encoding="utf-8")
