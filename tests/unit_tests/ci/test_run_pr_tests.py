from __future__ import annotations

import importlib.util
import io
import json
import subprocess
from pathlib import Path

import defusedxml.ElementTree as ET
import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "run-pr-tests.py"
MODULE_SPEC = importlib.util.spec_from_file_location("run_pr_tests", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise AssertionError(f"Unable to load run-pr-tests from {MODULE_PATH}")
run_pr_tests = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(run_pr_tests)


def _targets_cover(targets, *references):
    return all(
        any(reference == t or (t.endswith("/") and reference.startswith(t)) for t in targets)
        for reference in references
    )


def test_select_impacted_unit_tests_maps_source_prefixes():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "datus/agent/workflow.py",
            "datus/storage/document/store.py",
            "datus/__init__.py",
        ]
    )

    assert impacted == ["tests/unit_tests/"]


def test_select_impacted_unit_tests_includes_changed_unit_tests_and_dedupes():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "./tests/unit_tests/tools/test_registry.py",
            "tests/unit_tests/tools/test_registry.py",
            "tests/unit_tests/tools/conftest.py",
            "ci/run-pr-tests.py",
        ]
    )

    assert set(impacted) == {
        "tests/unit_tests/tools/",
        "tests/unit_tests/ci/",
    }


def test_select_impacted_unit_tests_collapses_parent_directory_over_child_files():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "tests/unit_tests/storage/conftest.py",
            "tests/unit_tests/storage/metric/test_metric_init.py",
            "tests/unit_tests/storage/metric/test_store.py",
            "tests/unit_tests/storage/test_knowledge_provenance.py",
        ]
    )

    assert impacted == ["tests/unit_tests/storage/"]


def test_select_impacted_unit_tests_retains_connector_and_shared_consumers():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "datus/tools/db_tools/sqlite_connector.py",
            "datus/tools/db_tools/db_manager.py",
        ]
    )

    assert _targets_cover(
        impacted,
        "tests/unit_tests/tools/db_tools/",
        "tests/unit_tests/tools/func_tool/test_database.py",
        "tests/unit_tests/agent/node/test_gen_sql_agentic_node.py",
    )


def test_select_impacted_unit_tests_keeps_prompt_node_and_skill_consumers():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "datus/prompts/prompt_templates/semantic_modeling_system_1.0.j2",
            "datus/resources/skills/dosi-semantic-authoring/SKILL.md",
        ]
    )

    assert _targets_cover(
        impacted,
        "tests/unit_tests/agent/node/",
        "tests/unit_tests/prompts/",
        "tests/unit_tests/tools/skill_tools/",
    )


def test_select_impacted_unit_tests_retains_cross_package_consumers():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "datus/agent/node/gen_sql_agentic_node.py",
            "datus/api/services/chat_task_manager.py",
            "datus/storage/metric/metric_init.py",
            "datus/storage/semantic_model/auto_create.py",
            "datus/tools/func_tool/database.py",
            "datus/tools/skill_tools/skill_registry.py",
            "datus/validation/target_extractor.py",
        ]
    )

    assert _targets_cover(
        impacted,
        "tests/unit_tests/agent/node/",
        "tests/unit_tests/agent/test_workflow.py",
        "tests/unit_tests/api/",
        "tests/unit_tests/cli/",
        "tests/unit_tests/tools/func_tool/",
        "tests/unit_tests/tools/skill_tools/",
        "tests/unit_tests/tools/db_tools/",
        "tests/unit_tests/storage/metric/",
        "tests/unit_tests/storage/semantic_model/",
        "tests/unit_tests/validation/",
    )


def test_select_impacted_unit_tests_maps_non_python_files_to_parent_directory():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "tests/unit_tests/tools/fixtures/data.json",
            "tests/unit_tests/fixtures/sample.yaml",
        ]
    )

    assert impacted == ["tests/unit_tests/"]


def test_filter_existing_paths_drops_missing_files(tmp_path):
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    real_file = tmp_path / "real_file.py"
    real_file.write_text("")

    filtered = run_pr_tests._filter_existing_paths(
        [
            str(real_dir) + "/",
            str(tmp_path / "missing_dir") + "/",
            str(real_file),
            str(tmp_path / "missing_file.py"),
        ]
    )

    assert filtered == [str(real_dir) + "/", str(real_file)]


def test_run_pytest_suite_isolates_nested_runner_basetemp(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("DATUS_TEST_STORAGE_BACKENDS", "postgresql+postgresql")

    class FakeProcess:
        pid = 12345
        stdout = io.StringIO("")

        def wait(self, timeout):
            return 0

    def fake_popen(command, **kwargs):
        calls.append({"command": command, "env": kwargs["env"]})
        return FakeProcess()

    monkeypatch.setattr(run_pr_tests, "DEFAULT_PYTEST_BASETEMP", str(tmp_path / "pytest-root"))
    monkeypatch.setattr(run_pr_tests, "DEFAULT_COVERAGE_DB", str(tmp_path / ".coverage"))
    monkeypatch.setattr(run_pr_tests.subprocess, "Popen", fake_popen)

    log_file = io.StringIO()
    assert (
        run_pr_tests._run_pytest_suite(
            ["tests/unit_tests/ci/"],
            str(tmp_path / "results.xml"),
            log_file,
            suite_name="impacted unit tests",
        )
        == 0
    )

    suite_basetemp = tmp_path / "pytest-root" / "impacted-unit-tests"
    assert calls[0]["command"][:3] == [run_pr_tests.sys.executable, "-m", "pytest"]
    assert f"--basetemp={suite_basetemp}" in calls[0]["command"]
    assert calls[0]["env"][run_pr_tests.PYTEST_BASETEMP_ENV] == str(suite_basetemp / "_nested-runners")
    assert calls[0]["env"]["DATUS_TEST_STORAGE_BACKENDS"] == "sqlite+lance"


def test_merge_and_parse_junit_results_across_multiple_suites(tmp_path):
    suite_a = tmp_path / "suite-a.xml"
    suite_b = tmp_path / "suite-b.xml"
    merged = tmp_path / "merged.xml"

    suite_a.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="acceptance" tests="2" failures="1" errors="0" skipped="0" time="1.2">
  <testcase classname="tests.a" name="test_ok" time="0.1" />
  <testcase classname="tests.a" name="test_fail" time="0.2">
    <failure message="boom">stacktrace-a</failure>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )
    suite_b.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="unit" tests="3" failures="0" errors="1" skipped="1" time="2.4">
  <testcase classname="tests.b" name="test_ok" time="0.1" />
  <testcase classname="tests.b" name="test_skip" time="0.1">
    <skipped />
  </testcase>
  <testcase classname="tests.b" name="test_error" time="0.3">
    <error message="kaput">stacktrace-b</error>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    run_pr_tests.merge_junit_results([str(suite_a), str(suite_b)], output_path=str(merged))
    parsed = run_pr_tests.parse_test_results([str(merged)])

    merged_root = ET.parse(merged).getroot()

    assert merged_root.attrib["tests"] == "5"
    assert merged_root.attrib["failures"] == "1"
    assert merged_root.attrib["errors"] == "1"
    assert merged_root.attrib["skipped"] == "1"
    assert parsed["total"] == 5
    assert parsed["passed"] == 2
    assert parsed["failed"] == 1
    assert parsed["errors"] == 1
    assert parsed["skipped"] == 1
    assert [failure["name"] for failure in parsed["failures"]] == ["test_fail", "test_error"]


def test_run_tests_treats_empty_impacted_collection_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pr_tests, "_reset_report_outputs", lambda: None)
    monkeypatch.setattr(run_pr_tests, "DEFAULT_PYTEST_LOG", str(tmp_path / "pytest-coverage.txt"))
    monkeypatch.setattr(run_pr_tests, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(run_pr_tests, "DEFAULT_COVERAGE_DB", str(tmp_path / ".coverage"))
    monkeypatch.setattr(run_pr_tests, "resolve_impacted_unit_tests", lambda base_ref: ["tests/unit_tests/"])
    monkeypatch.setattr(run_pr_tests, "merge_junit_results", lambda junit_xml_paths: None)
    monkeypatch.setattr(run_pr_tests, "verify_gate", lambda *args: 0)

    exit_codes = iter([0, 5])
    monkeypatch.setattr(
        run_pr_tests,
        "_run_pytest_suite",
        lambda *args, **kwargs: next(exit_codes),
    )

    exit_code, junit_paths = run_pr_tests.run_tests(base_ref="main")

    assert exit_code == 0
    assert junit_paths == [
        str(tmp_path / "test-results-acceptance.xml"),
        str(tmp_path / "test-results-impacted-unit.xml"),
    ]


def test_pr_harness_marker_expressions_route_component_tests():
    assert run_pr_tests.PR_HARNESS_MARK_EXPR == "(acceptance or component or llm_harness) and not quarantine"
    assert run_pr_tests.IMPACTED_UNIT_MARK_EXPR == (
        "not acceptance and not component and not llm_harness and not nightly and not quarantine"
    )


@pytest.mark.parametrize(
    ("cpu_count", "expected_workers"),
    [
        (None, "1"),
        (1, "1"),
        (2, "1"),
        (3, "1"),
        (4, "2"),
        (8, "4"),
    ],
)
def test_parallel_worker_default_uses_half_available_cpus(monkeypatch, cpu_count, expected_workers):
    monkeypatch.delenv("IMPACTED_UNIT_PARALLEL_WORKERS", raising=False)
    monkeypatch.setattr(run_pr_tests.os, "cpu_count", lambda: cpu_count)

    assert run_pr_tests._resolve_parallel_workers() == expected_workers


def test_parallel_worker_env_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("IMPACTED_UNIT_PARALLEL_WORKERS", "6")
    monkeypatch.setattr(run_pr_tests.os, "cpu_count", lambda: 16)

    assert run_pr_tests._resolve_parallel_workers() == "6"


def test_parallel_suite_uses_configured_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pr_tests, "PARALLEL_WORKERS", "4")
    command = run_pr_tests._build_pytest_command(
        ["tests/unit_tests/"],
        str(tmp_path / "junit.xml"),
        parallel=True,
    )

    worker_flag = command.index("-n")
    assert command[worker_flag : worker_flag + 2] == ["-n", "4"]
    assert "--dist" not in command


def test_parallel_suite_emits_dist_mode_after_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pr_tests, "PARALLEL_WORKERS", "4")
    command = run_pr_tests._build_pytest_command(
        ["tests/unit_tests/"],
        str(tmp_path / "junit.xml"),
        parallel=True,
        dist="loadfile",
    )

    worker_flag = command.index("-n")
    assert command[worker_flag : worker_flag + 5] == ["-n", "4", "--max-worker-restart=0", "--dist", "loadfile"]


def test_parallel_suite_writes_failure_report_when_a_worker_crashes(tmp_path, monkeypatch):
    """Completed file scopes must not strand loadfile's worker recovery queue."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_large.py").write_text(
        "import time, pytest\n@pytest.mark.parametrize('item', range(80))\ndef test_large(item): time.sleep(0.005)\n"
    )
    for index in range(8):
        (tmp_path / f"test_small_{index}.py").write_text(
            "import pytest\n@pytest.mark.parametrize('item', range(4))\ndef test_small(item): pass\n"
        )
    (tmp_path / "test_crash.py").write_text(
        "import os, pytest\n"
        "@pytest.mark.parametrize('item', range(3))\n"
        "def test_crash(item):\n"
        "    if item == 1: os._exit(17)\n"
    )
    monkeypatch.chdir(tmp_path)
    # The synthetic nested pytest owns its temp directory and plugins. Nightly's
    # outer --basetemp can be a parent of tmp_path and is invalid in this cwd.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setattr(run_pr_tests, "PARALLEL_WORKERS", "2")
    monkeypatch.setattr(run_pr_tests, "TEST_CMD_TIMEOUT", 15)
    monkeypatch.setattr(run_pr_tests, "DEFAULT_PYTEST_BASETEMP", str(tmp_path / "pytest-temp"))
    monkeypatch.setattr(run_pr_tests, "DEFAULT_COVERAGE_DB", str(tmp_path / ".coverage"))
    monkeypatch.setattr(run_pr_tests, "DEFAULT_COVERAGE_XML", str(tmp_path / "coverage.xml"))
    junit = tmp_path / "results.xml"
    log = io.StringIO()

    exit_code = run_pr_tests._run_pytest_suite(
        [str(tmp_path)], str(junit), log, suite_name="worker crash regression", parallel=True, dist="loadfile"
    )

    assert exit_code == 1, log.getvalue()
    results = run_pr_tests.parse_test_results(str(junit))
    assert results["failed"] + results["errors"] == 1
    assert "worker" in results["failures"][0]["text"]
    assert "TIMEOUT" not in log.getvalue()
    execution = json.loads(Path(str(junit) + ".execution.json").read_text())
    assert execution["duration_seconds"] > 0


def test_serial_suite_ignores_dist_mode(tmp_path):
    command = run_pr_tests._build_pytest_command(
        ["tests/unit_tests/"],
        str(tmp_path / "junit.xml"),
        parallel=False,
        dist="loadfile",
    )

    assert "-n" not in command
    assert "--dist" not in command


def test_emit_reports_produces_only_cobertura_xml(tmp_path):
    command = run_pr_tests._build_pytest_command(
        ["tests/unit_tests/"],
        str(tmp_path / "junit.xml"),
    )

    cov_reports = [arg for arg in command if arg.startswith("--cov-report")]
    assert cov_reports == [f"--cov-report=xml:{run_pr_tests.DEFAULT_COVERAGE_XML}"]


def test_ci_output_overrides_verbose_uncaptured_project_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -s -vv\nlog_cli = true\n")
    (tmp_path / "conftest.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def pytest_sessionstart(session):\n"
        "    Path(__file__).with_name('options.json').write_text(json.dumps({"
        "'verbosity': session.config.get_verbosity('test_cases'), 'capture': session.config.option.capture}))\n"
    )
    (tmp_path / "test_sample.py").write_text("def test_pass():\n    print('test output')\n    assert True\n")
    monkeypatch.setattr(run_pr_tests, "DEFAULT_COVERAGE_DB", str(tmp_path / ".coverage"))
    monkeypatch.setattr(run_pr_tests, "DEFAULT_PYTEST_BASETEMP", str(tmp_path / "temp"))
    log = io.StringIO()
    assert (
        run_pr_tests._run_pytest_suite(
            [str(tmp_path)], str(tmp_path / "results.xml"), log, suite_name="output", emit_reports=False
        )
        == 0
    )
    assert json.loads((tmp_path / "options.json").read_text()) == {"verbosity": 0, "capture": "fd"}


def test_suppressed_reports_disable_cov_output(tmp_path):
    command = run_pr_tests._build_pytest_command(
        ["tests/unit_tests/"],
        str(tmp_path / "junit.xml"),
        emit_reports=False,
    )

    cov_reports = [arg for arg in command if arg.startswith("--cov-report")]
    assert cov_reports == ["--cov-report="]


def test_run_tests_runs_acceptance_parallel_with_per_file_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pr_tests, "_reset_report_outputs", lambda: None)
    monkeypatch.setattr(run_pr_tests, "DEFAULT_PYTEST_LOG", str(tmp_path / "pytest-coverage.txt"))
    monkeypatch.setattr(run_pr_tests, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(run_pr_tests, "DEFAULT_COVERAGE_DB", str(tmp_path / ".coverage"))
    monkeypatch.setattr(run_pr_tests, "resolve_impacted_unit_tests", lambda base_ref: ["tests/unit_tests/"])
    monkeypatch.setattr(run_pr_tests, "merge_junit_results", lambda junit_xml_paths: None)
    monkeypatch.setattr(run_pr_tests, "verify_gate", lambda *args: 0)

    suite_calls = []

    def fake_suite(targets, junit_xml, log_file, **kwargs):
        suite_calls.append(kwargs)
        return 0

    monkeypatch.setattr(run_pr_tests, "_run_pytest_suite", fake_suite)

    exit_code, _ = run_pr_tests.run_tests(base_ref="main")

    assert exit_code == 0
    acceptance_call = next(call for call in suite_calls if call["suite_name"] == "acceptance")
    assert acceptance_call["parallel"] is True
    assert acceptance_call["dist"] == "loadfile"
    impacted_call = next(call for call in suite_calls if call["suite_name"] == "impacted unit tests")
    assert impacted_call["parallel"] is True
    assert impacted_call.get("dist") is None


def test_resolve_explicit_compare_ref_prefers_origin_for_branch_name(monkeypatch):
    monkeypatch.setattr(run_pr_tests, "_git_ref_exists", lambda ref: ref == "origin/main")

    assert run_pr_tests._resolve_explicit_compare_ref("main") == "origin/main"


def test_resolve_explicit_compare_ref_prefers_origin_for_slash_branch_name(monkeypatch):
    monkeypatch.setattr(run_pr_tests, "_git_ref_exists", lambda ref: ref == "origin/release/0.3.2")

    assert run_pr_tests._resolve_explicit_compare_ref("release/0.3.2") == "origin/release/0.3.2"


def test_resolve_explicit_compare_ref_accepts_remote_ref(monkeypatch):
    monkeypatch.setattr(run_pr_tests, "_git_ref_exists", lambda ref: ref == "upstream/main")

    assert run_pr_tests._resolve_explicit_compare_ref("upstream/main") == "upstream/main"


def test_find_compare_branch_caches_resolved_explicit_base_ref(monkeypatch):
    monkeypatch.setattr(run_pr_tests, "_COMPARE_BRANCH_CACHE", {})
    calls = []

    def fake_ref_exists(ref):
        calls.append(ref)
        return ref == "upstream/main"

    monkeypatch.setattr(run_pr_tests, "_git_ref_exists", fake_ref_exists)

    assert run_pr_tests.find_compare_branch("upstream/main") == "upstream/main"
    assert run_pr_tests.find_compare_branch("upstream/main") == "upstream/main"
    assert calls == ["upstream/main"]


def test_selection_report_preserves_owners_even_when_a_fallback_subsumes_them(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pr_tests, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(run_pr_tests, "list_changed_files", lambda base: ["datus/mcp_server.py", "datus/new_module.py"])
    assert run_pr_tests.resolve_impacted_unit_tests("main") == ["tests/unit_tests/"]
    plan = json.loads((tmp_path / "unit-test-plan.json").read_text())
    assert [selection["rule"] for selection in plan["selections"]] == ["mcp-runtime", "datus"]
    assert plan["selections"][1]["fallback"] is True


def test_pr_diff_includes_both_owners_of_a_renamed_module(tmp_path, monkeypatch):
    from ci.harness import test_plan

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    git("init")
    git("config", "user.email", "mapping-test@example.invalid")
    git("config", "user.name", "Mapping Test")
    old = tmp_path / "datus/models/mcp_utils.py"
    old.parent.mkdir(parents=True)
    old.write_text("VALUE = 42\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    new = tmp_path / "datus/utils/mcp_execution.py"
    new.parent.mkdir()
    git("mv", old.relative_to(tmp_path).as_posix(), new.relative_to(tmp_path).as_posix())
    git("commit", "-m", "move")
    monkeypatch.setattr(test_plan, "ROOT", tmp_path)
    monkeypatch.setattr(run_pr_tests, "find_compare_branch", lambda ref: base)
    changed = run_pr_tests.list_changed_files("base")
    assert set(changed) == {"datus/models/mcp_utils.py", "datus/utils/mcp_execution.py"}
    targets = run_pr_tests.select_impacted_unit_tests(changed)
    assert "tests/unit_tests/models/test_mcp_utils.py" in targets
    assert "tests/unit_tests/utils/test_mcp_execution.py" in targets


def test_main_returns_test_exit_code(monkeypatch):
    monkeypatch.setattr(
        run_pr_tests.argparse.ArgumentParser, "parse_args", lambda self: type("Args", (), {"base_ref": "main"})()
    )
    monkeypatch.setattr(run_pr_tests, "run_tests", lambda base_ref="": (3, ["report.xml"]))
    monkeypatch.setattr(
        run_pr_tests,
        "parse_test_results",
        lambda junit_xml_paths=None: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "failures": [],
        },
    )
    monkeypatch.setattr(run_pr_tests, "write_test_report", lambda test_results, output_path=None: "")
    monkeypatch.setattr(run_pr_tests, "extract_coverage", lambda base_ref: (0.0, 0.0))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert run_pr_tests.main() == 3
