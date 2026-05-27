from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import defusedxml.ElementTree as ET

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "run-pr-tests.py"
MODULE_SPEC = importlib.util.spec_from_file_location("run_pr_tests", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise AssertionError(f"Unable to load run-pr-tests from {MODULE_PATH}")
run_pr_tests = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(run_pr_tests)


def test_select_impacted_unit_tests_maps_source_prefixes():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "datus/agent/workflow.py",
            "datus/storage/document/store.py",
            "datus/__init__.py",
        ]
    )

    assert impacted == [
        "tests/unit_tests/agent/",
        "tests/unit_tests/storage/",
        "tests/unit_tests/",
    ]


def test_select_impacted_unit_tests_includes_changed_unit_tests_and_dedupes():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "./tests/unit_tests/tools/test_registry.py",
            "datus/tools/registry.py",
            "datus/tools/search.py",
            "ci/run-pr-tests.py",
        ]
    )

    assert impacted == [
        "tests/unit_tests/tools/test_registry.py",
        "tests/unit_tests/tools/",
        "tests/unit_tests/ci/",
    ]


def test_select_impacted_unit_tests_maps_db_tools_to_db_tools_tests():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "datus/tools/db_tools/sqlite_connector.py",
            "datus/tools/db_tools/db_manager.py",
        ]
    )

    assert impacted == ["tests/unit_tests/tools/db_tools/"]


def test_select_impacted_unit_tests_maps_non_python_files_to_parent_directory():
    impacted = run_pr_tests.select_impacted_unit_tests(
        [
            "tests/unit_tests/tools/fixtures/data.json",
            "tests/unit_tests/fixtures/sample.yaml",
        ]
    )

    assert impacted == [
        "tests/unit_tests/tools/fixtures/",
        "tests/unit_tests/fixtures/",
    ]


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
    assert run_pr_tests.PR_HARNESS_MARK_EXPR == "acceptance or component or llm_harness"
    assert run_pr_tests.IMPACTED_UNIT_MARK_EXPR == (
        "not acceptance and not component and not llm_harness and not nightly and not quarantine"
    )


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
