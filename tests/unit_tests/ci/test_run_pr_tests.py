from __future__ import annotations

import importlib.util
from pathlib import Path

import defusedxml.ElementTree as ET

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "run-pr-tests.py"
MODULE_SPEC = importlib.util.spec_from_file_location("run_pr_tests", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
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
