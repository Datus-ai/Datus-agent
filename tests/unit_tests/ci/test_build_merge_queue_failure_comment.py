from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "build_merge_queue_failure_comment.py"


@pytest.fixture()
def build_merge_queue_failure_comment(monkeypatch):
    module_name = "_test_build_merge_queue_failure_comment"
    module_spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError(f"Unable to load build_merge_queue_failure_comment from {MODULE_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    module_spec.loader.exec_module(module)
    return module


def test_build_comment_includes_suite_and_junit_failure_details(tmp_path, build_merge_queue_failure_comment):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "merge-queue-results.json").write_text(
        dedent(
            """
            {
              "results": [
                {
                  "suite": "acceptance-integration",
                  "status": "failure",
                  "exit_code": 1,
                  "targets": ["tests/integration/cli/test_cli_commands.py"]
                }
              ]
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    nested_artifacts_dir = artifacts_dir / "nested"
    nested_artifacts_dir.mkdir()
    (nested_artifacts_dir / "merge-queue-results.json").write_text(
        dedent(
            """
            {
              "results": [
                {
                  "suite": "unit-tests",
                  "status": "success",
                  "exit_code": 0,
                  "targets": ["tests/unit_tests"]
                }
              ]
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "test-results-merge-acceptance-integration.xml").write_text(
        dedent(
            """
            <testsuites>
              <testsuite name="acceptance-integration" tests="1" failures="1">
                <testcase classname="tests.integration.cli.test_cli_commands" name="test_save_command">
                  <failure message="assert '/tmp/test_output.json' in stdout">
                    AssertionError: assert '/tmp/test_output.json' in 'Save Output'
                  </failure>
                </testcase>
              </testsuite>
            </testsuites>
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    comment = build_merge_queue_failure_comment.build_comment(
        artifacts_dir=artifacts_dir,
        repository="Datus-ai/Datus-agent",
        workflow="Merge Queue Gate",
        event_name="merge_group",
        ref="refs/heads/gh-readonly-queue/main/pr-814-abc123",
        sha="c315c3536ce61aab2729364679c14de62c0ef954",
        run_number="195",
        run_url="https://github.com/Datus-ai/Datus-agent/actions/runs/26038673148",
        failed_jobs="merge-queue-gate",
    )

    assert build_merge_queue_failure_comment.COMMENT_MARKER in comment
    assert "## Merge Queue Failure" in comment
    assert "synthetic queue ref" in comment
    assert "- Event: `merge_group`" in comment
    assert "- Queue ref: `refs/heads/gh-readonly-queue/main/pr-814-abc123`" in comment
    assert "- Failed jobs: `merge-queue-gate`" in comment
    assert "- Run: [#195](https://github.com/Datus-ai/Datus-agent/actions/runs/26038673148)" in comment
    assert "- `acceptance-integration`: failed, exit code `1`, targets `1`" in comment
    assert "- `unit-tests`: passed, exit code `0`, targets `1`" in comment
    assert "`tests.integration.cli.test_cli_commands::test_save_command`" in comment
    assert "assert '/tmp/test_output.json' in stdout" in comment
    assert "AssertionError: assert '/tmp/test_output.json' in 'Save Output'" in comment


def test_build_comment_explains_missing_artifacts(tmp_path, build_merge_queue_failure_comment):
    comment = build_merge_queue_failure_comment.build_comment(
        artifacts_dir=tmp_path / "missing-artifacts",
        repository="Datus-ai/Datus-agent",
        workflow="Merge Queue Gate",
        event_name="merge_group",
        ref="refs/heads/gh-readonly-queue/main/pr-814-abc123",
        sha="c315c3536ce61aab2729364679c14de62c0ef954",
        run_number="195",
        run_url="https://github.com/Datus-ai/Datus-agent/actions/runs/26038673148",
        failed_jobs="merge-queue-gate",
    )

    assert "- No merge queue suite summary artifact was found." in comment
    assert "No JUnit failure details were found in the merge queue artifacts. Use the linked run logs." in comment


def test_failed_jobs_from_needs_extracts_failed_jobs(build_merge_queue_failure_comment):
    needs_json = dedent(
        """
        {
          "merge-queue-gate": {"result": "failure"},
          "other-job": {"result": "success"}
        }
        """
    )

    assert build_merge_queue_failure_comment.failed_jobs_from_needs(needs_json) == "merge-queue-gate"


def test_main_writes_comment_file(tmp_path, build_merge_queue_failure_comment):
    output = tmp_path / "comment.md"

    assert (
        build_merge_queue_failure_comment.main(
            [
                "--artifacts-dir",
                str(tmp_path / "missing-artifacts"),
                "--output",
                str(output),
                "--repository",
                "Datus-ai/Datus-agent",
                "--workflow",
                "Merge Queue Gate",
                "--event",
                "merge_group",
                "--ref",
                "refs/heads/gh-readonly-queue/main/pr-814-abc123",
                "--sha",
                "c315c3536ce61aab2729364679c14de62c0ef954",
                "--run-number",
                "195",
                "--run-url",
                "https://github.com/Datus-ai/Datus-agent/actions/runs/26038673148",
                "--failed-jobs",
                "merge-queue-gate",
            ]
        )
        == 0
    )

    assert build_merge_queue_failure_comment.COMMENT_MARKER in output.read_text(encoding="utf-8")
