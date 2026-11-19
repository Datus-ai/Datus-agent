"""Merged-tree consistency failures must reach the required queue check."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("merge_queue_tests", ROOT / "ci/run-merge-queue-tests.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("catalog_exit,map_exit", [(0, 0), (1, 0), (0, 1)])
def test_consistency_runs_both_checks_and_reports_failures(tmp_path, monkeypatch, catalog_exit, map_exit):
    for path, code in [
        ("datus/tools/registry/catalog_builder.py", catalog_exit),
        ("ci/harness/validate_coverage_map.py", map_exit),
    ]:
        script = tmp_path / path
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f"raise SystemExit({code})\n")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    report = tmp_path / "result.json"
    assert runner.main(["--report", str(report)]) == int(bool(catalog_exit or map_exit))
    result = json.loads(report.read_text())
    assert result["checks"] == [
        {"name": "generated-catalog", "exit_code": catalog_exit},
        {"name": "coverage-map", "exit_code": map_exit},
    ]


def test_timeout_fails_the_consistency_gate(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.run_checks(tmp_path, 3)
    assert result["status"] == "failure"
    assert len(result["checks"]) == 2
    assert all(check["exit_code"] == 1 for check in result["checks"])


@pytest.mark.parametrize("status,expected", [("success", 0), ("failure", 1), ("skipped", 1), ("cancelled", 1)])
def test_required_context_propagates_consistency_result(status, expected):
    workflow = yaml.safe_load((ROOT / ".github/workflows/merge-queue.yml").read_text())
    command = workflow["jobs"]["merge-queue-gate"]["steps"][0]["run"]
    result = subprocess.run(["bash", "-c", command], env={"DETERMINISTIC_RESULT": status}, timeout=5)
    assert result.returncode == expected
