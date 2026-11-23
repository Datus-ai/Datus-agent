"""Compose teardown errors reach the final gate while other projects clean up."""

import json
import os
import subprocess

import pytest

from tests.unit_tests.ci.test_superset_nightly_isolation import SCRIPT, source


@pytest.mark.parametrize("initial,expected", [(0, 1), (7, 7)])
def test_real_compose_cleanup_failure_reaches_exit_receipt_and_github_output(tmp_path, initial, expected):
    result, events, output, receipt = run_cleanup(tmp_path, initial=initial)
    assert result.returncode == expected, result.stdout + result.stderr
    assert events == ["owned-postgresql|down -v --remove-orphans", "owned-mysql|down -v --remove-orphans"]
    assert output.read_text().splitlines()[-1] == f"test_exit_code={expected}"
    assert json.loads(receipt.read_text()) == {
        "status": "failed",
        "cleanup_status": "failed",
        "initial_exit_code": initial,
        "cleanup_exit_code": 1,
        "final_exit_code": expected,
    }


def test_cleanup_only_returns_failure_and_continues_other_owned_projects(tmp_path):
    result, events, _, _ = run_cleanup(tmp_path, cleanup_only=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert events == ["owned-postgresql|down -v --remove-orphans", "owned-mysql|down -v --remove-orphans"]


def test_finish_preserves_test_failure_when_special_cleanup_sets_exit_code():
    result = subprocess.run(
        [
            "bash",
            "-c",
            "test_exit_code=7\n"
            "cleanup_hana_nightly_environment() { test_exit_code=1; return 1; }\n"
            + source("finish_compose_suite")
            + '\nfinish_compose_suite "HANA Agent Tests" unused owned\nstatus=$?\nprintf "%s:%s" "$status" "$test_exit_code"',
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "1:7"


def run_cleanup(tmp_path, *, initial=0, cleanup_only=False):
    compose = tmp_path / "compose.yml"
    compose.touch()
    repo = tmp_path / "repo"
    (repo / "ci").mkdir(parents=True)
    cleanup = repo / "ci/cleanup-run-testcontainers.sh"
    cleanup.write_text("#!/bin/sh\nexit 0\n")
    cleanup.chmod(0o700)
    events = tmp_path / "events"
    output = tmp_path / "output"
    receipt = tmp_path / "receipt.json"
    output.write_text("test_exit_code=0\n")
    functions = [
        "set -uo pipefail",
        "COMPOSE_GROUPS=('PostgreSQL Adapter Tests' 'MySQL Adapter Tests')",
        "has_docker_compose() { return 0; }",
        'compose_cmd() { printf "%s|%s %s %s\\n" "$1" "$3" "$4" "$5" >> "$EVENTS"; [ "$1" != owned-postgresql ]; }',
        "restore_agent_test_config() { :; }; cleanup_gaussdb_tls_host_artifacts() { :; }",
        "validate_pytest_basetemp() { return 1; }; cleanup_vision_nightly_environment() { return 0; }",
        source("compose_project_slug"),
        source("compose_project_name"),
        source("compose_down"),
        source("cleanup_all_compose"),
        source("cleanup_all"),
        source("finalize_nightly_exit"),
    ]
    if cleanup_only:
        entry = SCRIPT.read_text().split('if [ "${1:-}" = "--cleanup-only" ]; then', 1)[1].split("\nfi", 1)[0]
        functions.append(entry)
    else:
        functions += ["trap 'finalize_nightly_exit \"$?\"' EXIT", f"exit {initial}"]
    result = subprocess.run(
        ["bash", "-c", "\n".join(functions)],
        env={
            **os.environ,
            "NIGHTLY_SKIP_MANIFEST_FINALIZE": "1",
            "NIGHTLY_COMPOSE_PROJECT_PREFIX": "owned-",
            "POSTGRES_COMPOSE": str(compose),
            "MYSQL_COMPOSE": str(compose),
            "AGENT_TEST_CONFIG_BACKUP": str(tmp_path / "backup"),
            "NIGHTLY_PYTEST_BASETEMP": str(tmp_path / "basetemp"),
            "REPO_ROOT": str(repo),
            "EVENTS": str(events),
            "GITHUB_OUTPUT": str(output),
            "NIGHTLY_CLEANUP_STATUS_FILE": str(receipt),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result, events.read_text().splitlines(), output, receipt


@pytest.mark.parametrize(
    ("initial", "cleanup_fails", "expected_finalizations"),
    [
        (0, True, ["0", "1"]),
        (0, "sets-exit-code", ["0", "1"]),
        (0, False, ["0"]),
        (7, True, ["7"]),
    ],
)
def test_manifest_and_reports_carry_the_post_cleanup_exit_code(
    tmp_path, initial, cleanup_fails, expected_finalizations
):
    """Passing tests followed by failed teardown must not leave exit_code=0 in the evidence."""
    events = tmp_path / "events"
    repo = tmp_path / "repo"
    (repo / "ci").mkdir(parents=True)
    cleanup = repo / "ci/cleanup-run-testcontainers.sh"
    cleanup.write_text("#!/bin/sh\nexit 0\n")
    cleanup.chmod(0o700)
    functions = [
        "set -uo pipefail",
        f"test_exit_code={initial}",
        "NIGHTLY_MANIFEST_FINALIZED=0; PROVIDER_COVERAGE_MANIFEST_FINALIZED=0; NIGHTLY_FAILURE_CLASSIFICATION_FINALIZED=0",
        'manifest_update() { printf "manifest %s\\n" "$*" >> "$EVENTS"; }',
        'provider_coverage_update() { printf "provider\\n" >> "$EVENTS"; echo x > "$PROVIDER_COVERAGE_MANIFEST_FILE"; }',
        'failure_classification_update() { printf "classify %s\\n" "$*" >> "$EVENTS"; echo x > "$NIGHTLY_FAILURE_CLASSIFICATION_FILE"; }',
        "collect_nightly_trace_diagnostics() { :; }",
        "restore_agent_test_config() { :; }; cleanup_gaussdb_tls_host_artifacts() { :; }",
        "validate_pytest_basetemp() { return 1; }; cleanup_vision_nightly_environment() { return 0; }",
        # HANA teardown also flips test_exit_code while reporting its failure.
        "cleanup_all_compose() { test_exit_code=1; return 1; }"
        if cleanup_fails == "sets-exit-code"
        else f"cleanup_all_compose() {{ return {1 if cleanup_fails else 0}; }}",
        source("manifest_finalize"),
        source("provider_coverage_finalize"),
        source("failure_classification_finalize"),
        source("cleanup_all"),
        source("finalize_nightly_exit"),
        "trap 'finalize_nightly_exit \"$?\"' EXIT",
        f"exit {initial}",
    ]
    result = subprocess.run(
        ["bash", "-c", "\n".join(functions)],
        env={
            **os.environ,
            "NIGHTLY_MANIFEST_FILE": str(tmp_path / "manifest.json"),
            "PROVIDER_COVERAGE_MANIFEST_FILE": str(tmp_path / "provider.json"),
            "NIGHTLY_FAILURE_CLASSIFICATION_FILE": str(tmp_path / "classification.json"),
            "LOG_FILE": str(tmp_path / "log"),
            "AGENT_TEST_CONFIG_BACKUP": str(tmp_path / "backup"),
            "NIGHTLY_PYTEST_BASETEMP": str(tmp_path / "basetemp"),
            "REPO_ROOT": str(repo),
            "EVENTS": str(events),
            "NIGHTLY_CLEANUP_STATUS_FILE": str(tmp_path / "receipt.json"),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    final = int(expected_finalizations[-1])
    assert result.returncode == final, result.stdout + result.stderr
    lines = events.read_text().splitlines()
    manifests = [line.rsplit("--exit-code ", 1)[1] for line in lines if line.startswith("manifest finalize")]
    assert manifests == expected_finalizations
    classifications = [line for line in lines if line.startswith("classify")]
    assert len(classifications) == len(expected_finalizations)
    assert classifications[-1].endswith(f"--exit-code {final}")
    assert lines.count("provider") == len(expected_finalizations)
    assert json.loads((tmp_path / "receipt.json").read_text())["final_exit_code"] == final
