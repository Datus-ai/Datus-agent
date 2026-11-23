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
