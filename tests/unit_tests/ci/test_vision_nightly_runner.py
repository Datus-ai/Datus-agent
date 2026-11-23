"""Execute the nightly shell lifecycle with observed provisioning/test seams."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "ci/run-nightly-tests.sh"
pytestmark = pytest.mark.component


def source(name):
    text = SCRIPT.read_text()
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 3]


def run_lifecycle(tmp_path, *, failure="", filtered=False):
    events = tmp_path / "events.jsonl"
    observer = tmp_path / "observe.py"
    observer.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "action=sys.argv[1]\nargs=sys.argv[2:]\n"
        "with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps({'action':action,'args':args,'model':os.environ.get('DATUS_SELF_HOSTED_MODEL'),'layer':os.environ.get('DATUS_TEST_LAYER')})+'\\n')\n"
        "if action=='prepare':\n"
        " work=Path(args[args.index('--work')+1]);work.mkdir(parents=True,exist_ok=True)\n"
        " (work/'evidence.json').write_text('{}')\n"
        " if os.environ['FAIL_AT']=='prepare':sys.exit(8)\n"
        " (work/'environment.sh').write_text('export DATUS_SELF_HOSTED_MODEL=real-fixture\\nexport DATUS_SELF_HOSTED_BASE_URL=http://127.0.0.1:12345/v1\\n')\n"
        "if action=='cleanup':\n"
        " work=Path(args[args.index('--work')+1]);(work/'server.log').write_text('server evidence')\n"
        "if action=='pytest':Path(args[args.index('--gate-report')+1]).write_text('{}')\n"
        "if action==os.environ['FAIL_AT']:sys.exit(7)\n"
    )
    env = {
        **os.environ,
        "EVENTS": str(events),
        "PYTHON": sys.executable,
        "OBSERVER": str(observer),
        "FAIL_AT": failure,
        "FILTERED": str(int(filtered)),
        "RUNNER_TEMP": str(tmp_path),
        "REPO_ROOT": str(tmp_path / "repo"),
        "LOG_FILE": str(tmp_path / "nightly.log"),
        "NIGHTLY_HOME": str(tmp_path / "home"),
        "NIGHTLY_PROJECT_ROOT": str(tmp_path / "workspace"),
        "PRODUCT_EXECUTION_DIR": str(tmp_path / "execution"),
        "NIGHTLY_MANIFEST_FILE": str(tmp_path / "manifest.json"),
        "NIGHTLY_TRACE_REFERENCES_FILE": str(tmp_path / "trace.jsonl"),
        "NIGHTLY_COMPOSE_PROJECT_PREFIX": "owned-project-",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "NIGHTLY_GROUP_FILTER": "selected" if filtered else "",
    }
    for key in ("DATUS_SELF_HOSTED_MODEL", "DATUS_SELF_HOSTED_BASE_URL"):
        env.pop(key, None)
    harness = "\n".join(
        [
            "set -uo pipefail; test_exit_code=0; last_command_exit_code=0",
            'log() { echo "$*" >> "$LOG_FILE"; }',
            "utc_now() { echo now; }",
            'should_run_group() { [ "$FILTERED" = 0 ]; }',
            'manifest_record_suite() { printf "suite:%s:%s\\n" "$1" "$4" >> "$LOG_FILE"; }',
            "manifest_update() { :; }",
            "command_contains_pytest() { return 0; }",
            'collect_pytest_suite() { [ "$DATUS_SELF_HOSTED_MODEL" = real-fixture ] || exit 22; "$PYTHON" "$OBSERVER" collect; }',
            'run_with_agent_home() { shift 2; "$@"; }',
            'env() { export "$1"; shift; "$@"; }',
            'uv() { shift; if [ "$1" = python ]; then shift 2; "$PYTHON" "$OBSERVER" "$@"; else "$PYTHON" "$OBSERVER" "$@"; fi; }',
            source("compose_project_slug"),
            source("compose_project_name"),
            source("nightly_trace_expected_for_group"),
            source("vision_nightly_state_root"),
            source("cleanup_vision_nightly_environment"),
            source("run_logged_unfiltered"),
            source("run_logged"),
            source("run_self_hosted_vision_suite"),
            "run_self_hosted_vision_suite",
            '"$PYTHON" "$OBSERVER" unrelated',
            'exit "$test_exit_code"',
        ]
    )
    result = subprocess.run(["bash", "-c", harness], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    observed = [json.loads(line) for line in events.read_text().splitlines()]
    return result, observed, env


@pytest.mark.parametrize(
    "failure,expected,actions",
    [
        ("", 0, ["prepare", "collect", "pytest", "cleanup", "unrelated"]),
        ("prepare", 1, ["prepare", "cleanup", "unrelated"]),
        ("pytest", 7, ["prepare", "collect", "pytest", "cleanup", "unrelated"]),
        ("cleanup", 1, ["prepare", "collect", "pytest", "cleanup", "unrelated"]),
    ],
)
def test_shell_provisions_before_collection_and_continues_after_failures(tmp_path, failure, expected, actions):
    result, observed, env = run_lifecycle(tmp_path, failure=failure)
    assert result.returncode == expected, result.stdout + result.stderr
    assert [event["action"] for event in observed] == actions
    prepare = observed[0]["args"]
    state = Path(prepare[prepare.index("--work") + 1])
    assert "owned-project-self-hosted-vision-tests-123-2" in str(state)
    assert prepare[prepare.index("--cache") + 1] == str(tmp_path / "home/vision-models")
    assert not (state / "environment.sh").exists()
    artifacts = list((tmp_path / "repo/nightly-vision-evidence").rglob("*"))
    assert {path.name for path in artifacts if path.is_file()} == {"evidence.json", "server.log"}
    pytest_events = [event for event in observed if event["action"] == "pytest"]
    assert len(pytest_events) == actions.count("pytest")
    target = "tests/integration/models/test_custom_model_preflight.py::test_self_hosted_model_reads_code_from_image"
    assert [
        {
            "layer": event["layer"],
            "model": event["model"],
            "strict_skip": "--fail-on-skip" in event["args"],
            "execution_evidence": "ci.pytest_gate_plugin" in event["args"],
            "vision_case": target in event["args"],
            "reruns": "--reruns" in event["args"],
        }
        for event in pytest_events
    ] == [
        {
            "layer": "nightly",
            "model": "real-fixture",
            "strict_skip": True,
            "execution_evidence": True,
            "vision_case": True,
            "reruns": False,
        }
    ] * actions.count("pytest")
    assert Path(env["LOG_FILE"]).is_relative_to(tmp_path)


def test_filter_never_downloads_starts_or_collects_vision(tmp_path):
    result, observed, env = run_lifecycle(tmp_path, filtered=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [event["action"] for event in observed] == ["unrelated"]
    assert "suite:Self-hosted Vision Tests:skipped" in Path(env["LOG_FILE"]).read_text()
    assert not list(tmp_path.glob("datus-agent-nightly-vision-*"))


@pytest.mark.parametrize("cleanup_failure,expected", [(False, 0), (True, 1)])
def test_cleanup_only_entrypoint_invokes_run_scoped_vision_cleanup(tmp_path, cleanup_failure, expected):
    fake_repo = tmp_path / "repo"
    (fake_repo / "ci").mkdir(parents=True)
    script = fake_repo / "ci/cleanup-run-testcontainers.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o700)
    entry = SCRIPT.read_text().split('if [ "${1:-}" = "--cleanup-only" ]; then', 1)[1].split("\nfi", 1)[0]
    harness = "\n".join(
        [
            "set -uo pipefail",
            "restore_agent_test_config() { :; }",
            "cleanup_gaussdb_tls_host_artifacts() { :; }",
            "validate_pytest_basetemp() { return 1; }",
            "cleanup_all_compose() { return 0; }",
            f'cleanup_vision_nightly_environment() {{ echo vision >> "$EVENTS"; return {int(cleanup_failure)}; }}',
            source("cleanup_all"),
            entry,
        ]
    )
    env = {
        **os.environ,
        "EVENTS": str(tmp_path / "events"),
        "REPO_ROOT": str(fake_repo),
        "NIGHTLY_PYTEST_BASETEMP": str(tmp_path / "basetemp"),
        "AGENT_TEST_CONFIG_BACKUP": str(tmp_path / "backup"),
    }
    result = subprocess.run(["bash", "-c", harness], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == expected, result.stdout + result.stderr
    assert Path(env["EVENTS"]).read_text() == "vision\n"


@pytest.mark.parametrize("initial,cleanup,expected", [(0, 3, 3), (7, 0, 7), (7, 3, 7), (0, 0, 0)])
def test_exit_trap_reports_cleanup_failure(tmp_path, initial, cleanup, expected):
    trap = next(line for line in SCRIPT.read_text().splitlines() if line.startswith("trap 'finalize_nightly_exit"))
    output = tmp_path / "github-output"
    receipt = tmp_path / "nightly-cleanup-status.json"
    output.write_text("log_file=run.log\ntest_exit_code=0\n")
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "NIGHTLY_CLEANUP_STATUS_FILE": str(receipt),
        "REPO_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"cleanup_all() {{ return {cleanup}; }}\n{source('finalize_nightly_exit')}\n{trap}\nexit {initial}",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["test_exit_code"] == str(expected)
    assert json.loads(receipt.read_text()) == {
        "status": "passed" if expected == 0 else "failed",
        "cleanup_status": "passed" if cleanup == 0 else "failed",
        "initial_exit_code": initial,
        "cleanup_exit_code": cleanup,
        "final_exit_code": expected,
    }
