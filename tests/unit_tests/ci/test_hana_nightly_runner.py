"""HANA nightly owns a private Express lifecycle and never needs a live host here."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit_tests.ci.test_hive_nightly_runner import ROOT, SCRIPT


def _source(name):
    if f"{name}() {{" in SCRIPT.read_text():
        source = SCRIPT.read_text()
        start = source.index(f"{name}() {{")
        end = source.index("\n}\n", start) + len("\n}\n")
        return source[start:end]
    return f"{name}() {{ return 127; }}"


def _environment(tmp_path):
    adapter = tmp_path / "adapters" / "datus-hana"
    scripts = adapter / "scripts"
    scripts.mkdir(parents=True)
    (adapter / "docker-compose.yml").write_text("services:\n  hana:\n    image: test-hana\n")
    (scripts / "prepare_express.py").write_text(
        "import json,os,shutil,stat,sys\nfrom pathlib import Path\n"
        "root=Path(__file__).resolve().parents[1]\n"
        "cleanup='--cleanup' in sys.argv\n"
        "with open(os.environ['EVENTS'],'a') as f: f.write(json.dumps({'event':'cleanup' if cleanup else 'prepare','root':str(root),'root_mode':stat.S_IMODE(root.stat().st_mode),'marker_mode':stat.S_IMODE((root/'.nightly-project').stat().st_mode),'args':sys.argv[1:]})+'\\n')\n"
        "if cleanup:\n"
        " shutil.rmtree(root/'.express',ignore_errors=True)\n"
        "else:\n"
        " if os.environ.get('FAIL_AT')=='prepare': sys.exit(9)\n"
        " assert len(os.environ['DATUS_TEST_HANA_PASSWORD'])>=8\n"
        " p=root/'.express';p.mkdir(mode=0o700)\n"
        " (p/'passwords.json').write_text('private bootstrap')\n"
    )
    (scripts / "wait_for_hana.py").write_text(
        "import json,os,sys\n"
        "with open(os.environ['EVENTS'],'a') as f: f.write(json.dumps({'event':'ready','host':os.environ['HANA_HOST'],'port':os.environ['HANA_PORT'],'user':os.environ['HANA_USER'],'schema':os.environ['HANA_SCHEMA'],'require':os.environ['HANA_REQUIRE_AGENT_E2E']})+'\\n')\n"
        "sys.exit(8 if os.environ.get('FAIL_AT')=='ready' else 0)\n"
    )
    observer = tmp_path / "observer.py"
    observer.write_text(
        "import json,os,sys\n"
        "with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps({'event':'pytest','args':sys.argv[1:],'password_length':len(os.environ['HANA_PASSWORD'])})+'\\n')\n"
        "sys.exit(7 if os.environ.get('FAIL_AT')=='pytest' else 0)\n"
    )
    env = {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "REPO_ROOT": str(tmp_path / "agent"),
        "DB_ADAPTERS_ROOT": str(adapter.parent),
        "HANA_COMPOSE": str(adapter / "docker-compose.yml"),
        "NIGHTLY_COMPOSE_PROJECT_PREFIX": "datus-nightly-run-123-",
        "NIGHTLY_GROUP_FILTER": "",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "NIGHTLY_HOME": str(tmp_path / "home"),
        "NIGHTLY_PROJECT_ROOT": str(tmp_path / "workspace"),
        "LOG_FILE": str(tmp_path / "log"),
        "NIGHTLY_LOG_FILE": str(tmp_path / "log"),
        "EVENTS": str(tmp_path / "events.jsonl"),
        "PYTHON": sys.executable,
        "OBSERVER": str(observer),
        "HANA_HOST_PORT": "23942",
        "GITHUB_ACTIONS": "false",
    }
    for key in ("DATUS_TEST_HANA_PASSWORD", "HANA_PASSWORD", "HANA_USERKEY", "HANA_DATABASE"):
        env.pop(key, None)
    return env


def _harness(command=None):
    return "\n".join(
        [
            "set -uo pipefail",
            "test_exit_code=0; last_command_exit_code=0",
            'log() { echo "$*" >> "$LOG_FILE"; }',
            "utc_now() { echo now; }",
            'should_run_group() { [ "${FAIL_AT:-}" != filtered ]; }',
            'manifest_record_suite() { printf "suite:%s:%s\\n" "$1" "$4" >> "$LOG_FILE"; }',
            "manifest_record_compose_project() { :; }",
            "check_compose_host_ports_available() { return 0; }",
            "wait_for_service_health() { return 0; }",
            "dump_compose_diagnostics() { :; }",
            'run_with_agent_home() { shift 2; "$@"; }',
            'sudo() { printf "sudo:%s\\n" "$*" >> "$LOG_FILE"; shift; if [[ "$1" == --preserve-env=* ]]; then shift; fi; "$@"; }',
            'uv() { shift; if [ "$1" = --no-sync ]; then shift; fi; if [ "$1" = pytest ]; then shift; "$PYTHON" "$OBSERVER" "$@"; else shift; "$PYTHON" "$@"; fi; }',
            'compose_cmd() { printf "compose:%s:%s\\n" "$1" "$3" >> "$LOG_FILE"; if [ "$3" = down ] && [ "${FAIL_AT:-}" = cleanup ]; then return 6; fi; return 0; }',
            'compose_up() { printf "up:%s\\n" "$2" >> "$LOG_FILE"; if [ "${FAIL_AT:-}" = up ]; then test_exit_code=5; return 5; fi; }',
            'compose_down() { compose_cmd "$2" "$1" down || true; }',
            'run_logged() { shift; "$@"; last_command_exit_code=$?; if [ "$last_command_exit_code" -ne 0 ]; then test_exit_code=$last_command_exit_code; fi; }',
            _source("compose_project_slug"),
            _source("compose_project_name"),
            _source("compose_host_port_specs"),
            _source("hana_nightly_state_root"),
            _source("prepare_hana_nightly_environment"),
            _source("cleanup_hana_nightly_environment"),
            _source("finish_compose_suite"),
            _source("wait_for_compose_client_readiness"),
            _source("run_compose_suite"),
            command
            or 'run_compose_suite "HANA Agent Tests" "$HANA_COMPOSE" "hana:1200" -- "$PYTHON" "$OBSERVER" --fail-on-skip',
            'echo "exit:$test_exit_code" >> "$LOG_FILE"',
            'exit "$test_exit_code"',
        ]
    )


@pytest.mark.parametrize(
    "failure,expected,event_names",
    [
        ("", 0, ["prepare", "ready", "pytest", "cleanup"]),
        ("prepare", 1, ["prepare"]),
        ("up", 5, ["prepare", "cleanup"]),
        ("ready", 1, ["prepare", "ready", "cleanup"]),
        ("pytest", 7, ["prepare", "ready", "pytest", "cleanup"]),
    ],
)
def test_hana_owns_private_bootstrap_and_cleans_after_every_exit(tmp_path, failure, expected, event_names):
    env = _environment(tmp_path)
    env["FAIL_AT"] = failure
    result = subprocess.run(["bash", "-c", _harness()], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == expected, result.stdout + result.stderr
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    assert [event["event"] for event in events] == event_names
    root = Path(events[0]["root"])
    assert root.parent == tmp_path
    assert events[0]["root_mode"] == 0o700
    assert events[0]["marker_mode"] == 0o600
    assert root != Path(env["DB_ADAPTERS_ROOT"]) / "datus-hana"
    assert not root.exists()
    assert not (Path(env["DB_ADAPTERS_ROOT"]) / "datus-hana" / ".express").exists()
    ready = [event for event in events if event["event"] == "ready"]
    executed = [event for event in events if event["event"] == "pytest"]
    assert ready == [
        {
            "event": "ready",
            "host": "127.0.0.1",
            "port": "23942",
            "user": "DATUS_TEST",
            "schema": "DATUS_TEST",
            "require": "1",
        }
    ] * event_names.count("ready")
    assert [event["args"] for event in executed] == [["--fail-on-skip"]] * event_names.count("pytest")
    assert [event["password_length"] >= 24 for event in executed] == [True] * event_names.count("pytest")


def test_hana_cleanup_only_recovers_owned_state_without_password(tmp_path):
    env = _environment(tmp_path)
    harness = "\n".join(
        [
            _harness().split('run_compose_suite "HANA Agent Tests"')[0],
            "prepare_hana_nightly_environment || exit 1",
            "unset HANA_PASSWORD DATUS_TEST_HANA_PASSWORD",
            "cleanup_hana_nightly_environment || exit 1",
        ]
    )
    result = subprocess.run(["bash", "-c", harness], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    assert [event["event"] for event in events] == ["prepare", "cleanup"]
    assert not Path(events[0]["root"]).exists()


def test_hana_cleanup_failure_preserves_private_state_for_recovery(tmp_path):
    env = _environment(tmp_path)
    env["FAIL_AT"] = "cleanup"
    result = subprocess.run(["bash", "-c", _harness()], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    assert Path(events[0]["root"]).is_dir()
    assert "cleanup" not in [event["event"] for event in events]


def test_hana_filtered_group_never_prepares_or_starts_a_service(tmp_path):
    env = _environment(tmp_path)
    env["FAIL_AT"] = "filtered"
    result = subprocess.run(["bash", "-c", _harness()], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(env["EVENTS"]).exists()
    assert "suite:HANA Agent Tests:skipped" in Path(env["LOG_FILE"]).read_text()


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_actual_cleanup_only_entrypoint_recovers_run_without_bootstrap_password(tmp_path, fail_cleanup):
    env = _environment(tmp_path)
    prefix = _harness().split('run_compose_suite "HANA Agent Tests"')[0]
    prepare = subprocess.run(
        ["bash", "-c", prefix + "\nprepare_hana_nightly_environment"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert prepare.returncode == 0, prepare.stdout + prepare.stderr
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    state_root = Path(events[0]["root"])
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    docker = mock_bin / "docker"
    docker.write_text(
        '#!/bin/sh\nprintf "docker:%s\\n" "$*" >> "$LOG_FILE"\n'
        'if [ "${FAIL_AT:-}" = cleanup ]; then case " $* " in *" down "*) exit 6;; esac; fi\nexit 0\n'
    )
    docker.chmod(0o755)
    sudo = mock_bin / "sudo"
    sudo.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    sudo.chmod(0o755)
    env.update(
        REPO_ROOT=str(ROOT),
        PATH=str(mock_bin) + os.pathsep + env["PATH"],
        UV_CACHE_DIR=str(tmp_path / "uv-cache"),
        NIGHTLY_PYTEST_BASETEMP=str(tmp_path / "pytest-cleanup"),
        AGENT_TEST_CONFIG_BACKUP=str(tmp_path / "config-backup"),
        DATUS_TEST_RUN_ID="",
        FAIL_AT="cleanup" if fail_cleanup else "",
    )
    cleaned = subprocess.run(
        ["bash", str(SCRIPT), "--cleanup-only"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert cleaned.returncode == int(fail_cleanup), cleaned.stdout + cleaned.stderr
    assert state_root.exists() == fail_cleanup
    assert "unused-for-compose-down" not in cleaned.stdout + cleaned.stderr


def test_hana_cleanup_refuses_unowned_or_symlink_state(tmp_path):
    env = _environment(tmp_path)
    prefix = _harness().split('run_compose_suite "HANA Agent Tests"')[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("unrelated")
    command = (
        prefix
        + '\nstate_root="$(hana_nightly_state_root)"\n'
        + f'ln -s "{outside}" "$state_root"\n'
        + "cleanup_hana_nightly_environment"
    )
    result = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert sentinel.read_text() == "unrelated"
    assert "compose:" not in Path(env["LOG_FILE"]).read_text()


def test_private_password_never_enters_recorded_commands_or_local_output(tmp_path):
    env = _environment(tmp_path)
    password = "Example9_private_bootstrap"
    env["DATUS_TEST_HANA_PASSWORD"] = password
    result = subprocess.run(["bash", "-c", _harness()], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert password not in result.stdout + result.stderr + Path(env["LOG_FILE"]).read_text()
    assert password not in Path(env["EVENTS"]).read_text()


def test_cleanup_does_not_stop_or_remove_another_run_with_the_same_workflow_prefix(tmp_path):
    env = _environment(tmp_path)
    prefix = _harness().split('run_compose_suite "HANA Agent Tests"')[0]
    prepare = subprocess.run(
        ["bash", "-c", prefix + "\nprepare_hana_nightly_environment"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert prepare.returncode == 0, prepare.stdout + prepare.stderr
    events_before = Path(env["EVENTS"]).read_text()
    state_root = Path(json.loads(events_before.splitlines()[0])["root"])
    other_run = {**env, "GITHUB_RUN_ID": "456"}
    other_cleanup = subprocess.run(
        ["bash", "-c", prefix + "\ncleanup_hana_nightly_environment"],
        cwd=ROOT,
        env=other_run,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert other_cleanup.returncode == 0, other_cleanup.stdout + other_cleanup.stderr
    assert state_root.is_dir()
    assert Path(env["EVENTS"]).read_text() == events_before
    own_cleanup = subprocess.run(
        ["bash", "-c", prefix + "\ncleanup_hana_nightly_environment"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert own_cleanup.returncode == 0, own_cleanup.stdout + own_cleanup.stderr
    assert not state_root.exists()


def test_scheduled_group_runs_all_hana_agent_cases_after_other_compose_groups(tmp_path):
    source = SCRIPT.read_text()
    command = next(line for line in source.splitlines() if line.startswith('run_compose_suite "HANA Agent Tests" '))
    env = _environment(tmp_path)
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    uv = mock_bin / "uv"
    uv.write_text('#!/bin/sh\nshift 2\nexec "$PYTHON" "$OBSERVER" "$@"\n')
    uv.chmod(0o755)
    env["PATH"] = str(mock_bin) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["bash", "-c", _harness(command)], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    executed = next(event for event in events if event["event"] == "pytest")
    assert "tests/integration/tools/test_hana_semantic_verification.py" in executed["args"]
    assert "tests/integration/agent/test_hana_chat_agentic.py" in executed["args"]
    assert "--fail-on-skip" in executed["args"]
    assert source.index(command) > source.index('run_compose_suite "GaussDB Adapter Tests" ')
