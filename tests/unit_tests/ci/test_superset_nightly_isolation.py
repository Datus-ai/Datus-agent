"""Exercise the real nightly shell dispatch with owned service/pytest fakes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ci.bi_gate import NODEIDS

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "ci/run-nightly-tests.sh"
GROUPS = ["Superset Product Tests - " + suffix for suffix in ("Chart Lifecycle", "Bootstrap", "Dashboard")]


def source(name):
    text = SCRIPT.read_text()
    start = text.index(name + "() {")
    return text[start : text.index("\n}\n", start) + 3]


def run_suites(tmp_path, group_filter="", failure="", targeted=False):
    observer = tmp_path / "observer.py"
    observer.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]; report=Path(args[args.index('--gate-report')+1])\n"
        "report.write_text(json.dumps({'nodeids':[x for x in args if x.startswith('tests/')]})+'\\n')\n"
        "with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps({'action':'pytest','group':os.environ['DATUS_NIGHTLY_SUITE_NAME'],'home':os.environ['DATUS_TEST_HOME'],'workspace':os.environ['FAKE_WORKSPACE'],'report':str(report),'trace':os.environ['DATUS_NIGHTLY_TRACE_EXPECTED'],'args':args})+'\\n')\n"
        "sys.exit(7 if os.environ['FAILURE'] in ('pytest','pytest-cleanup') and os.environ['DATUS_NIGHTLY_SUITE_NAME'].endswith('Chart Lifecycle') else 0)\n"
    )
    dispatch = SCRIPT.read_text().split("# Scheduled Nightly runs the explicit Superset", 1)[1]
    dispatch = dispatch[dispatch.index('if [ -n "${PRODUCT_TEST_BASE_REF:-}" ]; then') :]
    dispatch = dispatch.split('run_compose_suite "BI Orchestration Contracts"', 1)[0]
    harness = "\n".join(
        [
            "set -uo pipefail; test_exit_code=0; last_command_exit_code=0; chart_down_calls=0",
            "utc_now() { echo now; }; log() { :; }",
            'event() { printf \'["%s","%s"]\\n\' "$1" "$2" >> "$EVENTS"; }',
            'manifest_record_suite() { event "suite:$1" "$4"; }; manifest_record_compose_project() { :; }; manifest_update() { :; }',
            "collect_pytest_suite() { :; }; command_contains_pytest() { return 0; }",
            'run_with_agent_home() { export DATUS_TEST_HOME="$1" FAKE_WORKSPACE="$2"; shift 2; "$@"; }',
            'env() { export "$1"; shift; "$@"; }',
            'uv() { shift; "$PYTHON" "$OBSERVER" "$@"; }',
            'compose_cmd() { [ "$3 $4 $5" = "down -v --remove-orphans" ] || return 97; event down "$1"; if [[ "$1" = *chart-lifecycle* ]]; then chart_down_calls=$((chart_down_calls + 1)); if [[ ( "$FAILURE" = cleanup || "$FAILURE" = pytest-cleanup ) && "$chart_down_calls" = 2 || "$FAILURE" = stale && "$chart_down_calls" = 1 ]]; then return 8; fi; fi; }',
            'compose_up() { event up "$1"; if [[ "$FAILURE" = service && "$1" = *chart-lifecycle* ]]; then test_exit_code=5; return 5; fi; }',
            'wait_for_service_health() { event "health:$3" "$1"; }',
            "check_compose_host_ports_available() { return 0; }; dump_compose_diagnostics() { :; }",
            'wait_for_http_readiness() { event http "$1"; }',
            source("should_run_group"),
            source("compose_project_slug"),
            source("compose_project_name"),
            source("compose_down"),
            source("compose_host_port_specs"),
            source("nightly_trace_expected_for_group"),
            source("wait_for_compose_client_readiness"),
            source("finish_compose_suite"),
            source("run_logged_unfiltered"),
            source("run_logged"),
            source("run_compose_suite"),
            dispatch,
            'exit "$test_exit_code"',
        ]
    )
    env = {
        **os.environ,
        "PRODUCT_TEST_BASE_REF": "base" if targeted else "",
        "NIGHTLY_GROUP_FILTER": group_filter,
        "NIGHTLY_HOME": str(tmp_path / "home"),
        "NIGHTLY_PROJECT_ROOT": str(tmp_path / "workspace"),
        "NIGHTLY_PYTEST_BASETEMP": str(tmp_path / "basetemp"),
        "NIGHTLY_MANIFEST_FILE": str(tmp_path / "manifest.json"),
        "NIGHTLY_TRACE_REFERENCES_FILE": str(tmp_path / "traces"),
        "NIGHTLY_COMPOSE_PROJECT_PREFIX": "owned-",
        "SUPERSET_COMPOSE": str(tmp_path / "compose.yml"),
        "SUPERSET_URL": "http://127.0.0.1:18088",
        "PRODUCT_EXECUTION_DIR": "",
        "DASHBOARD_MARK_EXPR": "nightly",
        "REPO_ROOT": str(ROOT),
        "LOG_FILE": str(tmp_path / "log"),
        "EVENTS": str(tmp_path / "events"),
        "PYTHON": sys.executable,
        "OBSERVER": str(observer),
        "FAILURE": failure,
    }
    result = subprocess.run(["bash", "-c", harness], env=env, text=True, capture_output=True, timeout=20)
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    return result, events


@pytest.mark.parametrize("group_filter", ["", "^Superset Product Tests$", "Superset Product Tests"])
def test_full_superset_cases_have_fresh_compose_and_independent_execution_evidence(tmp_path, group_filter):
    result, events = run_suites(tmp_path, group_filter)
    assert result.returncode == 0, result.stdout + result.stderr
    executed = [event for event in events if isinstance(event, dict)]
    assert [event["group"] for event in executed] == GROUPS
    assert [[arg for arg in event["args"] if arg.startswith("tests/")] for event in executed] == [[n] for n in NODEIDS]
    for field in ("home", "workspace", "report"):
        assert len({event[field] for event in executed}) == 3
    assert len({arg for event in executed for arg in event["args"] if arg.startswith("--basetemp=")}) == 3
    assert [event["trace"] for event in executed] == ["1"] * 3
    for event in executed:
        assert "--fail-on-skip" in event["args"]
        assert "--reruns" not in event["args"]
        assert json.loads(Path(event["report"]).read_text())["nodeids"] == [NODEIDS[GROUPS.index(event["group"])]]
    projects = [event[1] for event in events if isinstance(event, list) and event[0] == "up"]
    assert len(set(projects)) == 3
    assert [event for event in events if isinstance(event, list) and event[0] in ("down", "up")] == [
        [action, project] for project in projects for action in ("down", "up", "down")
    ]
    assert [event for event in events if isinstance(event, list) and event[0].startswith("health:")] == [
        [action, project] for project in projects for action in ("health:postgres", "health:superset")
    ]
    assert [event for event in events if isinstance(event, list) and event[0] == "http"] == [["http", "Superset"]] * 3


@pytest.mark.parametrize("failure,expected,groups", [("service", 5, GROUPS[1:]), ("pytest", 7, GROUPS)])
def test_failed_superset_case_cleans_its_project_and_does_not_suppress_other_cases(tmp_path, failure, expected, groups):
    result, events = run_suites(tmp_path, failure=failure)
    assert result.returncode == expected, result.stdout + result.stderr
    assert [event["group"] for event in events if isinstance(event, dict)] == groups
    projects = [event[1] for event in events if isinstance(event, list) and event[0] == "up"]
    assert len(set(projects)) == 3
    assert [event for event in events if isinstance(event, list) and event[0] in ("down", "up")] == [
        [action, project] for project in projects for action in ("down", "up", "down")
    ]


def test_exact_child_filter_starts_only_that_superset_case(tmp_path):
    result, events = run_suites(tmp_path, "^Superset Product Tests - Bootstrap$")
    assert result.returncode == 0, result.stderr
    assert [event["group"] for event in events if isinstance(event, dict)] == [GROUPS[1]]
    assert len([event for event in events if isinstance(event, list) and event[0] == "up"]) == 1


@pytest.mark.parametrize(
    "failure,groups,expected", [("cleanup", GROUPS, 1), ("stale", GROUPS[1:], 1), ("pytest-cleanup", GROUPS, 7)]
)
def test_superset_volume_cleanup_failure_is_blocking_and_other_cases_continue(tmp_path, failure, groups, expected):
    result, events = run_suites(tmp_path, failure=failure)
    assert result.returncode == expected, result.stdout + result.stderr
    assert [event["group"] for event in events if isinstance(event, dict)] == groups
    assert ["suite:" + GROUPS[0], "failed"] in events
    assert ["suite:" + GROUPS[1], "passed"] in events
    assert ["suite:" + GROUPS[2], "passed"] in events


def test_targeted_evaluation_keeps_existing_combined_parent_suite(tmp_path):
    result, events = run_suites(tmp_path, "^Superset Product Tests$", targeted=True)
    assert result.returncode == 0, result.stderr
    assert [event["group"] for event in events if isinstance(event, dict)] == ["Superset Product Tests"]
    assert len([event for event in events if isinstance(event, list) and event[0] == "up"]) == 1


def test_cleanup_recognizes_each_superset_child_and_removes_only_its_project(tmp_path):
    compose = tmp_path / "compose.yml"
    compose.touch()
    harness = "\n".join(
        [
            "set -uo pipefail",
            "COMPOSE_GROUPS=(" + " ".join(json.dumps(group) for group in GROUPS) + ")",
            "has_docker_compose() { return 0; }",
            'compose_cmd() { printf "%s|%s|%s|%s|%s\\n" "$1" "$2" "$3" "$4" "$5"; }',
            source("compose_project_slug"),
            source("compose_project_name"),
            source("compose_down"),
            source("cleanup_all_compose"),
            "cleanup_all_compose",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={**os.environ, "SUPERSET_COMPOSE": str(compose), "NIGHTLY_COMPOSE_PROJECT_PREFIX": "owned-"},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert [line for line in result.stdout.splitlines() if "|" in line] == [
        f"owned-superset-{suffix}|{compose}|down|-v|--remove-orphans"
        for suffix in ("chart-lifecycle", "bootstrap", "dashboard")
    ]
