"""Hive nightly connects semantic acceptance to the provisioned compose service."""

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "ci/run-nightly-tests.sh"


def _function(name):
    source = SCRIPT.read_text()
    start = source.index(f"{name}() {{")
    following = re.search(r"(?m)^[a-z_]+\(\) \{", source[start + len(name) + 5 :])
    end = start + len(name) + 5 + following.start() if following else len(source)
    return source[start:end].rstrip()


def _observer(tmp_path, status):
    observer = tmp_path / "observe.py"
    observer.write_text(
        "import json,os,stat,sys\n"
        "from pathlib import Path\n"
        "p=Path(os.environ['DOSI_TEST_HIVE_CONNECTIONS'])\n"
        "data={'path':str(p),'mode':stat.S_IMODE(p.stat().st_mode),'config':json.loads(p.read_text()),"
        "'connection':os.environ['DOSI_TEST_HIVE_CONNECTION'],'args':sys.argv[1:]}\n"
        "Path(os.environ['OBSERVED']).write_text(json.dumps(data))\n"
        f"sys.exit({status})\n"
    )
    return observer


def _environment(tmp_path):
    return {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "NIGHTLY_HOME": str(tmp_path / "home"),
        "NIGHTLY_PROJECT_ROOT": str(tmp_path / "workspace"),
        "OBSERVED": str(tmp_path / "observed.json"),
        "HIVE_HOST": "127.0.0.1",
        "HIVE_PORT": "21000",
        "HIVE_USERNAME": "nightly",
        "HIVE_PASSWORD": "local-value-with-'quote",
        "HIVE_DATABASE": "default",
    }


@pytest.mark.parametrize("status", [0, 7])
def test_hive_connections_use_provisioned_endpoint_and_clean_up(tmp_path, status):
    observer = _observer(tmp_path, status)
    harness = "\n".join(
        [
            'run_with_agent_home() { shift 2; "$@"; }',
            _function("run_hive_adapter_tests"),
            'run_hive_adapter_tests "$1" "$2" --collect-only',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", harness, "hive-runner", sys.executable, str(observer)],
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == status, result.stderr
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert observed["connection"] == "hive"
    assert observed["mode"] == stat.S_IRUSR | stat.S_IWUSR
    assert observed["config"] == {
        "datasources": {
            "hive": {
                "type": "hive",
                "server_type": "hive",
                "host": "127.0.0.1",
                "port": 21000,
                "username": "nightly",
                "password": "local-value-with-'quote",
                "database": "default",
            }
        }
    }
    assert observed["args"] == ["--collect-only"]
    assert not Path(observed["path"]).exists()


def test_scheduled_hive_group_runs_adapter_and_semantics_on_same_connection(tmp_path):
    observer = _observer(tmp_path, 0)
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    uv = mock_bin / "uv"
    uv.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" "{observer}" "$@"\n')
    uv.chmod(0o755)
    command = next(
        line for line in SCRIPT.read_text().splitlines() if line.startswith('run_compose_suite "Hive Adapter Tests" ')
    )
    harness = "\n".join(
        [
            'run_with_agent_home() { shift 2; "$@"; }',
            'run_compose_suite() { shift 2; while [ "$1" != "--" ]; do shift; done; shift; "$@"; }',
            _function("run_hive_adapter_tests") if "run_hive_adapter_tests() {" in SCRIPT.read_text() else "",
            command,
        ]
    )
    env = _environment(tmp_path)
    env.update({"PATH": str(mock_bin) + os.pathsep + env["PATH"], "HIVE_COMPOSE": "hive-compose.yml"})
    result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert "tests/integration/adapters/test_hive.py" in observed["args"]
    assert "tests/integration/tools/test_dosi_hive.py" in observed["args"]
    excluded = observed["args"][observed["args"].index("--deselect") + 1]
    assert excluded == "tests/integration/tools/test_dosi_hive.py::test_native_inceptor_system_varchar2_metrics"
    assert observed["config"]["datasources"]["hive"]["server_type"] == "hive"
    assert not Path(observed["path"]).exists()
