"""Host preparation operates only in CodeBuild; all kernel commands are fakes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "ci/prepare-codebuild-nightly-host.sh"
BUILD_ID = "datus-agent-nightly:12345678-1234-1234-1234-123456789abc"


def run_prepare(
    tmp_path,
    *,
    codebuild=BUILD_ID,
    system="Linux",
    arch="x86_64",
    uid="0",
    mode="",
    high=False,
    build_arn=None,
    project_arn="arn:aws:codebuild:us-east-1:123456789012:project/datus-agent-nightly",
    region="us-east-1",
):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    state = tmp_path / "state.json"
    initial = {
        "vm.max_map_count": 200000000 if high else 1048576,
        "fs.aio-max-nr": 524288 if high else 65536,
        "fs.file-max": 30000000 if high else 1000000,
        "vm.memory_failure_early_kill": 1 if high else 0,
    }
    state.write_text(json.dumps(initial))
    log = tmp_path / "commands.jsonl"
    log.write_text("")
    python = sys.executable
    for name, body in {
        "uname": 'case "$1" in -s) echo "$FAKE_SYSTEM";; -m) echo "$FAKE_ARCH";; *) exit 99;; esac',
        "id": 'echo "$FAKE_UID"',
        "sudo": 'printf \'["sudo"]\\n\' >> "$COMMAND_LOG"\n[ "$1" = -n ] || exit 98\nshift\nexec "$@"',
    }.items():
        target = binaries / name
        target.write_text("#!/bin/sh\n" + body + "\n")
        target.chmod(0o700)
    sysctl = binaries / "sysctl"
    sysctl.write_text(
        f"#!{python}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "state=Path(os.environ['STATE']); data=json.loads(state.read_text())\n"
        "with open(os.environ['COMMAND_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1]=='-n': print(data[sys.argv[2]])\n"
        "elif sys.argv[1]=='-w':\n"
        " key,value=sys.argv[2].split('=',1)\n"
        " if os.environ['MODE']=='write-failure':sys.exit(9)\n"
        " if os.environ['MODE']!='readback-failure':data[key]=int(value);state.write_text(json.dumps(data))\n"
        "else:sys.exit(98)\n"
    )
    sysctl.chmod(0o700)
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.defpath,
        "CODEBUILD_BUILD_ID": codebuild,
        "CODEBUILD_BUILD_ARN": build_arn or f"arn:aws:codebuild:us-east-1:123456789012:build/{codebuild}",
        "CODEBUILD_PROJECT_ARN": project_arn,
        "AWS_REGION": region,
        "FAKE_SYSTEM": system,
        "FAKE_ARCH": arch,
        "FAKE_UID": uid,
        "STATE": str(state),
        "COMMAND_LOG": str(log),
        "MODE": mode,
    }
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=10)
    return result, initial, json.loads(state.read_text()), [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("overrides", [{"codebuild": ""}, {"system": "Darwin"}, {"arch": "aarch64"}])
def test_preparation_refuses_non_codebuild_linux_x64_without_kernel_access(tmp_path, overrides):
    result, initial, state, commands = run_prepare(tmp_path, **overrides)
    assert result.returncode != 0
    assert "CodeBuild Linux x86_64" in result.stderr
    assert commands == []
    assert state == initial


@pytest.mark.parametrize(
    "overrides",
    [
        {"codebuild": "backend:12345678-1234-1234-1234-123456789abc"},
        {"codebuild": "datus-agent-nightly:not-a-build-uuid"},
        {"build_arn": "arn:aws:codebuild:us-east-1:123456789012:build/another:run"},
        {"build_arn": "invalid"},
        {"project_arn": "arn:aws:codebuild:us-east-1:123456789012:project/backend"},
        {"region": "us-west-2"},
    ],
)
def test_preparation_rejects_wrong_or_inconsistent_project_identifiers(tmp_path, overrides):
    result, initial, state, commands = run_prepare(tmp_path, **overrides)
    assert result.returncode != 0
    assert "CodeBuild" in result.stderr
    assert commands == []
    assert state == initial


def test_preparation_accepts_valid_build_when_optional_project_and_region_are_absent(tmp_path):
    result, _, state, _ = run_prepare(tmp_path, project_arn="", region="")
    assert result.returncode == 0, result.stderr
    assert state["vm.max_map_count"] == 135217728


@pytest.mark.parametrize("uid", ["0", "1000"])
def test_preparation_raises_required_values_and_verifies_every_write(tmp_path, uid):
    result, _, state, commands = run_prepare(tmp_path, uid=uid)
    assert result.returncode == 0, result.stderr
    assert state == {
        "vm.max_map_count": 135217728,
        "fs.aio-max-nr": 262144,
        "fs.file-max": 20000000,
        "vm.memory_failure_early_kill": 1,
    }
    writes = [command for command in commands if command[0] == "-w"]
    assert writes == [["-w", f"{key}={value}"] for key, value in state.items()]
    assert commands.count(["sudo"]) == (4 if uid != "0" else 0)
    assert [command for command in commands if command[0] == "-n"] == [["-n", key] for key in state for _ in range(2)]


def test_preparation_keeps_already_sufficient_limits(tmp_path):
    result, initial, state, commands = run_prepare(tmp_path, high=True)
    assert result.returncode == 0, result.stderr
    assert state == initial
    assert [command for command in commands if command[0] == "-w"] == []


@pytest.mark.parametrize("mode", ["write-failure", "readback-failure"])
def test_preparation_fails_closed_when_setting_does_not_take_effect(tmp_path, mode):
    result, initial, state, commands = run_prepare(tmp_path, mode=mode)
    assert result.returncode != 0
    assert state == initial
    assert [command for command in commands if command[0] == "-w"] == [["-w", "vm.max_map_count=135217728"]]


def test_workflow_uses_ephemeral_codebuild_runner_and_prepares_before_vendor():
    workflow = (ROOT / ".github/workflows/run-nightly.yml").read_text()
    assert "runs-on: codebuild-datus-agent-nightly-${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "8"' in workflow
    assert workflow.index("name: Ensure Docker runtime") < workflow.index("bash ci/prepare-codebuild-nightly-host.sh")
    assert workflow.index("bash ci/prepare-codebuild-nightly-host.sh") < workflow.index("name: Vendor GaussDB client")
