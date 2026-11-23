"""Nightly config is supplied privately, validated together and never echoed."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ci import prepare_nightly_config as config_helper

SCRIPT = Path(__file__).resolve().parents[3] / "ci" / "prepare_nightly_config.py"
TARGETS = {
    "DATUS_NIGHTLY_AGENT_CONFIG": "conf/agent.yml",
    "DATUS_NIGHTLY_CI_ENV": ".env",
    "DATUS_NIGHTLY_SERVICE_ENV": ".nightly.env",
}


@pytest.fixture
def payloads():
    return {
        "DATUS_NIGHTLY_AGENT_CONFIG": "agent:\n  models:\n    primary:\n      api_key: yaml-private-key\n      model: visible-model\n",
        "DATUS_NIGHTLY_CI_ENV": "# Model credentials\nMODEL_KEY=dotenv-private-key\n",
        "DATUS_NIGHTLY_SERVICE_ENV": "SERVICE_PASSWORD=service-private-password\n",
    }


def _run(root, payloads, *, gha=False, explicit_root=True):
    command = [sys.executable, str(SCRIPT)]
    command += ["--repo-root", str(root)] if explicit_root else []
    return subprocess.run(
        command,
        cwd=root,
        env={**os.environ, **payloads, "GITHUB_ACTIONS": "true" if gha else "false"},
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("explicit_root", [False, True])
def test_valid_payloads_write_only_fixed_private_files(tmp_path, payloads, explicit_root):
    result = _run(tmp_path, payloads, explicit_root=explicit_root)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == set(TARGETS.values())
    for name, relative in TARGETS.items():
        path = tmp_path / relative
        assert path.read_text() == payloads[name]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("missing", list(TARGETS))
def test_missing_payload_preserves_all_existing_files(tmp_path, payloads, missing):
    for relative in TARGETS.values():
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("existing-private-content")
    payloads[missing] = "  \n"
    result = _run(tmp_path, payloads)
    assert result.returncode == 1
    assert missing in result.stderr
    assert result.stdout == ""
    assert {p.read_text() for p in tmp_path.rglob("*") if p.is_file()} == {"existing-private-content"}


@pytest.mark.parametrize(
    "name,invalid",
    [
        ("DATUS_NIGHTLY_AGENT_CONFIG", "agent:\n  password: INVALID_SECRET_SENTINEL\n  broken: ["),
        ("DATUS_NIGHTLY_AGENT_CONFIG", "[INVALID_SECRET_SENTINEL]"),
        ("DATUS_NIGHTLY_AGENT_CONFIG", "agent:\n  password: INVALID_SECRET_SENTINEL\n  token: 2026-99-99\n"),
        ("DATUS_NIGHTLY_CI_ENV", "VALID=INVALID_SECRET_SENTINEL\nINVALID-NAME=value\n"),
        ("DATUS_NIGHTLY_SERVICE_ENV", "export PASSWORD=INVALID_SECRET_SENTINEL\n"),
        ("DATUS_NIGHTLY_SERVICE_ENV", "PASSWORD=INVALID_SECRET_SENTINEL\nSECOND_LINE_WITHOUT_EQUALS\n"),
    ],
)
@pytest.mark.parametrize("gha", [False, True])
def test_invalid_payload_is_private_and_writes_nothing(tmp_path, payloads, name, invalid, gha):
    payloads[name] = invalid
    result = _run(tmp_path, payloads, gha=gha)
    assert result.returncode == 1
    assert name in result.stderr
    assert "INVALID_SECRET_SENTINEL" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_github_masks_sensitive_context_and_escapes_workflow_commands(tmp_path, payloads):
    sensitive = "first%line\r\n::error::not-a-command"
    payloads["DATUS_NIGHTLY_AGENT_CONFIG"] = yaml.safe_dump(
        {
            "agent": {
                "model": "visible-model",
                "api_key": sensitive,
                "credentials": {"nested": ["nested-private-value"]},
            },
        }
    )
    payloads["DATUS_NIGHTLY_CI_ENV"] = "MODEL_KEY=dotenv%private\n"
    result = _run(tmp_path, payloads, gha=True)
    assert result.returncode == 0, result.stderr
    masks = result.stdout.splitlines()
    assert "::add-mask::first%25line%0D%0A::error::not-a-command" in masks
    assert "::add-mask::nested-private-value" in masks
    assert "::add-mask::dotenv%25private" in masks
    assert "::add-mask::service-private-password" in masks
    assert all(line.startswith("::add-mask::") for line in masks)
    assert "::add-mask::visible-model" not in masks


@pytest.mark.parametrize("relative", ["conf", "conf/agent.yml", ".env", ".nightly.env"])
def test_fixed_paths_cannot_follow_symlinks_outside_repo(tmp_path, payloads, relative):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "untouched"
    marker.write_text("outside-private-content")
    destination = outside if relative == "conf" else marker
    (root / relative).parent.mkdir(exist_ok=True)
    (root / relative).symlink_to(destination, target_is_directory=relative == "conf")
    result = _run(root, payloads)
    assert result.returncode == 1
    assert marker.read_text() == "outside-private-content"
    assert not (outside / "agent.yml").exists()
    assert result.stdout == ""


def test_replacing_existing_files_also_sets_private_permissions(tmp_path, payloads):
    for relative in TARGETS.values():
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("old")
        path.chmod(0o644)
    result = _run(tmp_path, payloads)
    assert result.returncode == 0, result.stderr
    assert all(stat.S_IMODE((tmp_path / relative).stat().st_mode) == 0o600 for relative in TARGETS.values())


def test_workflow_uses_env_secrets_and_declares_reusable_requirements():
    workflow = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/run-nightly.yml").read_text())
    steps = workflow["jobs"]["run-nightly"]["steps"]
    setup = next(step for step in steps if step["name"] == "Setup config file")
    assert setup["env"] == {name: "${{ secrets." + name + " }}" for name in TARGETS}
    assert "prepare_nightly_config.py" in setup["run"]
    assert "${{" not in setup["run"]
    triggers = workflow.get("on", workflow.get(True))
    assert all(triggers["workflow_call"]["secrets"][name]["required"] for name in TARGETS)


def test_staging_failure_cleans_private_temps_and_preserves_existing_files(monkeypatch, tmp_path, payloads, capsys):
    for name, value in payloads.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    for relative in TARGETS.values():
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("existing-private-content")
    mkstemp = config_helper.tempfile.mkstemp
    created = []

    def disk_failure(*args, **kwargs):
        if created:
            raise OSError("STAGING_PRIVATE_SENTINEL")
        descriptor, path = mkstemp(*args, **kwargs)
        created.append(path)
        return descriptor, path

    monkeypatch.setattr(config_helper.tempfile, "mkstemp", disk_failure)
    assert config_helper.main(["--repo-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "STAGING_PRIVATE_SENTINEL" not in captured.err
    assert {p.read_text() for p in tmp_path.rglob("*") if p.is_file()} == {"existing-private-content"}
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == set(TARGETS.values())


def test_dotenv_transport_normalization_preserves_literal_values(tmp_path, payloads):
    payloads["DATUS_NIGHTLY_CI_ENV"] = '  \r\n# comment\r\nMODEL_KEY="quoted%private"\r\n'
    result = _run(tmp_path, payloads, gha=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".env").read_bytes() == b'\n# comment\nMODEL_KEY="quoted%private"\n'
    assert "::add-mask::quoted%25private" in result.stdout.splitlines()
    assert '::add-mask::"quoted%25private"' in result.stdout.splitlines()
