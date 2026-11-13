"""SQL policy acceptance must follow source changes and prove the built combination."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ci.release_artifact import sha256, verify_wheel_payload
from ci.sql_policy_gate import (
    GROUP,
    REQUIRED_STAGES,
    ROOT,
    build_environment,
    load_config,
    load_profile,
    plan_gate,
    policy_engine_revision,
    snapshot,
    source_id,
    verify_result,
    write_json,
)

pytestmark = pytest.mark.component


@pytest.mark.parametrize(
    "path",
    [
        "datus/plugins/store.py",
        "datus/cli/plugin_service.py",
        "datus/tools/policy_runtime.py",
        "datus/tools/func_tool/database.py",
        "datus/tools/semantic_tools/dosi/runtime.py",
        "tests/integration/conftest.py",
        "tests/integration/plugins/test_sql_policy_plugin.py",
    ],
)
def test_policy_boundaries_require_real_host_acceptance(path):
    assert plan_gate([path])["required"]


@pytest.mark.parametrize(
    "path", ["pyproject.toml", "uv.lock", "ci/harness/sql-policy-host.json", "ci/sql_policy_gate.py"]
)
def test_build_changes_require_cold_acceptance(path):
    assert plan_gate([path])["cold"]


def test_docs_keep_the_gate_optional_and_scheduled_runs_require_cold_builds():
    assert not plan_gate(["docs/index.md"])["required"]
    assert not plan_gate(["docs/index.md"])["cold"]
    assert plan_gate([], force=True)["required"]
    assert plan_gate([], force=True)["cold"]


def test_host_gate_uses_the_reviewed_release_policy_source():
    assert plan_gate([])["sql_policies"] == load_profile()["release_sources"]["sql_policies"]


@pytest.mark.parametrize(
    "field,value", [("schema_version", 2), ("sql_policies", "main"), ("rust_toolchain", "stable"), ("cold_paths", [])]
)
def test_incomplete_or_mutable_configuration_blocks_planning(monkeypatch, tmp_path, field, value):
    from ci import sql_policy_gate

    config = {**load_config(), field: value}
    path = tmp_path / "config.json"
    write_json(path, config)
    monkeypatch.setattr(sql_policy_gate, "CONFIG", path)
    with pytest.raises(ValueError):
        plan_gate(["datus/plugins/store.py"])


@pytest.mark.parametrize("revisions", [["a" * 40], ["main"], ["a" * 40, "b" * 40]])
def test_policy_engine_binding_requires_one_immutable_revision(tmp_path, revisions):
    (tmp_path / "Cargo.toml").write_text(
        "[workspace.dependencies]\n"
        + "\n".join(
            f'dosi-{index} = {{ git = "https://github.com/Datus-ai/osi-engine", rev = "{revision}" }}'
            for index, revision in enumerate(revisions)
        )
    )
    if revisions == ["a" * 40]:
        assert policy_engine_revision(tmp_path) == "a" * 40
    else:
        with pytest.raises(ValueError):
            policy_engine_revision(tmp_path)


def test_cold_build_ignores_populated_host_caches_and_preserves_rustup(monkeypatch, tmp_path):
    for name in ("UV_CACHE_DIR", "CARGO_HOME", "CARGO_TARGET_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "preexisting-cache"))
    monkeypatch.setenv("RUSTUP_HOME", str(tmp_path / "toolchain"))
    env = build_environment(tmp_path / "cold", "cold", tmp_path / "engine")
    assert env["RUSTUP_HOME"] == str(tmp_path / "toolchain")
    for name in ("UV_CACHE_DIR", "CARGO_HOME", "CARGO_TARGET_DIR"):
        directory = Path(env[name])
        assert directory.is_relative_to(tmp_path / "cold")
        assert list(directory.iterdir()) == []
    (Path(env["CARGO_HOME"]) / "stale-build").touch()
    with pytest.raises(ValueError, match="must start empty"):
        build_environment(tmp_path / "cold", "cold", tmp_path / "engine")


def test_warm_build_reuses_configured_cache_and_fetches_engine_without_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "cached-cargo"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "builder")
    env = build_environment(tmp_path / "work", "warm", tmp_path / "engine")
    assert env["CARGO_HOME"] == str(tmp_path / "cached-cargo")
    assert env["GIT_CONFIG_KEY_0"] == "user.name"
    assert env["GIT_CONFIG_VALUE_1"] == "https://github.com/Datus-ai/osi-engine"
    assert env["GIT_CONFIG_KEY_1"] == f"url.{(tmp_path / 'engine').as_uri()}.insteadOf"


@pytest.mark.parametrize("suffix", ["", ".git"])
def test_native_dependency_fetch_reads_the_pinned_local_checkout(tmp_path, suffix):
    engine = tmp_path / "engine"
    engine.mkdir()
    for command in (
        ["git", "init", "-q", str(engine)],
        [
            "git",
            "-C",
            str(engine),
            "-c",
            "user.name=probe",
            "-c",
            "user.email=probe@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "pinned source",
        ],
    ):
        subprocess.run(command, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    cache = tmp_path / "fetch"
    subprocess.run(["git", "init", "-q", str(cache)], check=True, capture_output=True)
    env = build_environment(tmp_path / "work", "warm", engine)
    env["GIT_ALLOW_PROTOCOL"] = "file"
    fetched = subprocess.run(
        ["git", "-C", str(cache), "fetch", "https://github.com/Datus-ai/osi-engine" + suffix, sha],
        env=env,
        capture_output=True,
        text=True,
    )
    assert fetched.returncode == 0, fetched.stderr
    assert subprocess.check_output(["git", "-C", str(cache), "rev-parse", "FETCH_HEAD"], text=True).strip() == sha


def test_snapshot_binds_lockfiles_and_rejects_a_different_engine(monkeypatch, tmp_path):
    from ci import sql_policy_gate

    agent, policy, engine = (tmp_path / name for name in ("agent", "policy", "engine"))
    for root, paths in (
        (agent, ("pyproject.toml", "uv.lock")),
        (policy, ("pyproject.toml", "Cargo.toml", "Cargo.lock")),
        (engine, ("Cargo.toml", "Cargo.lock", "crates/dosi-py/pyproject.toml")),
    ):
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source input")
    (policy / "Cargo.toml").write_text(
        '[workspace.dependencies]\ndosi-sql = { git = "https://github.com/Datus-ai/osi-engine", rev = "'
        + "a" * 40
        + '" }\n'
    )
    monkeypatch.setattr(sql_policy_gate, "revision", lambda _: "a" * 40)
    monkeypatch.setattr(sql_policy_gate.subprocess, "check_output", lambda *args, **kwargs: b"")
    original = snapshot(agent, policy, engine)
    (policy / "Cargo.lock").write_text("changed dependency input")
    assert source_id(snapshot(agent, policy, engine)) != source_id(original)
    monkeypatch.setattr(sql_policy_gate, "revision", lambda root: ("b" if root == engine else "a") * 40)
    with pytest.raises(ValueError, match="pinned engine"):
        snapshot(agent, policy, engine)


def _evidence(tmp_path):
    sources = {name: {"sha": name[0] * 40} for name in ("agent", "sql_policies", "osi_engine")}
    wheels = {name: tmp_path / (name + ".whl") for name in sources}
    for wheel in wheels.values():
        wheel.write_bytes(b"built wheel")
    receipt = {
        "sources": sources,
        "source_id": source_id(sources),
        "artifacts": {name: {"sha256": sha256(wheel)} for name, wheel in wheels.items()},
        "stages": [{"name": name, "exit_code": 0} for name in REQUIRED_STAGES],
    }
    nodes = load_profile()["groups"][GROUP]["nodeids"]
    report = {
        "source_id": receipt["source_id"],
        "collected": nodes,
        "outcomes": {node: "passed" for node in nodes},
        "exit_code": 0,
    }
    path = tmp_path / "execution.json"
    write_json(path, report)
    return sources, receipt, wheels, report, path


def test_identified_builds_and_successful_execution_are_accepted(tmp_path):
    sources, receipt, wheels, _, path = _evidence(tmp_path)
    assert verify_result(sources, receipt, path, wheels)["status"] == "success"


@pytest.mark.parametrize("outcome", ["skipped", "xfailed", "failed", "not_executed"])
def test_missing_or_unsuccessful_p0_execution_blocks_acceptance(tmp_path, outcome):
    sources, receipt, wheels, report, path = _evidence(tmp_path)
    report["outcomes"] = {node: outcome for node in report["collected"]}
    write_json(path, report)
    with pytest.raises(ValueError):
        verify_result(sources, receipt, path, wheels)


@pytest.mark.parametrize(
    "changed", ["source", "artifact", "execution", "missing-report", "missing-stage", "failed-stage"]
)
def test_stale_source_artifact_or_execution_evidence_is_rejected(tmp_path, changed):
    sources, receipt, wheels, report, path = _evidence(tmp_path)
    if changed == "source":
        sources = {**sources, "agent": {"sha": "f" * 40}}
    elif changed == "artifact":
        wheels["sql_policies"].write_bytes(b"rebuilt after verification")
    elif changed == "execution":
        report["source_id"] = "old-combination"
        write_json(path, report)
    elif changed == "missing-report":
        path.unlink()
    elif changed == "missing-stage":
        receipt["stages"].pop()
    else:
        receipt["stages"][0]["exit_code"] = 1
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_result(sources, receipt, path, wheels)


@pytest.mark.parametrize("changed", ["extension", "bundled-library"])
def test_installed_native_payload_must_match_the_identified_wheel(tmp_path, changed):
    wheel = tmp_path / "policy.whl"
    package = tmp_path / "installed" / "datus_sql_policies"
    package.mkdir(parents=True)
    (package / "_native.so").write_bytes(b"native artifact")
    bundled = package.parent / "datus_sql_policies.libs" / "libdependency.so"
    bundled.parent.mkdir()
    bundled.write_bytes(b"bundled native library")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("datus_sql_policies/_native.so", b"native artifact")
        archive.writestr("datus_sql_policies.libs/libdependency.so", b"bundled native library")
    verify_wheel_payload(wheel, package.parent, package.name)
    (bundled if changed == "bundled-library" else package / "_native.so").write_bytes(b"unrelated cached native file")
    with pytest.raises(ValueError, match="payload differs"):
        verify_wheel_payload(wheel, package.parent, package.name)


@pytest.mark.parametrize("inherit_pytest_addopts", [False, True])
def test_execution_report_records_the_frozen_source_identity(tmp_path, monkeypatch, inherit_pytest_addopts):
    if inherit_pytest_addopts:
        monkeypatch.setenv("PYTEST_ADDOPTS", f"--basetemp={tmp_path.parent}")
    else:
        monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    (tmp_path / "test_probe.py").write_text("def test_probe(): pass\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    report = tmp_path / "execution.json"
    env = {
        **os.environ,
        "DATUS_GATE_SOURCE_ID": "source-combination",
        "PYTHONPATH": str(ROOT),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    # The outer pytest basetemp is an ancestor of this subprocess's working directory.
    env.pop("PYTEST_ADDOPTS", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_probe.py",
            "-p",
            "ci.pytest_gate_plugin",
            "--gate-report",
            str(report),
            "--rootdir",
            str(tmp_path),
            "-c",
            str(tmp_path / "pytest.ini"),
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(report.read_text())["source_id"] == "source-combination"


@pytest.mark.parametrize(
    "path", ["ci/release_artifact.py", "ci/pytest_gate_plugin.py", "ci/p0_plan.py", "ci/stage_p0.py"]
)
def test_verifier_changes_require_installed_host_acceptance(path):
    assert plan_gate([path])["required"] is True
