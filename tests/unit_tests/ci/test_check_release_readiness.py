from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess
from textwrap import dedent

import pytest
from packaging.version import Version

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "check_release_readiness.py"


@pytest.fixture()
def check_release_readiness(monkeypatch):
    module_name = "_test_check_release_readiness"
    module_spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError(f"Unable to load check_release_readiness from {MODULE_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    module_spec.loader.exec_module(module)
    return module


def _write_release_repo(tmp_path: Path, *, version: str = "0.2.6", init_version: str | None = None) -> Path:
    repo_root = tmp_path
    (repo_root / "datus").mkdir()
    (repo_root / "datus" / "__init__.py").write_text(
        f'__version__ = "{init_version or version}"\n',
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        dedent(
            f"""
            [project]
            name = "datus-agent"
            version = "{version}"
            dependencies = [
                "datus-db-core>=0.1.3",
                "datus-semantic-core>=0.2.0",
                "datus-bi-core>=0.1.2",
                "datus-scheduler-core>=0.1.1",
            ]
            """
        ).strip(),
        encoding="utf-8",
    )
    (repo_root / "requirements.txt").write_text(
        dedent(
            """
            datus-db-core>=0.1.3
            datus-semantic-core>=0.2.0
            datus-bi-core>=0.1.2
            datus-scheduler-core>=0.1.1
            """
        ).strip(),
        encoding="utf-8",
    )
    return repo_root


def test_source_version_consistency_accepts_matching_versions(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path)

    errors = check_release_readiness.check_source_version_consistency(repo_root)

    assert errors == []


def test_source_version_consistency_rejects_mismatched_init_version(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path, version="0.2.6", init_version="0.2.5")

    errors = check_release_readiness.check_source_version_consistency(repo_root)

    assert "Version mismatch" in errors[0]
    assert "0.2.6" in errors[0]
    assert "0.2.5" in errors[0]


def test_source_version_consistency_rejects_unexpected_release_version(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path, version="0.2.6")

    errors = check_release_readiness.check_source_version_consistency(repo_root, expected_version="0.2.7")

    assert errors == ["Expected release version 0.2.7, but pyproject.toml has 0.2.6"]


def test_installed_distribution_version_rejects_mismatch(monkeypatch, check_release_readiness):
    monkeypatch.setattr(check_release_readiness.metadata, "version", lambda _name: "0.2.5")

    errors = check_release_readiness.check_installed_distribution_version(Version("0.2.6"))

    assert errors == ["Installed distribution 'datus-agent' has version 0.2.5, expected 0.2.6"]


def test_console_script_versions_rejects_mismatched_output(check_release_readiness):
    def runner(args, **_kwargs):
        return CompletedProcess(args=args, returncode=0, stdout="Datus CLI 0.2.5\n", stderr="")

    errors = check_release_readiness.check_console_script_versions(Version("0.2.6"), commands=("datus",), runner=runner)

    assert errors == ["datus --version output 'Datus CLI 0.2.5' does not contain expected version 0.2.6"]


def test_console_script_versions_reports_missing_command(check_release_readiness):
    def runner(_args, **_kwargs):
        raise FileNotFoundError("missing datus")

    errors = check_release_readiness.check_console_script_versions(Version("0.2.6"), commands=("datus",), runner=runner)

    assert errors == ["datus --version failed to execute: missing datus"]


def test_adapter_dependency_consistency_accepts_matching_lower_bounds(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path)

    checks, errors = check_release_readiness.check_adapter_dependency_consistency(repo_root)

    assert errors == []
    assert {check.name: check.pyproject_lower_bound for check in checks} == {
        "datus-db-core": Version("0.1.3"),
        "datus-semantic-core": Version("0.2.0"),
        "datus-bi-core": Version("0.1.2"),
        "datus-scheduler-core": Version("0.1.1"),
    }


def test_adapter_dependency_consistency_rejects_mismatched_requirements_lower_bound(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path)
    (repo_root / "requirements.txt").write_text(
        dedent(
            """
            datus-db-core>=0.1.2
            datus-semantic-core>=0.2.0
            datus-bi-core>=0.1.2
            datus-scheduler-core>=0.1.1
            """
        ).strip(),
        encoding="utf-8",
    )

    _checks, errors = check_release_readiness.check_adapter_dependency_consistency(repo_root)

    assert errors == ["datus-db-core lower bound mismatch: pyproject.toml has 0.1.3, requirements.txt has 0.1.2"]


def test_adapter_dependency_consistency_requires_lower_bounds(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path)
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(pyproject.replace('"datus-db-core>=0.1.3"', '"datus-db-core"'))

    _checks, errors = check_release_readiness.check_adapter_dependency_consistency(repo_root)

    assert errors == ["datus-db-core in pyproject.toml must declare a >= lower bound"]


def test_adapter_latest_check_rejects_stale_lower_bound(tmp_path, check_release_readiness):
    repo_root = _write_release_repo(tmp_path)
    checks, errors = check_release_readiness.check_adapter_dependency_consistency(repo_root)
    assert errors == []

    _latest_checks, latest_errors = check_release_readiness.check_adapter_dependencies_are_latest(
        checks,
        lambda package_name: {
            "datus-db-core": Version("0.1.4"),
            "datus-semantic-core": Version("0.2.0"),
            "datus-bi-core": Version("0.1.2"),
            "datus-scheduler-core": Version("0.1.1"),
        }[package_name],
    )

    assert latest_errors == ["datus-db-core lower bound must match latest PyPI release: declared 0.1.3, latest 0.1.4"]


def test_main_offline_checks_pass_for_current_repo(check_release_readiness):
    assert check_release_readiness.main(["--repo-root", str(check_release_readiness.REPO_ROOT)]) == 0
