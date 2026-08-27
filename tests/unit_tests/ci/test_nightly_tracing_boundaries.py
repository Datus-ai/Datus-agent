from __future__ import annotations

import os
from pathlib import Path

import pytest

from ci.pytest_trace_reference_plugin import _append_jsonl
from tests.unit_tests import conftest as unit_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
NIGHTLY_SCRIPT = REPO_ROOT / "ci" / "run-nightly-tests.sh"


def _nightly_script_text() -> str:
    return NIGHTLY_SCRIPT.read_text(encoding="utf-8")


def _nightly_shell_commands() -> list[str]:
    commands: list[str] = []
    fragments: list[str] = []

    for raw_line in _nightly_script_text().splitlines():
        line = raw_line.strip()
        continued = line.endswith("\\")
        fragments.append(line[:-1].rstrip() if continued else line)
        if continued:
            continue

        commands.append(" ".join(fragment for fragment in fragments if fragment))
        fragments = []

    assert not fragments
    return commands


def _pytest_script_line_containing(label: str) -> str:
    matches = [line for line in _nightly_script_text().splitlines() if label in line and " uv run pytest " in line]
    assert len(matches) == 1
    return matches[0]


def test_nightly_broad_suites_do_not_collect_unit_tests():
    script = _nightly_script_text()

    assert "NIGHTLY_PYTEST_ROOTS=(tests/integration tests/regression)" in script

    for suite_name in (
        "Main Nightly Tests",
        "Product E2E Nightly Tests",
        "Provider Health Tests",
    ):
        line = _pytest_script_line_containing(suite_name)
        assert '"${NIGHTLY_PYTEST_ROOTS[@]}"' in line
        assert " tests/ " not in line


def test_nightly_pytest_commands_set_explicit_test_layer():
    pytest_commands = [
        command
        for command in _nightly_shell_commands()
        if (" uv run pytest " in command or " uv run python -m pytest " in command)
        and any(wrapper in command for wrapper in ("run_logged", "run_compose_suite", "run_with_agent_home"))
    ]

    assert len(pytest_commands) == 26
    assert any("tests/integration/adapters/test_doris.py" in command for command in pytest_commands)
    assert any("tests/integration/adapters/test_gaussdb.py" in command for command in pytest_commands)
    assert any("tests/integration/adapters/test_tidb.py" in command for command in pytest_commands)
    for group in (
        "P0 PostgreSQL Agent Storage Contracts",
        "P0 SQL Policy Plugin Contracts",
        "P0 Dosi Semantic Modeling E2E",
        "P0 Dashboard Bootstrap Skill E2E",
    ):
        assert any(group in command for command in pytest_commands)
    for command in pytest_commands:
        expected_layer = "unit" if "Full Unit Tests" in command else "nightly"
        assert f"env DATUS_TEST_LAYER={expected_layer}" in command


@pytest.mark.parametrize(
    ("test_layer", "expected_cleanup_enabled"),
    [
        (None, True),
        ("", True),
        ("unit", True),
        ("ci", True),
        ("nightly", False),
        (" Nightly ", False),
        ("integration", False),
        ("regression", False),
        ("product_e2e", False),
        ("provider_health", False),
    ],
)
def test_unit_conftest_keeps_external_tracing_for_non_unit_layers(test_layer, expected_cleanup_enabled):
    assert unit_conftest._external_tracing_cleanup_enabled(test_layer) == expected_cleanup_enabled


def _restore_unit_conftest_langfuse_state(saved_env: dict[str, str | None], stripped: bool) -> None:
    unit_conftest._saved_langfuse_env.clear()
    unit_conftest._saved_langfuse_env.update(saved_env)
    unit_conftest._langfuse_env_stripped = stripped


def test_unit_conftest_pytest_configure_keeps_langfuse_env_for_nightly_layer(monkeypatch):
    original_saved_env = dict(unit_conftest._saved_langfuse_env)
    original_stripped = unit_conftest._langfuse_env_stripped
    expected_env = {
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
        "LANGFUSE_BASE_URL": "https://langfuse.test",
    }

    try:
        unit_conftest._saved_langfuse_env.clear()
        monkeypatch.setenv("DATUS_TEST_LAYER", "nightly")
        for key, value in expected_env.items():
            monkeypatch.setenv(key, value)

        unit_conftest.pytest_configure(config=None)

        observed_env = {key: os.environ[key] for key in expected_env}
        assert observed_env == expected_env
        assert unit_conftest._saved_langfuse_env == {}
        assert unit_conftest._langfuse_env_stripped is False

        unit_conftest.pytest_unconfigure(config=None)

        observed_env = {key: os.environ[key] for key in expected_env}
        assert observed_env == expected_env
    finally:
        _restore_unit_conftest_langfuse_state(original_saved_env, original_stripped)


def test_unit_conftest_pytest_configure_strips_and_restores_langfuse_env_for_unit_layer(monkeypatch):
    original_saved_env = dict(unit_conftest._saved_langfuse_env)
    original_stripped = unit_conftest._langfuse_env_stripped
    expected_env = {
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
        "LANGFUSE_BASE_URL": "https://langfuse.test",
    }
    expected_saved_env = {**expected_env, "LANGFUSE_HOST": None}

    try:
        unit_conftest._saved_langfuse_env.clear()
        monkeypatch.delenv("DATUS_TEST_LAYER", raising=False)
        for key, value in expected_env.items():
            monkeypatch.setenv(key, value)

        unit_conftest.pytest_configure(config=None)

        observed_env = {key: os.environ.get(key) for key in expected_env}
        assert observed_env == dict.fromkeys(expected_env)
        assert unit_conftest._saved_langfuse_env == expected_saved_env
        assert unit_conftest._langfuse_env_stripped is True

        unit_conftest.pytest_unconfigure(config=None)

        observed_env = {key: os.environ[key] for key in expected_env}
        assert observed_env == expected_env
    finally:
        _restore_unit_conftest_langfuse_state(original_saved_env, original_stripped)


def test_trace_reference_jsonl_write_is_warn_only(monkeypatch, tmp_path):
    output = tmp_path / "missing-parent" / "trace.jsonl"

    def fail_mkdir(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.warns(RuntimeWarning, match="Failed to write nightly trace reference"):
        _append_jsonl(output, {"trace_id": "trace-1"})

    assert not output.exists()
