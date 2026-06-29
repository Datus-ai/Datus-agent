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
    pytest_command_lines = [
        line
        for line in _nightly_script_text().splitlines()
        if " uv run pytest " in line and ("run_logged" in line or "run_compose_suite" in line)
    ]

    assert len(pytest_command_lines) == 20
    for line in pytest_command_lines:
        expected_layer = "unit" if "Full Unit Tests" in line else "nightly"
        assert f"env DATUS_TEST_LAYER={expected_layer} uv run pytest" in line


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
