from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "check_flaky_registry.py"
MODULE_SPEC = importlib.util.spec_from_file_location("check_flaky_registry", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load check_flaky_registry module from {MODULE_PATH}")
check_flaky_registry = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(check_flaky_registry)


def _registry(allowed_until: str = "2999-01-01") -> dict:
    return {
        "version": 1,
        "entries": [
            {
                "id": "kimi-mcp-provider-529",
                "type": "test",
                "nodeid": "tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp",
                "owner": "agent-runtime",
                "layer": "product_e2e",
                "reason": "Provider overload rerun is known.",
                "allowed_until": allowed_until,
                "allowed_in": ["nightly"],
            },
            {
                "id": "async-pending-task",
                "type": "log_pattern",
                "pattern": "Task was destroyed but it is pending",
                "owner": "agent-runtime",
                "layer": "product_e2e",
                "reason": "Known teardown warning.",
                "allowed_until": allowed_until,
                "allowed_in": ["nightly"],
            },
        ],
    }


def test_validate_registry_accepts_owned_unexpired_entries():
    registry = check_flaky_registry.validate_registry(_registry(), today=date(2026, 5, 6), strict=True)

    assert registry.test_nodeids == {
        "tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp"
    }
    assert [entry_id for entry_id, _pattern in registry.log_patterns] == ["async-pending-task"]


def test_validate_registry_rejects_expired_entries():
    try:
        check_flaky_registry.validate_registry(_registry("2026-05-05"), today=date(2026, 5, 6))
    except ValueError as exc:
        assert "expired" in str(exc)
    else:
        raise AssertionError("expired flaky entries must fail validation")


def test_classify_log_splits_registered_and_unregistered_reruns(tmp_path):
    registry = check_flaky_registry.validate_registry(_registry(), today=date(2026, 5, 6))
    log_file = tmp_path / "nightly.log"
    log_file.write_text(
        "\n".join(
            [
                "RERUN tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp",
                "RERUN tests/integration/models/test_qwen_model.py::TestQwenModel::test_generate_with_mcp",
                "Task was destroyed but it is pending",
            ]
        ),
        encoding="utf-8",
    )

    registered, unregistered, patterns = check_flaky_registry.classify_log(log_file, registry)

    assert registered == ["tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp"]
    assert unregistered == ["tests/integration/models/test_qwen_model.py::TestQwenModel::test_generate_with_mcp"]
    assert patterns == ["async-pending-task"]


def test_main_returns_nonzero_for_unregistered_rerun(tmp_path):
    registry_file = tmp_path / "registry.yml"
    log_file = tmp_path / "nightly.log"
    registry_file.write_text(
        """
version: 1
entries:
  - id: kimi-mcp-provider-529
    type: test
    nodeid: tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp
    owner: agent-runtime
    layer: product_e2e
    reason: Provider overload rerun is known.
    allowed_until: 2999-01-01
    allowed_in:
      - nightly
""",
        encoding="utf-8",
    )
    log_file.write_text(
        "RERUN tests/integration/models/test_qwen_model.py::TestQwenModel::test_generate_with_mcp\n",
        encoding="utf-8",
    )

    assert check_flaky_registry.main(["--registry", str(registry_file), "--log-file", str(log_file)]) == 1
