from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parents[3] / "ci"
sys.path.insert(0, str(CI_DIR))
MODULE_PATH = CI_DIR / "classify_nightly_failures.py"
MODULE_SPEC = importlib.util.spec_from_file_location("classify_nightly_failures", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load classify_nightly_failures module from {MODULE_PATH}")
classify_nightly_failures = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(classify_nightly_failures)


def _registry() -> str:
    return """
version: 1
entries:
  - id: kimi-mcp-provider-529
    type: test
    nodeid: tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp
    owner: agent-runtime
    layer: provider_health
    reason: Provider overload rerun is known.
    allowed_until: 2999-01-01
    allowed_in:
      - nightly
  - id: litellm-async-pending-task-teardown
    type: log_pattern
    pattern: "Task was destroyed but it is pending"
    owner: agent-runtime
    layer: provider_health
    reason: LiteLLM teardown warning.
    allowed_until: 2999-01-01
    allowed_in:
      - nightly
"""


def test_classifies_manifest_suite_failures_and_known_flaky(tmp_path):
    manifest = {
        "schema_version": 1,
        "suites": [
            {
                "name": "PostgreSQL Adapter Tests",
                "mode": "blocking",
                "kind": "compose",
                "status": "failed",
                "exit_code": 1,
                "command": ["uv", "run", "pytest", "tests/integration/adapters/test_postgresql.py"],
            },
            {
                "name": "Provider Health Tests",
                "mode": "warn-only",
                "kind": "command",
                "status": "failed",
                "exit_code": 1,
                "command": ["uv", "run", "pytest", "-m", "provider_health"],
            },
        ],
    }
    manifest_file = tmp_path / "nightly-manifest.json"
    log_file = tmp_path / "nightly.log"
    registry_file = tmp_path / "flaky-registry.yml"
    output_file = tmp_path / "classification.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    registry_file.write_text(_registry(), encoding="utf-8")
    log_file.write_text(
        "\n".join(
            [
                "Host port is already in use PostgreSQL: 15432",
                "RERUN tests/integration/models/test_other_models.py::TestKimiModel::test_generate_with_mcp",
                "Task was destroyed but it is pending",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        classify_nightly_failures.main(
            [
                "--manifest",
                str(manifest_file),
                "--log-file",
                str(log_file),
                "--registry",
                str(registry_file),
                "--output",
                str(output_file),
                "--exit-code",
                "1",
            ]
        )
        == 0
    )

    classification = json.loads(output_file.read_text(encoding="utf-8"))

    assert classification["status"] == "failed"
    assert classification["summary"]["blocking_category_counts"]["docker_infra"] >= 1
    assert classification["summary"]["category_counts"]["provider_outage"] >= 1
    assert classification["summary"]["category_counts"]["known_flaky"] == 2
    assert any(finding["details"].get("suite") == "PostgreSQL Adapter Tests" for finding in classification["findings"])


def test_preserves_unknown_failure_when_exit_code_has_no_signal(tmp_path):
    manifest_file = tmp_path / "nightly-manifest.json"
    log_file = tmp_path / "nightly.log"
    registry_file = tmp_path / "flaky-registry.yml"
    output_file = tmp_path / "classification.json"
    manifest_file.write_text(json.dumps({"schema_version": 1, "suites": []}), encoding="utf-8")
    registry_file.write_text("version: 1\nentries: []\n", encoding="utf-8")
    log_file.write_text("nightly stopped unexpectedly\n", encoding="utf-8")

    assert (
        classify_nightly_failures.main(
            [
                "--manifest",
                str(manifest_file),
                "--log-file",
                str(log_file),
                "--registry",
                str(registry_file),
                "--output",
                str(output_file),
                "--exit-code",
                "2",
            ]
        )
        == 0
    )

    classification = json.loads(output_file.read_text(encoding="utf-8"))

    assert classification["summary"]["blocking_category_counts"] == {"unknown_failure": 1}
    assert classification["findings"][0]["details"]["log_tail"] == ["nightly stopped unexpectedly"]


def test_non_object_manifest_is_harness_setup_failure(tmp_path):
    manifest_file = tmp_path / "nightly-manifest.json"
    log_file = tmp_path / "nightly.log"
    registry_file = tmp_path / "flaky-registry.yml"
    output_file = tmp_path / "classification.json"
    manifest_file.write_text("[]\n", encoding="utf-8")
    registry_file.write_text("version: 1\nentries: []\n", encoding="utf-8")
    log_file.write_text("", encoding="utf-8")

    assert (
        classify_nightly_failures.main(
            [
                "--manifest",
                str(manifest_file),
                "--log-file",
                str(log_file),
                "--registry",
                str(registry_file),
                "--output",
                str(output_file),
                "--exit-code",
                "1",
            ]
        )
        == 0
    )

    classification = json.loads(output_file.read_text(encoding="utf-8"))

    assert classification["summary"]["blocking_category_counts"] == {"harness_setup": 1}
    assert classification["findings"][0]["details"]["error"] == "expected json object, got list"


def test_collection_failure_does_not_duplicate_suite_failure(tmp_path):
    manifest_file = tmp_path / "nightly-manifest.json"
    log_file = tmp_path / "nightly.log"
    registry_file = tmp_path / "flaky-registry.yml"
    output_file = tmp_path / "classification.json"
    manifest_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": [
                    {
                        "name": "Main Nightly Tests",
                        "mode": "blocking",
                        "kind": "command",
                        "status": "failed",
                        "exit_code": 2,
                        "collection": {"status": "failed", "exit_code": 2, "raw_tail": ["collection error"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry_file.write_text("version: 1\nentries: []\n", encoding="utf-8")
    log_file.write_text("collection failed\n", encoding="utf-8")

    assert (
        classify_nightly_failures.main(
            [
                "--manifest",
                str(manifest_file),
                "--log-file",
                str(log_file),
                "--registry",
                str(registry_file),
                "--output",
                str(output_file),
                "--exit-code",
                "2",
            ]
        )
        == 0
    )

    classification = json.loads(output_file.read_text(encoding="utf-8"))

    assert classification["summary"]["blocking_category_counts"] == {"harness_setup": 1}
    assert len(classification["findings"]) == 1


def test_passed_provider_suite_header_does_not_create_provider_finding(tmp_path):
    manifest_file = tmp_path / "nightly-manifest.json"
    log_file = tmp_path / "nightly.log"
    registry_file = tmp_path / "flaky-registry.yml"
    output_file = tmp_path / "classification.json"
    manifest_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": [
                    {
                        "name": "Provider Health Tests",
                        "mode": "warn-only",
                        "kind": "command",
                        "status": "passed",
                        "exit_code": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry_file.write_text("version: 1\nentries: []\n", encoding="utf-8")
    log_file.write_text("=== Provider Health Tests (warn-only) ===\n1 passed\n", encoding="utf-8")

    assert (
        classify_nightly_failures.main(
            [
                "--manifest",
                str(manifest_file),
                "--log-file",
                str(log_file),
                "--registry",
                str(registry_file),
                "--output",
                str(output_file),
                "--exit-code",
                "0",
            ]
        )
        == 0
    )

    classification = json.loads(output_file.read_text(encoding="utf-8"))

    assert classification["summary"]["finding_count"] == 0


def test_verbose_token_test_names_do_not_create_credential_finding(tmp_path):
    manifest_file = tmp_path / "nightly-manifest.json"
    log_file = tmp_path / "nightly.log"
    registry_file = tmp_path / "flaky-registry.yml"
    output_file = tmp_path / "classification.json"
    manifest_file.write_text(json.dumps({"schema_version": 1, "suites": []}), encoding="utf-8")
    registry_file.write_text("version: 1\nentries: []\n", encoding="utf-8")
    log_file.write_text(
        "tests/unit_tests/auth/test_token_storage.py::test_save_token PASSED\n"
        "tests/unit_tests/auth/test_claude_credential.py::test_source PASSED\n",
        encoding="utf-8",
    )

    assert (
        classify_nightly_failures.main(
            [
                "--manifest",
                str(manifest_file),
                "--log-file",
                str(log_file),
                "--registry",
                str(registry_file),
                "--output",
                str(output_file),
                "--exit-code",
                "0",
            ]
        )
        == 0
    )

    classification = json.loads(output_file.read_text(encoding="utf-8"))

    assert classification["summary"]["finding_count"] == 0
