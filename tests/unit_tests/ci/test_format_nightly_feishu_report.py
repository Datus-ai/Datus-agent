import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_node_executable() -> str:
    if node := shutil.which("node"):
        return node

    search_roots = []
    for env_name in ("GITHUB_WORKSPACE", "RUNNER_TEMP"):
        if value := os.environ.get(env_name):
            search_roots.extend(Path(value).resolve().parents)

    for root in search_roots:
        externals_dir = root / "externals"
        for candidate in externals_dir.glob("node*/bin/node"):
            if candidate.is_file():
                return str(candidate)

    pytest.fail("node executable is required to test the nightly Feishu formatter")


def test_nightly_feishu_report_separates_blocking_failures_from_diagnostics(tmp_path):
    (tmp_path / "nightly-manifest.json").write_text(
        json.dumps(
            {
                "summary": {
                    "suite_count": 20,
                    "collected_nodeid_count": 13557,
                    "status_counts": {"failed": 3, "passed": 17},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "nightly-failure-classification.json").write_text(
        json.dumps(
            {
                "summary": {
                    "blocking_category_counts": {
                        "credential_or_config": 1,
                        "docker_infra": 1,
                        "product_regression": 2,
                        "provider_outage": 1,
                    },
                    "category_counts": {
                        "credential_or_config": 1,
                        "docker_infra": 1,
                        "known_flaky": 2,
                        "product_regression": 2,
                        "provider_outage": 2,
                        "unknown_flaky": 4,
                    },
                },
                "findings": [
                    {
                        "blocking": True,
                        "category": "product_regression",
                        "title": "Gen Agent Tests failed",
                        "details": {"suite": "Gen Agent Tests", "exit_code": 1},
                    },
                    {
                        "blocking": True,
                        "category": "provider_outage",
                        "title": "Product E2E Nightly Tests failed",
                        "details": {"suite": "Product E2E Nightly Tests", "exit_code": 1},
                    },
                    {
                        "blocking": False,
                        "category": "known_flaky",
                        "title": "Registered rerun observed",
                        "details": {
                            "nodeid": (
                                "tests/integration/tools/test_bi_dashboard.py::"
                                "TestE2EIntegration::test_complete_workflow"
                            )
                        },
                    },
                    {
                        "blocking": False,
                        "category": "unknown_flaky",
                        "title": "Unregistered rerun observed",
                        "details": {
                            "nodeid": (
                                "tests/integration/agent/test_gen_metrics_agentic.py::"
                                "TestGenMetricsAgentic::test_execute_stream_generates_metric"
                            )
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    script = f"""
const {{ buildNightlyFeishuMessage }} = require({json.dumps(str(REPO_ROOT / "ci/format-nightly-feishu-report.js"))});
const message = buildNightlyFeishuMessage({{
  status: 'FAILED',
  runNumber: '75',
  runUrl: 'https://example.test/run/75',
  date: '2026-05-18',
  workspace: {json.dumps(str(tmp_path))},
  logContent: '',
}});
console.log(message);
"""
    result = subprocess.run([_find_node_executable(), "-e", script], check=True, capture_output=True, text=True)
    message = result.stdout

    assert (
        "**Blocking:** credential_or_config: 1, docker_infra: 1, product_regression: 2, provider_outage: 1." in message
    )
    assert "**Diagnostics:** known_flaky: 2, provider_outage: 1, unknown_flaky: 4." in message
    assert "### Blocking Failures" in message
    assert "### Diagnostic Signals" in message
    assert "all findings" not in message
    assert "### Failure Classification" not in message
