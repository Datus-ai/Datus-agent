"""Daily scope follows declared frequency and target, never credential presence."""

import json
import os
import subprocess
import sys

import pytest

from ci import nightly_manifest
from tests.unit_tests.ci.test_hive_nightly_runner import ROOT, _function

INIT = "tests/integration/cli/test_init_live.py::test_qwen_scoped_init_supplements_existing_documents"
SCALE = "tests/integration/cli/test_init_live.py::test_qwen_init_large_catalog_in_one_loop"
CAPACITY = "tests/integration/models/test_qwen_capacity_live.py::test_qwen_long_input_and_automatic_compaction"
INCEPTOR = "tests/integration/tools/test_dosi_hive.py::test_native_inceptor_system_varchar2_metrics"


@pytest.mark.parametrize("scale,capacity", [(False, False), (True, False), (False, True), (True, True)])
def test_weekly_scenarios_are_scheduled_out_unless_explicitly_requested(monkeypatch, scale, capacity):
    monkeypatch.setenv("DATUS_RUN_LIVE_INIT_SCALE", "1" if scale else "0")
    monkeypatch.setenv("DATUS_RUN_LIVE_CAPACITY", "1" if capacity else "0")
    # Unavailable credentials must not make core flows disappear from the plan.
    for name in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cases = {case["nodeid"]: case for case in nightly_manifest.nightly_case_scope()}
    assert cases[INIT]["status"] == "required"
    assert cases[INIT]["frequency"] == "nightly"
    assert cases[SCALE]["status"] == ("required" if scale else "scheduled_out")
    assert cases[CAPACITY]["status"] == ("required" if capacity else "scheduled_out")
    assert cases[SCALE]["frequency"] == cases[CAPACITY]["frequency"] == "weekly"
    assert cases[INCEPTOR]["status"] == "not_applicable"
    assert cases[INCEPTOR]["target"] == "apache-hive"
    assert all(case["reason"] for case in cases.values())


@pytest.mark.parametrize("enabled", [False, True])
def test_daily_shell_selection_consumes_the_same_plan_as_the_manifest(tmp_path, enabled):
    env = {
        **os.environ,
        "DATUS_RUN_LIVE_INIT_SCALE": "1" if enabled else "0",
        "DATUS_RUN_LIVE_CAPACITY": "1" if enabled else "0",
    }
    harness = "\n".join(
        [
            f'uv() {{ shift 2; "{sys.executable}" "$@"; }}',
            _function("prepare_nightly_case_scope"),
            "prepare_nightly_case_scope || exit 1",
            'printf "%s\\n" "${NIGHTLY_FREQUENCY_DESELECTS[@]}"',
        ]
    )
    result = subprocess.run(["bash", "-c", harness], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    expected = [INCEPTOR] if enabled else [SCALE, CAPACITY, INCEPTOR]
    assert result.stdout.splitlines() == [value for nodeid in expected for value in ("--deselect", nodeid)]

    output = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "ci/nightly_manifest.py",
            "init",
            "--repo-root",
            str(ROOT),
            "--external-repos-root",
            str(tmp_path / "external"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    manifest = json.loads(output.read_text())
    excluded = [case["nodeid"] for case in manifest["case_scope"] if case["status"] != "required"]
    assert excluded == expected
