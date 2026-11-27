"""CI provider exclusions are explicit and independent of credentials."""

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
def test_non_deepseek_cases_stay_excluded_even_with_weekly_opt_ins(monkeypatch, scale, capacity):
    monkeypatch.setenv("DATUS_RUN_LIVE_INIT_SCALE", "1" if scale else "0")
    monkeypatch.setenv("DATUS_RUN_LIVE_CAPACITY", "1" if capacity else "0")
    # Unavailable credentials must not make core flows disappear from the plan.
    for name in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cases = {case["nodeid"]: case for case in nightly_manifest.nightly_case_scope()}
    assert cases[INIT]["status"] == "not_applicable"
    assert cases[INIT]["frequency"] == "nightly"
    assert cases[SCALE]["status"] == "not_applicable"
    assert cases[CAPACITY.split("::")[0]]["status"] == "not_applicable"
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
    expected = [case["nodeid"] for case in nightly_manifest.nightly_case_scope() if case["status"] != "required"]
    assert result.stdout.splitlines() == [
        value for nodeid in expected for value in ("--deselect" if "::" in nodeid else "--ignore", nodeid)
    ]

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


def test_actual_shell_scope_keeps_only_deepseek_and_local_contracts(tmp_path):
    """Execute the same exclusions as CI, including before-import module ignores."""
    cases = nightly_manifest.nightly_case_scope()
    for case in cases:
        if case["status"] == "required":
            continue
        path, *suffix = case["nodeid"].split("::")
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix:
            with target.open("a") as handle:
                handle.write(f"def {suffix[0]}():\n    raise AssertionError('excluded real model executed')\n")
        else:
            target.write_text("raise AssertionError('excluded provider imported')\n")
    model_root = tmp_path / "tests/integration/models"
    (model_root / "test_deepseek_model.py").write_text("def test_deepseek():\n    assert True\n")
    local = model_root / "test_custom_model_preflight.py"
    with local.open("a") as handle:
        handle.write("def test_local_http_contract():\n    assert True\n")
    harness = "\n".join(
        [
            f'uv() {{ shift 3; "{sys.executable}" "{ROOT}/ci/nightly_manifest.py" "$@"; }}',
            _function("prepare_nightly_case_scope"),
            "prepare_nightly_case_scope || exit 1",
            f'"{sys.executable}" -m pytest tests/integration --noconftest -c /dev/null --rootdir . '
            "-q -p ci.pytest_gate_plugin --gate-report execution.json "
            '-p ci.pytest_fail_on_skip_plugin --fail-on-skip "${NIGHTLY_FREQUENCY_DESELECTS[@]}"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "", "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    report = json.loads((tmp_path / "execution.json").read_text())
    assert report["completed"] is True and report["exit_code"] == 0
    assert report["outcomes"] == {
        "tests/integration/models/test_deepseek_model.py::test_deepseek": "passed",
        "tests/integration/models/test_custom_model_preflight.py::test_local_http_contract": "passed",
    }


def test_provider_health_and_broad_groups_use_the_same_exclusions():
    script = (ROOT / "ci/run-nightly-tests.sh").read_text()
    for group in ("Main Nightly Tests", "Product E2E Nightly Tests", "Provider Health Tests"):
        command = next(line for line in script.splitlines() if line.startswith("run_") and f'"{group}"' in line)
        assert '"${NIGHTLY_FREQUENCY_DESELECTS[@]}"' in command
    assert "\nrun_self_hosted_vision_suite\n" not in script
    assert 'manifest_record_suite "Self-hosted Vision Tests" blocking command skipped' in script


def test_only_deepseek_claims_live_provider_coverage():
    import yaml

    providers = yaml.safe_load((ROOT / "ci/provider-coverage.yml").read_text())["providers"]
    live = {key for key, value in providers.items() if value["live_provider_health"].get("nodeids")}
    assert live == {"deepseek"}
    assert providers["qwen"]["live_provider_health"] == {"mode": "not-tested", "nodeids": []}
    assert all(
        value["deterministic"].get("nodeids") or value["deterministic"].get("inherits_from")
        for value in providers.values()
    )
