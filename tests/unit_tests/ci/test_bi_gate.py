"""BI gates must execute the affected real-service contract and reject missing receipts."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ci.bi_gate import GROUP, compose_document, required, verify
from ci.p0_plan import load_profile

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize(
    "path",
    [
        "datus/tools/func_tool/bi_tools.py",
        "datus/validation/hook.py",
        "datus/resources/skills/superset-dashboard/SKILL.md",
        "datus/resources/skills/bi-validation/SKILL.md",
        "tests/integration/tools/test_bi_chart_lifecycle.py",
        "uv.lock",
        ".github/workflows/bi-gate.yml",
    ],
)
def test_bi_change_requires_real_service_acceptance(path):
    assert required([path])


def test_unrelated_docs_do_not_require_bi():
    assert not required(["docs/changelog.md"])


def test_compose_isolates_names_ports_and_fixture_mounts(tmp_path):
    source = tmp_path / "compose.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "services": {
                    name: {"container_name": name, "ports": ["8088:8088"], "volumes": ["./init.sh:/init.sh:ro"]}
                    for name in ["superset", "postgres"]
                }
            }
        )
    )
    document = compose_document(source, 18089, 15439)
    assert document["services"]["superset"]["ports"] == ["127.0.0.1:18089:8088"]
    assert document["services"]["postgres"]["ports"] == ["127.0.0.1:15439:5432"]
    assert all("container_name" not in service for service in document["services"].values())
    assert document["services"]["postgres"]["volumes"] == [str(tmp_path / "init.sh") + ":/init.sh:ro"]


def test_bi_receipt_requires_all_p0_tests_and_same_sources(tmp_path):
    sources = {"agent": {"sha": "a" * 40}}
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    (tmp_path / "result.json").write_text(json.dumps({"sources": sources, "source_id": identity, "status": "success"}))
    nodeids = load_profile()["groups"][GROUP]["nodeids"]
    report = {"collected": nodeids, "outcomes": dict.fromkeys(nodeids, "passed"), "exit_code": 0, "source_id": identity}
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(report))
    assert verify(tmp_path, sources)["status"] == "success"
    for outcome in ["skipped", "failed", "xfailed"]:
        report["outcomes"][nodeids[0]] = outcome
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="did not pass"):
            verify(tmp_path, sources)
    report["outcomes"] = {}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify(tmp_path, sources)
    with pytest.raises(ValueError, match="source identity"):
        verify(tmp_path, {})
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify(tmp_path, sources)


def test_bi_gate_keeps_real_p0_regression():
    root = Path(__file__).resolve().parents[3]
    gate = (root / ".github/workflows/bi-gate.yml").read_text()
    assert "ci/bi_gate.py run" in gate
    assert "ci/bi_gate.py verify" in gate
    assert any("test_bi_chart_lifecycle.py" in n for n in load_profile()["groups"][GROUP]["nodeids"])


def test_privileged_pr_context_cannot_call_product_gate_workflows():
    root = Path(__file__).resolve().parents[3]
    queue = yaml.load((root / ".github/workflows/merge-queue.yml").read_text(), Loader=yaml.BaseLoader)
    context = yaml.load((root / ".github/workflows/merge-queue-pr.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(queue["on"]) == {"merge_group", "workflow_dispatch"}
    assert set(context["on"]) == {"pull_request_target"}
    (job,) = context["jobs"].values()
    assert job["name"] == "Merge Queue Gate"
    assert "uses" not in job
    assert all("uses" not in step for step in job["steps"])
    assert all(set(step) == {"name", "run"} for step in job["steps"])


def test_targeted_bi_suite_runs_every_required_file_with_services_and_excludes_broad_collection():
    import shlex

    root = Path(__file__).resolve().parents[3]
    script = (root / "ci/run-nightly-tests.sh").read_text()
    (command,) = [
        line
        for line in script.splitlines()
        if line.startswith('  run_compose_suite "P0 BI Product E2E"') and "pytest" in line
    ]
    targets = set(shlex.split(command))
    deselects = script.split("NIGHTLY_DEDICATED_SUITE_DESELECTS=(", 1)[1].split("\n)", 1)[0]
    broad_arguments = shlex.split(deselects)
    for nodeid in load_profile()["groups"][GROUP]["nodeids"]:
        path = nodeid.split("::", 1)[0]
        assert path in targets, f"Required BI contract missing from service-backed targeted suite: {nodeid}"
        assert any(broad_arguments[i : i + 2] == ["--deselect", path] for i in range(len(broad_arguments)))


def test_bi_job_budget_covers_script_timeouts_cleanup_and_setup():
    from ci.bi_gate import (
        COMPOSE_DOWN_TIMEOUT_SECONDS,
        COMPOSE_LOGS_TIMEOUT_SECONDS,
        COMPOSE_UP_TIMEOUT_SECONDS,
        PYTEST_TIMEOUT_SECONDS,
    )

    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/bi-gate.yml").read_text())
    script_budget = (
        COMPOSE_UP_TIMEOUT_SECONDS
        + PYTEST_TIMEOUT_SECONDS
        + COMPOSE_LOGS_TIMEOUT_SECONDS
        + COMPOSE_DOWN_TIMEOUT_SECONDS
    )
    setup_and_upload_margin = 15 * 60
    assert workflow["jobs"]["superset"]["timeout-minutes"] * 60 >= script_budget + setup_and_upload_margin
