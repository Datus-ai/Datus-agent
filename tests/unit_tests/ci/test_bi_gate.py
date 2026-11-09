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


def test_merge_queue_aggregates_bi_gate_and_real_p0_has_regression():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/merge-queue.yml").read_text())
    assert "bi" in workflow["jobs"]["merge-queue-gate"]["needs"]
    assert workflow["jobs"]["bi"]["uses"] == "./.github/workflows/bi-gate.yml"
    gate = (root / ".github/workflows/bi-gate.yml").read_text()
    assert "ci/bi_gate.py run" in gate
    assert "ci/bi_gate.py verify" in gate
    assert any("test_bi_chart_lifecycle.py" in n for n in load_profile()["groups"][GROUP]["nodeids"])


@pytest.mark.parametrize(
    "required,result,expected",
    [
        ("true", "success", 0),
        ("true", "failure", 1),
        ("true", "skipped", 1),
        ("false", "skipped", 0),
    ],
)
def test_final_required_context_checks_bi_result(required, result, expected):
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/merge-queue.yml").read_text())
    command = workflow["jobs"]["merge-queue-gate"]["steps"][0]["run"]
    completed = subprocess.run(
        ["bash", "-e", "-c", command],
        capture_output=True,
        env={
            **os.environ,
            "DETERMINISTIC_RESULT": "success",
            "POLICY_REQUIRED": "false",
            "POLICY_RESULT": "skipped",
            "BI_REQUIRED": required,
            "BI_RESULT": result,
        },
    )
    assert completed.returncode == expected
