"""BI gates must execute the affected real-service contract and reject missing receipts."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ci.bi_gate import NODEIDS, SOURCE_REVISIONS, compose_document, required, verify

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
        ".github/workflows/run-nightly.yml",
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


def test_bi_receipt_requires_all_product_tests_and_same_sources(tmp_path):
    sources = {"agent": {"sha": "a" * 40}}
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    (tmp_path / "result.json").write_text(json.dumps({"sources": sources, "source_id": identity, "status": "success"}))
    nodeids = NODEIDS
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


def test_local_bi_gate_pins_sources_and_keeps_real_product_regression():
    assert all(len(sha) == 40 for sha in SOURCE_REVISIONS.values())
    assert any("test_bi_chart_lifecycle.py" in n for n in NODEIDS)


def test_pr_queue_context_cannot_call_product_gate_workflows():
    root = Path(__file__).resolve().parents[3]
    queue = yaml.load((root / ".github/workflows/merge-queue.yml").read_text(), Loader=yaml.BaseLoader)
    context = yaml.load((root / ".github/workflows/code-quality.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(queue["on"]) == {"merge_group", "workflow_dispatch"}
    assert set(context["on"]) == {"pull_request"}
    job = context["jobs"]["merge-queue-context"]
    assert job["name"] == "Merge Queue Gate"
    assert "needs" not in job
    assert "if" not in job
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
        if line.startswith('  run_compose_suite "Superset Product Tests"') and '"$DASHBOARD_MARK_EXPR"' in line
    ]
    targets = set(shlex.split(command))
    deselects = script.split("NIGHTLY_DEDICATED_SUITE_DESELECTS=(", 1)[1].split("\n)", 1)[0]
    broad_arguments = shlex.split(deselects)
    for nodeid in NODEIDS:
        path = nodeid.split("::", 1)[0]
        assert path in targets, f"Required BI contract missing from service-backed targeted suite: {nodeid}"
        assert any(broad_arguments[i : i + 2] == ["--deselect", path] for i in range(len(broad_arguments)))
