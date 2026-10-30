"""Release P0 must execute its required cases against the identified artifact."""

import json
import sys
import zipfile
from types import SimpleNamespace

import pytest
import yaml

from ci.harness.test_plan import ROOT, load_map
from ci.p0_plan import check_p0_execution, load_profile, selected_groups
from ci.release_artifact import distributions, verify, verify_install
from ci.stage_p0 import stage

pytestmark = pytest.mark.component


def test_p0_reuses_all_core_journeys_and_includes_live_tasks():
    profile = load_profile()
    groups = profile["groups"]
    assert groups["P0 Core Product Contracts"]["nodeids"] == load_map()["execution"]["core_nodeids"]
    assert len(groups["P0 Core Real Model"]["nodeids"]) == 3
    all_nodeids = [n for group in groups.values() for n in group["nodeids"]]
    assert len(all_nodeids) == len(set(all_nodeids))
    for nodeid in all_nodeids:
        assert (ROOT / nodeid.split("::", 1)[0]).is_file(), nodeid
    assert all(len(sha) == 40 for sha in profile["release_sources"].values())


def test_p0_filter_selects_explicit_groups():
    groups = selected_groups(load_profile(), "^P0 Core ")
    assert set(groups) == {"P0 Core Product Contracts", "P0 Core Real Model"}
    with pytest.raises(ValueError, match="no P0 groups"):
        selected_groups(load_profile(), "^P0 DOES_NOT_EXIST$")


def test_required_postgresql_provider_cannot_skip_missing_dependencies(monkeypatch):
    from tests.integration import conftest

    def missing_dependency():
        raise ModuleNotFoundError("No module named 'testcontainers'")

    provider = SimpleNamespace(load=missing_dependency)
    monkeypatch.setattr(conftest, "_entry_points_for", lambda *args: [provider])
    with pytest.raises(pytest.fail.Exception, match="could not load.*testcontainers"):
        conftest._required_test_environment("datus.storage.rdb.testing", "postgresql")


def test_required_postgresql_setup_failure_cleans_both_providers(monkeypatch):
    from tests.integration import conftest

    calls = []

    def vector_setup():
        raise RuntimeError("Docker unavailable")

    rdb = SimpleNamespace(setup=lambda: None, teardown=lambda: calls.append("rdb"))
    vector = SimpleNamespace(setup=vector_setup, teardown=lambda: calls.append("vector"))
    monkeypatch.setattr(conftest, "_required_test_environment", lambda group, name: rdb if ".rdb." in group else vector)
    fixture = conftest.required_postgresql_storage.__wrapped__()
    with pytest.raises(pytest.fail.Exception, match="setup failed.*Docker unavailable"):
        next(fixture)
    assert calls == ["vector", "rdb"]


@pytest.mark.parametrize("outcome", ["passed", "skipped", "xfailed", "failed", "not_executed"])
def test_p0_requires_successful_execution_of_every_planned_case(tmp_path, outcome):
    path = tmp_path / "execution.json"
    path.write_text(
        json.dumps(
            {
                "collected": ["test_query.py::test_total"],
                "outcomes": {"test_query.py::test_total": outcome},
                "exit_code": 0,
            }
        )
    )
    groups = {"P0 Query": {"nodeids": ["test_query.py::test_total"]}}
    assert check_p0_execution([path], groups)["status"] == ("success" if outcome == "passed" else "failure")
    groups["P0 Query"]["nodeids"].append("test_query.py::test_followup")
    assert check_p0_execution([path], groups)["status"] == "failure"


def test_release_identity_verifies_source_and_distribution_hashes(tmp_path):
    wheel = tmp_path / "datus_agent-1.0-py3-none-any.whl"
    wheel.write_bytes(b"candidate wheel")
    receipt = {"source_sha": "source-sha", "distributions": distributions(tmp_path)}
    (tmp_path / "release-artifact.json").write_text(json.dumps(receipt))
    assert verify(tmp_path, "source-sha") == receipt
    with pytest.raises(ValueError, match="source SHA"):
        verify(tmp_path, "other-sha")
    wheel.write_bytes(b"rebuilt wheel")
    with pytest.raises(ValueError, match="hashes"):
        verify(tmp_path, "source-sha")


def test_installed_candidate_matches_the_receipt(monkeypatch, tmp_path):
    from ci import release_artifact

    wheel = tmp_path / "datus.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("datus/__init__.py", "__version__ = '1.0'\n")
    receipt = {"source_sha": "source-sha", "distributions": distributions(tmp_path)}
    site = tmp_path / "venv/site-packages"
    module = site / "datus/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("__version__ = '1.0'\n")
    monkeypatch.setitem(sys.modules, "datus", SimpleNamespace(__file__=str(module)))
    dist = SimpleNamespace(
        read_text=lambda _: json.dumps({"url": wheel.as_uri(), "archive_info": {}}),
        locate_file=lambda name: site / name,
    )
    monkeypatch.setattr(release_artifact.metadata, "distribution", lambda _: dist)
    result = verify_install(receipt, tmp_path / "checkout")
    assert result["installed_wheel_sha256"] == receipt["distributions"]["datus.whl"]
    assert result["import_path"].endswith("site-packages/datus/__init__.py")
    monkeypatch.setitem(sys.modules, "datus", SimpleNamespace(__file__=str(tmp_path / "checkout/datus/__init__.py")))
    with pytest.raises(ValueError, match="source checkout"):
        verify_install(receipt, tmp_path / "checkout")
    monkeypatch.setitem(sys.modules, "datus", SimpleNamespace(__file__=str(module)))
    module.write_text("__version__ = 'stale'\n")
    with pytest.raises(ValueError, match="payload differs"):
        verify_install(receipt, tmp_path / "checkout")


def test_staged_harness_retains_test_resources_and_uses_the_candidate_environment(tmp_path):
    source = tmp_path / "checkout"
    for name in ("tests", "ci", "conf", ".venv"):
        (source / name).mkdir(parents=True)
    for name in ("pyproject.toml", "pytest.ini", "tests/conftest.py", "conf/agent.yml"):
        (source / name).write_text("fixture content")
    destination = tmp_path / "outside-checkout"
    stage(source, destination)
    assert (destination / "tests/conftest.py").read_text() == "fixture content"
    assert (destination / "conf/agent.yml").read_text() == "fixture content"
    assert (destination / ".venv").resolve() == source / ".venv"


def test_publishing_consumes_the_validated_candidate():
    publish = yaml.safe_load((ROOT / ".github/workflows/publish-release.yml").read_text())
    assert publish["jobs"]["publish"]["needs"] == "validate-candidate"
    assert publish["jobs"]["validate-candidate"]["uses"] == "./.github/workflows/release-candidate.yml"
    steps = publish["jobs"]["publish"]["steps"]
    download = next(step for step in steps if step["name"] == "Download validated candidate distributions")
    assert download["with"]["name"] == "${{ needs.validate-candidate.outputs.artifact_name }}"
    verify_step = next(step for step in steps if step["name"] == "Verify and stage validated distributions")
    assert "release_artifact.py verify" in verify_step["run"]
