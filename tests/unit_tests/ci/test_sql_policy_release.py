"""Nightly consumes identified published wheels and never falls back to a build."""

import json
import subprocess
from pathlib import Path

import pytest
from packaging.tags import Tag

from ci import sql_policy_release as policy

pytestmark = pytest.mark.component
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def release_server(monkeypatch, tmp_path):
    wheel_name = "datus_sql_policies-0.5.3-cp312-abi3-manylinux_2_34_x86_64.whl"
    wheel_data = b"published native wheel"
    manifest = {
        "development": False,
        "version": "0.5.3",
        "platform": "linux-amd64",
        "source_commit": "a" * 40,
        "public_api_revision": "b" * 40,
        "hosts": {"dosi_requirement": "dosi-engine==0.2.1"},
    }
    release = {
        "id": 42,
        "tag_name": "v0.5.3",
        "html_url": "https://github.com/Datus-ai/datus-sql-policies/releases/tag/v0.5.3",
        "draft": False,
        "prerelease": False,
        "assets": [
            {"id": 1, "name": wheel_name},
            {"id": 2, "name": "BUILD-MANIFEST-linux-amd64.json"},
            {"id": 3, "name": "SHA256SUMS-linux-amd64"},
            {"id": 4, "name": "backend-" + wheel_name},
        ],
    }
    state = {"release": release, "manifest": manifest, "wheel_data": wheel_data, "requests": []}

    def request(endpoint, *, binary=False):
        state["requests"].append((endpoint, binary))
        if "/git/ref/tags/" in endpoint:
            return json.dumps({"object": {"type": "commit", "sha": "a" * 40}}).encode()
        if not binary:
            return json.dumps(release).encode()
        data = json.dumps(manifest).encode()
        return {
            "1": state["wheel_data"],
            "2": data,
            "3": (
                f"{policy.checksum(data)}  BUILD-MANIFEST-linux-amd64.json\n"
                f"{policy.checksum(wheel_data)}  {release['assets'][0]['name']}\n"
            ).encode(),
        }[endpoint.rsplit("/", 1)[-1]]

    monkeypatch.setattr(policy, "request", request)
    monkeypatch.setattr(policy, "platform_name", lambda: "linux-amd64")
    monkeypatch.setattr(policy, "sys_tags", lambda: iter([Tag("cp312", "abi3", "manylinux_2_34_x86_64")]))
    monkeypatch.setattr(policy.metadata, "version", lambda _: "0.2.1")
    return state


@pytest.mark.parametrize("selected,endpoint", [("", "latest"), ("v0.5.3", "tags/v0.5.3")])
def test_release_is_resolved_once_and_verified_without_source_access(tmp_path, release_server, selected, endpoint):
    work = tmp_path / "work"
    receipt = policy.prepare(work, release=selected, agent=ROOT)
    assert release_server["requests"][0] == (f"repos/{policy.REPOSITORY}/releases/{endpoint}", False)
    assert sum("/releases/" in name and not binary for name, binary in release_server["requests"]) == 1
    assert receipt["status"] == "success"
    assert receipt["tag"] == "v0.5.3"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["dosi_version"] == "0.2.1"
    assert receipt["wheel"]["sha256"] == policy.checksum(release_server["wheel_data"])
    assert (work / receipt["wheel"]["file"]).read_bytes() == release_server["wheel_data"]
    assert json.loads((work / "sql-policy-release.json").read_text()) == receipt


@pytest.mark.parametrize("field", ["draft", "prerelease"])
def test_unpublished_or_prerelease_is_rejected(tmp_path, release_server, field):
    release_server["release"][field] = True
    with pytest.raises(ValueError, match="published stable"):
        policy.prepare(tmp_path / "work")
    assert len(release_server["requests"]) == 1


def test_nightly_follows_a_new_stable_release(tmp_path, release_server):
    release_server["release"]["tag_name"] = "v0.5.4"
    release_server["release"]["assets"][0]["name"] = release_server["release"]["assets"][0]["name"].replace(
        "0.5.3", "0.5.4"
    )
    release_server["manifest"]["version"] = "0.5.4"
    receipt = policy.prepare(tmp_path / "work", agent=ROOT)
    assert receipt["tag"] == "v0.5.4"


@pytest.mark.parametrize(
    "case", ["missing-wheel", "incompatible-wheel", "missing-manifest", "missing-checksums", "duplicate-wheel"]
)
def test_missing_or_incompatible_release_fails_without_a_build(tmp_path, release_server, case):
    assets = release_server["release"]["assets"]
    if case == "incompatible-wheel":
        assets[0]["name"] = assets[0]["name"].replace("x86_64", "aarch64")
    elif case == "duplicate-wheel":
        assets.append(dict(assets[0]))
    else:
        assets[:] = [
            a for a in assets if a["id"] != {"missing-wheel": 1, "missing-manifest": 2, "missing-checksums": 3}[case]
        ]
    with pytest.raises(ValueError):
        policy.prepare(tmp_path / "work")


@pytest.mark.parametrize(
    "case", ["wheel-checksum", "asset-digest", "source", "development", "version", "platform", "dosi"]
)
def test_inconsistent_release_identity_is_rejected(tmp_path, release_server, monkeypatch, case):
    if case == "wheel-checksum":
        release_server["wheel_data"] = b"corrupted download"
    elif case == "asset-digest":
        release_server["release"]["assets"][0]["digest"] = "sha256:" + "0" * 64
    elif case == "source":
        release_server["manifest"]["source_commit"] = "c" * 40
    elif case == "dosi":
        monkeypatch.setattr(policy.metadata, "version", lambda _: "0.1.0")
    else:
        release_server["manifest"][case] = {"development": True, "version": "0.5.4", "platform": "macos-arm64"}[case]
    work = tmp_path / "work"
    with pytest.raises(ValueError):
        policy.prepare(work)
    receipt = json.loads((work / "sql-policy-release.json").read_text())
    assert receipt["status"] == "failure"
    assert receipt["error"]
    assert list(work.glob("*.whl")) == []


def test_reused_work_directory_does_not_reuse_previous_receipts(tmp_path, release_server):
    work = tmp_path / "work"
    policy.prepare(work, agent=ROOT)
    with pytest.raises(ValueError, match="fresh"):
        policy.prepare(work)


def test_failed_github_requests_are_bounded_and_do_not_attempt_a_build(monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(policy.subprocess, "run", fail)
    monkeypatch.setattr(policy.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="GitHub Release request failed"):
        policy.request("repos/example/releases/latest")
    assert len(calls) == 3
    assert all(command[0] == "gh" for command in calls)
