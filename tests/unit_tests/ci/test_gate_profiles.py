"""Documentation routing must preserve runtime and gate-configuration protection."""

import copy
import subprocess

import pytest
import yaml

from ci.harness.gate_profile import MAP_PATH, documentation_map_change, repository_profile, select_profile

pytestmark = pytest.mark.component


@pytest.mark.parametrize(
    "paths", [["docs/configuration/datasources/athena.md", "mkdocs.yml"], ["README.zh.md"], ["docs/assets/logo.svg"]]
)
def test_documentation_gets_the_docs_gate(paths):
    result = select_profile(paths)
    assert result["profile"] == "docs"
    assert result["ci_tests"] is False


@pytest.mark.parametrize(
    "path",
    [
        "datus/models/base.py",
        "datus/prompts/chat.md",
        "datus/resources/skills/init/SKILL.md",
        "docs/generate.py",
        ".github/workflows/merge-queue.yml",
        "uv.lock",
        "tests/conftest.py",
        "unknown.file",
    ],
)
def test_mixed_changes_keep_the_runtime_gate(path):
    assert select_profile(["docs/index.md", path])["profile"] == "code"


def _map():
    return {
        "execution": {"core_nodeids": ["test_core.py::test_query"]},
        "flow_groups": {
            "chat": {
                "flows": [{"id": "chat", "docs": ["docs/old.md"], "coverage": {"merge_queue": {"status": "covered"}}}]
            }
        },
    }


def test_documentation_registration_retains_ci_tests():
    before = _map()
    after = copy.deepcopy(before)
    after["flow_groups"]["chat"]["flows"][0]["docs"].append("docs/new.md")
    result = select_profile(
        [MAP_PATH, "docs/new.md"], map_before=yaml.safe_dump(before), map_after=yaml.safe_dump(after)
    )
    assert result["profile"] == "docs"
    assert result["ci_tests"] is True


@pytest.mark.parametrize("field", ["execution", "coverage"])
def test_gate_configuration_changes_keep_the_runtime_gate(field):
    before = _map()
    after = copy.deepcopy(before)
    if field == "execution":
        after["execution"]["core_nodeids"] = ["test_core.py::test_other_query"]
    else:
        after["flow_groups"]["chat"]["flows"][0]["coverage"]["merge_queue"]["status"] = "partial"
    assert not documentation_map_change(yaml.safe_dump(before), yaml.safe_dump(after))
    assert (
        select_profile([MAP_PATH], map_before=yaml.safe_dump(before), map_after=yaml.safe_dump(after))["profile"]
        == "code"
    )


def test_map_comparison_uses_the_real_git_base(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()

    git("init")
    git("config", "user.name", "Gate Test")
    git("config", "user.email", "test@example.invalid")
    path = tmp_path / MAP_PATH
    path.parent.mkdir(parents=True)
    before = _map()
    path.write_text(yaml.safe_dump(before))
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    before["flow_groups"]["chat"]["flows"][0]["docs"].append("docs/new.md")
    path.write_text(yaml.safe_dump(before))
    git("add", ".")
    git("commit", "-m", "register documentation")
    assert repository_profile([MAP_PATH], base, root=tmp_path)["profile"] == "docs"


def test_unknown_or_empty_change_set_keeps_the_runtime_gate():
    assert select_profile([])["profile"] == "code"
    assert select_profile([MAP_PATH])["profile"] == "code"
