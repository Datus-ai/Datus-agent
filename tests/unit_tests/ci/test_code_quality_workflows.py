"""Required check identities must survive PR metadata events and queue entry."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = {"format-check / format-check", "run-coverage / coverage"}


def workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


@pytest.mark.parametrize("event", ["opened", "synchronize", "reopened", "edited"])
def test_each_pr_event_publishes_real_required_checks(event):
    config = workflow("code-quality.yml")
    assert set(config["on"]) == {"pull_request"}
    assert event in config["on"]["pull_request"]["types"]
    contexts = set()
    for job_id, job in config["jobs"].items():
        # A skipped reusable call publishes no child context; a skipped job
        # must not satisfy a failed validation from a previous run either.
        assert "if" not in job
        name = job.get("name", job_id)
        assert "${{" not in name
        assert job["with"]["reuse_validation"] == "${{ github.event.action == 'edited' && !github.event.changes.base }}"
        called = workflow(Path(job["uses"]).name)
        for child_id, child in called["jobs"].items():
            child_name = child.get("name", child_id)
            contexts.add(f"{name} / {child_name}")
    assert contexts == REQUIRED
    assert config["jobs"]["run-coverage"]["needs"] == "format-check"


def test_merge_group_contexts_are_isolated_from_pr_events():
    config = workflow("merge-queue-code-quality.yml")
    assert set(config["on"]) == {"merge_group"}
    assert {job["name"] for job in config["jobs"].values()} == REQUIRED
    assert all("if" not in job for job in config["jobs"].values())


def test_metadata_mode_runs_tests_only_when_successful_evidence_is_not_reused():
    config = workflow("code-quality.yml")
    assert "github.event.pull_request.head.sha" in config["run-name"]
    assert "github.event.pull_request.base.ref" in config["run-name"]
    assert "github.event.pull_request.base.sha" in config["run-name"]
    for name in ("python-format-check.yml", "run-ut-and-coverage.yml"):
        called = workflow(name)
        assert called["on"]["workflow_call"]["inputs"]["reuse_validation"]["default"] == "false"
        for job in called["jobs"].values():
            assert "if" not in job
            steps = job["steps"]
            verifier = next(step for step in steps if step.get("name") == "Reuse verified validation")
            assert verifier["id"] == "reuse"
            assert verifier["if"] == "inputs.reuse_validation"
            assert "reuseValidation" in verifier["with"]["script"]
            assert "core.setOutput('reused', result.state === 'success')" in verifier["with"]["script"]
            for step in steps:
                if step is verifier or step.get("name") == "Checkout code":
                    continue
                assert "steps.reuse.outputs.reused != 'true'" in step["if"], step["name"]


def coverage_steps():
    return workflow("run-ut-and-coverage.yml")["jobs"]["coverage"]["steps"]


def test_all_comparisons_use_the_pinned_base():
    steps = coverage_steps()
    pin = next(step for step in steps if step.get("id") == "comparison")
    assert "github.event.pull_request.base.sha" in pin["env"]["BASE_SHA"]
    for name in ("Select PR profile", "Run tests and extract coverage", "Plan affected real-model product validation"):
        step = next(step for step in steps if step.get("name") == name)
        assert step["env"]["COMPARE_REF"] == "${{ steps.comparison.outputs.ref }}"
        assert '"$COMPARE_REF"' in step["run"]
        assert "BASE_REF" not in step["run"]


@pytest.fixture
def advancing_base(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()

    def git(*args, cwd=remote):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--initial-branch=main")
    git("config", "user.name", "CI Test")
    git("config", "user.email", "ci@example.invalid")
    (remote / "file").write_text("event base\n")
    git("add", ".")
    git("commit", "-m", "event base")
    base = git("rev-parse", "HEAD")
    checkout = tmp_path / "checkout"
    git("clone", str(remote), str(checkout))
    (remote / "file").write_text("branch advanced after event\n")
    git("commit", "-am", "advance base")
    assert git("rev-parse", "HEAD") != base
    return checkout, base


def pin_base(checkout, base_sha, *, pr_number="1619"):
    step = next(step for step in coverage_steps() if step.get("id") == "comparison")
    output = checkout / "step-output"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=checkout,
        env=os.environ
        | {"BASE_REF": "main", "BASE_SHA": base_sha, "PR_NUMBER": pr_number, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
    )
    return result, output.read_text() if output.exists() else ""


def test_base_fetch_uses_event_commit_when_branch_advances(advancing_base):
    checkout, base = advancing_base
    result, output = pin_base(checkout, base)
    assert result.returncode == 0, result.stderr
    assert output == f"ref={base}\n"


@pytest.mark.parametrize("base", ["", "main", "f" * 40])
def test_base_fetch_rejects_missing_invalid_or_unfetchable_pr_commit(advancing_base, base):
    checkout, _ = advancing_base
    result, output = pin_base(checkout, base)
    assert result.returncode != 0
    assert output == ""


def test_manual_validation_can_still_compare_against_a_branch(advancing_base):
    checkout, _ = advancing_base
    result, output = pin_base(checkout, "", pr_number="")
    assert result.returncode == 0, result.stderr
    assert output == "ref=origin/main\n"
