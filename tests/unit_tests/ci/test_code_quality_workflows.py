"""Required check identities must survive PR metadata events and queue entry."""

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


def test_metadata_mode_checks_evidence_without_installing_or_running_tests():
    config = workflow("code-quality.yml")
    assert "github.event.pull_request.head.sha" in config["run-name"]
    assert "github.event.pull_request.base.ref" in config["run-name"]
    for name in ("python-format-check.yml", "run-ut-and-coverage.yml"):
        called = workflow(name)
        assert called["on"]["workflow_call"]["inputs"]["reuse_validation"]["default"] == "false"
        for job in called["jobs"].values():
            assert "if" not in job
            steps = job["steps"]
            verifier = next(step for step in steps if step.get("name") == "Reuse verified validation")
            assert verifier["if"] == "inputs.reuse_validation"
            assert "reuseValidation" in verifier["with"]["script"]
            for step in steps:
                if step is verifier or step.get("name") == "Checkout code":
                    continue
                assert "!inputs.reuse_validation" in step["if"], step["name"]
