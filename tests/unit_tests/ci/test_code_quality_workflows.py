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
