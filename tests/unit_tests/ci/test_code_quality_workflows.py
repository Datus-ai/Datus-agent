"""PR triggers and required check identities must remain consistent."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = {"format-check / format-check", "run-coverage / coverage"}


def workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


@pytest.mark.parametrize(
    "name, event, expected_types",
    [
        ("code-quality.yml", "pull_request", {"opened", "synchronize", "reopened"}),
        ("title-check.yml", "pull_request_target", {"opened"}),
    ],
)
def test_quality_and_title_checks_ignore_pr_metadata_edits(name, event, expected_types):
    config = workflow(name)
    assert set(config["on"]) == {event}
    assert set(config["on"][event]["types"]) == expected_types


def test_test_audit_only_handles_pr_events_and_keeps_trusted_validation():
    config = workflow("test-audit.yml")
    assert set(config["on"]) == {"pull_request_target"}
    assert set(config["on"]["pull_request_target"]["types"]) == {"opened", "synchronize", "reopened", "edited"}
    job = config["jobs"]["test-audit"]
    assert job["if"] == (
        "${{ !startsWith(github.event.pull_request.title, '[Doc]') && "
        "!startsWith(github.event.pull_request.title, '[doc]') }}"
    )
    steps = job["steps"]
    audit = next(step for step in steps if step.get("id") == "audit")
    assert "python3 ../base/ci/audit_tests.py" in audit["run"]
    assert "--repo-root ." in audit["run"]
    assert audit["working-directory"] == "pr"
    failure = steps[-1]
    assert failure["if"] == (
        "${{ always() && (steps.audit.outcome != 'success' || steps.audit.outputs.audit_outcome == 'failure') }}"
    )
    assert "exit 1" in failure["run"]


@pytest.mark.parametrize("event", ["opened", "synchronize", "reopened"])
def test_each_pr_event_publishes_real_required_checks(event):
    config = workflow("code-quality.yml")
    assert set(config["on"]) == {"pull_request"}
    assert event in config["on"]["pull_request"]["types"]
    assert set(config["jobs"]) == {"format-check", "run-coverage", "merge-queue-context"}
    contexts = set()
    for job_id in ("format-check", "run-coverage"):
        job = config["jobs"][job_id]
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
    assert config["jobs"]["merge-queue-context"]["name"] == "Merge Queue Gate"
    assert config["jobs"]["run-coverage"]["needs"] == "format-check"


@pytest.mark.parametrize(
    "job_id,required_name,manual_name",
    [
        ("format-check-context", "format-check / format-check", "Manual queue format context"),
        ("coverage-context", "run-coverage / coverage", "Manual queue coverage context"),
    ],
)
def test_merge_group_contexts_are_isolated_from_pr_and_manual_checks(job_id, required_name, manual_name):
    config = workflow("merge-queue.yml")
    assert set(config["on"]) == {"merge_group", "workflow_dispatch"}
    job = config["jobs"][job_id]
    assert job["if"] == "${{ github.event_name == 'merge_group' }}"
    # Skipped jobs still publish checks, so dispatches must use distinct names.
    assert job["name"] == (
        "${{ github.event_name == 'merge_group' && '" + required_name + "' || '" + manual_name + "' }}"
    )
    assert manual_name not in REQUIRED | {"Merge Queue Gate"}


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


@pytest.mark.parametrize(
    "applicable,diff,outcome,expected",
    [
        ("true", "80.00", "success", 0),
        ("true", "79.99", "success", 1),
        ("false", "N/A", "success", 0),
        ("false", "N/A", "failure", 1),
        ("", "", "success", 1),
        ("true", "nan", "success", 1),
    ],
)
def test_threshold_gate_checks_tests_independently_of_application_coverage(applicable, diff, outcome, expected):
    step = next(step for step in coverage_steps() if step.get("name") == "Check coverage thresholds")
    script = step["run"]
    for key, value in {
        "steps.metrics.outputs.coverage_applicable": applicable,
        "steps.profile.outputs.coverage_applicable": applicable,
        "steps.metrics.outputs.diff": diff,
        "steps.metrics.outputs.test_outcome": outcome,
    }.items():
        script = script.replace("${{ " + key + " }}", value)
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False)
    assert result.returncode == expected, result.stdout + result.stderr


def test_version_only_validation_keeps_release_checks_and_runtime_setup_is_conditional():
    steps = coverage_steps()
    metadata = next(step for step in steps if step.get("name") == "Check release metadata")
    assert "run_tests" not in metadata["if"]
    assert "ci/check_release_readiness.py" in metadata["run"]
    assert "uv lock --check" in metadata["run"]
    for name in [
        "Setup config file",
        "Restore tracked test config",
        "Isolate test data directory",
        "Prepare small BIRD test fixtures",
    ]:
        step = next(step for step in steps if step.get("name") == name)
        assert "steps.profile.outputs.needs_runtime_fixtures == 'true'" in step["if"]


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
