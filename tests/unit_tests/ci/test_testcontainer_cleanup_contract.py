from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLEANUP_SCRIPT = REPO_ROOT / "ci" / "cleanup-run-testcontainers.sh"
NIGHTLY_SCRIPT = REPO_ROOT / "ci" / "run-nightly-tests.sh"
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-nightly.yml"


def test_cleanup_is_scoped_to_the_current_testcontainers_run():
    script = CLEANUP_SCRIPT.read_text(encoding="utf-8")

    assert "label=org.testcontainers=true" in script
    assert "label=com.datus.ci.run-id=${run_id}" in script
    assert 'docker rm -fv "$container_id"' in script


def test_nightly_exports_and_cleans_the_same_run_id():
    workflow = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
    nightly = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert "DATUS_TEST_RUN_ID: datus-agent-${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert '"$REPO_ROOT/ci/cleanup-run-testcontainers.sh"' in nightly
    assert "cleanup_status=$?" in nightly
    assert 'exit "$cleanup_status"' in nightly
