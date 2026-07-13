from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-nightly.yml"
NIGHTLY_SCRIPT = REPO_ROOT / "ci" / "run-nightly-tests.sh"


def test_nightly_preserves_checkout_packages_after_locked_sync():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'UV_NO_SYNC: "1"' in workflow
    assert "uv sync --locked" in workflow
    assert "uv run --no-sync python ci/verify_nightly_adapter_sources.py" in workflow
    assert "uv run --no-sync playwright install --with-deps chromium" in workflow


def test_nightly_kb_cache_tracks_all_adapter_checkouts():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for repo in (
        "external/datus-db-adapters",
        "external/datus-bi-adapters",
        "external/datus-scheduler-adapters",
        "external/datus-semantic-adapter",
        "external/datus-storage-adapters",
    ):
        assert repo in workflow

    assert 'git -C "$repo" rev-parse HEAD' in workflow
    assert "kb-v6-datus_agent_nightly" in workflow


def test_nightly_installs_storage_packages_from_latest_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: Datus-ai/datus-storage-adapters" in workflow
    assert "ref: main" in workflow
    assert "path: external/datus-storage-adapters" in workflow
    for package_name, package_path in (
        ("datus-storage-base", "./external/datus-storage-adapters/datus-storage-base"),
        ("datus-storage-postgresql", '"./external/datus-storage-adapters/datus-storage-postgresql[dev]"'),
    ):
        assert f"--reinstall-package {package_name}" in workflow
        assert package_path in workflow


def test_nightly_runs_postgresql_storage_adapter_tests_from_checkout():
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'STORAGE_ADAPTERS_ROOT="$(default_repo_root' in script
    assert 'run_logged "PostgreSQL Storage Adapter Tests"' in script
    assert 'uv run --no-sync pytest "$STORAGE_ADAPTERS_ROOT/datus-storage-postgresql/tests"' in script
    assert '"PostgreSQL Storage Adapter Tests"' in script.split("DOCKER_GROUPS=(", maxsplit=1)[1]
