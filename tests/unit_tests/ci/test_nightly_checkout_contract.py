from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-nightly.yml"


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
    ):
        assert repo in workflow

    assert 'git -C "$repo" rev-parse HEAD' in workflow
    assert "kb-v5-datus_agent_nightly" in workflow
