from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_prepare_release_allows_same_final_prerelease_for_rc_inputs():
    workflow = (REPO_ROOT / ".github" / "workflows" / "prepare-release.yml").read_text(encoding="utf-8")

    assert 'if [[ "${VERSION}" =~ rc[0-9]+$ ]]; then' in workflow
    assert "args+=(--allow-same-final-prerelease)" in workflow


def test_publish_release_skips_finalize_steps_for_prereleases():
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-release.yml").read_text(encoding="utf-8")

    assert 'description: "datus-agent version to publish, for example 0.3.8 or 0.3.8rc1"' in workflow
    assert "id: release_version" in workflow
    assert 'Version(os.environ["VERSION"])' in workflow
    assert 'if [[ "${IS_PRERELEASE}" == "true" ]]; then' in workflow
    assert "Skipping stable tag validation for prerelease ${VERSION}" in workflow

    stable_only_guard = "${{ inputs.repository == 'pypi' && steps.release_version.outputs.is_prerelease != 'true' }}"
    assert f"Create or verify release tag\n        if: {stable_only_guard}" in workflow
    assert f"Dispatch docs deploy\n        if: {stable_only_guard}" in workflow
    assert (
        "Create or update main metadata sync PR\n"
        "        if: ${{ inputs.repository == 'pypi' && inputs.create_main_metadata_pr "
        "&& steps.release_version.outputs.is_prerelease != 'true' }}"
    ) in workflow
    assert "Tag/docs/main metadata sync: skipped for prerelease" in workflow
