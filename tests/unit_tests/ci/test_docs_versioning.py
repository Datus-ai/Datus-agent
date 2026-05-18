from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "docs_versioning.py"
MODULE_SPEC = importlib.util.spec_from_file_location("docs_versioning", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load docs_versioning module from {MODULE_PATH}")
docs_versioning = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(docs_versioning)


def test_tag_patch_release_deploys_to_minor_docs_and_promotes_latest():
    metadata = docs_versioning.resolve_docs_metadata(
        event_name="push",
        github_ref_type="tag",
        github_ref_name="v0.3.7",
        published_versions_json=json.dumps([{"version": "0.2"}, {"version": "0.3.6"}]),
    )

    assert metadata.skip_deploy is False
    assert metadata.source_ref == "v0.3.7"
    assert metadata.docs_version == "0.3"
    assert metadata.aliases == ("latest",)
    assert metadata.set_default == "latest"
    assert metadata.edit_uri == "edit/v0.3.7/docs/"


def test_tag_patch_release_does_not_promote_latest_when_newer_minor_exists():
    metadata = docs_versioning.resolve_docs_metadata(
        event_name="push",
        github_ref_type="tag",
        github_ref_name="v0.3.8",
        published_versions_json=json.dumps([{"version": "0.4"}]),
    )

    assert metadata.docs_version == "0.3"
    assert metadata.aliases == ()
    assert metadata.set_default == ""


def test_published_versions_malformed_json_fails_closed():
    try:
        docs_versioning.resolve_docs_metadata(
            event_name="push",
            github_ref_type="tag",
            github_ref_name="v0.3.8",
            published_versions_json="{not-json",
        )
    except ValueError as exc:
        assert "Invalid published versions JSON" in str(exc)
    else:
        raise AssertionError("Expected malformed published versions JSON to fail")


def test_published_versions_non_list_json_fails_closed():
    try:
        docs_versioning.resolve_docs_metadata(
            event_name="push",
            github_ref_type="tag",
            github_ref_name="v0.3.8",
            published_versions_json=json.dumps({"version": "0.3"}),
        )
    except ValueError as exc:
        assert "expected a list" in str(exc)
    else:
        raise AssertionError("Expected non-list published versions JSON to fail")


def test_prerelease_tag_is_skipped():
    metadata = docs_versioning.resolve_docs_metadata(
        event_name="push",
        github_ref_type="tag",
        github_ref_name="v0.4.0rc1",
    )

    assert metadata.skip_deploy is True
    assert metadata.docs_version == ""
    assert metadata.edit_uri == "edit/v0.4.0rc1/docs/"


def test_main_push_deploys_dev_docs():
    metadata = docs_versioning.resolve_docs_metadata(
        event_name="push",
        github_ref_type="branch",
        github_ref_name="main",
    )

    assert metadata.skip_deploy is False
    assert metadata.source_ref == ""
    assert metadata.docs_version == "dev"
    assert metadata.edit_uri == "edit/main/docs/"


def test_manual_version_resolves_existing_tag():
    metadata = docs_versioning.resolve_docs_metadata(
        event_name="workflow_dispatch",
        github_ref_type="branch",
        github_ref_name="main",
        input_version="0.3.7",
        tag_exists=lambda tag: tag == "v0.3.7",
    )

    assert metadata.source_ref == "v0.3.7"
    assert metadata.docs_version == "0.3"


def test_manual_version_requires_stable_patch_release():
    try:
        docs_versioning.resolve_docs_metadata(
            event_name="workflow_dispatch",
            github_ref_type="branch",
            github_ref_name="main",
            input_version="v0.4.0rc1",
        )
    except ValueError as exc:
        assert "stable patch versions" in str(exc)
    else:
        raise AssertionError("Expected prerelease manual docs deploy to fail")


def test_github_env_output_contains_minor_version_and_aliases():
    metadata = docs_versioning.DocsVersionMetadata(
        skip_deploy=False,
        source_ref="v0.3.7",
        edit_uri="edit/v0.3.7/docs/",
        docs_version="0.3",
        aliases=("latest",),
        set_default="latest",
    )

    assert metadata.to_github_env().splitlines() == [
        "DOCS_SKIP_DEPLOY=false",
        "DOCS_BUILD_CONFIG=mkdocs.yml",
        "DOCS_SOURCE_REF=v0.3.7",
        "DOCS_EDIT_URI=edit/v0.3.7/docs/",
        "DOCS_VERSION=0.3",
        "DOCS_ALIASES=latest",
        "DOCS_SET_DEFAULT=latest",
    ]


def test_docs_version_visibility_hides_patch_and_old_minor_versions():
    min_visible_minor = docs_versioning.parse_minor("0.2")

    assert docs_versioning.should_hide_docs_version("dev", min_visible_minor) is False
    assert docs_versioning.should_hide_docs_version("0.1", min_visible_minor) is True
    assert docs_versioning.should_hide_docs_version("0.2", min_visible_minor) is False
    assert docs_versioning.should_hide_docs_version("0.3", min_visible_minor) is False
    assert docs_versioning.should_hide_docs_version("0.3.7", min_visible_minor) is True
    assert docs_versioning.should_hide_docs_version("0.4.0rc1", min_visible_minor) is True
    assert docs_versioning.should_hide_docs_version("latest", min_visible_minor) is False


def test_hide_legacy_cli_reports_invalid_minor(capsys):
    result = docs_versioning.main(["hide-legacy", "--min-visible-minor", "0.2.6"])

    captured = capsys.readouterr()
    assert result == 1
    assert "::error::Invalid --min-visible-minor" in captured.err
    assert "Invalid minor docs version: 0.2.6" in captured.err
