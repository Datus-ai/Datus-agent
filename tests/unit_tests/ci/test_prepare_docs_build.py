from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "prepare_docs_build.py"
MODULE_SPEC = importlib.util.spec_from_file_location("prepare_docs_build", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
prepare_docs_build = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(prepare_docs_build)


def test_load_yaml_resolves_env_tag(tmp_path, monkeypatch):
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text('edit_uri: !ENV [DOCS_EDIT_URI, "edit/main/docs/"]\n', encoding="utf-8")

    monkeypatch.setenv("DOCS_EDIT_URI", "edit/v0.2.6/docs/")

    loaded = prepare_docs_build.load_yaml(config_path)

    assert loaded["edit_uri"] == "edit/v0.2.6/docs/"


def test_merge_mkdocs_config_uses_source_nav_and_main_version_provider(tmp_path):
    base_config = {
        "nav": [{"Home": "index.md"}, {"Release Notes": "release_notes.md"}],
        "plugins": ["search", {"mike": {"alias_type": "redirect"}}],
        "extra": {
            "version": {"provider": "mike"},
            "social": [{"icon": "fontawesome/brands/github"}],
        },
        "edit_uri": "edit/main/docs/",
    }
    source_config = {
        "nav": [{"Home": "index.md"}, {"Subagent": "subagent/introduction.md"}],
        "docs_dir": "docs",
        "markdown_extensions": ["toc"],
        "extra": {"analytics": {"provider": "google"}},
        "plugins": ["search"],
    }

    merged = prepare_docs_build.merge_mkdocs_config(
        base_config,
        source_config,
        tmp_path / "tag-source",
    )

    assert merged["nav"] == source_config["nav"]
    assert merged["docs_dir"] == str((tmp_path / "tag-source" / "docs").resolve())
    assert merged["markdown_extensions"] == ["toc"]
    assert merged["plugins"] == base_config["plugins"]
    assert merged["edit_uri"] == "edit/main/docs/"
    assert merged["extra"]["analytics"]["provider"] == "google"
    assert merged["extra"]["version"]["provider"] == "mike"
