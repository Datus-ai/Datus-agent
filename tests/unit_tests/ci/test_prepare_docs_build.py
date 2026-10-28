from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "prepare_docs_build.py"
MODULE_SPEC = importlib.util.spec_from_file_location("prepare_docs_build", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise AssertionError(f"Unable to load prepare_docs_build from {MODULE_PATH}")
prepare_docs_build = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(prepare_docs_build)


def test_load_yaml_resolves_env_tag(tmp_path, monkeypatch):
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text('edit_uri: !ENV [DOCS_EDIT_URI, "edit/main/docs/"]\n', encoding="utf-8")

    monkeypatch.setenv("DOCS_EDIT_URI", "edit/v0.2.6/docs/")

    loaded = prepare_docs_build.load_yaml(config_path)

    assert loaded["edit_uri"] == "edit/v0.2.6/docs/"


def test_load_yaml_resolves_env_scalar_to_none_when_unset(tmp_path, monkeypatch):
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text("edit_uri: !ENV DOCS_EDIT_URI\n", encoding="utf-8")

    monkeypatch.delenv("DOCS_EDIT_URI", raising=False)

    loaded = prepare_docs_build.load_yaml(config_path)

    assert loaded["edit_uri"] is None


def test_load_yaml_resolves_env_default_when_unset(tmp_path, monkeypatch):
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text('edit_uri: !ENV [DOCS_EDIT_URI, "edit/main/docs/"]\n', encoding="utf-8")

    monkeypatch.delenv("DOCS_EDIT_URI", raising=False)

    loaded = prepare_docs_build.load_yaml(config_path)

    assert loaded["edit_uri"] == "edit/main/docs/"


def test_load_yaml_resolves_python_name_tag(tmp_path):
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text("path_type: !!python/name:pathlib.Path\n", encoding="utf-8")

    loaded = prepare_docs_build.load_yaml(config_path)

    assert loaded["path_type"] is Path


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


def test_merge_mkdocs_config_excludes_untranslated_default_pages_for_non_default_locale(tmp_path):
    source_root = tmp_path / "tag-source"
    docs_dir = source_root / "docs"
    (docs_dir / "API").mkdir(parents=True)
    (docs_dir / "index.md").write_text("English home", encoding="utf-8")
    (docs_dir / "index.zh.md").write_text("Chinese home", encoding="utf-8")
    (docs_dir / "API" / "models.md").write_text("English only", encoding="utf-8")
    (docs_dir / "chinese-only.zh.md").write_text("Chinese only", encoding="utf-8")
    base_config = {
        "plugins": [
            "search",
            {
                "i18n": {
                    "languages": [
                        {"locale": "en", "default": True},
                        {"locale": "zh"},
                    ]
                }
            },
        ],
        "exclude_docs": "drafts/**",
    }
    source_config = {"docs_dir": "docs"}

    merged = prepare_docs_build.merge_mkdocs_config(
        base_config,
        source_config,
        source_root,
        locale="zh",
    )

    assert merged["exclude_docs"].splitlines() == ["drafts/**", "API/models.md"]


def test_rewrite_legacy_chinese_docs_urls_updates_language_first_path(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    page = docs_dir / "release_notes.zh.md"
    page.write_text(
        "[旧链接](https://docs.datus.ai/0.3/zh/cli/reference/)\n"
        "[新链接](https://docs.datus.ai/zh/0.3/cli/reference/)\n",
        encoding="utf-8",
    )

    prepare_docs_build.rewrite_legacy_chinese_docs_urls(docs_dir)

    assert page.read_text(encoding="utf-8") == (
        "[旧链接](https://docs.datus.ai/zh/0.3/cli/reference/)\n[新链接](https://docs.datus.ai/zh/0.3/cli/reference/)\n"
    )


def test_main_refuses_to_delete_non_empty_output_root_without_force(tmp_path, monkeypatch):
    output_root = tmp_path / "build"
    output_root.mkdir()
    (output_root / "stale.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        prepare_docs_build,
        "parse_args",
        lambda: SimpleNamespace(
            base_config=tmp_path / "mkdocs.yml",
            output_root=output_root,
            source_ref="v0.2.6",
            force=False,
        ),
    )

    with pytest.raises(RuntimeError, match="Refusing to delete non-empty output root without --force"):
        prepare_docs_build.main()
