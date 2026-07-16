#!/usr/bin/env python3
"""Prepare a locale-specific MkDocs build from a source ref with the current config."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError

CONTENT_SENSITIVE_KEYS = (
    "nav",
    "docs_dir",
    "markdown_extensions",
    "extra_css",
    "extra_javascript",
    "hooks",
    "watch",
    "exclude_docs",
    "not_in_nav",
    "draft_docs",
    "validation",
)
LEGACY_CHINESE_DOCS_URL_RE = re.compile(r"https://docs\.datus\.ai/(?P<version>\d+(?:\.\d+)+)/zh/")


class MkDocsLoader(yaml.SafeLoader):
    """SafeLoader variant that resolves MkDocs-specific YAML tags."""


def _construct_env(loader: MkDocsLoader, node: yaml.Node) -> str | None:
    if isinstance(node, yaml.ScalarNode):
        env_names = [loader.construct_scalar(node)]
        default = None
    elif isinstance(node, yaml.SequenceNode):
        values = loader.construct_sequence(node)
        if not values:
            return None
        if len(values) == 1:
            # `!ENV [VAR]` means "look up VAR" without an explicit default.
            env_names = [values[0]]
            default = None
        else:
            # `!ENV [VAR1, VAR2, ..., default]` uses the last entry as fallback.
            *env_names, default = values
    else:
        raise TypeError(f"Unsupported !ENV node type: {type(node).__name__}")

    for env_name in env_names:
        value = os.environ.get(str(env_name))
        if value is not None:
            return value
    return default


def _construct_python_name(loader: MkDocsLoader, suffix: str, node: yaml.Node) -> object:
    if not suffix:
        raise ConstructorError(None, None, "Missing python/name target", node.start_mark)

    module_path, _, attr_name = suffix.rpartition(".")
    if not module_path or not attr_name:
        raise ConstructorError(None, None, f"Invalid python/name target: {suffix}", node.start_mark)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConstructorError(None, None, f"Could not import module for {suffix}: {exc}", node.start_mark) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ConstructorError(None, None, f"Could not resolve python/name target: {suffix}", node.start_mark) from exc


MkDocsLoader.add_constructor("!ENV", _construct_env)
MkDocsLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_python_name)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=MkDocsLoader) or {}


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle, sort_keys=False, allow_unicode=True)


def deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def default_locale(config: dict) -> str | None:
    for plugin in config.get("plugins", []):
        if not isinstance(plugin, dict) or "i18n" not in plugin:
            continue
        for language in plugin["i18n"].get("languages", []):
            if language.get("default"):
                return str(language["locale"])
    return None


def untranslated_default_pages(docs_dir: Path, locale: str, locales: set[str]) -> list[str]:
    localized_suffixes = tuple(f".{candidate}.md" for candidate in locales)
    untranslated = []
    for page in docs_dir.rglob("*.md"):
        if page.name.endswith(localized_suffixes):
            continue
        localized_page = page.with_name(f"{page.stem}.{locale}{page.suffix}")
        if not localized_page.exists():
            untranslated.append(page.relative_to(docs_dir).as_posix())
    return sorted(untranslated)


def add_excluded_docs(config: dict, paths: list[str]) -> None:
    if not paths:
        return
    existing = config.get("exclude_docs") or ""
    if isinstance(existing, str):
        patterns = existing.splitlines()
    else:
        patterns = [str(pattern) for pattern in existing]
    config["exclude_docs"] = "\n".join(dict.fromkeys([*patterns, *paths]))


def rewrite_legacy_chinese_docs_urls(docs_dir: Path) -> None:
    for page in docs_dir.rglob("*.md"):
        content = page.read_text(encoding="utf-8")
        updated = LEGACY_CHINESE_DOCS_URL_RE.sub(
            r"https://docs.datus.ai/zh/\g<version>/",
            content,
        )
        if updated != content:
            page.write_text(updated, encoding="utf-8")


def merge_mkdocs_config(
    base_config: dict,
    source_config: dict,
    source_root: Path,
    locale: str | None = None,
) -> dict:
    merged = copy.deepcopy(base_config)

    for key in CONTENT_SENSITIVE_KEYS:
        if key in source_config:
            merged[key] = copy.deepcopy(source_config[key])

    source_docs_dir = source_config.get("docs_dir", merged.get("docs_dir", "docs"))
    merged["docs_dir"] = str((source_root / source_docs_dir).resolve())

    source_extra = source_config.get("extra", {})
    base_extra = base_config.get("extra", {})
    if source_extra or base_extra:
        # Let main-config extra override tag/source extra on conflicts so mike-
        # related UI settings stay current; list values are replaced, not merged.
        merged["extra"] = deep_merge(source_extra, base_extra)

    if "plugins" in base_config:
        merged["plugins"] = copy.deepcopy(base_config["plugins"])

    if "edit_uri" in base_config:
        merged["edit_uri"] = copy.deepcopy(base_config["edit_uri"])

    primary_locale = default_locale(base_config)
    if locale and primary_locale and locale != primary_locale:
        languages = {
            str(language["locale"])
            for plugin in base_config.get("plugins", [])
            if isinstance(plugin, dict) and "i18n" in plugin
            for language in plugin["i18n"].get("languages", [])
        }
        docs_dir = Path(merged["docs_dir"])
        add_excluded_docs(
            merged,
            untranslated_default_pages(docs_dir, locale, languages),
        )

    return merged


def export_ref(source_ref: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", source_ref],
        stdout=subprocess.PIPE,
    )
    tar_failed = False
    try:
        subprocess.run(
            ["tar", "-xf", "-", "-C", str(destination)],
            stdin=archive.stdout,
            check=True,
        )
    except subprocess.CalledProcessError:
        tar_failed = True
        raise
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
        if tar_failed and archive.poll() is None:
            archive.kill()
        return_code = archive.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, archive.args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--force", action="store_true", help="Allow deleting a non-empty output root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    source_root = output_root / "source"

    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"Output root must be a directory: {output_root}")
        if any(output_root.iterdir()):
            if not args.force:
                raise RuntimeError(f"Refusing to delete non-empty output root without --force: {output_root}")
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    export_ref(args.source_ref, source_root)

    base_config = load_yaml(args.base_config.resolve())
    source_config = load_yaml(source_root / "mkdocs.yml")
    source_docs_dir = source_root / source_config.get("docs_dir", base_config.get("docs_dir", "docs"))
    rewrite_legacy_chinese_docs_urls(source_docs_dir)
    merged = merge_mkdocs_config(
        base_config,
        source_config,
        source_root,
        locale=os.environ.get("DOCS_LOCALE"),
    )
    dump_yaml(output_root / "mkdocs.yml", merged)

    print(output_root / "mkdocs.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
