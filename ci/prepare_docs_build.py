#!/usr/bin/env python3
"""Prepare a merged MkDocs config for building tagged docs with main config."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
from pathlib import Path

import yaml

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


class MkDocsLoader(yaml.UnsafeLoader):
    """UnsafeLoader variant that resolves MkDocs' !ENV tag."""


def _construct_env(loader: MkDocsLoader, node: yaml.Node) -> str | None:
    if isinstance(node, yaml.ScalarNode):
        env_names = [loader.construct_scalar(node)]
        default = None
    elif isinstance(node, yaml.SequenceNode):
        values = loader.construct_sequence(node)
        if not values:
            return None
        *env_names, default = values
        if len(values) == 1:
            env_names = [values[0]]
            default = None
    else:
        raise TypeError(f"Unsupported !ENV node type: {type(node).__name__}")

    for env_name in env_names:
        value = os.environ.get(str(env_name))
        if value is not None:
            return value
    return default


MkDocsLoader.add_constructor("!ENV", _construct_env)


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


def merge_mkdocs_config(base_config: dict, source_config: dict, source_root: Path) -> dict:
    merged = copy.deepcopy(base_config)

    for key in CONTENT_SENSITIVE_KEYS:
        if key in source_config:
            merged[key] = copy.deepcopy(source_config[key])

    source_docs_dir = source_config.get("docs_dir", merged.get("docs_dir", "docs"))
    merged["docs_dir"] = str((source_root / source_docs_dir).resolve())

    source_extra = source_config.get("extra", {})
    base_extra = base_config.get("extra", {})
    if source_extra or base_extra:
        merged["extra"] = deep_merge(source_extra, base_extra)

    if "plugins" in base_config:
        merged["plugins"] = copy.deepcopy(base_config["plugins"])

    if "edit_uri" in base_config:
        merged["edit_uri"] = copy.deepcopy(base_config["edit_uri"])

    return merged


def export_ref(source_ref: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", source_ref],
        stdout=subprocess.PIPE,
    )
    try:
        subprocess.run(
            ["tar", "-xf", "-", "-C", str(destination)],
            stdin=archive.stdout,
            check=True,
        )
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
    return_code = archive.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, archive.args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a merged MkDocs config for tagged docs builds.")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    source_root = output_root / "source"

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    export_ref(args.source_ref, source_root)

    base_config = load_yaml(args.base_config.resolve())
    source_config = load_yaml(source_root / "mkdocs.yml")
    merged = merge_mkdocs_config(base_config, source_config, source_root)
    dump_yaml(output_root / "mkdocs.yml", merged)

    print(output_root / "mkdocs.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
