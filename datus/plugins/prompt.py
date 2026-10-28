# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Render a plugin's system-prompt Jinja2 template.

The template named by ``manifest.system_prompt`` is rendered with a context of
``plugin_name``, ``profiles``, ``config_path`` and ``config_mutable``
(``config_path`` is ``None`` and ``config_mutable`` is ``False`` when the
runtime config is read-only, e.g. the multi-tenant chat API). Secret handling is
structural: :func:`strip_secret_fields` whitelists profile fields against the
manifest's ``config_schema`` BEFORE the template sees them — profile values
are env-expanded (real secrets) by the time prompts are built, so undeclared
or ``x-secret`` fields must never reach the template engine at all.

Every failure (missing template, path escape, syntax error, undefined
variable) is logged and resolves to ``None`` — one bad plugin must never break
prompt construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from datus.plugins.base import PluginManifest
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Bounds recursion into declared nested objects so a pathological schema can
# never hang secret stripping; anything deeper is dropped (fail closed).
_STRIP_MAX_DEPTH = 8


def _whitelist_fields(config: Dict[str, Any], properties: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """Keep only schema-declared, non-``x-secret`` fields, recursing into
    declared nested objects so a nested secret leaf is stripped too."""
    if depth >= _STRIP_MAX_DEPTH:
        return {}
    kept: Dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str) or key not in properties:
            continue
        spec = properties[key]
        if isinstance(spec, dict) and spec.get("x-secret") is True:
            continue
        nested_properties = spec.get("properties") if isinstance(spec, dict) else None
        if isinstance(spec, dict) and spec.get("type") == "object" and isinstance(nested_properties, dict):
            kept[key] = _whitelist_fields(value if isinstance(value, dict) else {}, nested_properties, depth + 1)
        else:
            kept[key] = value
    return kept


def strip_secret_fields(profiles: Any, config_schema: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Whitelist-filter profile fields for template rendering.

    Only properties declared in ``config_schema`` and NOT marked
    ``x-secret: true`` survive; undeclared fields are dropped. A declared
    ``type: object`` property with its own ``properties`` is filtered
    recursively under the same rules, so nested ``x-secret`` leaves and
    undeclared nested keys never reach the template either. Without a schema
    nothing is whitelisted, so profile names map to empty dicts — the template
    still sees which profiles exist, but no values.
    """
    if not isinstance(profiles, dict):
        return {}
    properties: Dict[str, Any] = {}
    if isinstance(config_schema, dict) and isinstance(config_schema.get("properties"), dict):
        properties = config_schema["properties"]
    stripped: Dict[str, Dict[str, Any]] = {}
    for profile_name, config in profiles.items():
        if not isinstance(profile_name, str):
            continue
        cfg = config if isinstance(config, dict) else {}
        stripped[profile_name] = _whitelist_fields(cfg, properties)
    return stripped


def render_plugin_prompt(
    manifest: PluginManifest,
    profiles: Any,
    config_path: Optional[str] = None,
    config_mutable: bool = True,
) -> Optional[str]:
    """Render ``manifest.system_prompt`` into a system-prompt section.

    ``profiles`` is the plugin's (already project-narrowed) profile mapping;
    it is secret-stripped here before rendering. ``config_mutable`` tells the
    template whether the agent may guide config edits — unconfigured plugins
    should point at their setup skill only when it is ``True`` and defer to
    the administrator otherwise. Returns the stripped rendered
    text, or ``None`` when the manifest declares no template, the template
    escapes the package dir, or rendering fails for any reason. Never raises.
    """
    if not manifest.system_prompt:
        return None
    package_dir = Path(manifest.package_dir).resolve()
    template_path = (package_dir / manifest.system_prompt).resolve()
    if not template_path.is_relative_to(package_dir):
        logger.warning(
            "Plugin %r system_prompt %r escapes the package directory; skipping.",
            manifest.name,
            manifest.system_prompt,
        )
        return None
    if not template_path.is_file():
        logger.warning("Plugin %r system_prompt template %s does not exist; skipping.", manifest.name, template_path)
        return None
    try:
        from jinja2 import FileSystemLoader, StrictUndefined
        from jinja2.sandbox import SandboxedEnvironment

        # A SandboxedEnvironment blocks unsafe attribute/object traversal, so a
        # plugin-controlled template can never execute arbitrary code during a
        # prompt build. autoescape stays off: templates emit markdown for the
        # LLM context, not HTML. StrictUndefined turns template typos into a
        # logged skip instead of silently corrupted prompt text.
        env = SandboxedEnvironment(
            loader=FileSystemLoader(str(package_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template(template_path.relative_to(package_dir).as_posix())
        rendered = template.render(
            plugin_name=manifest.name,
            profiles=strip_secret_fields(profiles, manifest.config_schema),
            config_path=config_path,
            config_mutable=config_mutable,
        )
    except Exception as exc:  # noqa: BLE001 - one bad template must not break prompt build
        logger.warning("Plugin %r system_prompt template failed to render: %s; skipping.", manifest.name, exc)
        return None
    rendered = rendered.strip()
    return rendered or None
