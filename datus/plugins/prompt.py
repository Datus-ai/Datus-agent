# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Render a plugin's system-prompt Jinja2 template.

The template named by ``manifest.system_prompt`` is rendered with a context of
``plugin_name``, ``profiles`` and ``config_path``. Secret handling is
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


def strip_secret_fields(profiles: Any, config_schema: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Whitelist-filter profile fields for template rendering.

    Only properties declared in ``config_schema`` and NOT marked
    ``x-secret: true`` survive; undeclared fields are dropped. Without a
    schema nothing is whitelisted, so profile names map to empty dicts — the
    template still sees which profiles exist, but no values.
    """
    if not isinstance(profiles, dict):
        return {}
    allowed: set = set()
    if isinstance(config_schema, dict):
        properties = config_schema.get("properties")
        if isinstance(properties, dict):
            for prop_name, spec in properties.items():
                if not isinstance(prop_name, str):
                    continue
                if isinstance(spec, dict) and spec.get("x-secret") is True:
                    continue
                allowed.add(prop_name)
    stripped: Dict[str, Dict[str, Any]] = {}
    for profile_name, config in profiles.items():
        if not isinstance(profile_name, str):
            continue
        cfg = config if isinstance(config, dict) else {}
        stripped[profile_name] = {k: v for k, v in cfg.items() if k in allowed}
    return stripped


def render_plugin_prompt(
    manifest: PluginManifest,
    profiles: Any,
    config_path: Optional[str] = None,
) -> Optional[str]:
    """Render ``manifest.system_prompt`` into a system-prompt section.

    ``profiles`` is the plugin's (already project-narrowed) profile mapping;
    it is secret-stripped here before rendering. Returns the stripped rendered
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
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        # autoescape stays off: templates emit markdown for the LLM context,
        # not HTML. StrictUndefined turns template typos into a logged skip
        # instead of silently corrupted prompt text.
        env = Environment(
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
        )
    except Exception as exc:  # noqa: BLE001 - one bad template must not break prompt build
        logger.warning("Plugin %r system_prompt template failed to render: %s; skipping.", manifest.name, exc)
        return None
    rendered = rendered.strip()
    return rendered or None
