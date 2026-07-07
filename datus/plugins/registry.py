# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Discovery of ``datus.plugins`` entry points.

A plugin package declares one entry point under the ``datus.plugins`` group,
e.g. ``hello = datus_hello.plugin:HelloPlugin``. This module loads those
classes and exposes their bundled skill directories, system-prompt sections,
and CLI bash-permission declarations. See :mod:`datus.plugins.base` for the
plugin contract.

Every lookup is defensive: a broken or missing plugin must never crash the CLI
or block skill discovery.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.tools.permission.bash_rules import BashCommandRules

logger = get_logger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "datus.plugins"

# Entry-point names that may contribute CLI permission patterns. A name with
# spaces or glob metacharacters could shift what the literal ``datus <name>``
# anchor matches, breaking the namespace confinement guarantee.
_SAFE_PLUGIN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_CLI_PERMISSION_PROFILES = ("normal", "auto")
_CLI_PERMISSION_ACTIONS = ("allow", "ask", "deny")


def entry_points_for_group(group: str, name: Optional[str] = None) -> list:
    """Return entry points in ``group`` (optionally filtered by ``name``).

    Mirrors the py3.10 ``select`` / pre-3.10 dict-shaped fallback used across
    the codebase (``service_adapter_installer.hot_reload_adapter``). Any error
    resolves to an empty list — discovery must never raise. Shared by every
    entry-point consumer (plugin registry, ``datus.cli_commands`` dispatch,
    ``datus.skills`` discovery) so the compatibility shim lives in one place.
    """
    try:
        import importlib.metadata as importlib_metadata

        eps = importlib_metadata.entry_points()
        if hasattr(eps, "select"):
            if name is not None:
                return list(eps.select(group=group, name=name))
            return list(eps.select(group=group))
        # pragma: no cover - legacy Python
        group_eps = eps.get(group, [])
        if name is not None:
            return [ep for ep in group_eps if ep.name == name]
        return list(group_eps)
    except Exception as exc:  # noqa: BLE001 - defensive: never crash discovery
        logger.debug("%s entry-point lookup failed: %s", group, exc)
        return []


def iter_plugin_entry_points() -> list:
    """Return all entry points registered under ``datus.plugins``."""
    return entry_points_for_group(PLUGIN_ENTRY_POINT_GROUP)


def plugin_entry_point_exists(name: str) -> bool:
    """True when a ``datus.plugins`` entry point named ``name`` is installed.

    Metadata-only: never imports the plugin package, so callers can consult it
    BEFORE the ``plugins_enabled`` master switch has been checked without
    executing third-party module-level code.
    """
    return bool(entry_points_for_group(PLUGIN_ENTRY_POINT_GROUP, name=name))


def load_plugin_class(name: str) -> Optional[type]:
    """Load the plugin class registered as ``name``, or ``None``.

    Returns ``None`` when no plugin claims ``name`` so callers fall through to
    other dispatch paths. A load failure (broken adapter) is logged and also
    yields ``None`` rather than propagating.
    """
    candidates = entry_points_for_group(PLUGIN_ENTRY_POINT_GROUP, name=name)
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning("Multiple datus.plugins entry points named %r; using the first.", name)
    try:
        return candidates[0].load()
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not crash the CLI
        logger.error("Failed to load plugin '%s': %s", name, exc)
        return None


# Process-level cache of ``(entry_point_name, plugin_class_or_None)`` pairs.
# Installed plugins cannot change mid-process, and the four collection
# functions below all run on hot paths (skill discovery, prompt build,
# per-node transformer wrapping, permission collection) — without the cache
# each of them re-enumerates distribution metadata and re-imports every
# plugin package on every call. ``None`` marks a failed load so consumers
# skip it without retrying the import.
_PLUGIN_CACHE: Optional[List[Tuple[Optional[str], Optional[type]]]] = None


def _loaded_plugins() -> List[Tuple[Optional[str], Optional[type]]]:
    """Return cached ``(name, plugin_cls)`` pairs for all installed plugins."""
    global _PLUGIN_CACHE
    if _PLUGIN_CACHE is None:
        pairs: List[Tuple[Optional[str], Optional[type]]] = []
        for ep in iter_plugin_entry_points():
            name = getattr(ep, "name", None)
            try:
                pairs.append((name, ep.load()))
            except Exception as exc:  # noqa: BLE001 - broken plugin skipped
                logger.debug("datus.plugins entry point %r failed to load: %s", name, exc)
                pairs.append((name, None))
        _PLUGIN_CACHE = pairs
    return _PLUGIN_CACHE


def invalidate_plugin_cache() -> None:
    """Drop the cached plugin classes (tests / in-process plugin installs)."""
    global _PLUGIN_CACHE
    _PLUGIN_CACHE = None


def _resolve_class_hook(plugin_cls: type, attr_name: str, expected_desc: str, expected_types: tuple) -> Optional[Any]:
    """Resolve an optional class-level plugin hook to a validated value.

    The attribute may be a classmethod/staticmethod/function or a plain value
    (``skills_dir``, ``tool_transformers`` and ``cli_permissions`` all share
    this contract). Returns ``None`` — never raises — when the hook is absent,
    resolves to ``None``, raises, or yields an unexpected type: one bad plugin
    must never break collection.
    """
    attr = getattr(plugin_cls, attr_name, None)
    if attr is None:
        return None
    plugin_repr = getattr(plugin_cls, "__name__", plugin_cls)
    try:
        value = attr() if callable(attr) else attr
    except Exception as exc:  # noqa: BLE001 - one bad plugin must not break collection
        logger.warning("plugin %r %s() failed: %s", plugin_repr, attr_name, exc)
        return None
    if value is None:
        # A hook explicitly returning None means "nothing to contribute".
        return None
    if not isinstance(value, expected_types):
        logger.warning(
            "plugin %r %s must return %s, got %s; ignoring",
            plugin_repr,
            attr_name,
            expected_desc,
            type(value).__name__,
        )
        return None
    return value


def _skill_dir_of(plugin_cls: type) -> Optional[str]:
    """Return a plugin class's bundled skill directory if it exposes one.

    ``skills_dir`` may be a callable (classmethod/staticmethod/function) or a
    plain string/``Path`` attribute. Returns the path only when it resolves to
    an existing directory.
    """
    value = _resolve_class_hook(plugin_cls, "skills_dir", "a str or PathLike", (str, os.PathLike))
    if value is None:
        return None
    candidate = str(value)
    if candidate and Path(candidate).expanduser().is_dir():
        return candidate
    return None


def plugin_skill_directories() -> List[str]:
    """Discover skill directories contributed by installed plugins.

    Iterates ``datus.plugins`` entry points, loads each plugin class, and
    collects its ``skills_dir()`` when it points at an existing directory.
    Every failure is swallowed so skill discovery never blocks startup.
    """
    found: List[str] = []
    for _name, plugin_cls in _loaded_plugins():
        if plugin_cls is None:
            continue
        skill_dir = _skill_dir_of(plugin_cls)
        if skill_dir and skill_dir not in found:
            found.append(skill_dir)
    return found


def _agent_config_location() -> Optional[str]:
    """Path of the agent config file actually loaded this process, or ``None``.

    Prefers the ``ConfigurationManager`` singleton (which reflects an explicit
    ``--config``) and falls back to the default resolution order. Never raises
    — prompt construction must not break on a missing config.
    """
    try:
        from datus.configuration import agent_config_loader

        mgr = agent_config_loader.CONFIGURATION_MANAGER
        if mgr is not None:
            return str(mgr.config_path)
        return str(agent_config_loader.parse_config_path())
    except Exception as exc:  # noqa: BLE001 - defensive: never break prompt build
        logger.debug("agent config location lookup failed: %s", exc)
        return None


def _plugin_config_preamble() -> str:
    """datus-owned header prepended to the plugin prompt sections.

    Tells the agent where plugin profiles live so it can add or edit them
    (e.g. when a setup skill walks the user through configuration). Contains
    only the file path and the config shape — never profile values.
    """
    config_path = _agent_config_location()
    location = f"`{config_path}` " if config_path else "the agent config file (agent.yml) "
    return (
        "## Plugins\n"
        f"Plugin profiles are configured in {location}under "
        "`agent.plugins.<plugin>.<profile>` (use `${ENV_VAR}` placeholders for secrets). "
        "`datus <plugin>` commands reload the file on every invocation, so config edits "
        "take effect immediately without restarting this session."
    )


def plugin_system_prompt_sections(agent_config) -> List[str]:
    """Collect system-prompt sections contributed by installed plugins.

    Each plugin may expose an *optional* class-level
    ``system_prompt(profiles) -> str | None`` (classmethod/staticmethod). It is
    resolved at prompt-build time, i.e. **without an active profile instance**,
    so it must be reachable at the class level — mirroring ``skills_dir()``.

    datus passes the plugin's *full* profile mapping (all environments, not
    just the active one) taken from ``agent_config.plugin_services[<ep.name>]``;
    an installed-but-unconfigured plugin receives ``{}`` and may return setup
    guidance. The plugin decides which non-secret fields to surface. datus
    never splices profile values itself. Every failure is swallowed so one bad
    plugin never blocks prompt construction.

    When at least one plugin contributes a section, a datus-owned ``## Plugins``
    preamble naming the loaded config file location is prepended so the agent
    knows where profiles are added or edited.
    """
    plugin_services = getattr(agent_config, "plugin_services", None) or {}
    sections: List[str] = []
    for name, plugin_cls in _loaded_plugins():
        if plugin_cls is None:
            continue
        attr = getattr(plugin_cls, "system_prompt", None)
        if not callable(attr):
            continue
        profiles = plugin_services.get(name, {})
        try:
            section = attr(profiles)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break prompt build
            logger.debug("plugin %r system_prompt() failed: %s", name, exc)
            continue
        if isinstance(section, str) and section.strip():
            sections.append(section.strip())
    if sections:
        sections.insert(0, _plugin_config_preamble())
    return sections


def _tool_transformers_of(plugin_cls: type) -> Optional[dict]:
    """Resolve a plugin class's optional ``tool_transformers`` declaration.

    Accepts a classmethod/staticmethod/function or a plain dict attribute,
    mirroring ``_skill_dir_of``. Returns the dict, or ``None`` when absent,
    malformed, or raising.
    """
    return _resolve_class_hook(plugin_cls, "tool_transformers", "a dict", (dict,))


def collect_plugin_tool_transformers() -> Dict[str, List]:
    """Collect tool argument transformers declared by installed plugins.

    Iterates ``datus.plugins`` entry points, resolves each class's optional
    ``tool_transformers()`` hook, and accumulates a mapping of tool pattern
    (proxy syntax: ``"execute_sql"``, ``"db_tools.*"``) to a flat transformer
    list. A declaration value may be a single callable or a list of callables;
    non-callable entries and non-string patterns are warned about and skipped.
    Every failure is logged and skipped; collection never raises.

    Transformer semantics (rewrite/deny, fail-closed) are documented in
    :mod:`datus.plugins.base` and enforced by
    :mod:`datus.tools.middleware.tool_middleware`.
    """
    accumulated: Dict[str, List] = {}
    for name, plugin_cls in _loaded_plugins():
        if plugin_cls is None:
            continue
        declared = _tool_transformers_of(plugin_cls)
        if declared is None:
            continue
        for pattern, value in declared.items():
            if not isinstance(pattern, str) or not pattern.strip():
                logger.warning("Plugin %r tool_transformers has invalid pattern %r; ignoring.", name, pattern)
                continue
            transformers = value if isinstance(value, list) else [value]
            valid = [t for t in transformers if callable(t)]
            if len(valid) != len(transformers):
                logger.warning(
                    "Plugin %r tool_transformers[%r] contains non-callable entries; skipping those.",
                    name,
                    pattern,
                )
            if valid:
                accumulated.setdefault(pattern.strip(), []).extend(valid)
    return accumulated


def _prefix_cli_pattern(ep_name: str, pattern: str) -> Optional[str]:
    """Confine a namespace-relative pattern to ``datus <ep_name> ...``.

    The prefix part (before the first ``:``) gets the two literal anchor
    tokens prepended; a glob part is reattached verbatim:

    * ``"greet:*"``     -> ``"datus hello greet:*"``
    * ``"greet"``       -> ``"datus hello greet"`` (stays an exact match)
    * ``":*"``          -> ``"datus hello:*"`` (the whole namespace)

    Because ``command_matches_pattern`` fnmatches token-by-token anchored at
    ``argv[0]``, the literal ``datus <ep_name>`` anchor means a plugin can
    never produce a rule matching commands outside its own namespace. Returns
    ``None`` for an empty/whitespace-only pattern (caller warns and skips).
    """
    raw = pattern.strip()
    if not raw:
        return None
    if ":" in raw:
        prefix, glob = raw.split(":", 1)
    else:
        prefix, glob = raw, None
    prefix = prefix.strip()
    new_prefix = f"datus {ep_name} {prefix}" if prefix else f"datus {ep_name}"
    return f"{new_prefix}:{glob}" if glob is not None else new_prefix


def _cli_permissions_of(plugin_cls: type) -> Optional[dict]:
    """Resolve a plugin class's optional ``cli_permissions`` declaration.

    Accepts a classmethod/staticmethod/function or a plain dict attribute,
    mirroring ``_skill_dir_of``. Returns the dict, or ``None`` when absent,
    malformed, or raising.
    """
    return _resolve_class_hook(plugin_cls, "cli_permissions", "a dict", (dict,))


def collect_plugin_cli_permissions() -> Dict[str, "BashCommandRules"]:
    """Collect per-profile bash rules declared by installed plugins.

    Iterates ``datus.plugins`` entry points, resolves each class's optional
    ``cli_permissions()`` hook, validates the shape (profile name ->
    ``{allow/ask/deny: [patterns]}``), and prefixes every pattern with
    ``datus <entry-point-name> `` so a plugin can only shape permissions for
    its own CLI namespace.

    Only ``normal`` and ``auto`` profile keys are accepted; ``dangerous``
    ignores all bash rules by design and is warned about explicitly. The
    returned rulesets carry **only** allow/ask/deny lists — never ``default``
    or ``classifier`` — so plugin declarations can never change a profile's
    default posture. Every failure is logged and skipped; collection never
    raises.
    """
    from datus.tools.permission.bash_rules import BashCommandRules

    accumulated: Dict[str, Dict[str, List[str]]] = {}
    seen_names: set = set()
    for name, plugin_cls in _loaded_plugins():
        if not isinstance(name, str) or name in seen_names or not name:
            if name in seen_names:
                logger.warning("Duplicate datus.plugins entry point %r; using the first for cli_permissions.", name)
            continue
        seen_names.add(name)
        if not _SAFE_PLUGIN_NAME_RE.match(name):
            logger.warning("Plugin entry-point name %r is not a safe CLI token; skipping its cli_permissions.", name)
            continue
        if plugin_cls is None:
            continue
        declared = _cli_permissions_of(plugin_cls)
        if declared is None:
            continue
        for profile_key, actions in declared.items():
            if profile_key == "dangerous":
                logger.warning(
                    "Plugin %r declares cli_permissions for 'dangerous'; the dangerous profile "
                    "ignores bash rules — entry dropped.",
                    name,
                )
                continue
            if profile_key not in _CLI_PERMISSION_PROFILES:
                logger.warning("Plugin %r cli_permissions has unknown profile key %r; ignoring.", name, profile_key)
                continue
            if not isinstance(actions, dict):
                logger.warning("Plugin %r cli_permissions[%r] must be a dict; ignoring.", name, profile_key)
                continue
            for action, patterns in actions.items():
                if action not in _CLI_PERMISSION_ACTIONS:
                    logger.warning(
                        "Plugin %r cli_permissions[%r] has unknown action %r; ignoring.", name, profile_key, action
                    )
                    continue
                if not isinstance(patterns, list):
                    logger.warning(
                        "Plugin %r cli_permissions[%r][%r] must be a list; ignoring.", name, profile_key, action
                    )
                    continue
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        logger.warning(
                            "Plugin %r cli_permissions[%r][%r] entries must be strings, got %s; skipping one.",
                            name,
                            profile_key,
                            action,
                            type(pattern).__name__,
                        )
                        continue
                    prefixed = _prefix_cli_pattern(name, pattern)
                    if prefixed is None:
                        logger.warning(
                            "Plugin %r cli_permissions[%r][%r] contains an empty pattern; skipping.",
                            name,
                            profile_key,
                            action,
                        )
                        continue
                    accumulated.setdefault(profile_key, {}).setdefault(action, []).append(prefixed)

    return {
        profile: BashCommandRules(
            allow=actions.get("allow", []),
            ask=actions.get("ask", []),
            deny=actions.get("deny", []),
        )
        for profile, actions in accumulated.items()
    }
