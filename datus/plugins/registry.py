# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Discovery of ``datus.plugins`` entry points and their manifests.

A plugin package declares one entry point under the ``datus.plugins`` group
whose value is the package name, e.g. ``hello = datus_plugin_hello``. This
module locates each package directory via ``importlib.util.find_spec``
(WITHOUT executing the package), reads its ``datus-plugin.yml`` manifest, and
exposes the declared skill directories, system-prompt templates, CLI
bash-permission rules, tool transformers and config schemas. Plugin code is
imported only lazily, through :func:`resolve_code_ref`, when a declared ``cli``
or ``tool_transformers`` entry actually runs. See :mod:`datus.plugins.base`
for the manifest contract.

Every lookup is defensive: a broken or missing plugin must never crash the CLI
or block skill discovery.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from datus.plugins.base import (
    MANIFEST_FILENAME,
    PluginCommand,
    PluginManifest,
    parse_code_ref,
    read_manifest_file,
)
from datus.plugins.prompt import render_plugin_prompt
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


def _package_dir_for_entry_point(ep) -> Optional[Path]:
    """Locate the package directory an entry point names, WITHOUT importing it.

    The entry-point value must be a bare package name (``datus_plugin_foo``).
    A legacy ``pkg.module:Class`` value (non-empty attribute part) is rejected
    with a migration warning. ``find_spec`` only consults path finders for a
    top-level package, so no plugin code runs here. Returns ``None`` — never
    raises — when the package cannot be located or is not a directory-backed
    package (single ``.py`` module, zipimport archive, namespace package
    without a manifest).
    """
    name = getattr(ep, "name", None)
    value = getattr(ep, "value", None) or ""
    module_name, _, attr = value.partition(":")
    module_name = module_name.strip()
    if attr.strip():
        logger.warning(
            "Plugin %r entry point %r targets an object (legacy class-based contract); "
            "rebuild it against the datus-plugin.yml manifest contract. Skipping.",
            name,
            value,
        )
        return None
    if not module_name:
        return None
    if "." in module_name:
        # A dotted target (``parent.child``) would make ``find_spec`` import the
        # parent package to locate the child, executing plugin code before the
        # manifest is ever read. The entry-point value must be a bare top-level
        # package name.
        logger.warning(
            "Plugin %r entry point %r must reference a bare top-level package, not a dotted module; skipping.",
            name,
            value,
        )
        return None
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not crash discovery
        logger.warning("Plugin %r package %r could not be located: %s; skipping.", name, module_name, exc)
        return None
    if spec is None:
        logger.warning("Plugin %r package %r is not importable; skipping.", name, module_name)
        return None
    if spec.origin:
        origin = Path(spec.origin)
        if origin.name == "__init__.py":
            return origin.parent
        logger.warning(
            "Plugin %r entry point must reference a package containing %s, not module %r; skipping.",
            name,
            MANIFEST_FILENAME,
            module_name,
        )
        return None
    # Namespace package: no origin, possibly several search locations.
    for location in getattr(spec, "submodule_search_locations", None) or []:
        candidate = Path(location)
        if (candidate / MANIFEST_FILENAME).is_file():
            return candidate
    logger.warning(
        "Plugin %r package %r has no %s in any of its locations; skipping.", name, module_name, MANIFEST_FILENAME
    )
    return None


# Process-level cache of ``(entry_point_name, manifest_or_None)`` pairs.
# Installed plugins cannot change mid-process, and the collection functions
# below all run on hot paths (skill discovery, prompt build, per-node
# transformer wrapping, permission collection) — without the cache each of
# them re-enumerates distribution metadata and re-reads every manifest on
# every call. ``None`` marks a missing/rejected manifest so consumers skip it
# without retrying.
_PLUGIN_CACHE: Optional[List[Tuple[Optional[str], Optional[PluginManifest]]]] = None

# Memoizes resolved ``tool_transformers`` callables per ``(plugin, ref)`` so
# the underlying import happens once, not on every per-node collection.
# ``None`` marks a failed resolution (import error / non-callable).
_RESOLVED_TRANSFORMER_CACHE: Dict[Tuple[str, str], Optional[Any]] = {}


def _loaded_manifests() -> List[Tuple[Optional[str], Optional[PluginManifest]]]:
    """Return cached ``(name, manifest)`` pairs for all installed plugins."""
    global _PLUGIN_CACHE
    if _PLUGIN_CACHE is None:
        pairs: List[Tuple[Optional[str], Optional[PluginManifest]]] = []
        for ep in iter_plugin_entry_points():
            name = getattr(ep, "name", None)
            manifest: Optional[PluginManifest] = None
            if isinstance(name, str) and name:
                package_dir = _package_dir_for_entry_point(ep)
                if package_dir is not None:
                    manifest = read_manifest_file(package_dir, name)
            pairs.append((name, manifest))
        _PLUGIN_CACHE = pairs
    return _PLUGIN_CACHE


def load_plugin_manifest(name: str) -> Optional[PluginManifest]:
    """Return the parsed manifest of the plugin registered as ``name``.

    Returns ``None`` when no plugin claims ``name`` (callers fall through to
    other dispatch paths) or when its manifest is missing/rejected (already
    warned about during loading). Never raises.
    """
    matches = [manifest for ep_name, manifest in _loaded_manifests() if ep_name == name]
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("Multiple datus.plugins entry points named %r; using the first.", name)
    return matches[0]


def invalidate_plugin_cache() -> None:
    """Drop the cached manifests and resolved transformers (tests / installs)."""
    global _PLUGIN_CACHE
    _PLUGIN_CACHE = None
    _RESOLVED_TRANSFORMER_CACHE.clear()


def resolve_code_ref(ref: str, plugin_name: str) -> Optional[Any]:
    """Import and return the object named by a dotted ref ``"module:attr"``.

    This is the ONLY place plugin code is imported — everything else in the
    registry works off manifest data. Returns ``None`` — never raises — on an
    invalid ref, import failure, or missing attribute.
    """
    parsed = parse_code_ref(ref)
    if parsed is None:
        logger.warning("Plugin %r code ref %r is invalid; skipping.", plugin_name, ref)
        return None
    module_name, attr_path = parsed
    try:
        module = importlib.import_module(module_name)
        target: Any = module
        for part in attr_path.split("."):
            target = getattr(target, part)
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not crash the caller
        logger.warning("Plugin %r failed to load code ref %r: %s", plugin_name, ref, exc)
        return None
    return target


def _is_active(name: Optional[str], active_names: Optional[Set[str]]) -> bool:
    """Whether plugin ``name`` passes the project activation filter.

    ``active_names is None`` means "no filter" (every plugin active — the
    ``plugins:`` section was absent). Otherwise only names in the set pass, so
    a project's ``plugins:`` whitelist gates which installed plugins may
    contribute skills / prompt sections / tool transformers / bash rules.
    """
    if active_names is None:
        return True
    return isinstance(name, str) and name in active_names


def _active_names_from_config(agent_config) -> Optional[Set[str]]:
    """Read the activation whitelist off an ``AgentConfig``, defensively.

    Returns ``None`` (no filter) when the config predates the accessor or the
    lookup raises, so prompt/skill collection degrades to "all active" rather
    than crashing.
    """
    getter = getattr(agent_config, "active_plugin_names", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception as exc:  # noqa: BLE001 - never break collection on a bad config
        logger.debug("active_plugin_names() failed: %s", exc)
        return None


def _active_profiles_subset(agent_config, name: str, profiles: dict) -> dict:
    """Narrow ``profiles`` to the ones the project activated for ``name``.

    ``agent_config.active_plugin_profiles(name)`` returns ``None`` (no pin — pass
    every profile) or the pinned profile names. A pin is applied as an
    intersection; a pin that matches nothing configured (stale) falls back to
    the full mapping so a typo never blanks the plugin out of the prompt.
    Defensive: any error or a config without the accessor passes ``profiles``
    unchanged.
    """
    if not isinstance(profiles, dict) or not profiles:
        return profiles
    getter = getattr(agent_config, "active_plugin_profiles", None)
    if not callable(getter):
        return profiles
    try:
        pinned = getter(name)
    except Exception as exc:  # noqa: BLE001 - never break prompt build on a bad config
        logger.debug("active_plugin_profiles(%r) failed: %s", name, exc)
        return profiles
    if not pinned:  # None (no pin) or [] (defensive) → surface everything
        return profiles
    narrowed = {p: cfg for p, cfg in profiles.items() if p in pinned}
    return narrowed or profiles


def _confined_package_path(manifest: PluginManifest, relative: str, key: str) -> Optional[Path]:
    """Resolve ``relative`` against the manifest's package dir, refusing escapes."""
    package_dir = Path(manifest.package_dir).resolve()
    candidate = (package_dir / relative).resolve()
    if not candidate.is_relative_to(package_dir):
        logger.warning("Plugin %r %s %r escapes the package directory; skipping.", manifest.name, key, relative)
        return None
    return candidate


def plugin_skill_directories(active_names: Optional[Set[str]] = None) -> List[str]:
    """Discover skill directories contributed by installed plugins.

    Iterates ``datus.plugins`` manifests and collects each ``skills`` path
    that resolves to an existing directory inside the plugin's package.
    ``active_names`` gates which plugins contribute (``None`` = every plugin;
    otherwise only names in the project's activation whitelist). No plugin
    code is executed. Every failure is swallowed so skill discovery never
    blocks startup.
    """
    found: List[str] = []
    for name, manifest in _loaded_manifests():
        if manifest is None or not manifest.skills:
            continue
        if not _is_active(name, active_names):
            continue
        candidate = _confined_package_path(manifest, manifest.skills, "skills")
        if candidate is None or not candidate.is_dir():
            continue
        path_str = str(candidate)
        if path_str not in found:
            found.append(path_str)
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


def _plugin_config_preamble(config_mutable: bool = True) -> str:
    """datus-owned header prepended to the plugin prompt sections.

    Tells the agent where plugin profiles live so it can add or edit them
    (e.g. when a setup skill walks the user through configuration). Contains
    only the file path and the config shape — never profile values.

    When ``config_mutable`` is ``False`` (multi-tenant chat API / gateway) the
    header must not name the config file nor describe its shape — the agent is
    told configuration is administrator-managed instead.
    """
    if not config_mutable:
        return (
            "## Plugins\n"
            "Plugin profiles are managed by the deployment administrator and the "
            "agent configuration is read-only in this environment. Never suggest "
            "editing configuration files and never walk the user through plugin "
            "setup; if a plugin is not configured, tell the user to contact their "
            "administrator. Configured `datus <plugin>` commands work normally.\n"
            "Run at most one `datus <plugin>` command per bash call, with `datus` as "
            "the command word: never wrap it in `$(...)`/backticks, and never behind "
            "`timeout`, `env`, `xargs` or `sh -c`. Pipes, redirection, `&&`/`;` lists "
            "and `--profile` are all fine; `--config` is rejected."
        )
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

    Renders each manifest's ``system_prompt`` Jinja2 template with the
    plugin's profile mapping taken from
    ``agent_config.plugin_services[<ep.name>]``, **narrowed to the profiles
    the project activated** (``plugins.<name>.active_profile`` in
    ``./.datus/config.yml``) and **secret-stripped against the manifest's
    config schema** — undeclared or ``x-secret`` fields never reach the
    template. An installed-but-unconfigured plugin renders with ``{}`` and may
    emit setup guidance. Every failure is swallowed so one bad plugin never
    blocks prompt construction.

    When at least one plugin contributes a section, a datus-owned ``## Plugins``
    preamble naming the loaded config file location is prepended so the agent
    knows where profiles are added or edited. When
    ``agent_config.config_mutable`` is ``False`` (multi-tenant chat API /
    gateway) the preamble switches to read-only wording, ``config_path`` is
    withheld from templates (a server-local path must not leak to tenants) and
    templates receive ``config_mutable=False`` so they can defer to the
    administrator instead of pointing at their setup skill.
    """
    plugin_services = getattr(agent_config, "plugin_services", None) or {}
    active_names = _active_names_from_config(agent_config)
    config_mutable = bool(getattr(agent_config, "config_mutable", True))
    config_path = _agent_config_location() if config_mutable else None
    sections: List[str] = []
    for name, manifest in _loaded_manifests():
        if manifest is None or not manifest.system_prompt:
            continue
        if not _is_active(name, active_names):
            continue
        profiles = _active_profiles_subset(agent_config, name, plugin_services.get(name, {}))
        section = render_plugin_prompt(manifest, profiles, config_path=config_path, config_mutable=config_mutable)
        if section:
            sections.append(section)
    if sections:
        sections.insert(0, _plugin_config_preamble(config_mutable))
    return sections


def _resolved_transformer(plugin_name: str, ref: str) -> Optional[Any]:
    """Resolve (and memoize) one declared transformer ref to a callable."""
    key = (plugin_name, ref)
    if key not in _RESOLVED_TRANSFORMER_CACHE:
        target = resolve_code_ref(ref, plugin_name)
        if target is not None and not callable(target):
            logger.warning("Plugin %r tool_transformers ref %r is not callable; skipping.", plugin_name, ref)
            target = None
        _RESOLVED_TRANSFORMER_CACHE[key] = target
    return _RESOLVED_TRANSFORMER_CACHE[key]


def collect_plugin_tool_transformers(active_names: Optional[Set[str]] = None) -> Dict[str, List]:
    """Collect tool argument transformers declared by installed plugins.

    Iterates ``datus.plugins`` manifests and lazily resolves each declared
    ``tool_transformers`` code ref, accumulating a mapping of tool pattern
    (proxy syntax: ``"execute_sql"``, ``"db_tools.*"``) to a flat transformer
    list. A ref that fails to import or is not callable is warned about and
    skipped — the plugin's remaining transformers still apply.
    ``active_names`` gates which plugins contribute (``None`` = every plugin).
    Collection never raises.

    Transformer semantics (rewrite/deny, fail-closed) are documented in
    :mod:`datus.plugins.base` and enforced by
    :mod:`datus.tools.middleware.tool_middleware`.
    """
    accumulated: Dict[str, List] = {}
    for name, manifest in _loaded_manifests():
        if manifest is None or not manifest.tool_transformers:
            continue
        if not _is_active(name, active_names):
            continue
        for pattern, refs in manifest.tool_transformers.items():
            valid: List[Any] = []
            for ref in refs:
                target = _resolved_transformer(str(name), ref)
                if target is not None:
                    valid.append(target)
            if valid:
                accumulated.setdefault(pattern, []).extend(valid)
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


def collect_plugin_cli_permissions(active_names: Optional[Set[str]] = None) -> Dict[str, "BashCommandRules"]:
    """Collect per-profile bash rules declared by installed plugins.

    Iterates ``datus.plugins`` manifests, validates each ``permissions``
    declaration's shape (profile name -> ``{allow/ask/deny: [patterns]}``),
    and prefixes every pattern with ``datus <entry-point-name> `` so a plugin
    can only shape permissions for its own CLI namespace. No plugin code is
    executed.

    ``active_names`` gates which plugins contribute (``None`` = every plugin).
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
    for name, manifest in _loaded_manifests():
        if not isinstance(name, str) or name in seen_names or not name:
            if name in seen_names:
                logger.warning("Duplicate datus.plugins entry point %r; using the first for permissions.", name)
            continue
        seen_names.add(name)
        if not _is_active(name, active_names):
            continue
        if not _SAFE_PLUGIN_NAME_RE.match(name):
            logger.warning("Plugin entry-point name %r is not a safe CLI token; skipping its permissions.", name)
            continue
        if manifest is None:
            continue
        declared = manifest.permissions
        if not declared:
            continue
        for profile_key, actions in declared.items():
            if profile_key == "dangerous":
                logger.warning(
                    "Plugin %r declares permissions for 'dangerous'; the dangerous profile "
                    "ignores bash rules — entry dropped.",
                    name,
                )
                continue
            if profile_key not in _CLI_PERMISSION_PROFILES:
                logger.warning("Plugin %r permissions has unknown profile key %r; ignoring.", name, profile_key)
                continue
            if not isinstance(actions, dict):
                logger.warning("Plugin %r permissions[%r] must be a dict; ignoring.", name, profile_key)
                continue
            for action, patterns in actions.items():
                if action not in _CLI_PERMISSION_ACTIONS:
                    logger.warning(
                        "Plugin %r permissions[%r] has unknown action %r; ignoring.", name, profile_key, action
                    )
                    continue
                if not isinstance(patterns, list):
                    logger.warning("Plugin %r permissions[%r][%r] must be a list; ignoring.", name, profile_key, action)
                    continue
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        logger.warning(
                            "Plugin %r permissions[%r][%r] entries must be strings, got %s; skipping one.",
                            name,
                            profile_key,
                            action,
                            type(pattern).__name__,
                        )
                        continue
                    prefixed = _prefix_cli_pattern(name, pattern)
                    if prefixed is None:
                        logger.warning(
                            "Plugin %r permissions[%r][%r] contains an empty pattern; skipping.",
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


# A config_schema may nest ``type: object`` properties; the TUI flattens them
# into dotted field names. The cap bounds recursion so a pathological schema
# (e.g. a self-referential YAML anchor that slipped past meta-validation) can
# never hang field derivation.
_SCHEMA_FLATTEN_MAX_DEPTH = 8


def _flatten_schema_properties(
    plugin_name: str,
    properties: Dict[str, Any],
    required_names: Set[str],
    *,
    prefix: str = "",
    inherited_required: bool = True,
    inherited_secret: bool = False,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """Flatten (possibly nested) schema properties into ordered field specs."""
    if depth >= _SCHEMA_FLATTEN_MAX_DEPTH:
        logger.warning(
            "Plugin %r config_schema nests deeper than %d levels; deeper fields are skipped.",
            plugin_name,
            _SCHEMA_FLATTEN_MAX_DEPTH,
        )
        return []
    normalized: List[Dict[str, Any]] = []
    for field_name, prop in properties.items():
        if not isinstance(field_name, str) or not field_name.strip():
            logger.warning("Plugin %r config_schema property name %r is invalid; skipping.", plugin_name, field_name)
            continue
        dotted = f"{prefix}{field_name.strip()}"
        spec_source = prop if isinstance(prop, dict) else {}
        secret = inherited_secret or spec_source.get("x-secret") is True
        required = inherited_required and field_name in required_names
        nested_properties = spec_source.get("properties")
        if spec_source.get("type") == "object" and isinstance(nested_properties, dict):
            nested_required = spec_source.get("required")
            nested_required_names = (
                {entry for entry in nested_required if isinstance(entry, str)}
                if isinstance(nested_required, list)
                else set()
            )
            normalized.extend(
                _flatten_schema_properties(
                    plugin_name,
                    nested_properties,
                    nested_required_names,
                    prefix=f"{dotted}.",
                    inherited_required=required,
                    inherited_secret=secret,
                    depth=depth + 1,
                )
            )
            continue
        spec: Dict[str, Any] = {
            "name": dotted,
            "description": str(spec_source.get("description") or ""),
            "required": required,
            "secret": secret,
        }
        if "default" in spec_source:
            spec["default"] = spec_source["default"]
        normalized.append(spec)
    return normalized


def plugin_config_schema(name: str) -> List[Dict[str, Any]]:
    """Return the normalized config-field specs derived from plugin ``name``'s
    manifest ``config_schema`` (a JSON Schema).

    Each returned entry has ``name`` (str), ``description`` (str), ``required``
    (bool), ``secret`` (bool — the ``x-secret: true`` property marker) and an
    optional ``default``. A ``type: object`` property with its own
    ``properties`` is expanded into one spec per leaf with a dotted name
    (``s3.secret_access_key``); ``x-secret`` on the object marks every leaf
    secret, and a leaf is ``required`` only when its whole ancestor path is
    required too. Property insertion order is preserved, so it defines the
    ``/plugins`` TUI field order. A plugin without a schema yields an empty
    list — the TUI then falls back to free-form key/value editing. Never
    raises.
    """
    manifest = load_plugin_manifest(name)
    if manifest is None or not isinstance(manifest.config_schema, dict):
        return []
    properties = manifest.config_schema.get("properties")
    if not isinstance(properties, dict):
        return []
    required = manifest.config_schema.get("required")
    required_names = {entry for entry in required if isinstance(entry, str)} if isinstance(required, list) else set()
    return _flatten_schema_properties(name, properties, required_names)


def iter_plugin_manifests() -> List[Tuple[str, PluginManifest]]:
    """Return ``(name, manifest)`` for every installed plugin with a valid manifest.

    Skips entry points whose name is missing or whose manifest was rejected
    during loading. Backed by the process-level manifest cache, so callers such
    as the ``!<plugin>`` completer can enumerate plugins cheaply. Never raises.
    """
    out: List[Tuple[str, PluginManifest]] = []
    for name, manifest in _loaded_manifests():
        if isinstance(name, str) and name and manifest is not None:
            out.append((name, manifest))
    return out


def plugin_commands(name: str) -> List[PluginCommand]:
    """Return the declarative CLI ``commands`` from plugin ``name``'s manifest.

    Descriptive metadata used by the ``!<plugin>`` completer / argument hints and
    ``datus plugin info`` — never used to parse the plugin's arguments. Returns an
    empty list when the plugin is unknown or declares no commands. Never raises.
    """
    manifest = load_plugin_manifest(name)
    if manifest is None:
        return []
    return list(manifest.commands)


def plugin_validate_profile(name: str, profile: Dict[str, Any]) -> List[str]:
    """Validate a candidate profile dict against plugin ``name``'s
    ``config_schema``.

    Returns human-readable ``"field: message"`` errors (empty = valid, or no
    schema). String values containing ``${`` are treated as opaque
    ``${ENV_VAR}`` placeholders and exempted from value-level validation —
    ``required`` violations (whose instance is the whole dict) still fire.
    A validator failure resolves to "no errors" so a broken schema never
    blocks the TUI save — the plugin CLI remains the final runtime guard.
    """
    manifest = load_plugin_manifest(name)
    if manifest is None or not isinstance(manifest.config_schema, dict):
        return []
    try:
        from jsonschema import Draft202012Validator

        errors: List[str] = []
        for error in Draft202012Validator(manifest.config_schema).iter_errors(profile):
            if isinstance(error.instance, str) and "${" in error.instance:
                continue
            path = ".".join(str(part) for part in error.path) or "profile"
            errors.append(f"{path}: {error.message}")
        return errors
    except Exception as exc:  # noqa: BLE001 - a broken schema must not block the TUI
        logger.warning("Plugin %r config_schema validation failed to run: %s", name, exc)
        return []
