# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill configuration models for AgentSkills integration.

Provides:
- SkillConfig: Global skills configuration from agent.yml
- SkillMetadata: Parsed metadata from SKILL.md frontmatter
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from datus.validation.report import TargetFilter

logger = logging.getLogger(__name__)


def _builtin_skills_dir() -> Optional[str]:
    """Return the packaged ``datus/resources/skills`` directory as a string.

    Resolves the path lazily via :func:`package_data_path` so it works for
    editable installs, wheel installs, and ``uv run`` from a checkout. Returns
    ``None`` when the resource cannot be materialised on the local filesystem
    (e.g. running from a zipped distribution that exposes resources only as
    a non-Path Traversable). Failures are swallowed because skill resolution
    must never block startup.
    """
    try:
        from datus.utils.resource_utils import package_data_path

        path = package_data_path("resources/skills")
        if path is None:
            return None
        try:
            if not Path(str(path)).exists():
                return None
        except (TypeError, OSError):
            return None
        return str(path)
    except Exception as exc:  # noqa: BLE001 - defensive: never break import
        logger.debug("Built-in skills directory resolution failed: %s", exc)
        return None


def _entry_point_skill_directories() -> List[str]:
    """Discover skill directories contributed by adapter packages.

    Adapter packages (e.g. ``datus-hello``) expose a bundled skill
    directory through the ``datus.skills`` entry-point group. Each entry point
    loads to either a string/``Path`` directory or a zero-arg callable that
    returns one. Every failure is swallowed — skill discovery must never block
    startup — and only existing directories are returned.
    """
    found: List[str] = []
    try:
        # Lazy import to avoid a cycle; the registry owns the one shared
        # py3.10 ``select`` / pre-3.10 dict-fallback compatibility shim.
        from datus.plugins.registry import entry_points_for_group

        candidates = entry_points_for_group("datus.skills")
    except Exception as exc:  # noqa: BLE001 - defensive: never break import
        logger.debug("datus.skills entry-point lookup failed: %s", exc)
        return found

    for ep in candidates:
        try:
            obj = ep.load()
            value = obj() if callable(obj) else obj
            candidate = str(value)
            if Path(candidate).expanduser().is_dir() and not _contains_directory(found, candidate):
                found.append(candidate)
        except Exception as exc:  # noqa: BLE001 - one bad adapter must not break discovery
            logger.debug("datus.skills entry point %r failed to resolve: %s", getattr(ep, "name", ep), exc)
    return found


def _plugins_system_enabled() -> bool:
    """Whether the datus-plugin system is enabled in the loaded agent config.

    ``agent.plugins_enabled: false`` is the master switch that disables all
    plugin functionality, including plugin-bundled skills. This reads the
    already-loaded ``ConfigurationManager`` singleton (never triggers a config
    load) because ``SkillConfig`` may be constructed without an ``AgentConfig``
    in reach (default factory, ``skill_cli``). Defaults to enabled when no
    config has been loaded yet (e.g. unit tests building SkillConfig directly).
    """
    try:
        from datus.configuration import agent_config_loader

        # Same coercion as ``AgentConfig.plugins_enabled`` so the two readers
        # of this key can never disagree on a value like ``"off"``.
        from datus.configuration.agent_config import _coerce_bool  # noqa: PLC2701

        mgr = agent_config_loader.CONFIGURATION_MANAGER
        if mgr is None:
            return True
        value = mgr.data.get("plugins_enabled")
    except Exception as exc:  # noqa: BLE001 - defensive: never break discovery
        logger.debug("plugins_enabled lookup failed: %s", exc)
        return True
    return _coerce_bool(value, True)


def _plugin_skill_directories() -> List[str]:
    """Skill directories contributed by ``datus.plugins`` packages.

    Delegates to :func:`datus.plugins.registry.plugin_skill_directories` (lazy
    import to avoid an import cycle). Returns an empty list when the plugin
    system is disabled (``agent.plugins_enabled: false``). Any failure resolves
    to an empty list so skill discovery never blocks startup.
    """
    if not _plugins_system_enabled():
        return []
    try:
        from datus.plugins.registry import plugin_skill_directories

        return plugin_skill_directories()
    except Exception as exc:  # noqa: BLE001 - defensive: never break discovery
        logger.debug("plugin skill-directory discovery failed: %s", exc)
        return []


def _adapter_skill_directories() -> List[str]:
    """Directories contributed by plugins (``datus.plugins``) then legacy
    adapters (``datus.skills``), de-duplicated with plugins taking precedence."""
    dirs: List[str] = []
    for ep_dir in _plugin_skill_directories():
        if not _contains_directory(dirs, ep_dir):
            dirs.append(ep_dir)
    for ep_dir in _entry_point_skill_directories():
        if not _contains_directory(dirs, ep_dir):
            dirs.append(ep_dir)
    return dirs


def _default_skill_directories() -> List[str]:
    """Default scan order: project → user → plugin/adapter entry points → packaged built-ins.

    The first two entries match the documented user-facing locations and always
    win (first-wins semantics in :class:`SkillRegistry`). Contributed
    directories — plugins (``datus.plugins``) then legacy adapters
    (``datus.skills``) — come next so an installed plugin can shadow a stale
    built-in, but never a user override. The packaged directory is appended last.
    """
    dirs: List[str] = ["./.datus/skills", "~/.datus/skills"]
    for ep_dir in _adapter_skill_directories():
        if not _contains_directory(dirs, ep_dir):
            dirs.append(ep_dir)
    builtin = _builtin_skills_dir()
    if builtin and not _contains_directory(dirs, builtin):
        dirs.append(builtin)
    return dirs


def _same_directory(left: str, right: str) -> bool:
    """Compare directory paths after user and relative-path normalization."""
    try:
        return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()
    except (TypeError, OSError, RuntimeError):
        return str(left) == str(right)


def _contains_directory(directories: List[str], candidate: str) -> bool:
    return any(_same_directory(directory, candidate) for directory in directories)


class SkillConfig(BaseModel):
    """Global skills configuration from agent.yml.

    Configures where to discover skills and global skill behavior.

    Example configuration:
        skills:
          directories:
            - ./.datus/skills
            - ~/.datus/skills
          warn_duplicates: true
          whitelist_from_compaction: true

    Attributes:
        directories: List of directories to scan for skills. Project-level
            directories (``./.datus/skills``) take precedence over the user
            override (``~/.datus/skills``), which in turn takes precedence
            over the packaged built-ins shipped under
            ``datus/resources/skills`` (always appended last so first-party
            skills like ``init`` are discoverable without any deployment
            step).
        warn_duplicates: Warn when duplicate skill names are found
        whitelist_from_compaction: Preserve skill content during session compaction
    """

    directories: List[str] = Field(
        default_factory=_default_skill_directories,
        description=(
            "Directories scanned for SKILL.md files. The packaged "
            "datus/resources/skills directory is appended automatically so "
            "built-in skills are always available."
        ),
    )
    warn_duplicates: bool = Field(default=True, description="Warn on duplicate skill names")
    whitelist_from_compaction: bool = Field(
        default=True, description="Preserve skill responses during session compaction"
    )
    # Marketplace settings
    marketplace_url: str = Field(default="http://localhost:9000", description="Town backend URL for skill marketplace")
    auto_sync: bool = Field(default=False, description="Auto-sync promoted skills on startup")
    install_dir: str = Field(default="~/.datus/skills", description="Directory for marketplace-installed skills")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillConfig":
        """Create SkillConfig from dictionary (agent.yml format).

        Args:
            data: Dictionary with skills configuration

        Returns:
            SkillConfig instance
        """
        if not data:
            return cls()

        # Always append plugin/adapter-contributed (entry-point) and packaged
        # built-in directories to whatever the user listed so a deployer can
        # never accidentally drop first-party, plugin, or adapter skills (e.g.
        # ``init``, ``hello``) by overriding ``skills.directories`` in agent.yml.
        directories = data.get("directories")
        if directories is None:
            directories = _default_skill_directories()
        else:
            directories = list(directories)
            for ep_dir in _adapter_skill_directories():
                if not _contains_directory(directories, ep_dir):
                    directories.append(ep_dir)
            builtin = _builtin_skills_dir()
            if builtin and not _contains_directory(directories, builtin):
                directories.append(builtin)

        return cls(
            directories=directories,
            warn_duplicates=data.get("warn_duplicates", True),
            whitelist_from_compaction=data.get("whitelist_from_compaction", True),
            marketplace_url=data.get("marketplace_url", "http://localhost:9000"),
            auto_sync=data.get("auto_sync", False),
            install_dir=data.get("install_dir", "~/.datus/skills"),
        )


class SkillMetadata(BaseModel):
    """Metadata parsed from SKILL.md frontmatter.

    Represents a skill discovered from the filesystem. The content is lazily loaded
    only when the skill is actually used.

    Example SKILL.md frontmatter:
        ---
        name: sql-optimization
        description: SQL query optimization techniques
        tags: [sql, performance]
        version: 1.0.0
        allowed_commands:
          - "python:scripts/*.py"
          - "sh:*.sh"
        disable_model_invocation: false
        user_invocable: true
        allowed_agents:
          - gen_dashboard
        context: fork
        agent: Explore
        ---

    Attributes:
        name: Unique skill name (required)
        description: Human-readable description (required)
        location: Path to the skill directory
        tags: Optional tags for categorization
        version: Optional version string
        allowed_commands: Patterns for allowed script execution (Claude Code compatible)
        disable_model_invocation: If true, only user can invoke via /skill-name
        user_invocable: If false, hidden from menu, only model invokes
        allowed_agents: Node names (from ``AgenticNode.get_node_name()``) allowed
            to see and load this skill. Empty list means no restriction — every
            agent can see it.
        context: "fork" to run in isolated subagent
        agent: Subagent type when context=fork (Explore, Plan, general-purpose)
        content: Full SKILL.md content (lazy loaded)
    """

    name: str = Field(..., description="Unique skill name")
    description: str = Field(..., description="Human-readable description")
    location: Path = Field(..., description="Path to skill directory")
    tags: List[str] = Field(default_factory=list, description="Optional categorization tags")
    version: Optional[str] = Field(default=None, description="Optional version string")

    # Invocation control
    disable_model_invocation: bool = Field(default=False, description="If true, only user can invoke via /skill-name")
    user_invocable: bool = Field(default=True, description="If false, hidden from menu, only model invokes")
    # Agent scoping: empty list == no restriction; non-empty == whitelist of node names
    allowed_agents: List[str] = Field(
        default_factory=list,
        description="Agent node names allowed to see/load this skill; empty = unrestricted",
    )
    # Subagent execution
    context: Optional[str] = Field(default=None, description="'fork' to run in isolated subagent")
    agent: Optional[str] = Field(default=None, description="Subagent type when context=fork")

    # Validator skill extensions (ValidationHook infrastructure)
    # Skills with kind="validator" are NOT injected into the main agent's
    # prompt via SkillFuncTool — they are consumed exclusively by
    # ValidationHook which fires them at run end on matching targets.
    kind: Literal["skill", "validator"] = Field(
        default="skill",
        description="'skill' (default) is loaded by the main agent; 'validator' is driven by ValidationHook",
    )
    severity: Literal["blocking", "advisory", "off"] = Field(
        default="advisory",
        description="Blocking drives retry via on_end final_report; advisory reports only; off disables the validator",
    )
    mode: Literal["llm"] = Field(
        default="llm",
        description="Execution mode for the validator (future: 'declarative'); only 'llm' supported in current PR",
    )
    targets: List[TargetFilter] = Field(
        default_factory=list,
        description="Per-target filters (empty = match all); any matching filter activates the validator",
    )

    # Marketplace metadata
    license: Optional[str] = Field(default=None, description="License identifier (e.g. Apache-2.0)")
    compatibility: Optional[Dict[str, Any]] = Field(default=None, description="Compatibility map")
    source: Optional[str] = Field(default=None, description="'local' or 'marketplace'")
    marketplace_version: Optional[str] = Field(default=None, description="Version from marketplace")

    # Content (lazy loaded)
    content: Optional[str] = Field(default=None, description="Full SKILL.md content (lazy loaded)")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_frontmatter(cls, frontmatter: Dict[str, Any], location: Path) -> "SkillMetadata":
        """Create SkillMetadata from parsed YAML frontmatter.

        Args:
            frontmatter: Parsed YAML frontmatter dictionary
            location: Path to the skill directory

        Returns:
            SkillMetadata instance

        Raises:
            ValueError: If required fields (name, description) are missing
        """
        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name:
            raise ValueError(f"Skill at {location} missing required 'name' field")
        if not description:
            raise ValueError(f"Skill at {location} missing required 'description' field")

        # Validator fields: parsed with pydantic's TargetFilter so YAML "schema"
        # alias flows through to db_schema correctly.
        raw_targets = frontmatter.get("targets", []) or []
        parsed_targets: List[TargetFilter] = []
        for t in raw_targets:
            if isinstance(t, TargetFilter):
                parsed_targets.append(t)
            elif isinstance(t, dict):
                parsed_targets.append(TargetFilter.model_validate(t))
            else:
                from datus.utils.exceptions import DatusException, ErrorCode

                raise DatusException(
                    ErrorCode.SKILL_FRONTMATTER_INVALID,
                    message_args={
                        "location": str(location),
                        "error_message": f"invalid target entry (expected dict): {t!r}",
                    },
                )

        # YAML parses bare ``off`` / ``on`` as booleans (False / True). That
        # collides with our ``severity: off`` spelling — coerce back to string
        # so skill authors can write ``severity: off`` unquoted.
        raw_severity = frontmatter.get("severity", "advisory")
        if raw_severity is False:
            raw_severity = "off"
        elif raw_severity is True:
            raw_severity = "on"  # not a valid enum value — pydantic will flag it

        return cls(
            name=name,
            description=description,
            location=location,
            tags=frontmatter.get("tags", []),
            version=frontmatter.get("version"),
            disable_model_invocation=frontmatter.get("disable_model_invocation", False),
            user_invocable=frontmatter.get("user_invocable", True),
            allowed_agents=frontmatter.get("allowed_agents", []),
            context=frontmatter.get("context"),
            agent=frontmatter.get("agent"),
            kind=frontmatter.get("kind", "skill"),
            severity=raw_severity,
            mode=frontmatter.get("mode", "llm"),
            targets=parsed_targets,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
        )

    def is_model_invocable(self) -> bool:
        """Check if the model can invoke this skill.

        Returns:
            True if model can invoke (disable_model_invocation is False)
        """
        return not self.disable_model_invocation

    def runs_in_subagent(self) -> bool:
        """Check if this skill runs in an isolated subagent.

        Returns:
            True if context is 'fork'
        """
        return self.context == "fork"

    def is_allowed_for(self, *node_names: Optional[str]) -> bool:
        """Check whether an agent is allowed to see this skill.

        Empty ``allowed_agents`` means no restriction. A non-empty list is a
        whitelist matched against *any* of the supplied identifiers — callers
        typically pass both the node alias (``get_node_name()``) and the
        canonical class name (``get_node_class_name()``) so that a custom
        subagent alias (e.g. ``my_dashboard`` with ``node_class: gen_dashboard``)
        still matches a whitelist written in terms of the class name.

        Args:
            *node_names: One or more identifiers to test against. ``None`` /
                empty values are ignored.

        Returns:
            True if the skill has no scoping or any provided identifier is
            whitelisted.
        """
        if not self.allowed_agents:
            return True
        return any(name in self.allowed_agents for name in node_names if name)

    def is_validator(self) -> bool:
        """Return True if this skill is a validator driven by ValidationHook.

        Validator skills are excluded from the main agent's available-skills
        list and are invoked exclusively by ``ValidationHook``.
        """
        return self.kind == "validator"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation (excluding content for efficiency)
        """
        return {
            "name": self.name,
            "description": self.description,
            "location": str(self.location),
            "tags": self.tags,
            "version": self.version,
            "disable_model_invocation": self.disable_model_invocation,
            "user_invocable": self.user_invocable,
            "allowed_agents": self.allowed_agents,
            "context": self.context,
            "agent": self.agent,
            "kind": self.kind,
            "severity": self.severity,
            "mode": self.mode,
            "targets": [t.model_dump(by_alias=True, exclude_none=True) for t in self.targets],
            "license": self.license,
            "compatibility": self.compatibility,
            "source": self.source,
            "marketplace_version": self.marketplace_version,
        }
