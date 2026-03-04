# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill manager for coordinating skill discovery, permissions, and loading.

High-level coordinator that combines SkillRegistry with PermissionManager
to provide a unified interface for skill operations.
"""

import fnmatch
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.skill_tools.skill_config import SkillConfig, SkillMetadata
from datus.tools.skill_tools.skill_registry import SkillRegistry

if TYPE_CHECKING:
    from datus.tools.permission.permission_manager import PermissionManager

logger = logging.getLogger(__name__)


class SkillManager:
    """High-level skill manager coordinating registry and permissions.

    Provides a unified interface for:
    - Discovering available skills (filtered by permissions)
    - Loading skill content (with permission checks)
    - Generating XML context for system prompts
    - Filtering skills by patterns

    Example usage:
        manager = SkillManager(
            config=skills_config,
            permission_manager=permission_manager
        )

        # Get skills available for a node (respects DENY permissions)
        skills = manager.get_available_skills("chatbot", patterns=["sql-*"])

        # Load a skill (checks permissions)
        success, message, content = manager.load_skill("sql-optimization", "chatbot")

        # Generate XML for system prompt
        xml = manager.generate_available_skills_xml("chatbot")
    """

    def __init__(
        self,
        config: Optional[SkillConfig] = None,
        permission_manager: Optional["PermissionManager"] = None,
        registry: Optional[SkillRegistry] = None,
    ):
        """Initialize the skill manager.

        Args:
            config: Skills configuration
            permission_manager: Permission manager for access control
            registry: Optional pre-configured registry (for testing)
        """
        self.config = config or SkillConfig()
        self.permission_manager = permission_manager
        self.registry = registry or SkillRegistry(config=self.config)

        # Scan directories on initialization
        self.registry.scan_directories()

        logger.debug(f"SkillManager initialized with {self.registry.get_skill_count()} skills")

    def get_available_skills(
        self,
        node_name: str,
        patterns: Optional[List[str]] = None,
    ) -> List[SkillMetadata]:
        """Get skills available for a node, filtered by permissions and patterns.

        Skills with DENY permission are hidden. Skills with ALLOW or ASK are included.

        Args:
            node_name: Name of the agentic node
            patterns: Optional list of glob patterns to filter skills (e.g., ["sql-*", "data-*"])

        Returns:
            List of available SkillMetadata
        """
        all_skills = self.registry.list_skills()

        # Filter by patterns if provided
        if patterns:
            filtered_by_pattern = []
            for skill in all_skills:
                for pattern in patterns:
                    if pattern == "*" or fnmatch.fnmatch(skill.name, pattern):
                        filtered_by_pattern.append(skill)
                        break
            all_skills = filtered_by_pattern

        # Filter by permissions (hide DENY)
        if self.permission_manager:
            all_skills = self.permission_manager.filter_available_skills(all_skills, node_name)

        # Filter by model invocation (respect disable_model_invocation)
        all_skills = [s for s in all_skills if s.is_model_invocable()]

        logger.debug(f"Available skills for {node_name}: {[s.name for s in all_skills]}")
        return all_skills

    def load_skill(
        self,
        skill_name: str,
        node_name: str,
        check_permission: bool = True,
    ) -> Tuple[bool, str, Optional[str]]:
        """Load a skill's full content.

        Checks permissions before loading (unless disabled).
        Does NOT handle ASK permission prompts - caller should handle that.

        Args:
            skill_name: Name of the skill to load
            node_name: Name of the current agentic node
            check_permission: Whether to check permission (default True)

        Returns:
            Tuple of (success, message, content)
            - success: True if loaded successfully
            - message: Success or error message
            - content: Full SKILL.md content if successful, None otherwise
        """
        # Check if skill exists
        skill = self.registry.get_skill(skill_name)
        if not skill:
            return False, f"Skill '{skill_name}' not found", None

        # Check permission
        if check_permission and self.permission_manager:
            permission = self.permission_manager.check_permission("skills", skill_name, node_name)

            if permission == PermissionLevel.DENY:
                logger.warning(f"Skill '{skill_name}' denied for node '{node_name}'")
                return False, f"Permission denied for skill '{skill_name}'", None

            if permission == PermissionLevel.ASK:
                # Return special status - caller should handle user prompt
                return False, "ASK_PERMISSION", None

        # Load content
        content = self.registry.load_skill_content(skill_name)
        if not content:
            return False, f"Failed to load content for skill '{skill_name}'", None

        logger.info(f"Loaded skill '{skill_name}' for node '{node_name}'")
        return True, f"Skill '{skill_name}' loaded successfully", content

    def generate_available_skills_xml(
        self,
        node_name: str,
        patterns: Optional[List[str]] = None,
    ) -> str:
        """Generate XML context for available skills (for system prompt injection).

        Produces the <available_skills> XML block that lists skills the LLM can use.

        Args:
            node_name: Name of the agentic node
            patterns: Optional patterns to filter skills

        Returns:
            XML string for system prompt injection
        """
        skills = self.get_available_skills(node_name, patterns)

        if not skills:
            return ""

        lines = ["<available_skills>"]
        for skill in skills:
            lines.append(f'<skill name="{skill.name}">')
            lines.append(f"  <description>{skill.description}</description>")
            if skill.tags:
                lines.append(f"  <tags>{', '.join(skill.tags)}</tags>")
            lines.append("</skill>")
        lines.append("</available_skills>")
        lines.append("")
        lines.append('To use a skill, call: load_skill(skill_name="<skill_name>")')

        return "\n".join(lines)

    def get_skill(self, skill_name: str) -> Optional[SkillMetadata]:
        """Get skill metadata by name.

        Args:
            skill_name: Name of the skill

        Returns:
            SkillMetadata if found, None otherwise
        """
        return self.registry.get_skill(skill_name)

    def refresh(self) -> None:
        """Re-scan directories for skills.

        Useful when skills are added or modified at runtime.
        """
        self.registry.refresh()

    def get_skill_count(self) -> int:
        """Get total number of discovered skills.

        Returns:
            Number of skills
        """
        return self.registry.get_skill_count()

    def list_all_skills(self) -> List[SkillMetadata]:
        """List all discovered skills (ignoring permissions).

        For admin/debugging purposes.

        Returns:
            List of all SkillMetadata
        """
        return self.registry.list_skills()

    def parse_skill_patterns(self, patterns_str: str) -> List[str]:
        """Parse skill patterns string from configuration.

        Args:
            patterns_str: Comma-separated patterns (e.g., "sql-*, data-*")

        Returns:
            List of pattern strings
        """
        if not patterns_str:
            return []

        patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
        return patterns

    def check_skill_permission(self, skill_name: str, node_name: str) -> PermissionLevel:
        """Check permission level for a specific skill.

        Args:
            skill_name: Name of the skill
            node_name: Name of the agentic node

        Returns:
            PermissionLevel (ALLOW, DENY, or ASK)
        """
        if not self.permission_manager:
            return PermissionLevel.ALLOW

        return self.permission_manager.check_permission("skills", skill_name, node_name)

    # --- Marketplace operations ---

    def _get_marketplace_client(self):
        """Lazy-initialize and cache marketplace client."""
        if not hasattr(self, "_marketplace_client") or self._marketplace_client is None:
            from datus.tools.skill_tools.marketplace_client import SkillMarketplaceClient

            self._marketplace_client = SkillMarketplaceClient(base_url=self.config.marketplace_url)
        return self._marketplace_client

    def search_marketplace(self, query: str = "", tag: str = "") -> List[dict]:
        """Search skills in the remote marketplace.

        Args:
            query: Search query
            tag: Filter by tag

        Returns:
            List of skill info dicts from marketplace
        """
        try:
            client = self._get_marketplace_client()
            return client.search(query=query, tag=tag)
        except Exception as e:
            logger.error(f"Marketplace search failed: {e}")
            return []

    def install_from_marketplace(self, name: str, version: str = "latest") -> Tuple[bool, str]:
        """Install a skill from the marketplace.

        Downloads the bundle from Town and extracts it to the install directory.

        Args:
            name: Skill name
            version: Version to install (default: "latest")

        Returns:
            Tuple of (success, message)
        """
        try:
            from pathlib import Path

            client = self._get_marketplace_client()
            install_dir = Path(self.config.install_dir).expanduser()
            install_dir.mkdir(parents=True, exist_ok=True)

            # Download and extract
            dest = client.download_bundle(name, version, install_dir)

            # Register in local registry
            self.registry.install_skill(name, dest)

            return True, f"Installed {name}@{version} to {dest}"
        except Exception as e:
            logger.error(f"Failed to install {name}: {e}")
            return False, f"Installation failed: {e}"

    def publish_to_marketplace(self, skill_dir: str, owner: str = "") -> Tuple[bool, str]:
        """Publish a local skill to the marketplace.

        Args:
            skill_dir: Path to the skill directory, OR a skill name
                       (resolved from registry if not a valid path)
            owner: Optional owner name

        Returns:
            Tuple of (success, message)
        """
        try:
            from pathlib import Path

            path = Path(skill_dir)
            # If path doesn't exist or has no SKILL.md, try resolving as a skill name
            if not (path.is_dir() and (path / "SKILL.md").exists()):
                skill_meta = self.registry.get_skill(skill_dir)
                if skill_meta and skill_meta.location and (skill_meta.location / "SKILL.md").exists():
                    path = skill_meta.location
                    logger.info(f"Resolved skill name '{skill_dir}' to {path}")

            client = self._get_marketplace_client()
            result = client.publish_skill(path, owner=owner)
            name = result.get("name", "unknown")
            version = result.get("latest_version", "?")
            return True, f"Published {name}@{version} to marketplace"
        except Exception as e:
            logger.error(f"Failed to publish: {e}")
            return False, f"Publish failed: {e}"

    def sync_promoted_skills(self) -> List[str]:
        """Auto-sync promoted skills from marketplace.

        Downloads any promoted skills that aren't locally installed.

        Returns:
            List of newly synced skill names
        """
        synced = []
        try:
            client = self._get_marketplace_client()
            promoted = client.list_skills(promoted=True)
            for skill_info in promoted:
                name = skill_info.get("name")
                if name and not self.registry.skill_exists(name):
                    ok, _msg = self.install_from_marketplace(name)
                    if ok:
                        synced.append(name)
                        logger.info(f"Auto-synced promoted skill: {name}")
        except Exception as e:
            logger.error(f"Failed to sync promoted skills: {e}")
        return synced
