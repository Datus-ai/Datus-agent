# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill registry for discovering and managing skills from the filesystem.

Scans configured directories for SKILL.md files, parses their frontmatter,
and provides lazy loading of skill content.
"""

import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from datus.tools.skill_tools.skill_config import SkillConfig, SkillMetadata

logger = logging.getLogger(__name__)

# Regex to extract YAML frontmatter from SKILL.md
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillRegistry:
    """Registry for discovering and managing skills.

    Scans configured directories for SKILL.md files at startup,
    parses their YAML frontmatter for metadata, and provides
    lazy loading of full skill content.

    Features:
    - Thread-safe skill discovery and access
    - Duplicate skill name detection with warnings
    - Lazy content loading (metadata at startup, content on demand)
    - Support for multiple skill directories

    Example usage:
        registry = SkillRegistry(config)
        registry.scan_directories()

        # List all available skills
        skills = registry.list_skills()

        # Get specific skill metadata
        skill = registry.get_skill("sql-optimization")

        # Load full content when needed
        content = registry.load_skill_content("sql-optimization")
    """

    def __init__(self, config: Optional[SkillConfig] = None, directories: Optional[List[str]] = None):
        """Initialize the skill registry.

        Args:
            config: SkillConfig with directories and settings
            directories: Override directories (for testing)
        """
        self.config = config or SkillConfig()
        self._directories = directories or self.config.directories
        self._skills: Dict[str, SkillMetadata] = {}
        self._scanned = False
        self._lock = threading.Lock()

    def scan_directories(self) -> None:
        """Scan all configured directories for SKILL.md files.

        Discovers skills by finding SKILL.md files, parsing their
        YAML frontmatter, and storing metadata. Warns on duplicate
        skill names if configured.

        Thread-safe - uses lock to prevent concurrent scanning.
        """
        with self._lock:
            if self._scanned:
                logger.debug("Skill directories already scanned, skipping")
                return

            seen_names: Dict[str, Path] = {}
            total_found = 0

            for directory in self._directories:
                dir_path = Path(directory).expanduser().resolve()

                if not dir_path.exists():
                    logger.debug(f"Skills directory not found: {dir_path}")
                    continue

                if not dir_path.is_dir():
                    logger.warning(f"Skills path is not a directory: {dir_path}")
                    continue

                logger.debug(f"Scanning skills directory: {dir_path}")

                # Find all SKILL.md files (case-insensitive)
                for skill_file in dir_path.rglob("SKILL.md"):
                    try:
                        metadata = self._parse_skill_file(skill_file)
                        if metadata:
                            total_found += 1

                            # Check for duplicates
                            if metadata.name in seen_names:
                                if self.config.warn_duplicates:
                                    logger.warning(
                                        f"Duplicate skill name '{metadata.name}': "
                                        f"found at {skill_file.parent}, "
                                        f"already exists at {seen_names[metadata.name]}"
                                    )
                                continue

                            seen_names[metadata.name] = skill_file.parent
                            self._skills[metadata.name] = metadata
                            logger.debug(f"Discovered skill: {metadata.name} at {skill_file.parent}")

                    except Exception as e:
                        logger.error(f"Failed to parse skill file {skill_file}: {e}")

            self._scanned = True
            logger.info(f"Skill registry: discovered {len(self._skills)} skills from {total_found} SKILL.md files")

    def _parse_skill_file(self, path: Path) -> Optional[SkillMetadata]:
        """Parse a SKILL.md file and extract metadata from frontmatter.

        Args:
            path: Path to the SKILL.md file

        Returns:
            SkillMetadata if successfully parsed, None otherwise
        """
        try:
            content = path.read_text(encoding="utf-8")

            # Extract YAML frontmatter
            match = FRONTMATTER_PATTERN.match(content)
            if not match:
                logger.warning(f"No valid frontmatter found in {path}")
                return None

            frontmatter_text = match.group(1)
            frontmatter = yaml.safe_load(frontmatter_text)

            if not frontmatter:
                logger.warning(f"Empty frontmatter in {path}")
                return None

            # Create metadata from frontmatter
            metadata = SkillMetadata.from_frontmatter(frontmatter, path.parent)
            return metadata

        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {path}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid skill metadata in {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing {path}: {e}")
            return None

    def get_skill(self, name: str) -> Optional[SkillMetadata]:
        """Get skill metadata by name.

        Thread-safe - locks during access to prevent concurrent modification.

        Args:
            name: Skill name (from frontmatter)

        Returns:
            SkillMetadata if found, None otherwise
        """
        if not self._scanned:
            self.scan_directories()

        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> List[SkillMetadata]:
        """List all discovered skills.

        Thread-safe - returns a snapshot of skills list.

        Returns:
            List of SkillMetadata for all skills
        """
        if not self._scanned:
            self.scan_directories()

        with self._lock:
            return list(self._skills.values())

    def load_skill_content(self, name: str) -> Optional[str]:
        """Load full SKILL.md content for a skill.

        Lazy loads the content on first request and caches it.

        Args:
            name: Skill name

        Returns:
            Full SKILL.md content if found, None otherwise
        """
        skill = self.get_skill(name)
        if not skill:
            logger.warning(f"Skill not found: {name}")
            return None

        # Return cached content if available
        if skill.content is not None:
            return skill.content

        # Load content from file
        skill_file = skill.location / "SKILL.md"
        try:
            content = skill_file.read_text(encoding="utf-8")
            skill.content = content
            logger.debug(f"Loaded content for skill: {name}")
            return content
        except Exception as e:
            logger.error(f"Failed to load content for skill {name}: {e}")
            return None

    def refresh(self) -> None:
        """Re-scan directories for skills.

        Clears cached skills and re-scans all directories.
        Useful when skills are added/removed at runtime.
        """
        with self._lock:
            self._skills.clear()
            self._scanned = False

        self.scan_directories()
        logger.info(f"Skill registry refreshed: {len(self._skills)} skills")

    def skill_exists(self, name: str) -> bool:
        """Check if a skill exists.

        Thread-safe - locks during access.

        Args:
            name: Skill name

        Returns:
            True if skill exists
        """
        if not self._scanned:
            self.scan_directories()

        with self._lock:
            return name in self._skills

    def get_skills_by_tag(self, tag: str) -> List[SkillMetadata]:
        """Get all skills with a specific tag.

        Thread-safe - returns a snapshot of filtered skills.

        Args:
            tag: Tag to filter by

        Returns:
            List of skills with the given tag
        """
        if not self._scanned:
            self.scan_directories()

        with self._lock:
            return [skill for skill in self._skills.values() if tag in skill.tags]

    def get_skill_count(self) -> int:
        """Get total number of discovered skills.

        Thread-safe - locks during access.

        Returns:
            Number of skills in registry
        """
        if not self._scanned:
            self.scan_directories()

        with self._lock:
            return len(self._skills)
