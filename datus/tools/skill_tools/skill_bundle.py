# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Bundle utilities for skill marketplace.

Create and extract .tar.gz bundles containing SKILL.md, scripts/, and references/.
"""

import hashlib
import io
import logging
import tarfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Files and directories to include in a bundle
BUNDLE_INCLUDES = ["SKILL.md", "scripts", "references"]


def create_bundle(skill_dir: Path) -> bytes:
    """Create a .tar.gz bundle from a skill directory.

    Includes SKILL.md, scripts/, and references/ directories.

    Args:
        skill_dir: Path to the skill directory containing SKILL.md

    Returns:
        bytes of the .tar.gz archive

    Raises:
        FileNotFoundError: If SKILL.md is not found in skill_dir
    """
    skill_dir = Path(skill_dir).resolve()
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Always include SKILL.md
        tar.add(str(skill_md), arcname="SKILL.md")

        # Include optional directories
        for name in BUNDLE_INCLUDES[1:]:  # skip SKILL.md, already added
            path = skill_dir / name
            if path.exists() and path.is_dir():
                for item in path.rglob("*"):
                    if item.is_file():
                        arcname = str(item.relative_to(skill_dir))
                        tar.add(str(item), arcname=arcname)

    bundle_data = buf.getvalue()
    logger.info(f"Created bundle from {skill_dir}: {len(bundle_data)} bytes")
    return bundle_data


def extract_bundle(bundle_path: Path, dest_dir: Path) -> Path:
    """Extract a skill bundle to a destination directory.

    Args:
        bundle_path: Path to the .tar.gz bundle file
        dest_dir: Destination directory to extract into

    Returns:
        Path to the extracted skill directory (dest_dir)

    Raises:
        FileNotFoundError: If bundle_path does not exist
    """
    bundle_path = Path(bundle_path)
    dest_dir = Path(dest_dir)

    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(bundle_path), mode="r:gz") as tar:
        # Security: filter out absolute paths and path traversal
        members = _safe_members(tar)
        tar.extractall(path=str(dest_dir), members=members)

    logger.info(f"Extracted bundle to {dest_dir}")
    return dest_dir


def extract_bundle_from_bytes(data: bytes, dest_dir: Path) -> Path:
    """Extract a skill bundle from bytes to a destination directory.

    Args:
        data: Bundle bytes (.tar.gz)
        dest_dir: Destination directory

    Returns:
        Path to the extracted skill directory
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO(data)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        members = _safe_members(tar)
        tar.extractall(path=str(dest_dir), members=members)

    logger.info(f"Extracted bundle from bytes to {dest_dir}")
    return dest_dir


def calculate_sha256(data: bytes) -> str:
    """Calculate SHA256 hash for bundle integrity.

    Args:
        data: Bundle bytes

    Returns:
        Hex-encoded SHA256 hash string
    """
    return hashlib.sha256(data).hexdigest()


def list_bundle_contents(bundle_path: Path) -> List[str]:
    """List contents of a bundle without extracting.

    Args:
        bundle_path: Path to the .tar.gz bundle

    Returns:
        List of file names in the bundle
    """
    with tarfile.open(str(bundle_path), mode="r:gz") as tar:
        return tar.getnames()


def _safe_members(tar: tarfile.TarFile) -> List[tarfile.TarInfo]:
    """Filter tar members to prevent path traversal attacks."""
    safe = []
    for member in tar.getmembers():
        # Reject absolute paths
        if member.name.startswith("/") or member.name.startswith(".."):
            logger.warning(f"Skipping unsafe tar member: {member.name}")
            continue
        # Reject path traversal
        if ".." in member.name.split("/"):
            logger.warning(f"Skipping path traversal tar member: {member.name}")
            continue
        # Reject symlinks and hardlinks (CVE-2007-4559)
        if member.issym() or member.islnk():
            logger.warning(f"Skipping symlink/hardlink tar member: {member.name}")
            continue
        safe.append(member)
    return safe
