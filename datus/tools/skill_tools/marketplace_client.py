# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
HTTP client for Town Skills Marketplace.

Talks to the Town Backend's skills API to search, download, and publish skills.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

from datus.tools.skill_tools.skill_bundle import (
    create_bundle,
    extract_bundle_from_bytes,
)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

logger = logging.getLogger(__name__)


class SkillMarketplaceClient:
    """HTTP client for Town Skills Marketplace."""

    def __init__(self, base_url: str = "http://localhost:9000", token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = 60.0

        if token:
            self.token = token
        else:
            from datus.tools.skill_tools.marketplace_auth import load_token

            self.token = load_token(self.base_url)

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/skills{path}"

    def _auth_headers(self) -> Dict[str, str]:
        """Return Authorization header if a token is available."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _handle_response(self, resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Marketplace error ({resp.status_code}): {detail}")
        if resp.status_code == 204:
            return None
        return resp.json()

    def search(self, query: str = "", tag: str = "") -> List[Dict]:
        """Search skills in the marketplace."""
        params = {}
        if query:
            params["q"] = query
        if tag:
            params["tag"] = tag
        with httpx.Client(timeout=self.timeout, headers=self._auth_headers()) as client:
            resp = client.get(self._url("/search"), params=params)
            data = self._handle_response(resp)
            return data.get("skills", [])

    def list_skills(self, promoted: Optional[bool] = None) -> List[Dict]:
        """List all skills, optionally filtering by promoted status."""
        params = {}
        if promoted is not None:
            params["promoted"] = str(promoted).lower()
        with httpx.Client(timeout=self.timeout, headers=self._auth_headers()) as client:
            resp = client.get(self._url(""), params=params)
            data = self._handle_response(resp)
            return data.get("skills", [])

    def get_skill_info(self, name: str) -> Dict:
        """Get detailed skill info including version history."""
        with httpx.Client(timeout=self.timeout, headers=self._auth_headers()) as client:
            resp = client.get(self._url(f"/{name}"))
            return self._handle_response(resp)

    def get_version(self, name: str, version: str) -> Dict:
        """Get a specific version of a skill."""
        with httpx.Client(timeout=self.timeout, headers=self._auth_headers()) as client:
            resp = client.get(self._url(f"/{name}/{version}"))
            return self._handle_response(resp)

    def download_bundle(self, name: str, version: str, dest_dir: Path) -> Path:
        """Download and extract a skill bundle from the marketplace.

        Args:
            name: Skill name
            version: Version to download (use "latest" to resolve)
            dest_dir: Directory to extract into (e.g. ~/.datus/skills/{name})

        Returns:
            Path to the extracted skill directory
        """
        # Resolve "latest" version
        if version == "latest":
            info = self.get_skill_info(name)
            version = info.get("latest_version", version)

        with httpx.Client(timeout=self.timeout, headers=self._auth_headers()) as client:
            resp = client.get(self._url(f"/{name}/{version}/download"))
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"Download failed ({resp.status_code}): {detail}")

            data = resp.content
            dest = Path(dest_dir) / name
            extract_bundle_from_bytes(data, dest)
            logger.info(f"Downloaded and extracted {name}@{version} to {dest}")
            return dest

    def publish_skill(self, skill_dir: Path, owner: str = "") -> Dict:
        """Publish a local skill directory to the marketplace.

        Reads SKILL.md frontmatter, creates a bundle, and uploads both
        metadata and bundle to the Town backend.

        Args:
            skill_dir: Path to local skill directory containing SKILL.md
            owner: Optional owner name

        Returns:
            Published skill info dict
        """
        skill_dir = Path(skill_dir).resolve()
        skill_md_path = skill_dir / "SKILL.md"

        if not skill_md_path.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

        # Parse frontmatter
        content = skill_md_path.read_text(encoding="utf-8")
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError("No valid YAML frontmatter in SKILL.md")

        frontmatter = yaml.safe_load(match.group(1))
        name = frontmatter.get("name")
        version = frontmatter.get("version", "0.1.0")

        if not name:
            raise ValueError("SKILL.md missing 'name' field")

        # Build publish body
        body = {
            "name": name,
            "version": str(version),
            "description": frontmatter.get("description", ""),
            "owner": owner or frontmatter.get("owner"),
            "tags": frontmatter.get("tags", []),
            "license": frontmatter.get("license"),
            "compatibility": frontmatter.get("compatibility"),
            "skill_md_content": content,
            "allowed_commands": frontmatter.get("allowed_commands", []),
            "changelog": frontmatter.get("changelog"),
        }

        with httpx.Client(timeout=self.timeout, headers=self._auth_headers()) as client:
            # Step 1: Publish metadata
            resp = client.post(self._url(""), json=body)
            skill_data = self._handle_response(resp)

            # Step 2: Upload bundle if there are files
            try:
                bundle_data = create_bundle(skill_dir)
                files = {"file": (f"{name}-{version}.tar.gz", bundle_data, "application/gzip")}
                resp = client.post(self._url(f"/{name}/{version}/upload"), files=files)
                self._handle_response(resp)
                logger.info(f"Uploaded bundle for {name}@{version}")
            except FileNotFoundError:
                logger.debug("No bundle to upload (SKILL.md only)")

            return skill_data
