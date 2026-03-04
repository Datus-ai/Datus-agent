# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for skill marketplace client, auth, and CLI commands."""

import base64
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datus.tools.skill_tools.marketplace_auth import (
    _is_token_expired,
    clear_token,
    load_token,
    save_token,
)
from datus.tools.skill_tools.marketplace_client import SkillMarketplaceClient

SKILL_MD = """---
name: test-skill
description: A test skill
tags: [test]
version: "1.0.0"
---

# Test Skill
"""


class TestSkillMarketplaceClient:
    """Tests for SkillMarketplaceClient with mocked HTTP calls."""

    def test_init_default_url(self):
        client = SkillMarketplaceClient()
        assert client.base_url == "http://localhost:9000"

    def test_init_custom_url(self):
        client = SkillMarketplaceClient("http://custom:8080")
        assert client.base_url == "http://custom:8080"

    def test_url_construction(self):
        client = SkillMarketplaceClient("http://localhost:9000")
        assert client._url("/search") == "http://localhost:9000/api/skills/search"
        assert client._url("") == "http://localhost:9000/api/skills"

    @patch("datus.tools.skill_tools.marketplace_client.httpx.Client")
    def test_search(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "skills": [{"name": "sql-opt", "latest_version": "1.0"}],
            "total": 1,
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        client = SkillMarketplaceClient()
        results = client.search(query="sql")
        assert len(results) == 1
        assert results[0]["name"] == "sql-opt"

    @patch("datus.tools.skill_tools.marketplace_client.httpx.Client")
    def test_list_skills(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"skills": [], "total": 0}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        client = SkillMarketplaceClient()
        results = client.list_skills()
        assert results == []

    @patch("datus.tools.skill_tools.marketplace_client.httpx.Client")
    def test_get_skill_info(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"name": "test", "latest_version": "1.0"}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        client = SkillMarketplaceClient()
        info = client.get_skill_info("test")
        assert info["name"] == "test"

    @patch("datus.tools.skill_tools.marketplace_client.httpx.Client")
    def test_handle_error_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"detail": "Not found"}
        mock_resp.text = "Not found"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        client = SkillMarketplaceClient()
        with pytest.raises(RuntimeError, match="404"):
            client.get_skill_info("nonexistent")

    def test_publish_skill_missing_skill_md(self, tmp_path):
        client = SkillMarketplaceClient()
        with pytest.raises(FileNotFoundError):
            client.publish_skill(tmp_path)

    def test_publish_skill_missing_name_field(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("---\ndescription: test\n---\nBody")
        client = SkillMarketplaceClient()
        with pytest.raises(ValueError, match="name"):
            client.publish_skill(tmp_path)


class TestSkillManagerMarketplace:
    """Tests for SkillManager marketplace methods."""

    def test_search_marketplace_handles_error(self):
        from datus.tools.skill_tools.skill_config import SkillConfig
        from datus.tools.skill_tools.skill_manager import SkillManager

        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        # Use a temp directory to avoid scanning real skill dirs
        manager = SkillManager(config=config, registry=MagicMock())
        manager.registry.list_skills.return_value = []
        manager.registry.get_skill_count.return_value = 0

        results = manager.search_marketplace(query="test")
        assert results == []  # Should return empty on connection error

    def test_install_from_marketplace_handles_error(self):
        from datus.tools.skill_tools.skill_config import SkillConfig
        from datus.tools.skill_tools.skill_manager import SkillManager

        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=MagicMock())
        manager.registry.get_skill_count.return_value = 0

        ok, msg = manager.install_from_marketplace("test")
        assert ok is False
        assert "failed" in msg.lower()

    def test_publish_handles_error(self):
        from datus.tools.skill_tools.skill_config import SkillConfig
        from datus.tools.skill_tools.skill_manager import SkillManager

        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=MagicMock())
        manager.registry.get_skill_count.return_value = 0

        ok, _msg = manager.publish_to_marketplace("/nonexistent/path")
        assert ok is False


class TestSkillConfigMarketplace:
    """Tests for marketplace-related config fields."""

    def test_default_marketplace_url(self):
        from datus.tools.skill_tools.skill_config import SkillConfig

        config = SkillConfig()
        assert config.marketplace_url == "http://localhost:9000"
        assert config.auto_sync is False
        assert config.install_dir == "~/.datus/skills"

    def test_from_dict_marketplace(self):
        from datus.tools.skill_tools.skill_config import SkillConfig

        config = SkillConfig.from_dict(
            {
                "marketplace_url": "http://custom:8080",
                "auto_sync": True,
                "install_dir": "/custom/path",
            }
        )
        assert config.marketplace_url == "http://custom:8080"
        assert config.auto_sync is True
        assert config.install_dir == "/custom/path"


class TestSkillMetadataMarketplace:
    """Tests for marketplace metadata fields on SkillMetadata."""

    def test_new_fields_defaults(self):
        from datus.tools.skill_tools.skill_config import SkillMetadata

        meta = SkillMetadata(name="test", description="test", location=Path("/tmp/test"))
        assert meta.license is None
        assert meta.compatibility is None
        assert meta.source is None
        assert meta.marketplace_version is None

    def test_from_frontmatter_with_license(self):
        from datus.tools.skill_tools.skill_config import SkillMetadata

        fm = {
            "name": "test",
            "description": "test",
            "license": "MIT",
            "compatibility": {"datus": ">=0.2.0"},
        }
        meta = SkillMetadata.from_frontmatter(fm, Path("/tmp/test"))
        assert meta.license == "MIT"
        assert meta.compatibility == {"datus": ">=0.2.0"}

    def test_to_dict_includes_new_fields(self):
        from datus.tools.skill_tools.skill_config import SkillMetadata

        meta = SkillMetadata(
            name="test",
            description="test",
            location=Path("/tmp/test"),
            license="Apache-2.0",
            source="marketplace",
            marketplace_version="1.0.0",
        )
        d = meta.to_dict()
        assert d["license"] == "Apache-2.0"
        assert d["source"] == "marketplace"
        assert d["marketplace_version"] == "1.0.0"


def _make_jwt(exp: int) -> str:
    """Create a fake JWT with the given exp claim."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp, "sub": "test"}).encode()).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


class TestMarketplaceAuth:
    """Tests for marketplace_auth token persistence."""

    def test_save_and_load_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "marketplace_auth.json"
        monkeypatch.setattr("datus.tools.skill_tools.marketplace_auth.AUTH_FILE", auth_file)

        future_exp = int(time.time()) + 3600
        token = _make_jwt(future_exp)

        save_token(token, "http://localhost:9000", "user@test.com")
        loaded = load_token("http://localhost:9000")
        assert loaded == token

    def test_load_token_returns_none_when_missing(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "marketplace_auth.json"
        monkeypatch.setattr("datus.tools.skill_tools.marketplace_auth.AUTH_FILE", auth_file)

        assert load_token("http://localhost:9000") is None

    def test_load_token_returns_none_when_expired(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "marketplace_auth.json"
        monkeypatch.setattr("datus.tools.skill_tools.marketplace_auth.AUTH_FILE", auth_file)

        past_exp = int(time.time()) - 100
        token = _make_jwt(past_exp)

        save_token(token, "http://localhost:9000", "user@test.com")
        assert load_token("http://localhost:9000") is None

    def test_clear_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "marketplace_auth.json"
        monkeypatch.setattr("datus.tools.skill_tools.marketplace_auth.AUTH_FILE", auth_file)

        future_exp = int(time.time()) + 3600
        token = _make_jwt(future_exp)

        save_token(token, "http://localhost:9000", "user@test.com")
        assert clear_token("http://localhost:9000") is True
        assert load_token("http://localhost:9000") is None

    def test_clear_token_returns_false_when_missing(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "marketplace_auth.json"
        monkeypatch.setattr("datus.tools.skill_tools.marketplace_auth.AUTH_FILE", auth_file)

        assert clear_token("http://localhost:9000") is False

    def test_is_token_expired_invalid_format(self):
        assert _is_token_expired("not-a-jwt") is True

    def test_trailing_slash_normalization(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "marketplace_auth.json"
        monkeypatch.setattr("datus.tools.skill_tools.marketplace_auth.AUTH_FILE", auth_file)

        future_exp = int(time.time()) + 3600
        token = _make_jwt(future_exp)

        save_token(token, "http://localhost:9000/", "user@test.com")
        loaded = load_token("http://localhost:9000")
        assert loaded == token


class TestMarketplaceClientAuth:
    """Tests for SkillMarketplaceClient token integration."""

    def test_client_uses_explicit_token(self):
        client = SkillMarketplaceClient(token="my-token")
        assert client.token == "my-token"
        assert client._auth_headers() == {"Authorization": "Bearer my-token"}

    def test_client_no_token_returns_empty_headers(self):
        with patch("datus.tools.skill_tools.marketplace_auth.load_token", return_value=None):
            client = SkillMarketplaceClient()
            assert client.token is None
            assert client._auth_headers() == {}

    @patch("datus.tools.skill_tools.marketplace_client.httpx.Client")
    def test_search_sends_auth_header(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"skills": [], "total": 0}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        client = SkillMarketplaceClient(token="test-jwt")
        client.search(query="sql")

        mock_client_cls.assert_called_once_with(
            timeout=60.0,
            headers={"Authorization": "Bearer test-jwt"},
        )
