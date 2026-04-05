"""Tests for datus.api.legacy_auth — authentication service."""

import pytest

from datus.api.legacy_auth import AuthService, load_auth_config


class TestLoadAuthConfig:
    """Tests for load_auth_config."""

    def test_load_with_explicit_path(self, tmp_path):
        """load_auth_config loads from explicit path."""
        import yaml

        config_file = tmp_path / "auth.yml"
        config_file.write_text(yaml.dump({"clients": [{"id": "test", "secret": "pass"}]}))
        result = load_auth_config(str(config_file))
        assert result is not None
        assert "clients" in result

    def test_load_with_nonexistent_path(self):
        """load_auth_config returns defaults for nonexistent path."""
        result = load_auth_config("/nonexistent/auth.yml")
        assert result is not None
        assert "clients" in result

    def test_load_with_invalid_yaml(self, tmp_path):
        """load_auth_config handles invalid YAML gracefully."""
        config_file = tmp_path / "bad_auth.yml"
        config_file.write_text(":\n  - ][")
        result = load_auth_config(str(config_file))
        assert result is not None


class TestAuthService:
    """Tests for AuthService."""

    def test_validate_client_credentials(self):
        """validate_client_credentials returns True for valid credentials."""
        svc = AuthService()
        # Default config should have at least one client
        assert svc.validate_client_credentials("datus_client", "datus_secret_key") is True

    def test_validate_client_invalid(self):
        """validate_client_credentials returns False for invalid credentials."""
        svc = AuthService()
        assert svc.validate_client_credentials("bad", "bad") is False

    def test_validate_token_expired(self):
        """validate_token raises for expired tokens."""
        import time

        import jwt
        from fastapi import HTTPException

        svc = AuthService()
        # Create an already-expired token
        payload = {"client_id": "test", "exp": int(time.time()) - 3600}
        token = jwt.encode(payload, svc.jwt_secret, algorithm=svc.jwt_algorithm)
        with pytest.raises(HTTPException) as exc_info:
            svc.validate_token(token)
        assert exc_info.value.status_code == 401
