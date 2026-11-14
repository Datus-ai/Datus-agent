"""Fail-closed legacy JWT configuration and token contracts."""

import secrets
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import yaml
from fastapi import HTTPException

from datus.api.legacy_auth import AuthService, load_auth_config


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)


@pytest.fixture
def auth_config(tmp_path):
    path = tmp_path / "auth.yml"
    config = {"clients": {"test-client": secrets.token_urlsafe(32)}, "jwt": {"secret_key": secrets.token_urlsafe(32)}}
    path.write_text(yaml.safe_dump(config))
    return path, config


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "{}",
        "[]",
        "null",
        "secret: [",
        "clients: {}",
        "clients: []",
        "clients: {user: null}",
        "clients: {user: ''}",
        "clients: {user: '   '}",
        "clients: {'': value}",
        "clients: {123: value}",
        "clients: {user: value}\njwt: null",
        "clients: {user: value}\njwt: []",
        "clients: {user: value}\njwt: {secret_key: ''}",
        "clients: {user: value}\njwt: {secret_key: '   '}",
        "clients: {user: value}\njwt: {secret_key: 123}",
    ],
)
def test_invalid_configuration_disables_auth_without_creating_files(tmp_path, contents):
    path = tmp_path / "auth.yml"
    path.write_text(contents)
    service = AuthService(str(path))
    assert not service.available
    assert service.clients == {} and service.jwt_secret is None
    assert path.read_text() == contents
    with pytest.raises(HTTPException) as error:
        service.generate_access_token("datus_client")
    assert error.value.status_code == 503


def test_missing_configuration_stays_disabled(tmp_path):
    path = tmp_path / "missing.yml"
    service = AuthService(str(path))
    assert not service.available and not path.exists()
    for method, args in [
        (service.validate_client_credentials, ("datus_client", "datus_secret_key")),
        (service.validate_token, ("untrusted-token",)),
    ]:
        with pytest.raises(HTTPException) as error:
            method(*args)
        assert error.value.status_code == 503


def test_explicit_path_and_instance_isolation(auth_config, tmp_path):
    path, config = auth_config
    first = AuthService(str(path))
    other = tmp_path / "other.yml"
    other_config = {
        "clients": {"another-client": secrets.token_urlsafe(32)},
        "jwt": {"secret_key": secrets.token_urlsafe(32)},
    }
    other.write_text(yaml.safe_dump(other_config))
    second = AuthService(str(other))
    assert load_auth_config(str(path)) == config
    assert first.available and second.available
    assert first.validate_client_credentials("test-client", config["clients"]["test-client"])
    assert not second.validate_client_credentials("test-client", config["clients"]["test-client"])
    with pytest.raises(HTTPException) as error:
        second.validate_token(first.generate_access_token("test-client")["access_token"])
    assert error.value.status_code == 401


@pytest.mark.parametrize("override", [None, "", "   ", "valid"])
def test_environment_override(auth_config, monkeypatch, override):
    path, config = auth_config
    if override is not None:
        monkeypatch.setenv("JWT_SECRET_KEY", override)
    service = AuthService(str(path))
    assert service.available == (override not in ("", "   "))
    if service.available:
        assert service.jwt_secret == (config["jwt"]["secret_key"] if override is None else override)
    else:
        assert service.jwt_secret is None


def test_environment_secret_still_requires_clients(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    path = tmp_path / "auth.yml"
    path.write_text("jwt: {}")
    assert not AuthService(str(path)).available
    path.write_text("clients: {user: explicit-secret}")
    assert AuthService(str(path)).available


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "none"),
        ("algorithm", "invalid"),
        ("expiration_hours", 0),
        ("expiration_hours", -1),
        ("expiration_hours", True),
        ("expiration_hours", "two"),
        ("expiration_hours", float("nan")),
    ],
)
def test_invalid_jwt_options_disable_auth(auth_config, field, value):
    path, config = auth_config
    config["jwt"][field] = value
    path.write_text(yaml.safe_dump(config))
    assert not AuthService(str(path)).available


def test_credentials_and_token_round_trip(auth_config):
    path, config = auth_config
    service = AuthService(str(path))
    assert service.validate_client_credentials("test-client", config["clients"]["test-client"])
    assert not service.validate_client_credentials("test-client", "wrong")
    assert not service.validate_client_credentials("unknown-client", None)
    assert not service.validate_client_credentials(None, None)
    assert not service.validate_client_credentials("datus_client", "datus_secret_key")
    result = service.generate_access_token("test-client")
    assert result["token_type"] == "Bearer" and result["expires_in"] == 7200
    assert service.validate_token(result["access_token"])["client_id"] == "test-client"


@pytest.mark.parametrize(
    "failure", ["expired", "wrong_signature", "missing_identity", "empty_identity", "invalid_identity"]
)
def test_invalid_tokens_are_rejected(auth_config, failure):
    service = AuthService(str(auth_config[0]))
    payload = {"client_id": "test-client", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    if failure == "expired":
        payload["exp"] = datetime.now(timezone.utc) - timedelta(hours=1)
    elif failure == "missing_identity":
        del payload["client_id"]
    elif failure == "empty_identity":
        payload["client_id"] = "   "
    elif failure == "invalid_identity":
        payload["client_id"] = ["test-client"]
    key = secrets.token_urlsafe(32) if failure == "wrong_signature" else service.jwt_secret
    token = jwt.encode(payload, key, algorithm="HS256")
    with pytest.raises(HTTPException) as error:
        service.validate_token(token)
    assert error.value.status_code == 401


def test_invalid_configuration_never_logs_sensitive_details(tmp_path, caplog):
    path = tmp_path / "auth.yml"
    secret = secrets.token_urlsafe(32)
    path.write_text("clients: [" + secret)
    assert not AuthService(str(path)).available
    assert "Legacy JWT authentication is unavailable" in caplog.text
    assert secret not in caplog.text and str(path) not in caplog.text


@pytest.mark.parametrize("contents", [None, "", "{}", "clients: ["])
def test_importing_api_and_web_with_unavailable_auth_does_not_crash(tmp_path, contents):
    import subprocess
    import sys
    from pathlib import Path

    from datus.api import legacy_auth

    conf = tmp_path / "conf"
    conf.mkdir()
    if contents is not None:
        (conf / "auth_clients.yml").write_text(contents)
    script = """
import sys
from unittest.mock import patch
from datus.utils.path_manager import DatusPathManager
with patch('datus.utils.path_manager.get_path_manager', return_value=DatusPathManager(sys.argv[1])):
    from datus.api.legacy_auth import auth_service
    from datus.api.service import create_app
    from datus.cli.web.chatbot import create_web_app
assert not auth_service.available
assert callable(create_app) and callable(create_web_app)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(legacy_auth.__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_example_configuration_cannot_enable_authentication():
    from pathlib import Path

    from datus.api import legacy_auth

    example = Path(legacy_auth.__file__).parents[2] / "conf/auth_clients.yml.example"
    assert not AuthService(str(example)).available
