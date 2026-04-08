"""Unit tests for datus.api.auth.loader.load_auth_provider."""

import sys
import types

import pytest

from datus.api.auth.loader import load_auth_provider
from datus.api.auth.no_auth_provider import NoAuthProvider
from datus.utils.exceptions import DatusException


class _DummyAuth:
    """Implements the AuthProvider Protocol structurally."""

    def __init__(self, issuer: str = "", audience: str = ""):
        self.issuer = issuer
        self.audience = audience

    async def authenticate(self, request):  # pragma: no cover - not exercised here
        return None

    def on_evict(self, callback) -> None:  # pragma: no cover
        return None


class _NotAnAuth:
    def __init__(self):
        pass


@pytest.fixture
def fake_module():
    mod_name = "_datus_test_fake_auth_mod"
    mod = types.ModuleType(mod_name)
    mod.DummyAuth = _DummyAuth
    mod.NotAnAuth = _NotAnAuth
    sys.modules[mod_name] = mod
    yield mod_name
    sys.modules.pop(mod_name, None)


def test_default_when_empty():
    provider = load_auth_provider(None, namespace="default")
    assert isinstance(provider, NoAuthProvider)

    provider = load_auth_provider({}, namespace="default")
    assert isinstance(provider, NoAuthProvider)

    provider = load_auth_provider({"auth_provider": {}}, namespace="default")
    assert isinstance(provider, NoAuthProvider)


def test_load_custom_with_kwargs(fake_module):
    cfg = {
        "auth_provider": {
            "class": f"{fake_module}.DummyAuth",
            "kwargs": {"issuer": "iss", "audience": "aud"},
        }
    }
    provider = load_auth_provider(cfg, namespace="ns")
    assert isinstance(provider, _DummyAuth)
    assert provider.issuer == "iss"
    assert provider.audience == "aud"


def test_load_custom_colon_separator(fake_module):
    cfg = {"auth_provider": {"class": f"{fake_module}:DummyAuth"}}
    provider = load_auth_provider(cfg, namespace="ns")
    assert isinstance(provider, _DummyAuth)


def test_invalid_class_path():
    with pytest.raises(DatusException):
        load_auth_provider({"auth_provider": {"class": "NoModule"}}, namespace="ns")


def test_missing_module():
    with pytest.raises(DatusException):
        load_auth_provider({"auth_provider": {"class": "nonexistent_pkg_xyz.SomeClass"}}, namespace="ns")


def test_missing_class(fake_module):
    with pytest.raises(DatusException):
        load_auth_provider({"auth_provider": {"class": f"{fake_module}.MissingClass"}}, namespace="ns")


def test_not_implementing_protocol(fake_module):
    with pytest.raises(DatusException):
        load_auth_provider({"auth_provider": {"class": f"{fake_module}.NotAnAuth"}}, namespace="ns")


def test_constructor_failure(fake_module):
    with pytest.raises(DatusException):
        load_auth_provider(
            {"auth_provider": {"class": f"{fake_module}.DummyAuth", "kwargs": {"bad": 1}}},
            namespace="ns",
        )
