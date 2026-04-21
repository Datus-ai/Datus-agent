# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.gateway.channel.registry."""

import pytest

from datus.gateway.channel.base import ChannelAdapter
from datus.gateway.channel.registry import (
    _ADAPTER_TYPES,
    get_adapter_class,
    list_adapters,
    register_adapter,
    register_builtins,
)
from datus.gateway.models import OutboundMessage
from datus.utils.exceptions import DatusException


# Dummy adapter for testing
class _DummyAdapter(ChannelAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_message(self, message: OutboundMessage):
        pass


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean registry for each test."""
    saved = dict(_ADAPTER_TYPES)
    _ADAPTER_TYPES.clear()
    yield
    _ADAPTER_TYPES.clear()
    _ADAPTER_TYPES.update(saved)


class TestRegisterAdapter:
    def test_register_and_get(self):
        register_adapter("dummy", _DummyAdapter)
        assert get_adapter_class("dummy") is _DummyAdapter

    def test_get_unknown_raises(self):
        with pytest.raises(DatusException):
            get_adapter_class("nonexistent")

    def test_list_empty(self):
        assert list_adapters() == []

    def test_list_after_register(self):
        register_adapter("beta", _DummyAdapter)
        register_adapter("alpha", _DummyAdapter)
        assert list_adapters() == ["alpha", "beta"]

    def test_overwrite_registration(self):
        register_adapter("dup", _DummyAdapter)

        class _OtherAdapter(_DummyAdapter):
            pass

        register_adapter("dup", _OtherAdapter)
        assert get_adapter_class("dup") is _OtherAdapter


class TestRegisterBuiltins:
    def test_register_builtins_populates(self):
        register_builtins()
        names = list_adapters()
        assert "feishu" in names
        assert "slack" in names

    def test_register_builtins_idempotent(self):
        register_builtins()
        first = dict(_ADAPTER_TYPES)
        register_builtins()
        second = dict(_ADAPTER_TYPES)
        assert first == second

    def test_builtin_classes_are_channel_adapters(self):
        register_builtins()
        for name in ["feishu", "slack"]:
            cls = get_adapter_class(name)
            assert issubclass(cls, ChannelAdapter)
