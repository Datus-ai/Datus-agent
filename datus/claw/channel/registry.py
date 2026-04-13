# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Adapter registry with lazy loading for built-in adapters."""

from typing import Dict, List, Type

from datus.claw.channel.base import ChannelAdapter
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_ADAPTER_TYPES: Dict[str, Type[ChannelAdapter]] = {}


def register_adapter(name: str, cls: Type[ChannelAdapter]) -> None:
    """Register an adapter class under *name*."""
    _ADAPTER_TYPES[name] = cls
    logger.debug("Registered channel adapter: %s -> %s", name, cls.__name__)


def get_adapter_class(name: str) -> Type[ChannelAdapter]:
    """Return the adapter class for *name*, raising ``DatusException`` if unknown."""
    if name not in _ADAPTER_TYPES:
        raise DatusException(
            ErrorCode.COMMON_UNSUPPORTED,
            message_args={
                "your_value": name,
                "field_name": "channel adapter",
            },
        )
    return _ADAPTER_TYPES[name]


def list_adapters() -> List[str]:
    """Return sorted list of registered adapter names."""
    return sorted(_ADAPTER_TYPES.keys())


def register_builtins() -> None:
    """Lazily import and register built-in adapters."""
    from datus.claw.adapters.feishu import FeishuAdapter
    from datus.claw.adapters.slack import SlackAdapter

    for name, cls in [
        ("feishu", FeishuAdapter),
        ("slack", SlackAdapter),
    ]:
        if name not in _ADAPTER_TYPES:
            register_adapter(name, cls)
