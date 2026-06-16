# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Data-access policy extension points.

The open-source package owns the stable configuration shape and provider
loading contract. Concrete policy engines can live in separate packages and be
registered with ``agent.data_access.provider``.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, Dict, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from datus.utils.exceptions import DatusException, ErrorCode


class DataAccessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    provider: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "DataAccessConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise DatusException(ErrorCode.COMMON_FIELD_INVALID, message="agent.data_access must be a mapping")
        if not raw:
            return cls()

        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message="agent.data_access.enabled must be a boolean",
            )

        provider = raw.get("provider")
        provider_name = str(provider).strip() if provider is not None else None
        return cls(
            enabled=enabled,
            provider=provider_name or None,
            raw=deepcopy(raw),
        )


class EnforcementResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    sql: Optional[str] = None
    reason: Optional[str] = None
    applied_policies: list[str] = Field(default_factory=list)


class DataAccessEnforcer(Protocol):
    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: Optional[Dict[str, Any]],
    ) -> EnforcementResult:
        """Return a rewritten SQL statement or a denial."""


class DataAccessProviderError(DatusException):
    """Raised when enabled data-access policy cannot load its provider."""

    def __init__(self, message: str):
        super().__init__(ErrorCode.COMMON_CONFIG_ERROR, message_args={"config_error": message})


class NoopDataAccessEnforcer:
    def __init__(self, config: Optional[DataAccessConfig] = None):
        self.config = config or DataAccessConfig()

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: Optional[Dict[str, Any]],
    ) -> EnforcementResult:
        return EnforcementResult(allowed=True, sql=sql)


def load_data_access_enforcer(config: Optional[DataAccessConfig]) -> DataAccessEnforcer:
    config = config or DataAccessConfig()
    if not config.enabled:
        return NoopDataAccessEnforcer(config)
    if not config.provider:
        raise DataAccessProviderError(
            "agent.data_access.enabled is true but agent.data_access.provider is not configured"
        )

    provider_cls = _load_provider_class(config.provider)
    try:
        provider = provider_cls(config)
    except TypeError as e:
        raise DataAccessProviderError(f"Failed to initialize data-access provider {config.provider!r}: {e}") from e

    enforce_read = getattr(provider, "enforce_read", None)
    if not callable(enforce_read):
        raise DataAccessProviderError(f"Data-access provider {config.provider!r} must implement enforce_read")
    return provider


def _load_provider_class(provider: str) -> type:
    module_name, _, class_name = provider.partition(":")
    if not module_name or not class_name:
        raise DataAccessProviderError(
            "agent.data_access.provider must be a Python path like 'package.module:ProviderClass'"
        )
    try:
        module = importlib.import_module(module_name)
        provider_cls = getattr(module, class_name)
    except Exception as e:
        raise DataAccessProviderError(f"Failed to load data-access provider {provider!r}: {e}") from e
    if not isinstance(provider_cls, type):
        raise DataAccessProviderError(f"Data-access provider {provider!r} must reference a class")
    return provider_cls
