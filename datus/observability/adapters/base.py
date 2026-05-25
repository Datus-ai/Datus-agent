# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Adapter interface for external observability backends."""

from __future__ import annotations

from typing import Any, Protocol

from datus.observability.config import ObservabilityAdapterConfig, TracingConfig


class ObservabilityAdapter(Protocol):
    name: str
    capabilities: set[str]

    def setup(self, adapter_config: ObservabilityAdapterConfig, tracing_config: TracingConfig) -> None: ...

    def record_event(self, event: Any) -> None: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...
