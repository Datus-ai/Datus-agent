# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Stable trace reference returned by Datus observability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceReference:
    trace_id: str
    span_id: str | None = None
    run_id: str | None = None
    provider: str | None = None

    def to_metadata(self) -> dict[str, str]:
        metadata = {"trace_id": self.trace_id}
        if self.span_id:
            metadata["trace_span_id"] = self.span_id
        if self.run_id:
            metadata["trace_run_id"] = self.run_id
        if self.provider:
            metadata["trace_provider"] = self.provider
        return metadata
