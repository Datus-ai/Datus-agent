# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Trusted, request-local observations at the enforced read boundary.

The observer is installed by the host, never by model/tool arguments. ContextVar
propagation also covers asyncio.to_thread. Observers may refuse a read by raising;
the runtime must not turn a missing observation into evidence of enforcement.
No observation is logged, persisted, or exposed in a tool response here.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ReadObservation:
    phase: Literal["before_sql", "after_sql", "metric_compile"]
    agent_config: Any
    policy_context: dict[str, Any]
    datasource: str
    sql: str | None = None
    applied_policy_ids: tuple[str, ...] = ()
    runtime_names: tuple[str, ...] = ()
    datasource_configs: Any = None


_observer: ContextVar[Callable[[ReadObservation], None] | None] = ContextVar("read_observer", default=None)


@contextmanager
def observe_reads(observer: Callable[[ReadObservation], None]) -> Iterator[None]:
    """Scope an in-process observer to one operation, resetting even on failure."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def notify_read(observation: ReadObservation) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(observation)
