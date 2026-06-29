"""Metric-retrieval hook for hosts that embed datus-agent.

``MetricRAG.search_metrics`` emits a :class:`MetricRetrievalEvent` after every
search so a host (e.g. a SaaS wrapper) can count how often each Semantic Hub
metric is consumed by chatbot / NL2SQL. The hook is optional and
**fire-and-forget**: search never awaits it and swallows its errors, so a slow
or failing host can never break retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class MetricRetrievalEvent:
    """One metric search and the metrics it returned.

    ``metrics`` carries ``{"id", "name", "uid"}`` per hit; ``uid`` is the
    Semantic Hub stable node id and is empty for metrics not governed by the Hub
    (the host should ignore those when counting Hub consumption).
    """

    query_text: str
    metrics: List[Dict[str, Any]]
    datasource_id: Optional[str] = None
    project_name: Optional[str] = None
    sub_agent_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MetricRetrievalHook(Protocol):
    """Extension contract for counting metric retrieval consumption."""

    def on_retrieval(self, event: MetricRetrievalEvent) -> None:
        """Record a retrieval. Called fire-and-forget; must not raise."""
        ...


_metric_retrieval_hook: Optional[MetricRetrievalHook] = None


def set_metric_retrieval_hook(hook: Optional[MetricRetrievalHook]) -> None:
    """Register (or clear, with ``None``) the active metric-retrieval hook."""
    global _metric_retrieval_hook
    _metric_retrieval_hook = hook


def get_metric_retrieval_hook() -> Optional[MetricRetrievalHook]:
    """Return the active metric-retrieval hook, or ``None`` when unregistered."""
    return _metric_retrieval_hook


# Convenience factory for hosts that prefer a plain callable over a class.
RetrievalFn = Callable[[MetricRetrievalEvent], None]


def make_metric_retrieval_hook(on_retrieval: RetrievalFn) -> MetricRetrievalHook:
    """Build a :class:`MetricRetrievalHook` from a single callable."""

    class _LambdaHook:
        def on_retrieval(self, event: MetricRetrievalEvent) -> None:
            on_retrieval(event)

    return _LambdaHook()
