# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OpenAI Agents SDK tracing integration used by Datus observability."""

from __future__ import annotations

from typing import Any, cast


class DatusOpenAIAgentsInstrumentor:
    """Install OpenInference tracing while merging the first agent span into the trace root."""

    def instrument(self, *, tracer_provider: Any, config: Any) -> None:
        from agents import set_trace_processors
        from openinference.instrumentation import OITracer
        from openinference.instrumentation.openai_agents.version import __version__
        from opentelemetry import trace as trace_api
        from opentelemetry.trace import Tracer

        tracer = OITracer(
            trace_api.get_tracer("openinference.instrumentation.openai_agents", __version__, tracer_provider),
            config=config,
        )
        set_trace_processors([DatusOpenInferenceTracingProcessor(cast(Tracer, tracer))])

    def uninstrument(self) -> None:
        return None


def instrument_openai_agents(*, tracer_provider: Any, config: Any) -> DatusOpenAIAgentsInstrumentor:
    instrumentor = DatusOpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, config=config)
    return instrumentor


try:
    from openinference.instrumentation.openai_agents import _processor as _oi_processor
except Exception:  # pragma: no cover - import availability is checked during adapter setup.
    _oi_processor = None  # type: ignore[assignment]


_OpenInferenceTracingProcessorBase = (
    object if _oi_processor is None else _oi_processor.OpenInferenceTracingProcessor  # type: ignore[union-attr]
)


class DatusOpenInferenceTracingProcessor(_OpenInferenceTracingProcessorBase):  # type: ignore[misc]
    """OpenInference processor that avoids a duplicate root agent in Langfuse.

    The upstream processor emits one OpenTelemetry span for the Agents SDK trace and
    another span for the first AgentSpanData. Langfuse renders both as agent-like
    observations, which creates a duplicate-looking ``agent/chat -> chat`` tree.
    Datus uses the trace root as the first agent span and nests model/tool spans
    directly underneath it.
    """

    def __init__(self, tracer: Any) -> None:
        if _oi_processor is None:
            raise RuntimeError("openinference.instrumentation.openai_agents is required")
        super().__init__(tracer)
        self._merged_root_agent_span_ids: set[str] = set()
        self._merged_root_trace_ids: set[str] = set()

    def on_trace_end(self, trace: Any) -> None:
        self._merged_root_trace_ids.discard(trace.trace_id)
        super().on_trace_end(trace)

    def on_span_start(self, span: Any) -> None:
        if self._is_mergeable_root_agent_span(span):
            root_span = self._root_spans.get(span.trace_id)
            if root_span is None:
                super().on_span_start(span)
                return
            root_span.set_attribute(_oi_processor.LLM_SYSTEM, _oi_processor.OpenInferenceLLMSystemValues.OPENAI.value)
            self._otel_spans[span.span_id] = root_span
            self._tokens[span.span_id] = _oi_processor.attach(_oi_processor.set_span_in_context(root_span))
            self._merged_root_agent_span_ids.add(span.span_id)
            return

        super().on_span_start(span)

    def on_span_end(self, span: Any) -> None:
        if span.span_id not in self._merged_root_agent_span_ids:
            super().on_span_end(span)
            return

        self._merged_root_agent_span_ids.discard(span.span_id)
        if token := self._tokens.pop(span.span_id, None):
            _oi_processor.detach(token)  # type: ignore[arg-type]
        root_span = self._otel_spans.pop(span.span_id, None)
        if root_span is None:
            return
        data = span.span_data
        if isinstance(data, _oi_processor.AgentSpanData):
            root_span.set_attribute(_oi_processor.GRAPH_NODE_ID, data.name)
            key = f"{data.name}:{span.trace_id}"
            if parent_node := self._reverse_handoffs_dict.pop(key, None):
                root_span.set_attribute(_oi_processor.GRAPH_NODE_PARENT_ID, parent_node)

    def shutdown(self) -> None:
        self._merged_root_agent_span_ids.clear()
        self._merged_root_trace_ids.clear()
        super().shutdown()

    def _is_mergeable_root_agent_span(self, span: Any) -> bool:
        if span.parent_id is not None:
            return False
        if span.trace_id not in self._root_spans:
            return False
        if span.trace_id in self._merged_root_trace_ids:
            return False
        if not isinstance(span.span_data, _oi_processor.AgentSpanData):
            return False
        self._merged_root_trace_ids.add(span.trace_id)
        return True
