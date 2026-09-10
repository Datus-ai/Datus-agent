# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Per-call evidence shared by SDK adapters, native models, logs and traces.

The scope covers an adapter-visible request, not hidden SDK/remote retries.
Provider IDs are copied verbatim from headers or documented SDK fields.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import wraps
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import structlog

from datus.observability.manager import get_observability_manager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
_CURRENT_CALL: ContextVar[ModelCall | None] = ContextVar("datus_model_call", default=None)
_PHASE: ContextVar[str] = ContextVar("datus_model_phase", default="task")
_MAX_TOOL_BYTES = 1024 * 1024
_MAX_TOOL_ATTRIBUTES = 1536
_KNOWN_PROVIDER_HOSTS = {"api.openai.com", "api.anthropic.com", "api.deepseek.com", "chatgpt.com"}


@dataclass
class _RunEvidence:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    tools: dict[Any, tuple[str, dict[str, Any], Any]] = field(default_factory=dict)
    capability_events: list[str] = field(default_factory=list)
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    compact_id: str | None = None


_RUN: ContextVar[_RunEvidence | None] = ContextVar("datus_model_run", default=None)


@contextmanager
def observation_run(*, run_id: str | None = None):
    state = _RunEvidence(run_id=run_id) if run_id else _RunEvidence()
    token = _RUN.set(state)
    log_tokens = structlog.contextvars.bind_contextvars(run_id=state.run_id)
    try:
        yield state
    finally:
        structlog.contextvars.reset_contextvars(**log_tokens)
        _RUN.reset(token)


def capability_event(event: str, *, reason: str, **fields: Any) -> str:
    event_id = uuid4().hex
    if state := _RUN.get():
        state.capability_events.append(event_id)
        del state.capability_events[:-16]
    logger.warning(event, cause_event_id=event_id, reason=reason, **fields)
    return event_id


def consume_tool_call_identity(tool_call_id: str | None) -> dict[str, Any]:
    """Release a completed call while preserving other in-flight calls."""
    state = _RUN.get()
    return state.tool_calls.pop(tool_call_id, {}) if state is not None else {}


def remember_compact(compact_id: str) -> None:
    if state := _RUN.get():
        state.compact_id = compact_id


def current_model_call() -> ModelCall | None:
    return _CURRENT_CALL.get()


@contextmanager
def model_phase(phase: str):
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


def observe_model_phase(phase: str):
    def decorate(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            with model_phase(phase):
                return await func(*args, **kwargs)

        return wrapped

    return decorate


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _best_effort(field_name: str):
    """Optional response metadata must never change a model's return behavior."""

    def decorate(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "llm.capture_failed",
                    model_call_id=self.model_call_id,
                    field=field_name,
                    error_type=type(exc).__name__,
                )

        return wrapped

    return decorate


def _setting(value: Any) -> Any:
    # OpenAI/Anthropic NOT_GIVEN and omit are not actual request values.
    if value is None or type(value).__name__ in {"NotGiven", "Omit"}:
        return "omitted"
    return _plain(value)


def _tool_name(tool: Mapping[str, Any]) -> str:
    definition = tool.get("function", tool)
    return str(definition.get("name") or tool.get("type") or "unknown")


class ModelCall:
    def __init__(self, *, model: str, model_impl: str, protocol: str, endpoint: Any = None):
        self.model_call_id = uuid4().hex
        host = urlparse(str(endpoint or "")).hostname
        self.fields: dict[str, Any] = {
            "model_call_id": self.model_call_id,
            "model": str(model),
            "model_impl": model_impl,
            "protocol": protocol,
            "phase": _PHASE.get(),
            "endpoint_host": host or "provider_default",
            "request_id_coverage": "adapter_visible_response",
            "request_id_status": "not_observable",
            "request_id_issuer": "provider" if host in _KNOWN_PROVIDER_HOSTS else "unknown",
        }
        self._id_headers = dict(get_observability_manager().remote_id_headers.get(host or "", {}))
        if issuer := self._id_headers.get("issuer"):
            self.fields["request_id_issuer"] = issuer
        self.tools: list[dict[str, Any]] = []
        self._span = None
        self._start = time.monotonic()
        self._finished = False
        self._received = False
        self._tool_state = "not_observable"
        self._pending_span_end = None
        from datus.utils.trace_context import get_trace_context

        trace_ctx = get_trace_context()
        self._agent_name = trace_ctx.name if trace_ctx else "agent"
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        # Native calls without SDK agent spans still need task-local comparisons.
        self._agent_key: Any = (self._agent_name, task)
        if trace_ctx:
            if trace_ctx.session_id:
                self.fields["session_id"] = trace_ctx.session_id
            for key in ("run_id", "turn_id", "task_id", "node_id", "run_attempt"):
                if value := trace_ctx.metadata.get(key):
                    self.fields[key] = value
        if state := _RUN.get():
            self.fields.setdefault("run_id", state.run_id)
            if state.compact_id:
                self.fields["compact_id"] = state.compact_id
        from datus.observability.compaction import current_compact_id

        if compact_id := current_compact_id():
            self.fields["compact_id"] = compact_id

    def __enter__(self):
        self._token = _CURRENT_CALL.set(self)
        self._log_tokens = structlog.contextvars.bind_contextvars(model_call_id=self.model_call_id, phase=_PHASE.get())
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.finish(exc)
        finally:
            structlog.contextvars.reset_contextvars(**self._log_tokens)
            _CURRENT_CALL.reset(self._token)

    def bind_span(self, span: Any) -> None:
        self._span = span
        if span is not None:
            ctx = span.get_span_context()
            if ctx.is_valid:
                self.fields.update(trace_id=f"{ctx.trace_id:032x}", span_id=f"{ctx.span_id:016x}")

    def bind_sdk_span(self, span: Any) -> None:
        # SDK span objects support extension attributes; the processor reads the
        # object synchronously at span end, never an export-thread ContextVar.
        span._datus_model_call = self

    def bind_agent(self, span: Any) -> None:
        # Names can be reused by parallel child runs. NoOpSpan objects also have
        # distinct identities, so this works with SDK tracing disabled.
        self._agent_key = span
        self.fields["agent_name"] = span.span_data.name

    def request(self, params: Mapping[str, Any], *, boundary: str = "sdk_request") -> None:
        try:
            raw_tools = params.get("tools")
            self.tools = (
                copy.deepcopy([_plain(tool) for tool in raw_tools]) if isinstance(raw_tools, (list, tuple)) else []
            )
            self.fields.update(
                tools_count=len(self.tools),
                tool_names=[_tool_name(tool) for tool in self.tools],
                tool_choice=_setting(params.get("tool_choice")),
                parallel_tool_calls=_setting(params.get("parallel_tool_calls")),
                capture_boundary=boundary,
            )
            self._tool_state = "complete"
            self._compare_tools()
            logger.info("llm.started", **self._summary())
            logger.debug("llm.tools", **self.fields)
        except Exception as exc:
            self._tool_state = "failed"
            logger.warning(
                "llm.capture_failed", model_call_id=self.model_call_id, field="tools", error_type=type(exc).__name__
            )

    def _compare_tools(self) -> None:
        state = _RUN.get()
        if state is None or self.fields["phase"] != "task":
            return
        if state.capability_events:
            self.fields["capability_event_ids"] = list(state.capability_events)
        # Agent spans distinguish nested agents that share a trace context.
        key = self._agent_key
        definitions = {_tool_name(tool): tool for tool in self.tools}
        previous = state.tools.get(key)
        if previous is None:
            logger.info(
                "tools.available",
                agent_name=self.fields.get("agent_name", self._agent_name),
                tools_count=len(definitions),
                model_call_id=self.model_call_id,
            )
        else:
            previous_call, before, previous_choice = previous
            added = sorted(definitions.keys() - before.keys())
            removed = sorted(before.keys() - definitions.keys())
            changed = sorted(name for name in before.keys() & definitions.keys() if before[name] != definitions[name])
            choice_changed = previous_choice != self.fields["tool_choice"]
            if added or removed or changed or choice_changed:
                self.fields.update(
                    previous_model_call_id=previous_call,
                    tools_added=added,
                    tools_removed=removed,
                    tools_schema_changed=changed,
                    tool_choice_changed=choice_changed,
                    tools_change_reason="unknown",
                )
                # Nearby events are evidence to inspect, not proof of causality.
                self.fields["capability_event_ids"] = list(state.capability_events)
                emit = logger.warning if removed else logger.info
                emit("tools.changed", **self.fields)
        state.tools[key] = self.model_call_id, definitions, self.fields["tool_choice"]

    def response(self, response: Any, *, streaming: bool = False) -> None:
        """Accept only raw headers and documented ID properties, never body.id."""
        try:
            headers = getattr(response, "_response_headers", None)
            if not isinstance(headers, Mapping):
                raw = getattr(response, "response", None)
                headers = getattr(raw, "headers", None)
            if not isinstance(headers, Mapping):
                headers = getattr(response, "headers", None)
            source = None
            request_id = None
            if isinstance(headers, Mapping):
                normalized = {str(k).lower(): v for k, v in headers.items()}
                selected = self._id_headers.get("request_id_header")
                for header in [selected] if selected else ("x-request-id", "request-id"):
                    if isinstance(normalized.get(header), str) and normalized[header]:
                        request_id, source = normalized[header], f"header:{header}"
                        break
                if "request_id" not in self.fields:
                    self.fields["request_id_status"] = "absent"
                for option, field_name in (
                    ("trace_id_header", "provider_trace_id"),
                    ("gateway_request_id_header", "gateway_request_id"),
                ):
                    header = self._id_headers.get(option)
                    value = normalized.get(header) if header else None
                    if isinstance(value, str) and value:
                        if option == "trace_id_header" and self.fields["request_id_issuer"] != "provider":
                            field_name = "remote_trace_id"
                        self.fields[field_name] = value
                        self.fields[f"{field_name}_source"] = f"header:{header}"
            for attr in ("_request_id", "request_id"):
                value = getattr(response, attr, None)
                if (
                    request_id is None
                    and not self._id_headers.get("request_id_header")
                    and isinstance(value, str)
                    and value
                ):
                    request_id, source = value, f"sdk:{attr}"
            if request_id is not None:
                self.fields.update(request_id=request_id, request_id_source=source, request_id_status="captured")
                identity = {"provider": "provider_request_id", "gateway": "gateway_request_id"}.get(
                    self.fields["request_id_issuer"], "remote_request_id"
                )
                self.fields[identity] = request_id
            has_response = (
                not isinstance(response, BaseException) or isinstance(headers, Mapping) or request_id is not None
            )
            if has_response and not self._received:
                self.fields["response_headers_ms"] = round((time.monotonic() - self._start) * 1000, 2)
                if streaming:
                    logger.info("llm.response_received", **self._summary())
                self._received = True
            self.usage(getattr(response, "usage", None))
            self.export_to(self._span)
        except Exception as exc:
            self.fields["request_id_status"] = "capture_failed"
            logger.warning(
                "llm.capture_failed",
                model_call_id=self.model_call_id,
                field="request_id",
                error_type=type(exc).__name__,
            )

    @_best_effort("usage")
    def usage(self, usage: Any) -> None:
        usage = _plain(usage)
        if isinstance(usage, Mapping):
            for target, aliases in {
                "input_tokens": ("input_tokens", "prompt_tokens"),
                "output_tokens": ("output_tokens", "completion_tokens"),
                "total_tokens": ("total_tokens",),
            }.items():
                for key in aliases:
                    value = usage.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.fields[target] = value
                        break

    @_best_effort("tool_calls")
    def record_tool_calls(self, items: Any) -> None:
        if not isinstance(items, (list, tuple)):
            return
        state = _RUN.get()
        count = 0
        for item in items:
            item = _plain(item)
            if not isinstance(item, Mapping) or item.get("type") not in {"function_call", "tool_use"}:
                continue
            count += 1
            tool_call_id = item.get("call_id") or item.get("id")
            if state is not None and isinstance(tool_call_id, str):
                state.tool_calls[tool_call_id] = {
                    key: self.fields[key]
                    for key in ("model_call_id", "request_id", "session_id", "run_id")
                    if key in self.fields
                }
        self.fields["tool_calls_count"] = count

    def stream_event(self) -> None:
        if "first_event_ms" not in self.fields:
            self.fields["first_event_ms"] = round((time.monotonic() - self._start) * 1000, 2)

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        if error is not None:
            self.response(error)
            try:
                error._datus_model_call_id = self.model_call_id
            except (AttributeError, TypeError):
                pass
        cancelled = (
            isinstance(error, (asyncio.CancelledError, GeneratorExit)) or type(error).__name__ == "ExecutionInterrupted"
        )
        self.fields.update(
            status="cancelled" if cancelled else "error" if error else "success",
            duration_ms=round((time.monotonic() - self._start) * 1000, 2),
        )
        if error:
            self.fields["error_type"] = type(error).__name__
            self.fields["failure_stage"] = "after_response" if self._received else "before_response"
        self.export_to(self._span)
        if self._pending_span_end is not None:
            span, args, kwargs = self._pending_span_end
            self._pending_span_end = None
            span.end(*args, **kwargs)
        logger.info("llm.finished", **self._summary())

    def end_sdk_span(self, span: Any, *args: Any, **kwargs: Any) -> None:
        # The SDK exits its generation before the adapter can catch a stream or
        # conversion error. Defer only the OTel end until that same call exits.
        if self._finished:
            self.export_to(span)
            span.end(*args, **kwargs)
        else:
            self._pending_span_end = span, args, kwargs

    def output(self, value: Any) -> None:
        self.usage(getattr(value, "usage", None))
        self.record_tool_calls(getattr(value, "content", None))
        manager = get_observability_manager()
        if self._span is not None and self._span.is_recording() and manager.content_enabled("responses"):
            try:
                payload = _plain(value)
                if isinstance(payload, dict):
                    payload = {key: item for key, item in payload.items() if key != "tools"}
                self._span.set_attribute(
                    "output.value", json.dumps(manager.redact(payload), ensure_ascii=False, default=str)
                )
                self._span.set_attribute("output.mime_type", "application/json")
            except Exception:
                logger.warning("llm.capture_failed", model_call_id=self.model_call_id, field="output")

    def _summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.fields.items() if k not in {"tool_names", "tool_choice", "parallel_tool_calls"}}

    def export_to(self, span: Any) -> None:
        if span is None or not span.is_recording():
            return
        try:
            manager = get_observability_manager()
            state = self._tool_state
            count = 0
            definitions = {}
            size = 0
            if state == "complete" and not manager.content_enabled("tool_definitions"):
                state = "disabled"
            if state == "complete":
                for index, tool in enumerate(self.tools):
                    redacted = manager.redact(tool)
                    if redacted != tool:
                        state = "redacted"
                    # Langfuse recognizes `parameters` for flat definitions.
                    # Preserve Anthropic's original schema and expose its alias
                    # only in telemetry; the provider request is unchanged.
                    if "input_schema" in redacted and "parameters" not in redacted:
                        redacted = {**redacted, "parameters": redacted["input_schema"]}
                    definition = json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
                    size += len(definition.encode("utf-8"))
                    if size > _MAX_TOOL_BYTES or index * 3 >= _MAX_TOOL_ATTRIBUTES:
                        state = "truncated"
                        break
                    prefix = f"llm.tools.{index}.tool"
                    definitions[f"{prefix}.json_schema"] = definition
                    span.set_attribute(f"{prefix}.json_schema", definition)
                    span.set_attribute(f"{prefix}.name", _tool_name(redacted))
                    body = redacted.get("function", redacted)
                    if isinstance(body.get("description"), str):
                        span.set_attribute(f"{prefix}.description", body["description"])
                    count += 1
            # Write IDs and counts last so bounded OTel attributes retain them.
            span.set_attribute("datus.llm.tools_capture_state", state)
            span.set_attribute("datus.llm.tools_captured_count", count)
            for key, value in self.fields.items():
                if value is not None:
                    span.set_attribute(
                        f"datus.llm.{key}", json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                    )
            attributes = getattr(span, "attributes", None)
            if isinstance(attributes, Mapping) and count:
                retained = sum(attributes.get(key) == value for key, value in definitions.items())
                if retained != count:
                    span.set_attribute("datus.llm.tools_capture_state", "truncated")
                    span.set_attribute("datus.llm.tools_captured_count", retained)
        except Exception as exc:
            try:
                span.set_attribute("datus.llm.tools_capture_state", "failed")
            except Exception:
                pass  # A broken telemetry span must not fail the model call.
            logger.warning(
                "llm.capture_failed", model_call_id=self.model_call_id, field="span", error_type=type(exc).__name__
            )


@contextmanager
def native_model_call(*, model: str, model_impl: str, protocol: str, endpoint: Any, params: Mapping[str, Any]):
    """Reuse an existing native generation, or fill a direct-call span gap."""
    from agents.tracing import get_current_span

    current = get_current_span()
    call = ModelCall(model=model, model_impl=model_impl, protocol=protocol, endpoint=endpoint)
    manager = get_observability_manager()
    if current is not None and current.span_data.type in {"generation", "response"}:
        from opentelemetry.trace import get_current_span as get_otel_span

        with call:
            call.bind_sdk_span(current)
            call.bind_span(get_otel_span())
            call.request(params)
            yield call
    else:
        attrs = {
            "openinference.span.kind": "LLM",
            "llm.model_name": model,
            "datus.operation": "llm.generate",
            "langfuse.observation.type": "generation",
        }
        if manager.content_enabled("prompts"):
            prompt = {
                key: _plain(params[key])
                for key in ("messages", "input", "system", "instructions")
                if key in params and _setting(params[key]) != "omitted"
            }
            attrs["input.value"] = json.dumps(manager.redact(prompt), default=str)
            attrs["input.mime_type"] = "application/json"
        with manager.span("generation", attrs) as span, call:
            call.bind_span(span)
            call.request(params)
            yield call
