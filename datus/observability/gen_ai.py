# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Export captured OpenInference messages using the OTel GenAI convention too."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def add_gen_ai_attributes(span: Any) -> None:
    """Bridge already-masked attributes without consulting raw model payloads."""
    attributes = dict(getattr(span, "attributes", None) or {})
    if attributes.get("openinference.span.kind") != "LLM":
        return
    span.set_attribute("gen_ai.operation.name", "chat")
    for source, target in (
        ("llm.model_name", "gen_ai.request.model"),
        ("llm.provider", "gen_ai.provider.name"),
        ("llm.token_count.prompt", "gen_ai.usage.input_tokens"),
        ("llm.token_count.completion", "gen_ai.usage.output_tokens"),
        ("llm.token_count.total", "gen_ai.usage.total_tokens"),
        ("llm.token_count.prompt_details.cache_read", "gen_ai.usage.cache_read.input_tokens"),
        ("llm.token_count.prompt_details.cache_write", "gen_ai.usage.cache_creation.input_tokens"),
        ("llm.token_count.completion_details.reasoning", "gen_ai.usage.reasoning.output_tokens"),
    ):
        if source in attributes:
            span.set_attribute(target, attributes[source])
    for direction in ("input", "output"):
        messages = _messages(attributes, f"llm.{direction}_messages.")
        # Agents SDK returns one selected choice. OpenInference represents its
        # Responses text/reasoning/call items as separate assistant messages.
        if direction == "output" and messages and all(message["role"] == "assistant" for message in messages):
            messages = [{"role": "assistant", "parts": [part for message in messages for part in message["parts"]]}]
        if messages:
            span.set_attribute(f"gen_ai.{direction}.messages", json.dumps(messages, ensure_ascii=False))


def _indexed(attributes: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    for key, value in attributes.items():
        if not key.startswith(prefix):
            continue
        index, separator, suffix = key[len(prefix) :].partition(".")
        if separator and index.isdigit():
            groups.setdefault(int(index), {})[suffix] = value
    return [groups[index] for index in sorted(groups)]


def _messages(attributes: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    messages = []
    for fields in _indexed(attributes, prefix):
        role = fields.get("message.role")
        if not isinstance(role, str):
            continue
        parts: list[dict[str, Any]] = []
        for part in _indexed(fields, "message.contents."):
            kind = part.get("message_content.type")
            text = part.get("message_content.text")
            if kind in {"text", "reasoning"} and isinstance(text, str):
                parts.append({"type": kind, "content": text})
            elif isinstance(url := part.get("message_content.image.image.url"), str):
                parts.append({"type": "uri", "modality": "image", "uri": url})
        # Responses emits both structured parts and their flattened text summary.
        # Use the flat form only when there are no captured structured parts.
        if not parts and isinstance(text := fields.get("message.content"), str):
            parts.append({"type": "text", "content": text})
        for call in _indexed(fields, "message.tool_calls."):
            tool_call = {"type": "tool_call"}
            for source, target in (("id", "id"), ("function.name", "name"), ("function.arguments", "arguments")):
                if f"tool_call.{source}" in call:
                    value = call[f"tool_call.{source}"]
                    tool_call[target] = _parse_json(value) if target == "arguments" else value
            parts.append(tool_call)
        if role == "tool" and (call_id := fields.get("message.tool_call_id")):
            result = fields.get("message.content", parts)
            parts = [{"type": "tool_call_response", "id": call_id, "result": _parse_json(result)}]
        messages.append({"role": role, "parts": parts})
    return messages


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value
