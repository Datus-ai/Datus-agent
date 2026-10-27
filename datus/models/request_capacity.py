# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Request-local context budgets shared by all model transports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.image_content import replace_images
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Budget assumptions for models without published specifications, not detected
# server limits. Explicit deployment settings and known model specs take priority.
DEFAULT_CONTEXT_LENGTH = 262144
DEFAULT_MAX_TOKENS = 8192


class ContextCapacityError(DatusException):
    """A terminal preparation error: retrying the same request cannot fix it."""

    def __init__(self, reason: str, detail: str, **budget: Any):
        self.reason = reason
        self.budget = budget
        super().__init__(ErrorCode.MODEL_REQUEST_FAILED, message_args={"error_detail": f"{reason}: {detail}"})


@lru_cache(maxsize=16)
def _tokenizer(reference: str):
    if reference.startswith("tiktoken:"):
        import tiktoken

        return tiktoken.get_encoding(reference.removeprefix("tiktoken:"))
    from tokenizers import Tokenizer

    path = Path(reference)
    return Tokenizer.from_file(str(path)) if path.is_file() else Tokenizer.from_pretrained(reference)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class RequestCapacity:
    model: str
    context_length: int | None
    default_output: int | None
    tokenizer_ref: str | None = None
    supports_output_limit: bool = True
    output_ceiling: int | None = None

    @property
    def margin(self) -> int:
        return self.margin_for(self.context_length)

    @staticmethod
    def margin_for(context_length: int | None) -> int:
        return max(2048, math.ceil((context_length or 0) * 0.05))

    def output_limit(self, explicit: Any = None) -> int:
        value = self.default_output if explicit is None else explicit
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContextCapacityError("CONTEXT_SPEC_MISSING", f"Configure max_tokens for {self.model}.")
        if not self.supports_output_limit:
            # Subscription transports have no output-cap request field. A
            # requested smaller output cannot lower their service ceiling.
            return max(value, self.default_output or 0, self.output_ceiling or 0)
        if self.output_ceiling is not None and value > self.output_ceiling:
            raise ContextCapacityError(
                "OUTPUT_LIMIT_EXCEEDED",
                f"Requested output {value} exceeds the known maximum {self.output_ceiling} for {self.model}.",
            )
        return value

    def _counter_model(self) -> str | None:
        import litellm

        return next((name for name in (self.model, self.model.split("/", 1)[-1]) if name in litellm.model_cost), None)

    @property
    def counter_source(self) -> str:
        if self.tokenizer_ref:
            return self.tokenizer_ref
        import litellm

        return (
            "litellm_serialized_request"
            if self._counter_model() in litellm.open_ai_chat_completion_models
            else "utf8_byte_upper_bound"
        )

    def input_limit(self, output: int | None = None) -> int:
        if not isinstance(self.context_length, int) or self.context_length <= 0:
            raise ContextCapacityError("CONTEXT_SPEC_MISSING", f"Configure context_length for {self.model}.")
        return self.context_length - self.output_limit(output) - self.margin

    def count(self, payload: Any) -> int:
        """Count the complete serialized input, including schemas and framing.

        Without a known tokenizer, reserve one token per UTF-8 byte. The
        serialized representation deliberately includes field names and a
        per-message framing allowance. Images are accounted for separately by
        the matched provider counter, never by tokenizing their base64 bytes.
        """
        import litellm

        if _has_opaque_file(payload):
            raise ContextCapacityError(
                "CONTEXT_COUNT_UNAVAILABLE", "File inputs need extracted text or a provider token count before sending."
            )
        # The shared image walker accepts message/content envelopes, not a
        # whole API request. Preserve every other request field for text counting.
        text_view = replace_images(payload)
        if isinstance(payload, dict):
            text_view = {
                key: replace_images(value) if key in {"messages", "input"} else value
                for key, value in text_view.items()
            }
        text = _json(text_view)
        messages = payload.get("messages", payload.get("input", [])) if isinstance(payload, dict) else payload
        framing = 64 * (len(messages) if isinstance(messages, list) else 1) + 256
        counter_model = self._counter_model()
        try:
            if self.tokenizer_ref:
                tokenizer = _tokenizer(self.tokenizer_ref)
                encoded = (
                    tokenizer.encode(text, disallowed_special=())
                    if self.tokenizer_ref.startswith("tiktoken:")
                    else tokenizer.encode(text)
                )
                count = len(encoded.ids) if hasattr(encoded, "ids") else len(encoded)
            else:
                # For unknown aliases and non-OpenAI models, reserve UTF-8 bytes
                # rather than using LiteLLM's generic OpenAI tokenizer. This is
                # conservative for byte-level provider tokenizers; configure
                # tokenizer_ref to use the deployment's actual encoding.
                count = (
                    litellm.token_counter(model=counter_model, text=text)
                    if self.counter_source == "litellm_serialized_request"
                    else len(text.encode("utf-8"))
                )
            images = _image_messages(payload)
            if images:
                if counter_model is None:
                    raise ValueError("image counting requires a registered vision model")
                count += litellm.token_counter(
                    model=counter_model, messages=images, use_default_image_token_count=False
                )
        except ContextCapacityError:
            raise
        except Exception as exc:
            raise ContextCapacityError("CONTEXT_COUNT_UNAVAILABLE", f"Cannot count {self.model}: {exc}") from exc
        return count + framing

    def check(self, payload: Any, output: int | None = None) -> int:
        available = self.input_limit(output)
        count = self.count(payload)
        budget = {
            "model": self.model,
            "input_tokens": count,
            "max_output_tokens": self.output_limit(output),
            "context_length": self.context_length,
            "safety_margin": self.margin,
            "input_limit": available,
            "counter": self.counter_source,
        }
        logger.debug("Context capacity check: %s", budget)
        from datus.observability.model_call import current_model_call

        if call := current_model_call():
            call.fields.update({f"capacity_{key}": value for key, value in budget.items()})
        if count > available:
            parts = (
                {key: self.count({key: value}) for key, value in payload.items()} if isinstance(payload, dict) else {}
            )
            budget["input_parts"] = parts
            largest = ", ".join(f"{key}={value}" for key, value in sorted(parts.items(), key=lambda pair: -pair[1])[:3])
            raise ContextCapacityError(
                "CONTEXT_LIMIT_EXCEEDED",
                f"Input {count} + output {budget['max_output_tokens']} + margin {self.margin} "
                f"exceeds context_length {self.context_length} for {self.model}. "
                f"Largest input sections (including framing): {largest}. Reduce those sections or check model settings.",
                **budget,
            )
        return count


def effective_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the same top-level extra-body overrides as the SDK transport."""
    result = {k: v for k, v in payload.items() if k not in {"extra_body", "extra_args"}}
    for key in ("extra_args", "extra_body"):
        if isinstance(payload.get(key), dict):
            result.update(payload[key])
    return result


def guard_request(capacity: RequestCapacity, payload: dict[str, Any]) -> int:
    request = effective_request(payload)
    output = request.get("max_completion_tokens", request.get("max_output_tokens", request.get("max_tokens")))
    output = capacity.output_limit(output)
    content = {
        k: request[k]
        for k in (
            "system",
            "instructions",
            "messages",
            "input",
            "tools",
            "response_format",
            "text",
            "prompt",
            "tool_choice",
        )
        if k in request
    }
    return capacity.check(content, output)


def _image_messages(payload: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            kind = value.get("type")
            if kind in {"image_url", "input_image"}:
                url = value.get("image_url")
                detail = value.get("detail", "auto")
                if isinstance(url, dict):
                    detail = url.get("detail", detail)
                    url = url.get("url")
                if not isinstance(url, str) or not url.startswith("data:"):
                    raise ValueError("remote images need a provider token count before sending")
                blocks.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
                return
            if kind == "image":
                source = value.get("source", {})
                if source.get("type") != "base64":
                    raise ValueError("remote images need a provider token count before sending")
                blocks.append(
                    {"type": "image_url", "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"}}
                )
                return
            for item in value.values():
                visit(item)

    visit(payload)
    return [{"role": "user", "content": blocks}] if blocks else []


def _has_opaque_file(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_opaque_file(item) for item in value)
    if not isinstance(value, dict):
        return False
    kind = value.get("type")
    if kind == "input_file" or (kind == "file" and "file" in value):
        return True
    if kind == "document" and isinstance(value.get("source"), dict) and value["source"].get("type") != "text":
        return True
    return any(_has_opaque_file(item) for item in value.values())
