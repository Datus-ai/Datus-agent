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
from datus.utils.image_content import image_blocks, replace_images
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Budget assumptions for models without published specifications, not detected
# server limits. Explicit deployment settings and known model specs take priority.
DEFAULT_CONTEXT_LENGTH = 1048576
DEFAULT_MAX_TOKENS = 384000
DEFAULT_MEDIA_TOKENS = 4096


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
class TokenEstimate:
    tokens: int
    source: str
    estimated: bool
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestCapacity:
    model: str
    context_length: int | None
    default_output: int | None
    tokenizer_ref: str | None = None
    supports_output_limit: bool = True
    output_ceiling: int | None = None
    context_source: str = "config"
    output_source: str = "config"
    enforce_catalog_limits: bool = True

    def __post_init__(self) -> None:
        if self.context_length is None:
            object.__setattr__(self, "context_length", DEFAULT_CONTEXT_LENGTH)
            object.__setattr__(self, "context_source", "default")
        if self.default_output is None:
            usable = self.context_length - self.margin
            output = min(DEFAULT_MAX_TOKENS, max(1, usable // 2)) if self.supports_output_limit else DEFAULT_MAX_TOKENS
            object.__setattr__(self, "default_output", output)
            object.__setattr__(self, "output_source", "default")

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
        if self.enforce_catalog_limits and self.output_ceiling is not None and value > self.output_ceiling:
            raise ContextCapacityError(
                "OUTPUT_LIMIT_EXCEEDED",
                f"Requested output {value} exceeds the known maximum {self.output_ceiling} for {self.model}.",
            )
        return value

    def _counter_model(self) -> str | None:
        import litellm

        try:
            return next(
                (name for name in (self.model, self.model.split("/", 1)[-1]) if name in litellm.model_cost), None
            )
        except Exception:
            return None

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
        """Keep the integer counting API used by existing capacity consumers."""
        return self.estimate(payload).tokens

    def estimate(self, payload: Any) -> TokenEstimate:
        """Estimate input without requiring a provider catalog or optional tokenizer.

        Count a text-only view and account for media separately. Binary bytes
        never become text tokens, and the original request is not modified.
        """
        import litellm

        payload_view, files = _file_count_view(payload)
        # The shared image walker accepts message/content envelopes, not a
        # whole API request. Preserve every other request field for text counting.
        text_view = replace_images(payload_view)
        if isinstance(payload_view, dict):
            text_view = {
                key: replace_images(value) if key in {"messages", "input"} else value
                for key, value in text_view.items()
            }
        text = _json(text_view)
        messages = payload.get("messages", payload.get("input", [])) if isinstance(payload, dict) else payload
        framing = 64 * (len(messages) if isinstance(messages, list) else 1) + 256
        counter_model = self._counter_model()
        count = None
        source = "utf8_byte_upper_bound"
        estimated = True
        reasons = []
        if self.tokenizer_ref:
            try:
                tokenizer = _tokenizer(self.tokenizer_ref)
                encoded = (
                    tokenizer.encode(text, disallowed_special=())
                    if self.tokenizer_ref.startswith("tiktoken:")
                    else tokenizer.encode(text)
                )
                count = len(encoded.ids) if hasattr(encoded, "ids") else len(encoded)
                source, estimated = self.tokenizer_ref, False
            except Exception as exc:
                reasons.append(f"tokenizer unavailable ({type(exc).__name__})")
        if count is None and counter_model in litellm.open_ai_chat_completion_models:
            try:
                count = litellm.token_counter(model=counter_model, text=text)
                if type(count) is not int or count < 0:
                    raise ValueError("invalid token count")
                source = "litellm_serialized_request"
                estimated = not self.enforce_catalog_limits or bool(reasons)
            except Exception as exc:
                count = None
                reasons.append(f"text counter unavailable ({type(exc).__name__})")
        if count is None:
            count = len(text.encode("utf-8"))
            source, estimated = "utf8_byte_upper_bound", True

        content = (
            [payload[key] for key in ("messages", "input") if key in payload]
            if isinstance(payload, dict) and any(key in payload for key in ("messages", "input"))
            else payload
        )
        for block in image_blocks(content):
            try:
                if counter_model is None:
                    raise ValueError("unknown image model")
                image_tokens = litellm.token_counter(
                    model=counter_model, messages=_image_messages(block), use_default_image_token_count=False
                )
                if type(image_tokens) is not int or image_tokens < 0:
                    raise ValueError("invalid image token count")
                count += image_tokens
                estimated |= not self.enforce_catalog_limits
            except Exception as exc:
                count += DEFAULT_MEDIA_TOKENS
                estimated = True
                reasons.append(f"image counter unavailable ({type(exc).__name__})")
        if files:
            count += files * DEFAULT_MEDIA_TOKENS
            estimated = True
            reasons.append("file content estimated")
        fallback_reasons = tuple(dict.fromkeys(reasons))
        if fallback_reasons:
            logger.debug("Using estimated context tokens: model=%s reasons=%s", self.model, fallback_reasons)
        return TokenEstimate(count + framing, source, estimated, fallback_reasons)

    def check(self, payload: Any, output: int | None = None) -> int:
        available = self.input_limit(output)
        estimate = self.estimate(payload)
        count = estimate.tokens
        enforce = not estimate.estimated and (
            self.context_source == "config" or (self.enforce_catalog_limits and self.context_source == "catalog")
        )
        budget = {
            "model": self.model,
            "input_tokens": count,
            "max_output_tokens": self.output_limit(output),
            "context_length": self.context_length,
            "safety_margin": self.margin,
            "input_limit": available,
            "counter": estimate.source,
            "estimated": estimate.estimated,
            "context_source": self.context_source,
            "output_source": self.output_source,
            "fallback_reasons": estimate.fallback_reasons,
            "over_budget": count > available,
            "enforced": enforce,
        }
        logger.debug("Context capacity check: %s", budget)
        from datus.observability.model_call import current_model_call

        if call := current_model_call():
            call.fields.update({f"capacity_{key}": value for key, value in budget.items()})
        if count > available:
            if not enforce:
                logger.debug("Allowing request above estimated context budget: %s", budget)
                return count
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
    # Inspect the same message envelopes as replace_images; tool schemas and
    # business dictionaries are counted as text, not traversed as content.
    content = (
        [payload[key] for key in ("messages", "input") if key in payload]
        if isinstance(payload, dict) and ("messages" in payload or "input" in payload)
        else payload
    )
    for block in image_blocks(content):
        kind = block["type"]
        if kind in ("image_url", "input_image") or "image_url" in block or "file_id" in block:
            url = block.get("image_url")
            detail = block.get("detail", "auto")
            if isinstance(url, dict):
                detail = url.get("detail", detail)
                url = url.get("url")
            if not isinstance(url, str) or not url.startswith("data:"):
                raise ValueError("remote images need a provider token count before sending")
            blocks.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
        else:
            source = block.get("source", {})
            if source.get("type") != "base64":
                raise ValueError("remote images need a provider token count before sending")
            blocks.append(
                {"type": "image_url", "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"}}
            )
    return [{"role": "user", "content": blocks}] if blocks else []


def _file_count_view(payload: Any) -> tuple[Any, int]:
    """Replace opaque file content only inside message/tool content envelopes."""
    files = 0

    def walk(value: Any) -> Any:
        nonlocal files
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        if not isinstance(value, dict):
            return value
        kind = value.get("type")
        if (
            kind == "input_file"
            or (kind == "file" and "file" in value)
            or (kind == "document" and isinstance(value.get("source"), dict) and value["source"].get("type") != "text")
        ):
            files += 1
            return {
                "type": "text",
                "text": "[File content estimated]",
                **{k: value[k] for k in ("filename", "title") if k in value},
            }
        if kind == "function_call_output" and isinstance(value.get("call_id"), str):
            field = "output"
        elif value.get("role") in ("user", "assistant", "system", "developer", "tool") or (
            kind == "tool_result" and isinstance(value.get("tool_use_id"), str)
        ):
            field = "content"
        else:
            return value
        return {**value, field: walk(value[field])} if isinstance(value.get(field), (list, tuple)) else value

    if isinstance(payload, dict) and any(key in payload for key in ("messages", "input")):
        result = {key: walk(value) if key in {"messages", "input"} else value for key, value in payload.items()}
    else:
        result = walk(payload)
    return result, files
