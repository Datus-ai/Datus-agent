# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Conservative image-input metadata; unknown models remain selectable."""


def image_model_support(model: str, modalities: list[str] | None = None) -> bool | None:
    if modalities is not None:
        return "image" in modalities
    slug = model.split("/")[-1]
    supported = {
        "deepseek-v4-flash",
        "deepseek-flash",
        "deepseek-v4-flash-vision-exp",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-astra",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    }
    unsupported = {"deepseek-v3.2", "deepseek-v4-pro", "qwen3-coder-next", "qwen3-coder-plus"}
    if slug in supported:
        return True
    if slug in unsupported:
        return False
    return None
