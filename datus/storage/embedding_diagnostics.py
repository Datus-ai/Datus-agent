# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""User-facing diagnostics for embedding-dependent feature degradation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_FASTEMBED_REPO_ID = "qdrant/all-MiniLM-L6-v2-onnx"


def resolve_fastembed_cache_dir() -> str:
    """Return the fastembed cache directory used by Datus defaults."""
    if cache_path := os.getenv("FASTEMBED_CACHE_PATH"):
        return str(Path(cache_path).expanduser().resolve())
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return str((hf_home / "fastembed").resolve())


def fastembed_environment_details(cache_dir: Optional[str] = None) -> str:
    """Format cache-related environment details for logs and errors."""
    hf_home = os.environ.get("HF_HOME")
    fastembed_cache_path = os.environ.get("FASTEMBED_CACHE_PATH")
    effective_cache_dir = cache_dir or resolve_fastembed_cache_dir()
    default_hf_home = str((Path.home() / ".cache" / "huggingface").resolve())
    default_fastembed_cache = str((Path(default_hf_home) / "fastembed").resolve())
    return (
        f"cache_dir={effective_cache_dir}; "
        f"HF_HOME={hf_home or f'(not set, default {default_hf_home})'}; "
        f"FASTEMBED_CACHE_PATH={fastembed_cache_path or '(not set)'}; "
        f"default_fastembed_cache={default_fastembed_cache}"
    )


def format_fastembed_download_error(
    *,
    model_name: str,
    repo_id: Optional[str],
    cache_dir: Optional[str],
    cause: BaseException,
) -> str:
    """Build a clear FastEmbed cache/download failure message."""
    resolved_repo_id = repo_id or DEFAULT_FASTEMBED_REPO_ID
    return (
        "Embedding model cache is missing and the FastEmbed model could not be downloaded from Hugging Face. "
        f"model={model_name}; repo_id={resolved_repo_id}; {fastembed_environment_details(cache_dir)}. "
        "Remediation: configure a Hugging Face proxy or mirror, pre-cache the model artifacts, "
        "or configure an OpenAI-compatible embedding provider for embedding-backed stores. "
        f"Original error: {cause}"
    )


def format_context_degraded_warning(error: BaseException | str | None = None) -> str:
    """Build the non-fatal warning shown when context search is unavailable."""
    details = str(error).strip() if error else ""
    message = (
        "Context search and @ references are disabled because the embedding model is unavailable. "
        "Database tools and normal chat remain available. "
        "Remediation: configure a Hugging Face proxy or mirror, pre-cache the FastEmbed model, "
        "or configure an OpenAI-compatible embedding provider."
    )
    if details:
        message = f"{message} Details: {details}"
    return message


def is_embedding_unavailable_error(error: BaseException | str | None) -> bool:
    """Return True for embedding-cache/provider failures that should degrade search."""
    if not error:
        return False
    message = str(error)
    return any(
        marker in message
        for marker in (
            "MODEL_EMBEDDING_ERROR",
            "error_code=300019",
            "Embedding model cache is missing",
            "embedding model is unavailable",
        )
    )


DATASOURCE_SCOPE_MARKER = "datasource is required for datasource-scoped storage"


def is_datasource_scope_error(error: BaseException | str | None) -> bool:
    """Return True for the "no datasource selected" storage-scope failure."""
    return bool(error) and DATASOURCE_SCOPE_MARKER in str(error)


def format_context_unavailable(error: BaseException | str | None = None) -> str:
    """Build a context-degradation message that matches the actual cause.

    ``format_context_degraded_warning`` hardcodes the embedding-unavailable
    text; routing every failure through it blames the embedding model (and
    recommends Hugging Face mirrors) for unrelated causes such as a missing
    datasource. Classify first, then format.
    """
    if is_embedding_unavailable_error(error):
        return format_context_degraded_warning(error)
    if is_datasource_scope_error(error):
        return (
            "Context search and @ references are inactive because no datasource is selected. "
            "Database tools and normal chat remain available. "
            "Select a datasource (/database in the CLI, or --datasource) to enable them."
        )
    details = str(error).strip() if error else ""
    message = "Context search and @ references are disabled. Database tools and normal chat remain available."
    if details:
        message = f"{message} Details: {details}"
    return message
