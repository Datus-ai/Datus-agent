# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Platform-specific OTLP observability adapters."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from datus.observability.adapters.otlp import OtlpAdapter
from datus.observability.config import ObservabilityAdapterConfig
from datus.utils.exceptions import DatusException, ErrorCode


class LangSmithAdapter(OtlpAdapter):
    name = "langsmith"
    capabilities = {"traces", "otlp", "llm_observability", "projects", "attachments"}

    def resolve_adapter_config(self, adapter_config: ObservabilityAdapterConfig) -> ObservabilityAdapterConfig:
        endpoint = adapter_config.endpoint
        if not endpoint:
            base_url = _option_or_env(
                adapter_config,
                "base_url",
                "LANGSMITH_ENDPOINT",
                "LANGCHAIN_ENDPOINT",
                default="https://api.smith.langchain.com",
            )
            endpoint = _with_otel_trace_path(base_url)

        api_key = _option_or_env(adapter_config, "api_key", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
        if not api_key:
            raise DatusException(
                ErrorCode.COMMON_FIELD_REQUIRED,
                message="LangSmith adapter requires LANGSMITH_API_KEY or LANGCHAIN_API_KEY",
            )

        headers = {"x-api-key": api_key}
        project = _option_or_env(adapter_config, "project", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
        if project:
            headers["Langsmith-Project"] = project
        headers.update(adapter_config.headers)
        return replace(adapter_config, endpoint=endpoint, headers=headers)


class BraintrustAdapter(OtlpAdapter):
    name = "braintrust"
    capabilities = {"traces", "otlp", "llm_observability", "project_parent", "experiments", "evals"}

    def resolve_adapter_config(self, adapter_config: ObservabilityAdapterConfig) -> ObservabilityAdapterConfig:
        endpoint = adapter_config.endpoint or _option_or_env(adapter_config, "endpoint", "BRAINTRUST_OTEL_ENDPOINT")
        if not endpoint:
            api_url = _option_or_env(adapter_config, "api_url", "BRAINTRUST_API_URL")
            if api_url:
                endpoint = _with_otel_trace_path(api_url)
            elif (_option_or_env(adapter_config, "region", "BRAINTRUST_REGION") or "").lower() == "eu":
                endpoint = "https://api-eu.braintrust.dev/otel/v1/traces"
            else:
                endpoint = "https://api.braintrust.dev/otel/v1/traces"

        api_key = _option_or_env(adapter_config, "api_key", "BRAINTRUST_API_KEY")
        if not api_key:
            raise DatusException(
                ErrorCode.COMMON_FIELD_REQUIRED, message="Braintrust adapter requires BRAINTRUST_API_KEY"
            )

        parent = _option_or_env(adapter_config, "parent", "BRAINTRUST_PARENT")
        if not parent:
            project_id = _option_or_env(adapter_config, "project_id", "BRAINTRUST_PROJECT_ID")
            project_name = _option_or_env(adapter_config, "project_name", "BRAINTRUST_PROJECT_NAME")
            if project_id:
                parent = f"project_id:{project_id}"
            elif project_name:
                parent = f"project_name:{project_name}"
        if not parent:
            raise DatusException(
                ErrorCode.COMMON_FIELD_REQUIRED,
                message="Braintrust adapter requires parent, project_id, or project_name",
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "x-bt-parent": parent,
        }
        headers.update(adapter_config.headers)
        return replace(adapter_config, endpoint=endpoint, headers=headers)


class DatadogAdapter(OtlpAdapter):
    name = "datadog"
    capabilities = {"traces", "otlp", "apm", "llm_observability"}

    def resolve_adapter_config(self, adapter_config: ObservabilityAdapterConfig) -> ObservabilityAdapterConfig:
        endpoint = adapter_config.endpoint or _option_or_env(
            adapter_config,
            "endpoint",
            "DATADOG_OTLP_TRACES_ENDPOINT",
            "DD_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        )
        if not endpoint:
            agent_host = _option_or_env(adapter_config, "agent_host", "DD_AGENT_HOST", default="localhost")
            agent_port = _option_or_env(adapter_config, "agent_port", "DD_OTLP_HTTP_PORT", default="4318")
            endpoint = f"http://{agent_host}:{agent_port}/v1/traces"

        headers = {}
        api_key = _option_or_env(adapter_config, "api_key", "DD_API_KEY", "DATADOG_API_KEY")
        if api_key:
            headers["dd-api-key"] = api_key
        headers.update(adapter_config.headers)
        return replace(adapter_config, endpoint=endpoint, headers=headers)


def _option_or_env(
    adapter_config: ObservabilityAdapterConfig,
    option_key: str,
    *env_names: str,
    default: str | None = None,
) -> str | None:
    value = _clean_string(adapter_config.options.get(option_key))
    if value is not None:
        return value
    for env_name in env_names:
        value = _clean_string(os.environ.get(env_name))
        if value is not None:
            return value
    return default


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith("<MISSING:"):
        return None
    return text


def _join_url(base_url: str | None, path: str) -> str:
    if not base_url:
        raise DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message="OTLP platform adapter requires a base URL or endpoint",
        )
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _with_otel_trace_path(base_url: str | None) -> str:
    if not base_url:
        raise DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message="OTLP platform adapter requires a base URL or endpoint",
        )
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/traces"):
        return normalized
    if normalized.endswith("/otel"):
        return f"{normalized}/v1/traces"
    return f"{normalized}/otel/v1/traces"
