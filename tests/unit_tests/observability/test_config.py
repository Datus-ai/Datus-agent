# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest

from datus.observability.config import ObservabilityAdapterConfig, ObservabilityConfig
from datus.utils.exceptions import DatusException


def test_observability_absent_is_disabled_by_default():
    cfg = ObservabilityConfig.from_dict(None)

    assert cfg.explicit is False
    assert cfg.tracing.enabled is False
    assert cfg.tracing.capture_content is True


def test_empty_observability_does_not_make_tracing_explicit():
    cfg = ObservabilityConfig.from_dict({})

    assert cfg.explicit is False
    assert cfg.tracing.explicit is False
    assert cfg.tracing.enabled is False


def test_observability_parses_capture_overrides_and_headers():
    cfg = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "service_name": "datus-prod",
                "environment": "prod",
                "capture_content": True,
                "capture": {
                    "tool_results": False,
                    "artifacts": False,
                },
                "adapters": [
                    {
                        "type": "otlp",
                        "endpoint": "https://collector.example/v1/traces",
                        "headers": "x-api-key=abc,project=datus",
                    }
                ],
            }
        }
    )

    assert cfg.explicit is True
    assert cfg.tracing.enabled is True
    assert cfg.tracing.service_name == "datus-prod"
    assert cfg.tracing.environment == "prod"
    assert cfg.tracing.capture.prompts is True
    assert cfg.tracing.capture.tool_results is False
    assert cfg.tracing.capture.artifacts is False
    assert cfg.tracing.adapters[0].type == "otlp"
    assert cfg.tracing.adapters[0].headers == {"x-api-key": "abc", "project": "datus"}


def test_tracing_enabled_defaults_to_langfuse_adapter():
    cfg = ObservabilityConfig.from_dict({"tracing": {"enabled": True}})

    assert cfg.tracing.enabled is True
    assert len(cfg.tracing.adapters) == 1
    assert cfg.tracing.adapters[0].type == "langfuse"


def test_explicit_empty_adapters_are_respected():
    cfg = ObservabilityConfig.from_dict({"tracing": {"enabled": True, "adapters": []}})

    assert cfg.tracing.enabled is True
    assert cfg.tracing.adapters == []


def test_adapter_timeout_normalizes_blank_and_missing_placeholders():
    blank = ObservabilityAdapterConfig.from_dict({"type": "otlp", "timeout": " "})
    missing = ObservabilityAdapterConfig.from_dict({"type": "otlp", "timeout": "<MISSING:OTLP_TIMEOUT>"})

    assert blank.timeout is None
    assert missing.timeout is None


def test_adapter_timeout_invalid_value_uses_datus_exception():
    with pytest.raises(DatusException, match="timeout"):
        ObservabilityAdapterConfig.from_dict({"type": "otlp", "timeout": "soon"})


def test_capture_content_false_disables_content_fields_by_default():
    cfg = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "capture_content": False,
                "capture": {
                    "prompts": True,
                },
            }
        }
    )

    assert cfg.tracing.capture_content is False
    assert cfg.tracing.capture.prompts is True
    assert cfg.tracing.capture.responses is False
    assert cfg.tracing.capture.tool_args is False
