# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.observability.config import RedactConfig
from datus.observability.privacy import redact_value


def test_redaction_can_be_disabled():
    value = {"api_key": "secret"}

    assert redact_value(value, RedactConfig(enabled=False)) is value


def test_redaction_preserves_token_usage_metrics():
    redacted = redact_value(
        {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "totalTokens": 46,
            "access_token": "secret",
            "openai_api_key": "key",
            "nested": {"secretKey": "secret"},
        },
        RedactConfig(),
    )

    assert redacted["prompt_tokens"] == 12
    assert redacted["completion_tokens"] == 34
    assert redacted["totalTokens"] == 46
    assert redacted["access_token"] == "[REDACTED]"
    assert redacted["openai_api_key"] == "[REDACTED]"
    assert redacted["nested"]["secretKey"] == "[REDACTED]"


def test_redaction_handles_acronym_camel_case_field_boundaries():
    redacted = redact_value(
        {
            "APIKey": "secret",
            "openaiAPIKey": "secret",
            "notebook": "safe",
        },
        RedactConfig(fields=["api_key", "key"]),
    )

    assert redacted["APIKey"] == "[REDACTED]"
    assert redacted["openaiAPIKey"] == "[REDACTED]"
    assert redacted["notebook"] == "safe"


def test_redaction_applies_patterns_and_nested_sequences():
    redacted = redact_value(
        {
            "messages": ["call me at 555-0100", ("safe", "code 555-0199")],
        },
        RedactConfig(patterns=[r"555-\d{4}"]),
    )

    assert redacted == {"messages": ["call me at [REDACTED]", ("safe", "code [REDACTED]")]}


def test_redaction_ignores_invalid_patterns_and_exact_fields():
    redacted = redact_value(
        {
            "token": "secret",
            "prompt_tokens": 12,
            "message": "unterminated regex should not crash",
            "nested": {"safe": "value"},
        },
        RedactConfig(fields=["", "token"], patterns=["["]),
    )

    assert redacted["token"] == "[REDACTED]"
    assert redacted["prompt_tokens"] == 12
    assert redacted["message"] == "unterminated regex should not crash"
    assert redacted["nested"] == {"safe": "value"}
