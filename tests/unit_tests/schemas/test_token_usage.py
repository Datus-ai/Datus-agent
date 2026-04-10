# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/schemas/token_usage.py."""

from datus.schemas.token_usage import TokenUsage


class TestTokenUsageDefaults:
    def test_all_defaults_zero(self):
        tu = TokenUsage()
        assert tu.requests == 0
        assert tu.input_tokens == 0
        assert tu.output_tokens == 0
        assert tu.total_tokens == 0
        assert tu.cached_tokens == 0
        assert tu.reasoning_tokens == 0
        assert tu.cache_creation_tokens == 0
        assert tu.cache_hit_rate == 0.0
        assert tu.context_usage_ratio == 0.0
        assert tu.context_length == 0
        assert tu.session_total_tokens == 0

    def test_construct_with_kwargs(self):
        tu = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert tu.input_tokens == 100
        assert tu.output_tokens == 50
        assert tu.total_tokens == 150


class TestTokenUsageExtraIgnored:
    def test_extra_fields_ignored(self):
        tu = TokenUsage(input_tokens=10, unknown_field="should_be_ignored", another=42)
        assert tu.input_tokens == 10
        assert not hasattr(tu, "unknown_field")
        assert not hasattr(tu, "another")


class TestTokenUsageFromUsageDict:
    def test_from_usage_dict_basic(self):
        d = {
            "requests": 3,
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "cached_tokens": 500,
            "reasoning_tokens": 10,
            "cache_hit_rate": 0.5,
            "context_usage_ratio": 0.01,
        }
        tu = TokenUsage.from_usage_dict(d)
        assert tu.requests == 3
        assert tu.input_tokens == 1000
        assert tu.cached_tokens == 500
        assert tu.cache_hit_rate == 0.5

    def test_from_usage_dict_overrides(self):
        d = {"input_tokens": 100, "output_tokens": 50}
        tu = TokenUsage.from_usage_dict(d, context_length=128000, session_total_tokens=100)
        assert tu.input_tokens == 100
        assert tu.context_length == 128000
        assert tu.session_total_tokens == 100

    def test_from_usage_dict_override_takes_precedence(self):
        d = {"input_tokens": 100}
        tu = TokenUsage.from_usage_dict(d, input_tokens=999)
        assert tu.input_tokens == 999

    def test_from_usage_dict_extra_keys_ignored(self):
        d = {"input_tokens": 50, "some_extra_key": "ignored"}
        tu = TokenUsage.from_usage_dict(d)
        assert tu.input_tokens == 50

    def test_from_usage_dict_empty(self):
        tu = TokenUsage.from_usage_dict({})
        assert tu.total_tokens == 0


class TestTokenUsageSerialization:
    def test_model_dump(self):
        tu = TokenUsage(input_tokens=100, output_tokens=50)
        d = tu.model_dump()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert "context_length" in d
