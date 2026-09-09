# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/orcarouter_model.py.

CI-level: zero external dependencies. All LiteLLM / OpenAI SDK calls mocked.
"""

from unittest.mock import MagicMock

import pytest

from datus.models.orcarouter_model import OrcaRouterModel
from datus.utils.exceptions import DatusException, ErrorCode


def _make_model_config(model="deepseek/deepseek-chat", api_key=None, base_url=None):
    cfg = MagicMock()
    cfg.model = model
    cfg.type = "orcarouter"
    cfg.api_key = api_key
    cfg.base_url = base_url
    cfg.temperature = None
    cfg.top_p = None
    cfg.enable_thinking = False
    cfg.reasoning_effort = None
    cfg.default_headers = None
    cfg.max_retry = 3
    cfg.retry_interval = 0.0
    cfg.strict_json_schema = True
    cfg.save_llm_trace = False
    cfg.auth_type = "api_key"
    cfg.use_native_api = False
    cfg.ssl_verify = None
    return cfg


class TestOrcaRouterModel:
    def test_uses_configured_api_key_and_base_url(self):
        model = OrcaRouterModel(_make_model_config(api_key="sk-orca-test", base_url="https://custom.example/v1"))
        assert model.api_key == "sk-orca-test"
        assert model.base_url == "https://custom.example/v1"

    def test_api_key_falls_back_to_environment(self, monkeypatch):
        monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-env")
        model = OrcaRouterModel(_make_model_config(api_key=None))
        assert model.api_key == "sk-orca-env"

    def test_missing_api_key_raises_datus_exception(self, monkeypatch):
        monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
        with pytest.raises(DatusException) as exc_info:
            OrcaRouterModel(_make_model_config(api_key=None))
        assert exc_info.value.code == ErrorCode.COMMON_ENV

    def test_base_url_defaults_to_orcarouter_gateway(self, monkeypatch):
        monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-env")
        model = OrcaRouterModel(_make_model_config(api_key=None, base_url=None))
        assert model.base_url == "https://api.orcarouter.ai/v1"
