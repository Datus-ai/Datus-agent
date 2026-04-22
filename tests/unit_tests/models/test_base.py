# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.models.base.LLMBaseModel.test_connection``."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel

pytestmark = pytest.mark.ci


class _StubModel(LLMBaseModel):
    """Minimal concrete subclass exposing a controllable ``generate``.

    Inheriting from :class:`LLMBaseModel` without implementing every
    abstract method would trigger TypeError on instantiation, so the
    stub satisfies the full interface with no-op coroutines for the
    unused branches.
    """

    def __init__(self, responder):
        self.model_config = ModelConfig(type="stub", api_key="", model="stub", base_url=None)
        self._responder = responder

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs):  # type: ignore[override]
        return self._responder(prompt, **kwargs)

    def generate_with_json_output(self, prompt: Any, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def generate_with_tools(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def generate_with_tools_stream(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def token_count(self, prompt: str) -> int:  # type: ignore[override]
        return 0

    def context_length(self):  # type: ignore[override]
        return None


class TestTestConnection:
    def test_returns_ok_on_successful_generate(self):
        model = _StubModel(lambda _p, **_k: "pong")
        ok, err = asyncio.run(model.test_connection(timeout=1.0))
        assert ok is True
        assert err == ""

    def test_returns_false_on_empty_response(self):
        model = _StubModel(lambda _p, **_k: "   ")
        ok, err = asyncio.run(model.test_connection(timeout=1.0))
        assert ok is False
        assert "empty" in err.lower()

    def test_returns_false_on_exception(self):
        def _fail(_p, **_k):
            raise RuntimeError("down")

        model = _StubModel(_fail)
        ok, err = asyncio.run(model.test_connection(timeout=1.0))
        assert ok is False
        assert "down" in err

    def test_returns_false_on_timeout(self):
        model = _StubModel(MagicMock(return_value="ignored"))
        with patch("datus.models.base.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            ok, err = asyncio.run(model.test_connection(timeout=0.05))
        assert ok is False
        assert "timed out" in err.lower()


class TestCreateModelCache:
    """``create_model`` keeps an LRU cache so ``/model`` switches stay cheap."""

    def _agent_config(self, model_config: ModelConfig):
        cfg = MagicMock()
        cfg.active_model.return_value = model_config
        cfg.model_config.return_value = model_config
        cfg.models = {"custom": model_config}
        cfg.session_dir = None
        return cfg

    def _patch_constructor(self):
        sentinel = object()
        module = MagicMock()
        instance = MagicMock(name="LLMInstance")
        module.OpenAIModel = MagicMock(return_value=instance)
        return sentinel, module, instance

    def _fresh_cache(self):
        """Reset the process-wide model cache between tests."""
        LLMBaseModel._MODEL_CACHE.clear()

    def test_same_config_returns_cached_instance(self):
        self._fresh_cache()
        cfg_a = ModelConfig(type="openai", api_key="k", model="gpt-4.1", base_url="https://a")
        agent_cfg = self._agent_config(cfg_a)
        _, module, instance = self._patch_constructor()
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            first = LLMBaseModel.create_model(agent_cfg)
            second = LLMBaseModel.create_model(agent_cfg)
        assert first is second
        module.OpenAIModel.assert_called_once()

    def test_different_model_yields_new_instance(self):
        self._fresh_cache()
        cfg_a = ModelConfig(type="openai", api_key="k", model="gpt-4.1", base_url="https://a")
        cfg_b = ModelConfig(type="openai", api_key="k", model="gpt-4o", base_url="https://a")
        module = MagicMock()
        module.OpenAIModel = MagicMock(side_effect=lambda **kw: MagicMock(name="Instance"))
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            a1 = LLMBaseModel.create_model(self._agent_config(cfg_a))
            b1 = LLMBaseModel.create_model(self._agent_config(cfg_b))
            a2 = LLMBaseModel.create_model(self._agent_config(cfg_a))
        assert a1 is not b1
        assert a1 is a2, "switching back to the original model should hit the cache"
        assert module.OpenAIModel.call_count == 2

    def test_cache_eviction_respects_maxsize(self):
        self._fresh_cache()
        module = MagicMock()
        module.OpenAIModel = MagicMock(side_effect=lambda **kw: MagicMock(name="Instance"))
        with patch.dict("sys.modules", {"datus.models.openai_model": module}):
            for i in range(LLMBaseModel._MODEL_CACHE_MAXSIZE + 2):
                cfg = ModelConfig(type="openai", api_key="k", model=f"m{i}", base_url="https://a")
                LLMBaseModel.create_model(self._agent_config(cfg))
        assert len(LLMBaseModel._MODEL_CACHE) == LLMBaseModel._MODEL_CACHE_MAXSIZE
