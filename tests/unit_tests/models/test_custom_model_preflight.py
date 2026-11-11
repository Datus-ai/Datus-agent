"""Self-hosted requests must survive missing metadata and optional counters."""

import base64
import copy
import io
from unittest.mock import Mock

import litellm
import pytest
from PIL import Image

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.models.request_capacity import ContextCapacityError, RequestCapacity


@pytest.fixture
def image_url():
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize("kind", ["image_url", "input_image", "image"])
def test_unregistered_vision_alias_keeps_image_input(image_url, kind):
    block = {
        "image_url": {"type": "image_url", "image_url": {"url": image_url}},
        "input_image": {"type": "input_image", "image_url": image_url},
        "image": {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_url.split(",", 1)[1]},
        },
    }[kind]
    payload = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Read this"}, block]}]}
    original = copy.deepcopy(payload)
    capacity = RequestCapacity("openai/private-vision", 262144, 8192)

    assert capacity.check(payload) >= 4096
    assert payload == original


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image_url", "image_url": {"url": "https://images.example/chart.png"}},
        {"type": "input_image", "file_id": "file-image"},
        {"type": "image", "source": {"type": "url", "url": "https://images.example/chart.png"}},
        {"type": "input_file", "file_id": "file-report"},
        {"type": "input_file", "filename": "report.pdf", "file_data": "data:application/pdf;base64," + "A" * 50000},
        {"type": "file", "file": {"file_data": "A" * 50000, "filename": "report.pdf"}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "A" * 50000}},
    ],
)
def test_unmeasurable_attachments_are_estimated_without_changing_payload(block):
    capacity = RequestCapacity("gpt-4o", 8192, 1024)
    payload = {"messages": [{"role": "user", "content": [block]}]}
    original = copy.deepcopy(payload)

    assert 4096 <= capacity.check(payload) < 8192
    assert payload == original


def test_tokenizer_failure_uses_available_counter_then_byte_estimate(monkeypatch):
    monkeypatch.setattr("datus.models.request_capacity._tokenizer", Mock(side_effect=OSError("unavailable")))
    payload = {"input": "Explain the orders table"}
    for model in ("gpt-4o", "openai/private-deployment"):
        capacity = RequestCapacity(model, 8192, 1024, tokenizer_ref="/missing/tokenizer.json")
        assert capacity.check(payload) > 0

    monkeypatch.setattr(litellm, "token_counter", Mock(side_effect=ValueError("counter unavailable")))
    assert RequestCapacity("gpt-4o", 8192, 1024).check(payload) > 0


def test_failed_image_counter_keeps_successful_text_count(monkeypatch, image_url):
    def counter(*, text=None, **kwargs):
        if text is not None:
            return 123
        raise ValueError("vision metadata unavailable")

    monkeypatch.setattr(litellm, "token_counter", counter)
    capacity = RequestCapacity("gpt-4o", 8192, 1024)
    payload = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}]}

    assert capacity.check(payload) == 123 + 4096 + 64 + 256


def test_estimated_text_over_budget_is_allowed_but_known_count_is_enforced():
    payload = {"input": "\N{CJK UNIFIED IDEOGRAPH-8868}" * 5000}
    estimated = RequestCapacity("openai/private-deployment", 8192, 1024)
    assert estimated.check(payload) > estimated.input_limit()

    precise = RequestCapacity("gpt-4o", 8192, 1024, tokenizer_ref="tiktoken:cl100k_base")
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        precise.check({"input": "word " * 20000})


@pytest.mark.parametrize("name", ["gpt-4o", "gpt-4o-selfhost"])
def test_custom_endpoint_explicit_output_overrides_catalog_ceiling(name):
    model = OpenAIModel(
        ModelConfig(type="openai", model=name, api_key="test", base_url="https://private.example/v1", max_tokens=32768)
    )
    request = model.check_request({"messages": [{"role": "user", "content": "hello"}]})
    assert request["max_tokens"] == 32768
    assert model._build_agent("", str, False, {}, []).model_settings.max_tokens == 32768


def test_custom_catalog_window_is_a_hint_even_with_configured_tokenizer():
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="gpt-4o",
            api_key="test",
            base_url="https://private.example/v1",
            tokenizer_ref="tiktoken:cl100k_base",
        )
    )
    payload = {"input": "word " * 140000}
    assert model.request_capacity().check(payload) > model.request_capacity().input_limit()


def test_missing_limits_use_flash_budget_and_leave_room_for_input():
    model = OpenAIModel(
        ModelConfig(type="openai", model="private-deployment", api_key="test", base_url="https://private.example/v1")
    )
    assert model.context_length() == 1048576
    assert model.max_tokens() == 384000
    assert model.check_request({"input": "hello"})["max_tokens"] == 384000

    small = OpenAIModel(ModelConfig(type="openai", model="private-deployment", api_key="test", context_length=32768))
    assert 0 < small.max_tokens() < small.context_length() // 2

    missing = RequestCapacity("private-deployment", None, None)
    assert missing.check({"input": "hello"}) > 0


def test_metadata_lookup_failure_uses_defaults(monkeypatch):
    model = OpenAIModel(
        ModelConfig(type="openai", model="private-deployment", api_key="test", base_url="https://private.example/v1")
    )
    monkeypatch.setattr(model, "_lookup_spec", Mock(side_effect=OSError("catalog unavailable")))
    capacity = model.request_capacity()
    assert capacity.context_length == 1048576
    assert capacity.default_output == 384000
    assert model.check_request({"input": "hello"})["max_tokens"] == 384000
    assert model.max_tokens() == 384000
    assert model.context_length() == 1048576


@pytest.mark.parametrize("result", [None, -1, True, "unknown"])
def test_invalid_counter_result_falls_back_without_invalidating_request(monkeypatch, result):
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: result)
    capacity = RequestCapacity("gpt-4o", 8192, 1024)
    estimate = capacity.estimate({"input": "hello"})
    assert estimate.tokens > 0
    assert estimate.estimated
    assert estimate.source == "utf8_byte_upper_bound"


def test_file_estimation_preserves_tool_schemas_and_business_data():
    capacity = RequestCapacity("private-deployment", 8192, 1024)
    business = {"type": "input_file", "file_data": "A" * 10000}
    payload = {"messages": [{"role": "tool", "content": business}], "tools": [{"parameters": business}]}
    assert capacity.count(payload) > 20000


@pytest.mark.asyncio
async def test_estimated_pressure_reaches_existing_context_preparation():
    from datus.agent.node.context_rewriter import MidTurnCompactor
    from tests.unit_tests.agent.node.test_context_rewriter import _FakeNode

    node = _FakeNode()
    node.model = OpenAIModel(
        ModelConfig(type="openai", model="private-deployment", api_key="test", context_length=8192, max_tokens=1024)
    )
    raw = [{"role": "user", "content": "\N{CJK UNIFIED IDEOGRAPH-8868}" * 5000}]
    compactor = MidTurnCompactor(node)
    assert await compactor.rewrite_sdk_input(raw) == raw
    prepared = node.compact_mid_turn.await_args.kwargs
    assert prepared["base_tokens"] > node.model.request_capacity().input_limit()
