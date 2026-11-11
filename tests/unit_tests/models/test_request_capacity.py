import base64
import copy
import io
import json
from itertools import product
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import litellm
import pytest
from agents import ModelSettings
from agents.models.interface import ModelTracing
from PIL import Image

from datus.configuration.agent_config import ModelConfig, load_model_config
from datus.models.openai_model import OpenAIModel
from datus.models.request_capacity import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_TOKENS,
    ContextCapacityError,
    RequestCapacity,
    guard_request,
)
from datus.tools.registry.contracts import tool_contract

pytestmark = pytest.mark.component


@pytest.mark.parametrize(
    "attachment",
    [
        {"type": "image_url", "image_url": {"url": "https://example.invalid/chart.png"}},
        {"type": "input_image", "file_id": "file-image"},
        {"type": "input_file", "file_id": "file-report"},
    ],
)
def test_estimated_attachment_does_not_disable_trusted_text_limit(attachment):
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-vision",
            api_key="test",
            base_url="https://private.example/v1",
            context_length=8192,
            max_tokens=1024,
            tokenizer_ref="tiktoken:cl100k_base",
        )
    )
    text = {"type": "input_text", "text": "word " * 12000}
    payload = {"input": [{"role": "user", "content": [text, attachment]}]}
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        model.check_request(payload)


def test_estimated_media_can_exceed_budget_when_trusted_text_fits():
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-vision",
            api_key="test",
            base_url="https://private.example/v1",
            context_length=4096,
            max_tokens=1024,
            tokenizer_ref="tiktoken:cl100k_base",
        )
    )
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Read this image"},
                    {"type": "input_image", "file_id": "file-image"},
                ],
            }
        ]
    }
    capacity = model.request_capacity()
    assert capacity.check(payload) > capacity.input_limit()


def test_budget_uses_actual_output_and_complete_request():
    capacity = RequestCapacity("gpt-4o", 262144, 8192)
    assert capacity.input_limit() == 240844
    assert capacity.input_limit(4000) == 245036
    small = {"messages": [{"role": "user", "content": "hello"}]}
    assert capacity.count({**small, "tools": [{"description": "x " * 10000}]}) > capacity.count(small)
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED") as error:
        RequestCapacity("gpt-4o", 4096, 1024).check({"messages": [{"role": "user", "content": "word " * 10000}]})
    assert error.value.budget["max_output_tokens"] == 1024


def test_unresolved_capacity_and_unknown_tokenizers_have_defaults():
    assert RequestCapacity("unknown", None, 100).check({"input": "hello"}) > 0
    assert RequestCapacity("unknown-proxy-name", 262144, 100).check({"input": "hello"}) > 0
    assert RequestCapacity("unknown-proxy-name", 262144, 100, "tiktoken:o200k_base").check({"input": "hello"}) > 0


@pytest.mark.parametrize("context,output,tokenizer", product([None, 4096], [None, 512], [None, "tiktoken:o200k_base"]))
def test_custom_model_capacity_settings_are_independently_optional(context, output, tokenizer):
    config = {
        "type": "openai",
        "model": "private-deployment",
        "api_key": "test",
        "base_url": "https://proxy.example/v1",
    }
    config.update(
        {k: v for k, v in {"context_length": context, "max_tokens": output, "tokenizer_ref": tokenizer}.items() if v}
    )
    model = OpenAIModel(load_model_config(config))
    capacity = model.request_capacity()
    expected_context = context or DEFAULT_CONTEXT_LENGTH
    expected_output = output or (1024 if context == 4096 else DEFAULT_MAX_TOKENS)
    assert model.context_length() == capacity.context_length == expected_context
    assert model.max_tokens() == capacity.default_output == expected_output
    assert capacity.counter_source == (tokenizer or "utf8_byte_upper_bound")
    # Resolving defaults must not persist them as explicit user overrides.
    assert model.model_config.context_length == context
    assert model.model_config.max_tokens == output
    assert model.model_config.tokenizer_ref == tokenizer
    request = {"messages": [{"role": "user", "content": "Describe the order schema"}]}
    assert model.check_request(request)["max_tokens"] == expected_output
    assert model._build_agent("Follow the task", str, False, {}, []).model_settings.max_tokens == expected_output
    assert model.check_request({**request, "max_tokens": 256})["max_tokens"] == 256


@pytest.mark.parametrize(
    "context,output,expected_output", [(65536, 12000, 12000), (2097152, 12000, 12000), (4096, 8192, 1024)]
)
def test_catalogue_specs_precede_defaults_for_both_sdk_and_request_budget(
    monkeypatch, context, output, expected_output
):
    import litellm

    monkeypatch.setitem(
        litellm.model_cost, "catalogued-deployment", {"max_input_tokens": context, "max_output_tokens": output}
    )
    model = OpenAIModel(ModelConfig(type="openai", model="catalogued-deployment", api_key="test"))
    capacity = model.request_capacity()
    assert model.context_length() == capacity.context_length == context
    assert model.max_tokens() == capacity.default_output == expected_output
    assert model._build_agent("", str, False, {}, []).model_settings.max_tokens == expected_output
    model.check_request({"messages": [{"role": "user", "content": "hello"}]})


def test_unknown_model_counts_complete_utf8_request_without_guessing_a_tokenizer():
    capacity = RequestCapacity("private-deployment", 128000, 8192)
    payload = {
        "instructions": "Verify the order schema",
        "messages": [{"role": "user", "content": "SELECT amount FROM orders WHERE currency = '\N{EURO SIGN}'"}],
        "tools": [{"name": "describe_table", "description": "Read the table schema"}],
    }
    with patch("litellm.token_counter", side_effect=AssertionError("must not guess an alias tokenizer")):
        serialized_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        assert capacity.count(payload) == serialized_bytes + 64 + 256


def test_tool_schema_with_type_parameter_is_counted_as_text():
    capacity = RequestCapacity("deepseek/deepseek-v4-flash", 262144, 8192)
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "kind of result"},
                            "example": {"type": "image_url", "image_url": {"url": "https://example.com/example.png"}},
                        },
                    },
                },
            }
        ],
    }
    assert capacity.check(request) > capacity.check({"messages": request["messages"]})
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        RequestCapacity("deepseek/deepseek-v4-flash", 262144, 8192, allow_estimated_overflow=False).check(
            {**request, "tools": request["tools"] * 10000}
        )


def test_explicit_broken_tokenizer_falls_back_with_provenance():
    capacity = RequestCapacity("private-deployment", 128000, 8192, "/missing/tokenizer.json")
    with patch("datus.models.request_capacity._tokenizer", side_effect=OSError("Cannot read tokenizer")):
        estimate = capacity.estimate({"input": "hello"})
        assert estimate.tokens > 0
        assert estimate.estimated
        assert estimate.fallback_reasons == ("tokenizer unavailable (OSError)",)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["private-deployment", "z-ai/yumcode-pro"])
async def test_custom_model_without_new_settings_reaches_transport_despite_estimated_overflow(model_name):
    import litellm

    model = OpenAIModel(
        ModelConfig(type="openai", model=model_name, api_key="test", base_url="https://proxy.example/v1")
    )
    response = litellm.ModelResponse(
        model=model_name,
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response) as send:
        # Requests above the former 256K fallback must reach the transport.
        prompt = "Inspect the schema.\n" + "column_name " * 25000
        result = await model.generate_with_tools(prompt, max_turns=1)
        assert result["content"] == "done"
        assert send.await_args.kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
        send.reset_mock()
        result = await model.generate_with_tools("\N{EURO SIGN}" * DEFAULT_CONTEXT_LENGTH, max_turns=1)
        assert result["content"] == "done"
        send.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_tool_with_type_parameter_reaches_transport():
    import litellm
    from agents import function_tool

    @function_tool
    def search(type: str) -> str:
        """Search for a type of result."""
        return type

    model = OpenAIModel(
        ModelConfig(type="openai", model="private-deployment", api_key="test", base_url="https://proxy.example/v1")
    )
    response = litellm.ModelResponse(
        model="private-deployment",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response) as send:
        result = await model.generate_with_tools("hello", tools=[search], max_turns=1)
    assert result["content"] == "done"
    assert send.await_args.kwargs["tools"][0]["function"]["parameters"]["properties"]["type"]["type"] == "string"


def test_catalogued_non_openai_model_uses_explicit_conservative_count():
    capacity = RequestCapacity("claude-sonnet-4-5", 200000, 8192)
    assert capacity.counter_source == "utf8_byte_upper_bound"
    with patch("litellm.token_counter", side_effect=AssertionError("must not select a fallback tokenizer")):
        assert capacity.count({"input": "\N{EURO SIGN}" * 200}) >= 600


@pytest.mark.parametrize(
    "model,field,kind,detail",
    [
        ("gpt-4o", "messages", "image_url", "auto"),
        ("gpt-4o", "messages", "image_url", "low"),
        ("gpt-4o", "messages", "image_url", "high"),
        ("gpt-4o", "input", "input_image", "auto"),
        ("gpt-4o", "input", "input_image", "low"),
        ("gpt-4o", "input", "input_image", "high"),
        ("claude-sonnet-4-5", "messages", "image", "auto"),
    ],
)
def test_inline_image_budget_depends_on_pixels_not_base64_size(model, field, kind, detail):
    import litellm
    from PIL import Image

    from datus.utils.image_content import IMAGE_OMITTED

    capacity = RequestCapacity(model, 8192, 1024, allow_estimated_overflow=False)
    screenshot = Image.new("RGB", (512, 512), "blue")

    def request(compression):
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG", compress_level=compression)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        url = f"data:image/png;base64,{encoded}"
        blocks = {
            "image_url": {"type": "image_url", "image_url": {"url": url, "detail": detail}},
            "input_image": {"type": "input_image", "image_url": url, "detail": detail},
            "image": {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
        }
        payload = {
            "instructions": "Inspect the chart",
            field: [{"role": "user", "content": [{"type": "text", "text": "Explain"}, blocks[kind]]}],
            "tools": [{"name": "describe_table", "description": "Read structure"}],
        }
        return payload, url

    compressed, compressed_url = request(9)
    uncompressed, _ = request(0)
    original = copy.deepcopy(uncompressed)
    assert len(str(uncompressed)) > 100 * len(str(compressed))
    # Identical pixels have identical image cost even with a much larger encoding.
    with patch("litellm.token_counter", wraps=litellm.token_counter) as counter:
        measured = capacity.check(uncompressed)
    counted_images = [call.kwargs["messages"] for call in counter.call_args_list if "messages" in call.kwargs]
    assert len(counted_images) == 1
    assert counted_images[0][0]["content"][0]["image_url"].get("detail", "auto") == detail
    assert measured == capacity.count(compressed)
    assert uncompressed == original
    text_only = copy.deepcopy(uncompressed)
    text_only[field][0]["content"][1] = {
        "type": "input_text" if kind == "input_image" else "text",
        "text": IMAGE_OMITTED,
    }
    image_cost = litellm.token_counter(
        model=model,
        messages=[
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": compressed_url, "detail": detail}}]}
        ],
        use_default_image_token_count=False,
    )
    assert image_cost > 0
    assert measured == capacity.count(text_only) + image_cost
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        capacity.check({**uncompressed, "instructions": "large instructions " * 10000})
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        capacity.check({**uncompressed, "tools": [{"name": "large_tool", "description": "schema " * 10000}]})


@pytest.mark.parametrize("kind", ["function_call_output", "tool_result"])
def test_image_in_tool_result_is_counted(kind):
    from PIL import Image

    from datus.utils.image_content import IMAGE_OMITTED

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
    url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    image = {"type": "image_url", "image_url": {"url": url}}
    if kind == "function_call_output":
        item = {"type": kind, "call_id": "call-1", "output": [image]}
        text_item = {**item, "output": [{"type": "text", "text": IMAGE_OMITTED}]}
    else:
        item = {"role": "user", "content": [{"type": kind, "tool_use_id": "call-1", "content": [image]}]}
        text_item = {
            "role": "user",
            "content": [{"type": kind, "tool_use_id": "call-1", "content": [{"type": "text", "text": IMAGE_OMITTED}]}],
        }
    capacity = RequestCapacity("gpt-4o", 8192, 1024)
    assert capacity.count({"input": [item]}) > capacity.count({"input": [text_item]})


def test_small_context_default_reserves_both_margin_and_input():
    model = OpenAIModel(ModelConfig(type="openai", model="gpt-4o", api_key="test", context_length=4096))
    capacity = model.request_capacity()
    assert capacity.margin == 2048
    assert model.max_tokens() == capacity.default_output == 1024
    assert capacity.input_limit() == 1024
    request = {"messages": [{"role": "user", "content": "hello"}]}
    assert model.check_request(request)["max_tokens"] == 1024
    assert model._build_agent("Hello", str, False, {}, []).model_settings.max_tokens == 1024
    assert model.check_request({**request, "max_tokens": 1500})["max_tokens"] == 1500
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        model.check_request({**request, "max_tokens": 2048})
    model.model_config.max_tokens = 2048
    assert model.max_tokens() == 2048
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        model.check_request(request)


def test_known_output_ceiling_is_not_silently_clamped():
    capacity = RequestCapacity("gpt-4o", 262144, 8192, output_ceiling=16384)
    with pytest.raises(ContextCapacityError, match="OUTPUT_LIMIT_EXCEEDED"):
        capacity.output_limit(20000)
    assert capacity.output_limit(4000) == 4000


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_file", "file_id": "file-123"},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "input_image", "file_id": "file-123"},
    ],
)
def test_opaque_media_is_not_counted_as_a_short_identifier(block):
    assert RequestCapacity("gpt-4o", 8192, 1024).check({"input": [{"role": "user", "content": [block]}]}) >= 4096


def test_extra_body_effective_output_is_counted():
    capacity = RequestCapacity("gpt-4o", 8192, 100)
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        guard_request(
            capacity,
            {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100, "extra_body": {"max_tokens": 8192}},
        )


@pytest.mark.parametrize("field,value", [("context_length", 0), ("max_tokens", -1), ("context_length", True)])
def test_model_capacity_config_is_validated(field, value):
    with pytest.raises(ValueError):
        ModelConfig(type="openai", api_key="test", model="proxy", **{field: value})


def test_capacity_config_round_trips():
    model = load_model_config(
        {
            "type": "openai",
            "model": "proxy",
            "context_length": 262144,
            "max_tokens": 8192,
            "tokenizer_ref": "tiktoken:o200k_base",
        }
    )
    assert load_model_config(model.to_dict()) == model


@pytest.fixture
def model():
    return OpenAIModel(
        ModelConfig(
            type="openai",
            api_key="test",
            model="gpt-4o",
            base_url="https://proxy.example/v1",
            context_length=8192,
            max_tokens=1024,
            tokenizer_ref="tiktoken:o200k_base",
        )
    )


def test_direct_generation_is_blocked_before_http(model):
    with patch("datus.models.openai_compatible.litellm.completion") as send:
        with pytest.raises(ContextCapacityError):
            model.generate("word " * 20000)
    send.assert_not_called()


def test_sdk_and_summary_output_override_reaches_settings(model):
    agent = model._build_agent("", str, True, {}, [], max_tokens=400)
    assert agent.model_settings.max_tokens == 400
    assert model.max_tokens() == 1024


def test_rewriter_reserves_subscription_ceiling_instead_of_smaller_request(model):
    from types import SimpleNamespace

    from datus.agent.node.context_rewriter import MidTurnCompactor

    capacity = RequestCapacity("gpt-4o", 262144, 16384, supports_output_limit=False, output_ceiling=32768)
    with patch.object(model, "request_capacity", return_value=capacity):
        rewriter = MidTurnCompactor(SimpleNamespace(model=model))
        rewriter.set_request_context("system", [], 4000)
        assert rewriter._output_tokens == 32768


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_sdk_transport_blocks_before_http_for_both_modes(model, stream):
    agent = model._build_agent("", str, True, {}, [])
    with patch("litellm.acompletion", new_callable=AsyncMock) as send:
        with pytest.raises(ContextCapacityError):
            await agent.model._fetch_response(
                "system",
                [{"role": "user", "content": "word " * 20000}],
                ModelSettings(max_tokens=1024),
                [],
                None,
                [],
                MagicMock(),
                ModelTracing.DISABLED,
                stream=stream,
            )
    send.assert_not_called()


def test_response_transport_checks_merged_extra_body():
    from openai import AsyncOpenAI

    from datus.models.observed_model import ObservedResponsesModel

    transport = ObservedResponsesModel("gpt-4o", AsyncOpenAI(api_key="test"))
    transport._datus_capacity = RequestCapacity("gpt-4o", 8192, 100)
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        transport._build_response_create_kwargs(
            "system",
            [{"role": "user", "content": "hello"}],
            ModelSettings(max_tokens=100, extra_body={"max_output_tokens": 8192}),
            [],
            None,
            [],
        )


def test_native_transport_blocks_before_create():
    from datus.models.claude_model import ClaudeModel

    cfg = ModelConfig(
        type="claude",
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
        use_native_api=True,
        context_length=8192,
        max_tokens=1024,
    )
    native = ClaudeModel(cfg)
    with patch.object(native, "_raw_anthropic_messages_create") as send:
        with pytest.raises(ContextCapacityError):
            native._anthropic_messages_create(
                model=cfg.model, messages=[{"role": "user", "content": "word " * 20000}], max_tokens=1024
            )
    send.assert_not_called()


@pytest.mark.asyncio
async def test_summary_chunks_and_merge_each_fit_actual_budget(model):
    from datus.models.summary_chunks import bounded_summary, summary_payload

    items = [{"role": "user", "content": "word " * 3500}]
    with patch.object(
        model,
        "summarize_items",
        new_callable=AsyncMock,
        return_value={"content": "short summary", "usage": {"output_tokens": 3}},
    ) as summarize:
        text, used = await bounded_summary(
            model, items, instruction="summarize", prompt="keep progress", item_format="responses", agent_name="test"
        )
    assert text == "short summary"
    assert 2 < summarize.await_count <= 8
    assert used == summarize.await_count * 3
    for call in summarize.await_args_list:
        assert call.kwargs["max_tokens"] == 4000
        model.request_capacity().check(summary_payload(call.args[0], "summarize", "keep progress"), 4000)
    assert items[0]["content"] == "word " * 3500
    assert model.max_tokens() == 1024


@pytest.mark.asyncio
async def test_summary_chunks_fit_the_real_sdk_transport(model):
    import litellm

    from datus.models.summary_chunks import bounded_summary

    response = litellm.ModelResponse(
        model="gpt-4o",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "short summary"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    # Exercise the real summary method, SDK converter, and transport guard.
    # Only the external completion is replaced; an oversized planned chunk
    # fails in the actual transport before this mock can be reached.
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response) as send:
        summary, _ = await bounded_summary(
            model,
            [{"role": "assistant", "content": "catalog entry " * 3500}],
            instruction="Retain project decisions.",
            prompt="Summarize progress.",
            item_format="responses",
            agent_name="test",
        )
    assert summary == "short summary"
    assert 2 < send.await_count <= 8


@pytest.mark.asyncio
async def test_summary_planning_supports_official_responses_transport():
    from datus.models.summary_chunks import bounded_summary

    model = OpenAIModel(
        ModelConfig(type="openai", api_key="test", model="gpt-4o", context_length=8192, max_tokens=1024)
    )
    with patch.object(
        model, "summarize_items", new_callable=AsyncMock, return_value={"content": "short summary", "usage": {}}
    ) as summarize:
        text, _ = await bounded_summary(
            model,
            [{"role": "assistant", "content": "catalog entry " * 3500}],
            instruction="Retain project decisions.",
            prompt="Summarize progress.",
            item_format="responses",
            agent_name="test",
        )
    assert text == "short summary"
    assert 2 < summarize.await_count <= 8


@pytest.mark.asyncio
async def test_summary_budget_exhaustion_does_not_start_unbounded_calls(model):
    from datus.models.summary_chunks import bounded_summary

    with patch.object(model, "summarize_items", new_callable=AsyncMock) as summarize:
        with pytest.raises(ContextCapacityError, match="SUMMARY_LIMIT_EXCEEDED"):
            await bounded_summary(
                model,
                [{"role": "user", "content": "word " * 50000}],
                instruction="summarize",
                prompt="keep facts",
                item_format="responses",
                agent_name="test",
            )
    summarize.assert_not_awaited()


def test_subscription_budget_reserves_service_ceiling_without_wire_parameter():
    capacity = RequestCapacity("gpt-4o", 262144, 16384, supports_output_limit=False, output_ceiling=32768)
    assert capacity.output_limit(4000) == 32768
    assert capacity.input_limit(4000) == 216268


@pytest.mark.parametrize("structured", [False, True])
def test_subscription_capacity_failure_keeps_diagnostics_without_retry(structured):
    from datus.models.codex_model import CodexModel

    with patch("datus.models.codex_model.OAuthManager"):
        model = CodexModel(ModelConfig(type="codex", api_key="", model="gpt-5.3-codex", auth_type="oauth"))
    error = ContextCapacityError("CONTEXT_LIMIT_EXCEEDED", "too large", input_tokens=200000)
    with (
        patch.object(model, "_refresh_client_token"),
        patch.object(model, "_generate_direct_response", side_effect=error) as send,
    ):
        with pytest.raises(ContextCapacityError) as raised:
            (model.generate_with_json_output if structured else model.generate)("prompt")
    assert raised.value is error
    assert send.call_count == 1
    model.oauth_manager.refresh_tokens.assert_not_called()


@pytest.mark.asyncio
async def test_final_guard_survives_soft_compaction_failure(model):
    from types import SimpleNamespace

    from datus.agent.node.context_rewriter import MidTurnCompactor

    node = SimpleNamespace(
        model=model,
        compact_mid_turn=AsyncMock(side_effect=RuntimeError("summary failed")),
        context_length=8192,
        running_turn_usage=None,
    )
    rewriter = MidTurnCompactor(node)
    rewriter.set_request_context("system", [], 1024)
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        await rewriter.rewrite_sdk_input([{"role": "user", "content": "word " * 20000}])


@pytest.mark.asyncio
async def test_real_sdk_receives_native_page_without_reader(model, tmp_path, two_turn_agents_model):
    """The producer returns a page to both the model and tool events."""
    from types import SimpleNamespace

    from agents import Agent, Runner

    from datus.agent.node.context_rewriter import MidTurnCompactor
    from datus.models.base import LLMBaseModel
    from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
    from datus.utils.result_pagination import SnapshotStore, list_page, native_pagination
    from datus.utils.tool_archive import ToolArchive
    from datus.utils.tool_result_store import ToolResultStore, parse_result

    tables = [{"name": f"table_{i}"} for i in range(30000)]

    @tool_contract(access="read", direct_call=True, category="db_tools")
    @native_pagination()
    def list_tables(page_size: int | None = None):
        """Return a table inventory page, default 200, with no hard size cap."""
        rid = SnapshotStore().save("list_tables", tables, {}, {})
        return FuncToolResult(result=list_page(tables, rid, page_size, 0))

    store = ToolResultStore(ToolArchive("p", "s", base_dir=tmp_path), model.request_capacity().count)
    node = SimpleNamespace(
        model=model,
        context_length=8192,
        running_turn_usage=None,
        _get_tool_result_store=lambda: store,
        compact_mid_turn=AsyncMock(return_value={"mode": "noop", "success": True}),
    )
    rewriter = MidTurnCompactor(node)
    scripted = two_turn_agents_model(tool_name="list_tables")
    agent = Agent(name="paging", model=scripted, tools=[trans_to_function_tool(list_tables)])
    config = LLMBaseModel._build_run_config(model, context_rewriter=rewriter, agent_name="paging")
    config.tracing_disabled = True
    run = Runner.run_streamed(agent, input="inspect tables", run_config=config)
    events = [event async for event in run.stream_events()]
    assert scripted.turn == 2
    output = next(item for item in scripted.inputs[1] if item.get("type") == "function_call_output")
    page = parse_result(output["output"])
    assert page["result"]["has_more"] is True
    assert len(page["result"]["items"]) == 200
    raw_event = next(
        event.item for event in events if getattr(getattr(event, "item", None), "type", None) == "tool_call_output_item"
    )
    assert raw_event.output == page
    assert len(tables) == 30000


def _provider_model(provider, custom):
    from datus.models.claude_model import ClaudeModel
    from datus.models.deepseek_model import DeepSeekModel

    model_class, model_name, official_url = (
        (ClaudeModel, "claude-sonnet-4-5", "https://api.anthropic.com")
        if provider == "claude"
        else (DeepSeekModel, "deepseek-v4-flash", "https://api.deepseek.com")
    )
    return model_class(
        ModelConfig(
            type=provider,
            model=model_name,
            api_key="test",
            context_length=8192,
            max_tokens=1024,
            base_url="https://private.example/v1" if custom else official_url,
        )
    )


@pytest.mark.parametrize("provider", ["claude", "deepseek"])
def test_known_provider_preserves_standard_context_guard(provider):
    model = _provider_model(provider, custom=False)
    payload = {"messages": [{"role": "user", "content": "word " * 20000}]}
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        model.check_request(payload)


@pytest.mark.parametrize("provider", ["claude", "deepseek"])
def test_known_provider_private_endpoint_allows_estimated_overflow(provider):
    model = _provider_model(provider, custom=True)
    payload = {"messages": [{"role": "user", "content": "word " * 20000}]}
    assert model.request_capacity().check(payload) > model.request_capacity().input_limit()


def test_unknown_model_keeps_estimation_fallback_with_explicit_window():
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="unpublished-deployment",
            api_key="test",
            context_length=8192,
            max_tokens=1024,
        )
    )
    payload = {"messages": [{"role": "user", "content": "word " * 20000}]}
    assert model.request_capacity().check(payload) > model.request_capacity().input_limit()


@pytest.mark.parametrize(
    "setting,output",
    [("config", 16384), ("max_tokens", 16384), ("max_completion_tokens", 8192), ("max_output_tokens", 6144)],
)
def test_explicit_output_reservation_must_fit_configured_window(setting, output):
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-deployment",
            api_key="test",
            base_url="https://private.example/v1",
            context_length=8192,
            max_tokens=output if setting == "config" else 1024,
        )
    )
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    if setting != "config":
        payload[setting] = output
    with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
        model.check_request(payload)


@pytest.mark.parametrize("name", ["gpt-4o", "private-deployment"])
def test_inferred_custom_window_does_not_reject_explicit_output_reservation(name):
    model = OpenAIModel(
        ModelConfig(
            type="openai", model=name, api_key="test", base_url="https://private.example/v1", max_tokens=2000000
        )
    )
    assert model.check_request({"input": "hello"})["max_tokens"] == 2000000


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


def test_failed_tokenizer_load_is_reused_across_capacity_instances(monkeypatch):
    from datus.models import request_capacity

    loader = Mock(side_effect=OSError("Hub unavailable"))
    monkeypatch.setattr(request_capacity, "_tokenizer", loader)
    monkeypatch.setattr(request_capacity, "monotonic", lambda: 1000.0, raising=False)
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-deployment",
            api_key="test",
            tokenizer_ref="tests/unavailable-tokenizer-reuse",
        )
    )
    for _ in range(10):
        assert model.check_request({"input": "hello"})["max_tokens"] > 0
    loader.assert_called_once()


def test_failed_tokenizer_can_recover_after_failure_cache_expires(monkeypatch):
    from datus.models import request_capacity

    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3]
    loader = Mock(side_effect=[OSError("Hub unavailable"), tokenizer])
    now = [1000.0]
    monkeypatch.setattr(request_capacity, "_tokenizer", loader)
    monkeypatch.setattr(request_capacity, "monotonic", lambda: now[0], raising=False)
    capacity = RequestCapacity("private-deployment", 8192, 1024, tokenizer_ref="tests/unavailable-tokenizer-recover")
    assert capacity.estimate({"input": "hello"}).estimated
    assert capacity.estimate({"input": "hello"}).estimated
    loader.assert_called_once()
    now[0] += 3600
    estimate = capacity.estimate({"input": "hello"})
    assert not estimate.estimated
    assert estimate.source == capacity.tokenizer_ref
    assert loader.call_count == 2


def test_failed_tokenizer_does_not_block_other_references(monkeypatch):
    from datus.models import request_capacity

    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(request_capacity, "_tokenizer", Mock(side_effect=[OSError("Hub unavailable"), tokenizer]))
    failed = RequestCapacity("private-deployment", 8192, 1024, tokenizer_ref="tests/unavailable-tokenizer-isolated")
    available = RequestCapacity("private-deployment", 8192, 1024, tokenizer_ref="tests/available-tokenizer")
    assert failed.estimate({"input": "hello"}).estimated
    assert not available.estimate({"input": "hello"}).estimated


@pytest.mark.parametrize("name", ["gpt-4o", "private-deployment"])
def test_empty_tokenizer_encoding_uses_estimates_and_preserves_endpoint_policy(monkeypatch, name):
    tokenizer = Mock()
    tokenizer.encode.return_value = []
    monkeypatch.setattr("datus.models.request_capacity._tokenizer", Mock(return_value=tokenizer))
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model=name,
            api_key="test",
            context_length=8192,
            max_tokens=1024,
            tokenizer_ref="tests/empty-tokenizer-encoding",
        )
    )
    payload = {"input": "word " * 20000}
    capacity = model.request_capacity()
    estimate = capacity.estimate(payload)
    assert estimate.estimated
    assert estimate.trusted_tokens == 0
    assert estimate.tokens > capacity.input_limit()
    if name == "gpt-4o":
        assert estimate.source == "litellm_serialized_request"
        with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
            model.check_request(payload)
    else:
        assert estimate.source == "utf8_byte_upper_bound"
        assert model.check_request(payload)["max_tokens"] == 1024


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


@pytest.mark.parametrize("result", [None, 0, -1, True, "unknown"])
def test_invalid_counter_result_falls_back_without_invalidating_request(monkeypatch, result):
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: result)
    capacity = RequestCapacity("gpt-4o", 8192, 1024)
    estimate = capacity.estimate({"input": "hello"})
    assert estimate.tokens > 0
    assert estimate.estimated
    assert estimate.source == "utf8_byte_upper_bound"


def test_disabled_litellm_counter_estimates_text_and_images(monkeypatch, image_url):
    monkeypatch.setattr(litellm, "disable_token_counter", True)
    capacity = RequestCapacity("gpt-4o", 8192, 1024)
    payload = {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": image_url}]}]}
    estimate = capacity.estimate(payload)
    assert estimate.estimated
    assert estimate.source == "utf8_byte_upper_bound"
    assert estimate.tokens >= 4096
    assert estimate.trusted_tokens == 0


def test_file_estimation_preserves_tool_schemas_and_business_data():
    capacity = RequestCapacity("private-deployment", 8192, 1024)
    business = {"type": "input_file", "file_data": "A" * 10000}
    payload = {"messages": [{"role": "tool", "content": business}], "tools": [{"parameters": business}]}
    assert capacity.count(payload) > 20000
