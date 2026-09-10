# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Image decoding, file policy, and SDK output contract; no network calls."""

import base64
import io
import json
from types import SimpleNamespace

import pytest
from agents.items import ItemHelpers
from agents.tool import ToolOutputImage
from PIL import Image

from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.utils.image_content import (
    IMAGE_OMITTED,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_FILE_BYTES,
    anthropic_image_tool_content,
    contains_images,
    image_result_for_display,
    redact_image_payloads,
    replace_images,
    tool_images_as_user_input,
)


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "chart.png"
    Image.new("RGB", (80, 40), "red").save(path)
    return path


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_reads_actual_format_independent_of_filename(tmp_path, fmt):
    path = tmp_path / "image.bin"
    Image.new("RGB", (80, 40), "red").save(path, format=fmt)
    output = FilesystemFuncTool(root_path=str(tmp_path)).read_image(path.name)
    assert isinstance(output[1], ToolOutputImage)
    metadata = image_result_for_display(output)
    assert metadata["success"] == 1
    assert metadata["result"]["path"] == path.name
    payload = base64.b64decode(output[1].image_url.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as decoded:
        assert decoded.size == (80, 40)
        assert decoded.getpixel((0, 0))[0] > 240


def test_orientation_resize_and_metadata(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (3000, 1500), "white").save(path, exif=exif)
    output = FilesystemFuncTool(root_path=str(tmp_path)).read_image(path.name)
    metadata = image_result_for_display(output)["result"]
    assert (metadata["width"], metadata["height"]) == (1024, MAX_IMAGE_DIMENSION)
    with Image.open(io.BytesIO(base64.b64decode(output[1].image_url.split(",", 1)[1]))) as decoded:
        assert not decoded.getexif()


@pytest.mark.parametrize("kind", ["missing", "directory", "corrupt", "large", "pixels", "gif", "animated"])
def test_invalid_images_return_recoverable_errors(tmp_path, kind):
    path = tmp_path / "image.png"
    if kind == "directory":
        path.mkdir()
    elif kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "large":
        with path.open("wb") as stream:
            stream.truncate(MAX_IMAGE_FILE_BYTES + 1)
    elif kind == "pixels":
        Image.new("L", (5100, 5000)).save(path)
    elif kind == "gif":
        Image.new("RGB", (10, 10)).save(path, format="GIF")
    elif kind == "animated":
        Image.new("RGB", (10, 10), "red").save(path, save_all=True, append_images=[Image.new("RGB", (10, 10), "blue")])
    result = FilesystemFuncTool(root_path=str(tmp_path)).read_image(path.name)
    assert result.success == 0
    assert result.error


def test_strict_and_symlink_paths_use_existing_policy(tmp_path, image_file):
    root = tmp_path / "workspace"
    root.mkdir()
    tool = FilesystemFuncTool(root_path=str(root), strict=True)
    assert tool.read_image(str(image_file)).success == 0
    (root / "linked.png").symlink_to(image_file)
    assert tool.read_image("linked.png").success == 0
    assert tool.read_image("../chart.png").success == 0


def test_read_file_redirects_to_read_image(tmp_path, image_file):
    result = FilesystemFuncTool(root_path=str(tmp_path)).read_file(image_file.name)
    assert result.success == 0
    assert "read_image" in result.error


@pytest.mark.asyncio
async def test_function_invocation_returns_structured_sdk_image(tmp_path, image_file):
    tool = next(t for t in FilesystemFuncTool(root_path=str(tmp_path)).available_tools() if t.name == "read_image")
    output = await tool.on_invoke_tool(None, json.dumps({"path": image_file.name}))
    item = ItemHelpers.tool_call_output_item(SimpleNamespace(call_id="image_1"), output)
    assert item["output"][1]["type"] == "input_image"
    assert item["output"][1]["image_url"].startswith("data:image/png;base64,")


def test_native_conversion_and_display_do_not_mutate_image(tmp_path, image_file):
    output = FilesystemFuncTool(root_path=str(tmp_path)).read_image(image_file.name)
    native = anthropic_image_tool_content(output)
    assert native[1]["source"]["media_type"] == "image/png"
    assert native[1]["source"]["data"] == output[1].image_url.split(",", 1)[1]
    for value in (output, native, json.dumps(native)):
        display = redact_image_payloads(value)
        assert native[1]["source"]["data"] not in str(display)
    assert contains_images(output)
    assert contains_images(native)
    replaced = replace_images(native, IMAGE_OMITTED)
    assert replaced[1] == {"type": "text", "text": IMAGE_OMITTED}
    assert not contains_images(replaced)


def test_chat_transport_keeps_parallel_tool_results_together(tmp_path, image_file):
    output = FilesystemFuncTool(root_path=str(tmp_path)).read_image(image_file.name)
    item = ItemHelpers.tool_call_output_item(SimpleNamespace(call_id="image_1"), output)
    other = {"type": "function_call_output", "call_id": "sql_1", "output": "SELECT 1"}
    original = [item, other]
    converted = tool_images_as_user_input(original)
    assert [i.get("type") for i in converted[:2]] == ["function_call_output"] * 2
    assert converted[2]["role"] == "user"
    assert contains_images(converted[2])
    assert not contains_images(converted[:2])
    assert contains_images(original[0])
    assert tool_images_as_user_input(converted) == converted


def test_non_image_structured_results_keep_their_shape():
    value = {"type": {"name": "struct"}, "fields": ([{"type": ["int", "string"]}],)}
    assert not contains_images(value)
    assert redact_image_payloads(value) is value


def test_image_views_render_query_records_and_protocol_content(tmp_path, image_file):
    output = FilesystemFuncTool(root_path=str(tmp_path)).read_image(image_file.name)
    sdk_item = ItemHelpers.tool_call_output_item(SimpleNamespace(call_id="image_1"), output)
    row = {"type": "image_url", "count": 42}
    # A business column may itself contain data shaped like an image block.
    source = {"type": "input_image", "image_url": "https://example.invalid/chart.png"}
    result = {"success": 1, "result": {"data": [row, source]}}
    assert image_result_for_display(result) == result
    assert redact_image_payloads(result) == result
    assert replace_images(result) == result
    assert image_result_for_display([row, source]) == [row, source]

    mixed = [{"role": "user", "content": [{"type": "text", "text": json.dumps(result)}]}, sdk_item]
    rewritten = replace_images(mixed)
    assert rewritten[0] == mixed[0]
    assert rewritten[1]["output"][1] == {"type": "input_text", "text": IMAGE_OMITTED}


def test_trace_view_masks_native_image_bytes_and_preserves_other_data(tmp_path, image_file):
    native = anthropic_image_tool_content(FilesystemFuncTool(root_path=str(tmp_path)).read_image(image_file.name))
    value = {"business": {"type": "image_url", "data": "SALE42"}, "messages": [{"role": "user", "content": native}]}
    expected_source = {"type": "base64", "media_type": "image/png", "data": "[image data omitted]"}
    structured = redact_image_payloads(value)
    assert structured["business"] == value["business"]
    assert structured["messages"][0]["content"][1]["source"] == expected_source
    encoded = json.loads(redact_image_payloads(json.dumps(value)))
    assert encoded == structured


def test_hidden_images_are_not_read(tmp_path):
    hidden = tmp_path / ".datus" / "sessions"
    hidden.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(hidden / "chart.png")
    tool = FilesystemFuncTool(root_path=str(tmp_path), datus_home=str(tmp_path / ".datus"))
    assert tool.read_image(".datus/sessions/chart.png").success == 0


def test_payload_is_bounded_and_transparency_preserved(tmp_path, monkeypatch):
    import datus.utils.image_content as image_content

    path = tmp_path / "noise.png"
    pixels = Image.effect_noise((512, 512), 100).convert("RGBA")
    pixels.putalpha(128)
    pixels.save(path)
    monkeypatch.setattr(image_content, "MAX_IMAGE_PAYLOAD_BYTES", 4000)
    output = FilesystemFuncTool(root_path=str(tmp_path)).read_image(path.name)
    payload = base64.b64decode(output[1].image_url.split(",", 1)[1])
    assert len(payload) <= 4000
    with Image.open(io.BytesIO(payload)) as decoded:
        assert decoded.width < 512
        assert decoded.mode == "RGBA"
        assert decoded.getpixel((0, 0))[3] == 128


def test_log_redaction_omits_images_in_json_and_python_repr(tmp_path, image_file):
    from datus.utils.loggings import _redact_log_event

    native = anthropic_image_tool_content(FilesystemFuncTool(root_path=str(tmp_path)).read_image(image_file.name))
    data = native[1]["source"]["data"]
    for body in (json.dumps(native), str(native)):
        formatted = _redact_log_event(None, None, {"event": f"Request: {body}"})["event"]
        assert data not in formatted
        assert "chart.png" in formatted
    assert native[1]["source"]["data"] == data
