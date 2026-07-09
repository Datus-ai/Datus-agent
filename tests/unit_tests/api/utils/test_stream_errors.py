# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.api.utils.stream_errors — client-safe error humanization."""

from datus.api.utils.stream_errors import humanize_stream_error

# The exact litellm/AnthropicException shape reported in the field: a Python
# bytes-repr wrapping provider JSON whose message is byte-escaped Chinese
# ("网络错误 <id>，请稍后重试。").
LITELLM_BLOB = (
    "litellm.InternalServerError: AnthropicException - "
    'b\'{"type":"error","error":{"type":"api_error","code":"1234",'
    '"message":"[1234][\\xe7\\xbd\\x91\\xe7\\xbb\\x9c\\xe9\\x94\\x99\\xe8\\xaf\\xaf '
    "20260708110941e974e209bda24c95 \\xef\\xbc\\x8c\\xe8\\xaf\\xb7\\xe7\\xa8\\x8d"
    "\\xe5\\x90\\x8e\\xe9\\x87\\x8d\\xe8\\xaf\\x95\\xe3\\x80\\x82]"
    '[20260708110941e974e209bda24c95]"},"request_id":"20260708110941e974e209bda24c95"}\''
)


# Class names must match litellm's exactly — humanize_stream_error keys off the
# class name walked over the MRO, mirroring the real exception hierarchy.
class InternalServerError(Exception):
    """Stand-in mirroring litellm.InternalServerError's class name."""


class RateLimitError(Exception):
    pass


def test_decodes_litellm_blob_to_readable_chinese():
    error_type, message = humanize_stream_error(InternalServerError(LITELLM_BLOB))

    assert error_type == "UPSTREAM_ERROR"
    # The byte-escaped Chinese is decoded; the embedded request id is stripped.
    assert message == "网络错误 ，请稍后重试。"
    # No raw blob leaks through.
    assert "\\x" not in message
    assert "litellm" not in message
    assert "b'" not in message


def test_strips_hex_id_touching_cjk_without_spaces():
    # Some gateways emit "错误<id>，" with no separators. CJK chars are \w under
    # Unicode, so a \b-anchored pattern would leave the id in place.
    blob = (
        "litellm.InternalServerError: AnthropicException - "
        'b\'{"error":{"type":"api_error",'
        '"message":"\\xe9\\x94\\x99\\xe8\\xaf\\xaf'  # "错误" directly followed by the id
        "20260708110941e974e209bda24c95"
        "\\xef\\xbc\\x8c\\xe8\\xaf\\xb7\\xe7\\xa8\\x8d\\xe5\\x90\\x8e\\xe9\\x87\\x8d\\xe8\\xaf\\x95\\xe3\\x80\\x82\"}}'"
    )
    _, message = humanize_stream_error(InternalServerError(blob))

    assert message == "错误，请稍后重试。"
    assert "20260708110941e974e209bda24c95" not in message


def test_maps_known_class_to_stable_code():
    error_type, message = humanize_stream_error(RateLimitError("429 rate limited"))

    assert error_type == "UPSTREAM_RATE_LIMITED"
    assert message  # non-empty fallback sentence
    assert "\\x" not in message


def test_unknown_exception_falls_back_generically():
    error_type, message = humanize_stream_error(ValueError("boom"))

    assert error_type == "INTERNAL_ERROR"
    assert message == "Something went wrong while generating the response. Please try again."


def test_never_returns_empty_message():
    _, message = humanize_stream_error(InternalServerError(""))

    assert message
