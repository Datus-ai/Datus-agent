# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.tui.clipboard`."""

from __future__ import annotations

import base64
import io
import sys

import pytest

from datus.cli.tui import clipboard as clipboard_mod
from datus.cli.tui.clipboard import copy_to_clipboard


@pytest.fixture(autouse=True)
def _no_pyperclip_by_default(monkeypatch):
    """Force the OSC 52 fallback unless a test injects its own pyperclip stub.

    Lets tests target one backend at a time without the host machine's
    pyperclip behaviour leaking in.
    """

    def _fail_pyperclip(text):  # noqa: ANN001
        raise RuntimeError("pyperclip disabled for tests")

    fake = type("FakePyperclip", (), {"copy": staticmethod(_fail_pyperclip)})
    monkeypatch.setitem(sys.modules, "pyperclip", fake)


def test_empty_text_returns_false_without_writing():
    stream = io.StringIO()
    assert copy_to_clipboard("", stream=stream) is False
    assert stream.getvalue() == ""


def test_pyperclip_success_short_circuits(monkeypatch):
    """When pyperclip succeeds, no OSC 52 escape is written."""
    calls = []

    fake = type("FakePyperclip", (), {"copy": staticmethod(lambda text: calls.append(text))})
    monkeypatch.setitem(sys.modules, "pyperclip", fake)

    stream = io.StringIO()
    assert copy_to_clipboard("hello", stream=stream) is True
    assert calls == ["hello"]
    assert stream.getvalue() == ""  # OSC 52 not invoked


def test_osc52_fallback_writes_escape_to_stream(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    stream = io.StringIO()
    assert copy_to_clipboard("hello", stream=stream) is True
    written = stream.getvalue()
    assert written.startswith("\x1b]52;c;")
    assert written.endswith("\x07")
    # Decode the base64 payload and confirm round-trip.
    payload = written[len("\x1b]52;c;") : -1]
    assert base64.b64decode(payload).decode() == "hello"


def test_osc52_unicode_round_trip(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    stream = io.StringIO()
    copy_to_clipboard("你好 world", stream=stream)
    written = stream.getvalue()
    payload = written[len("\x1b]52;c;") : -1]
    assert base64.b64decode(payload).decode("utf-8") == "你好 world"


def test_osc52_tmux_wraps_with_pass_through(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
    stream = io.StringIO()
    assert copy_to_clipboard("hi", stream=stream) is True
    written = stream.getvalue()
    # tmux pass-through: outer wrapper, ESCs doubled inside.
    assert written.startswith("\x1bPtmux;")
    assert written.endswith("\x1b\\")
    # The original ESC (0x1b) bytes inside the inner payload must be
    # doubled so tmux unwraps cleanly. Only the opening ESC and any
    # internal ESCs are doubled; the trailing BEL (0x07) is a non-ESC
    # terminator so it stays single.
    inner = written[len("\x1bPtmux;") : -len("\x1b\\")]
    assert inner.startswith("\x1b\x1b]52;c;")
    assert inner.endswith("\x07")


def test_osc52_skipped_when_payload_too_large(monkeypatch):
    """Selections beyond 100 KiB encoded fall through silently."""
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(clipboard_mod, "_OSC52_MAX_BYTES", 8)
    stream = io.StringIO()
    # 9 bytes encoded → "MTIzNDU2Nzg5" (12 chars) > 8 limit.
    assert copy_to_clipboard("123456789", stream=stream) is False
    assert stream.getvalue() == ""
