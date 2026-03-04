# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.utils.terminal_utils — terminal control character suppression."""

import sys
from io import StringIO

import pytest

from datus.utils.terminal_utils import suppress_keyboard_input


class TestSuppressKeyboardInput:
    """Tests for suppress_keyboard_input context manager."""

    def test_noop_when_stdin_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context manager is a no-op when stdin has no fileno (e.g. Streamlit)."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        # StringIO has no fileno(), should yield without error
        with suppress_keyboard_input():
            pass  # no-op path

    def test_noop_when_stdin_has_no_fileno(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context manager is a no-op when stdin.fileno() is unavailable."""

        class NoFileNo:
            def fileno(self) -> int:
                raise AttributeError("no fileno")

        monkeypatch.setattr(sys, "stdin", NoFileNo())
        with suppress_keyboard_input():
            pass  # should not raise

    def test_context_manager_yields(self) -> None:
        """Context manager yields control even in non-terminal environments."""
        executed = False
        with suppress_keyboard_input():
            executed = True
        assert executed is True

    def test_noop_when_fileno_raises_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context manager is a no-op when fileno() raises OSError."""

        class BadFileNo:
            def fileno(self) -> int:
                raise OSError("not a tty")

        monkeypatch.setattr(sys, "stdin", BadFileNo())
        with suppress_keyboard_input():
            pass  # should not raise


class TestSuppressKeyboardInputEdgeCases:
    """Edge case tests for suppress_keyboard_input."""

    def test_exception_propagates_through_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exceptions inside the context manager propagate normally."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        with pytest.raises(ValueError, match="test error"):
            with suppress_keyboard_input():
                raise ValueError("test error")
