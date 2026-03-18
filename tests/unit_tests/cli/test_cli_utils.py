# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.cli._cli_utils — select_choice and select_list."""

from unittest.mock import MagicMock, patch

import pytest

from datus.cli._cli_utils import _FREE_TEXT_SENTINEL, select_choice, select_list

_KEY_ALIASES = {"enter": "c-m", "backspace": "c-h"}


def _find_handler(kb, key_name):
    """Find a key-binding handler by key name string.

    Handles prompt_toolkit aliases (e.g. 'enter' -> 'c-m').
    """
    targets = {key_name, _KEY_ALIASES.get(key_name, key_name)}
    for binding in kb.bindings:
        for key in binding.keys:
            key_str = key.value if hasattr(key, "value") else str(key)
            if key_str in targets:
                return binding.handler
    return None


def _make_event():
    """Create a mock event with trackable exit."""
    event = MagicMock()
    event.app.exit = MagicMock()
    event.data = ""
    return event


def _capture_kb(choices, default="", allow_free_text=False):
    """Run select_choice and capture the KeyBindings object."""
    captured = {}

    def fake_app(**kwargs):
        captured["kb"] = kwargs.get("key_bindings")
        app = MagicMock()
        app.run.return_value = default
        return app

    with patch("prompt_toolkit.Application", side_effect=fake_app):
        select_choice(MagicMock(), choices, default=default, allow_free_text=allow_free_text)

    return captured["kb"]


class TestSelectChoiceBasic:
    """Tests for select_choice basic behaviour."""

    @pytest.mark.ci
    def test_free_text_sentinel_constant(self):
        assert isinstance(_FREE_TEXT_SENTINEL, str)
        assert len(_FREE_TEXT_SENTINEL) > 0

    @pytest.mark.ci
    @patch("prompt_toolkit.Application")
    def test_returns_selected_key(self, mock_app_cls):
        mock_app_cls.return_value.run.return_value = "y"
        result = select_choice(MagicMock(), {"y": "Yes", "n": "No"}, default="y")
        assert result == "y"

    @pytest.mark.ci
    @patch("datus.cli._cli_utils.prompt_input", return_value="my custom answer")
    @patch("prompt_toolkit.Application")
    def test_free_text_custom_answer(self, mock_app_cls, _mock_prompt_input):
        mock_app_cls.return_value.run.return_value = _FREE_TEXT_SENTINEL
        result = select_choice(MagicMock(), {"1": "A", "2": "B"}, default="1", allow_free_text=True)
        assert result == "my custom answer"

    @pytest.mark.ci
    @patch("datus.cli._cli_utils.prompt_input", return_value="")
    @patch("prompt_toolkit.Application")
    def test_free_text_empty_answer(self, mock_app_cls, _mock_prompt_input):
        mock_app_cls.return_value.run.return_value = _FREE_TEXT_SENTINEL
        result = select_choice(MagicMock(), {"1": "A", "2": "B"}, default="1", allow_free_text=True)
        assert result == ""

    @pytest.mark.ci
    @patch("prompt_toolkit.Application")
    def test_free_text_pick_key(self, mock_app_cls):
        mock_app_cls.return_value.run.return_value = "2"
        result = select_choice(MagicMock(), {"1": "A", "2": "B"}, default="1", allow_free_text=True)
        assert result == "2"

    @pytest.mark.ci
    def test_error_returns_default(self):
        with patch("prompt_toolkit.Application", side_effect=RuntimeError("no terminal")):
            result = select_choice(MagicMock(), {"y": "Yes", "n": "No"}, default="n")
        assert result == "n"

    @pytest.mark.ci
    @patch("prompt_toolkit.Application")
    def test_without_free_text(self, mock_app_cls):
        mock_app_cls.return_value.run.return_value = "n"
        result = select_choice(MagicMock(), {"y": "Yes", "n": "No"}, default="y", allow_free_text=False)
        assert result == "n"

    @pytest.mark.ci
    @patch("prompt_toolkit.Application")
    def test_keyboard_interrupt_returns_default(self, mock_app_cls):
        mock_app_cls.return_value.run.side_effect = KeyboardInterrupt
        result = select_choice(MagicMock(), {"y": "Yes", "n": "No"}, default="n")
        assert result == "n"


class TestSelectChoiceKeyBindings:
    """Test key-binding handlers directly."""

    @pytest.mark.ci
    def test_enter_exits_with_selected_key(self):
        kb = _capture_kb({"y": "Yes", "n": "No"}, default="y")
        handler = _find_handler(kb, "enter")
        event = _make_event()
        handler(event)
        event.app.exit.assert_called_once_with(result="y")

    @pytest.mark.ci
    def test_cancel_exits_with_default(self):
        kb = _capture_kb({"y": "Yes", "n": "No"}, default="n")
        handler = _find_handler(kb, "c-c")
        event = _make_event()
        handler(event)
        event.app.exit.assert_called_once_with(result="n")

    @pytest.mark.ci
    def test_shortcut_key_exits(self):
        kb = _capture_kb({"y": "Yes", "n": "No"}, default="y")
        handler = _find_handler(kb, "n")
        event = _make_event()
        handler(event)
        event.app.exit.assert_called_once_with(result="n")

    @pytest.mark.ci
    def test_up_navigates(self):
        kb = _capture_kb({"y": "Yes", "n": "No"}, default="n")
        handler = _find_handler(kb, "up")
        event = _make_event()
        # Should not crash and should not call exit
        handler(event)
        event.app.exit.assert_not_called()

    @pytest.mark.ci
    def test_down_navigates(self):
        kb = _capture_kb({"y": "Yes", "n": "No"}, default="y")
        handler = _find_handler(kb, "down")
        event = _make_event()
        handler(event)
        event.app.exit.assert_not_called()

    @pytest.mark.ci
    def test_slash_exits_with_free_text_sentinel(self):
        """Pressing '/' exits the selector with the free-text sentinel."""
        kb = _capture_kb({"1": "A", "2": "B"}, default="1", allow_free_text=True)
        handler = _find_handler(kb, "/")
        event = _make_event()
        handler(event)
        event.app.exit.assert_called_once_with(result=_FREE_TEXT_SENTINEL)

    @pytest.mark.ci
    def test_up_navigates_with_free_text_enabled(self):
        """Up arrow still navigates normally when free text is enabled."""
        kb = _capture_kb({"1": "A", "2": "B"}, default="1", allow_free_text=True)
        event = _make_event()
        _find_handler(kb, "up")(event)
        event.app.exit.assert_not_called()

    @pytest.mark.ci
    def test_down_navigates_with_free_text_enabled(self):
        """Down arrow still navigates normally when free text is enabled."""
        kb = _capture_kb({"1": "A", "2": "B"}, default="1", allow_free_text=True)
        event = _make_event()
        _find_handler(kb, "down")(event)
        event.app.exit.assert_not_called()

    @pytest.mark.ci
    def test_ctrl_c_returns_default_with_free_text_enabled(self):
        """Ctrl-C exits with default even when free text is enabled."""
        kb = _capture_kb({"1": "A", "2": "B"}, default="1", allow_free_text=True)
        event = _make_event()
        _find_handler(kb, "c-c")(event)
        event.app.exit.assert_called_once_with(result="1")

    @pytest.mark.ci
    def test_shortcut_key_still_selects_option_with_free_text_enabled(self):
        """Existing shortcut keys still select the matching option immediately."""
        kb = _capture_kb({"1": "A", "2": "B"}, default="1", allow_free_text=True)
        event = _make_event()
        _find_handler(kb, "1")(event)
        event.app.exit.assert_called_once_with(result="1")

    @pytest.mark.ci
    def test_any_key_handler_absent_with_prompt_based_free_text(self):
        """Prompt-based free text mode no longer registers a raw <any> handler."""
        kb = _capture_kb({"1": "A", "2": "B"}, default="1", allow_free_text=True)
        assert _find_handler(kb, "<any>") is None

    @pytest.mark.ci
    def test_enter_on_sentinel_exits_with_sentinel(self):
        """Pressing enter when free-text sentinel is selected exits with the sentinel."""
        kb = _capture_kb({"1": "A"}, default="1", allow_free_text=True)
        # Navigate down to the sentinel
        _find_handler(kb, "down")(_make_event())
        event = _make_event()
        _find_handler(kb, "enter")(event)
        event.app.exit.assert_called_once_with(result=_FREE_TEXT_SENTINEL)


def _capture_list_kb(items, **kwargs):
    """Run select_list and capture the KeyBindings object and internal state."""
    captured = {}

    def fake_app(**app_kwargs):
        captured["kb"] = app_kwargs.get("key_bindings")
        app = MagicMock()
        app.run.return_value = 0
        return app

    with patch("prompt_toolkit.Application", side_effect=fake_app):
        select_list(MagicMock(), items, **kwargs)

    return captured.get("kb")


class TestSelectList:
    """Tests for select_list interactive list selector."""

    @pytest.mark.ci
    @patch("prompt_toolkit.Application")
    def test_returns_selected_index(self, mock_app_cls):
        """select_list returns the index selected by the user."""
        mock_app_cls.return_value.run.return_value = 2
        items = [["a", "1"], ["b", "2"], ["c", "3"]]
        result = select_list(MagicMock(), items)
        assert result == 2

    @pytest.mark.ci
    @patch("prompt_toolkit.Application")
    def test_cancel_returns_none(self, mock_app_cls):
        """select_list returns None when user cancels."""
        mock_app_cls.return_value.run.return_value = None
        items = [["a", "1"], ["b", "2"]]
        result = select_list(MagicMock(), items)
        assert result is None

    @pytest.mark.ci
    def test_empty_items_returns_none(self):
        """select_list returns None for empty item list."""
        result = select_list(MagicMock(), [])
        assert result is None

    @pytest.mark.ci
    def test_wrap_around_up(self):
        """Up arrow at top wraps to bottom."""
        items = [["a"], ["b"], ["c"]]
        kb = _capture_list_kb(items)
        assert kb is not None
        # Press up from index 0 → should wrap to index 2
        handler = _find_handler(kb, "up")
        event = _make_event()
        handler(event)
        # Then press enter to confirm
        enter = _find_handler(kb, "enter")
        enter_event = _make_event()
        enter(enter_event)
        enter_event.app.exit.assert_called_once_with(result=2)

    @pytest.mark.ci
    def test_wrap_around_down(self):
        """Down arrow at bottom wraps to top."""
        items = [["a"], ["b"]]
        kb = _capture_list_kb(items)
        assert kb is not None
        handler = _find_handler(kb, "down")
        # Press down twice from index 0 → index 1 → wraps to index 0
        handler(_make_event())
        handler(_make_event())
        enter_event = _make_event()
        _find_handler(kb, "enter")(enter_event)
        enter_event.app.exit.assert_called_once_with(result=0)

    @pytest.mark.ci
    def test_scrolling_viewport(self):
        """Scrolling offset adjusts when selected goes past max_visible."""
        items = [[str(i)] for i in range(20)]
        kb = _capture_list_kb(items, max_visible=5)
        assert kb is not None
        handler = _find_handler(kb, "down")
        # Navigate down 6 times (past max_visible=5)
        for _ in range(6):
            handler(_make_event())
        # Should not crash; confirm selection
        enter_event = _make_event()
        _find_handler(kb, "enter")(enter_event)
        enter_event.app.exit.assert_called_once_with(result=6)

    @pytest.mark.ci
    def test_error_returns_none(self):
        """select_list returns None on exception."""
        with patch("prompt_toolkit.Application", side_effect=RuntimeError("no terminal")):
            result = select_list(MagicMock(), [["a"], ["b"]])
        assert result is None

    @pytest.mark.ci
    def test_escape_cancels(self):
        """Pressing Escape exits with None."""
        items = [["a"], ["b"]]
        kb = _capture_list_kb(items)
        assert kb is not None
        handler = _find_handler(kb, "escape")
        event = _make_event()
        handler(event)
        event.app.exit.assert_called_once_with(result=None)

    @pytest.mark.ci
    def test_ctrl_c_cancels(self):
        """Pressing Ctrl+C exits with None."""
        items = [["a"], ["b"]]
        kb = _capture_list_kb(items)
        assert kb is not None
        handler = _find_handler(kb, "c-c")
        event = _make_event()
        handler(event)
        event.app.exit.assert_called_once_with(result=None)

    @pytest.mark.ci
    def test_with_headers(self):
        """select_list works correctly with headers provided."""
        items = [["val1", "val2"], ["val3", "val4"]]
        kb = _capture_list_kb(items, headers=["Col A", "Col B"])
        assert kb is not None
        # Confirm first item selected
        enter_event = _make_event()
        _find_handler(kb, "enter")(enter_event)
        enter_event.app.exit.assert_called_once_with(result=0)

    @pytest.mark.ci
    def test_pagedown(self):
        """PageDown jumps forward."""
        items = [[str(i)] for i in range(20)]
        kb = _capture_list_kb(items, max_visible=5)
        assert kb is not None
        _find_handler(kb, "pagedown")(_make_event())
        enter_event = _make_event()
        _find_handler(kb, "enter")(enter_event)
        enter_event.app.exit.assert_called_once_with(result=5)

    @pytest.mark.ci
    def test_pageup(self):
        """PageUp jumps backward."""
        items = [[str(i)] for i in range(20)]
        kb = _capture_list_kb(items, max_visible=5)
        assert kb is not None
        # Go down 10, then page up (max(0, 10-5) = 5)
        for _ in range(10):
            _find_handler(kb, "down")(_make_event())
        _find_handler(kb, "pageup")(_make_event())
        enter_event = _make_event()
        _find_handler(kb, "enter")(enter_event)
        enter_event.app.exit.assert_called_once_with(result=5)

    @pytest.mark.ci
    def test_clip_truncates_long_text(self):
        """Long text is clipped to terminal display width."""
        long_text = "A" * 200
        items = [[long_text]]
        captured_text = {}

        def fake_app(**app_kwargs):
            layout = app_kwargs.get("layout")
            if layout:
                window = layout.container
                control = window.content
                captured_text["lines"] = control.text()
            app = MagicMock()
            app.run.return_value = 0
            return app

        with (
            patch("prompt_toolkit.Application", side_effect=fake_app),
            patch("shutil.get_terminal_size", return_value=(20, 40)),
        ):
            select_list(MagicMock(), items)

        lines = captured_text.get("lines", [])
        primary_line = lines[0][1] if lines else ""
        # content_width = 20 - 6 = 14. Prefix "  -> " = 4 chars. Total visible = 14 of A's
        assert len(primary_line.strip()) <= 14 + 4  # clipped content + prefix

    @pytest.mark.ci
    def test_keyboard_interrupt_returns_none(self):
        """KeyboardInterrupt during Application.run returns None."""
        with patch("prompt_toolkit.Application") as mock_app_cls:
            mock_app_cls.return_value.run.side_effect = KeyboardInterrupt
            result = select_list(MagicMock(), [["a"], ["b"]])
        assert result is None

    @pytest.mark.ci
    def test_single_column_no_secondary(self):
        """Items with only primary line (no secondary) render without error."""
        items = [["only primary"]]
        captured_text = {}

        def fake_app(**app_kwargs):
            layout = app_kwargs.get("layout")
            if layout:
                window = layout.container
                control = window.content
                captured_text["lines"] = control.text()
            app = MagicMock()
            app.run.return_value = 0
            return app

        with patch("prompt_toolkit.Application", side_effect=fake_app):
            select_list(MagicMock(), items)

        lines = captured_text.get("lines", [])
        # Should have: primary, secondary (empty), blank, hint = 4 lines
        assert len(lines) >= 3

    @pytest.mark.ci
    def test_scroll_info_shown(self):
        """Scroll info is displayed when items exceed max_visible."""
        items = [[str(i)] for i in range(10)]
        captured_text = {}

        def fake_app(**app_kwargs):
            layout = app_kwargs.get("layout")
            if layout:
                window = layout.container
                control = window.content
                captured_text["lines"] = control.text()
            app = MagicMock()
            app.run.return_value = 0
            return app

        with patch("prompt_toolkit.Application", side_effect=fake_app):
            select_list(MagicMock(), items, max_visible=3)

        lines = captured_text.get("lines", [])
        assert any("1-3 of 10" in line[1] for line in lines)
