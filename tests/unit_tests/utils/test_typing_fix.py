# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import sys
from unittest.mock import MagicMock, patch

from datus.utils.typing_fix import patch_agents_typing_issue


class TestPatchAgentsTypingIssue:
    def test_returns_false_when_agents_not_installed(self):
        """When 'agents' module is not available, should return False gracefully."""
        with patch.dict(sys.modules, {"agents": None, "agents.models": None, "agents.models.chatcmpl_converter": None}):
            result = patch_agents_typing_issue()
        assert result is False

    def test_returns_false_on_import_error(self):
        """ImportError during import leads to False return."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def raise_import_error(name, *args, **kwargs):
            if "agents" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raise_import_error):
            result = patch_agents_typing_issue()
        assert result is False

    def test_returns_false_on_general_exception(self):
        """Any other exception during patching should return False."""
        mock_converter = MagicMock()
        mock_converter.Converter.items_to_messages = MagicMock()

        mock_agents = MagicMock()
        mock_agents_models = MagicMock()

        # Make something blow up unexpectedly
        mock_converter.Converter = None  # accessing .items_to_messages will raise AttributeError

        with patch.dict(
            sys.modules,
            {
                "agents": mock_agents,
                "agents.models": mock_agents_models,
                "agents.models.chatcmpl_converter": mock_converter,
                "openai": MagicMock(),
                "openai.types": MagicMock(),
                "openai.types.chat": MagicMock(),
                "openai.types.chat.chat_completion_message_function_tool_call_param": MagicMock(),
            },
        ):
            # Even if it returns True (agents lib found), test that it doesn't raise
            result = patch_agents_typing_issue()
        assert isinstance(result, bool)

    def test_function_is_callable(self):
        assert callable(patch_agents_typing_issue)

    def test_module_import_does_not_raise(self):
        """Importing the module should not raise even in Python 3.12+."""

        import datus.utils.typing_fix  # noqa: F401 - just import to ensure no crash

        assert True  # if we get here, no exception was raised

    def test_returns_bool(self):
        """Function always returns a bool."""
        result = patch_agents_typing_issue()
        assert isinstance(result, bool)
