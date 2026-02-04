# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Typing compatibility fixes for agents library in Python 3.12+
"""
import sys


def patch_agents_typing_issue():
    """
    Monkey patch to fix typing.Union instantiation issue in agents library.
    This fixes the ChatCompletionMessageToolCallParam instantiation error.
    """
    try:
        import agents.models.chatcmpl_converter as converter
        from openai.types.chat.chat_completion_message_function_tool_call_param import (
            ChatCompletionMessageFunctionToolCallParam,
        )

        # Store the original items_to_messages function
        original_items_to_messages = converter.Converter.items_to_messages

        @staticmethod
        def patched_items_to_messages(input_items):
            """Patched version that handles Union types correctly."""
            try:
                return original_items_to_messages(input_items)
            except TypeError as e:
                if "Cannot instantiate typing.Union" in str(e):
                    # This is a bit hacky, but we need to patch the specific line that fails
                    # We'll monkey patch the ChatCompletionMessageToolCallParam import
                    original_param = converter.ChatCompletionMessageToolCallParam

                    # Replace the Union with the concrete type
                    converter.ChatCompletionMessageToolCallParam = ChatCompletionMessageFunctionToolCallParam

                    try:
                        result = original_items_to_messages(input_items)
                        return result
                    finally:
                        # Restore the original
                        converter.ChatCompletionMessageToolCallParam = original_param
                else:
                    raise

        # Apply the patch
        converter.Converter.items_to_messages = patched_items_to_messages

        # Also patch the direct import
        converter.ChatCompletionMessageToolCallParam = ChatCompletionMessageFunctionToolCallParam

        return True

    except ImportError:
        # print(f"Warning: Could not apply agents typing patch: {e}")
        return False
    except Exception:
        # print(f"Error applying agents typing patch: {e}")
        return False


# Apply the patch when this module is imported
if sys.version_info >= (3, 12):
    patch_agents_typing_issue()
