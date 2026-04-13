# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Web interface for Datus Agent.

Serves a React-based chatbot frontend backed by FastAPI.
"""

from datus.cli.web.chat_executor import ChatExecutor
from datus.cli.web.chatbot import StreamlitChatbot, run_web_interface
from datus.cli.web.config_manager import ConfigManager
from datus.cli.web.ui_components import UIComponents

__all__ = [
    "ChatExecutor",
    "ConfigManager",
    "StreamlitChatbot",
    "UIComponents",
    "run_web_interface",
]
