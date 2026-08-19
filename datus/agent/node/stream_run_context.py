# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Per-invocation state container threaded through ``AgenticNode.execute_stream``.

A single :class:`StreamRunContext` is created at the top of every
``execute_stream`` call and passed to every hook so that subclass overrides
do not need long parameter lists. Subclasses may stash arbitrary scratch
state in :attr:`StreamRunContext.extras` when they need to share data
between hooks (e.g. ``_before_stream`` parsing ``gold_sql`` that
``_prepare_template_context`` later consumes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from agents.extensions.memory import AdvancedSQLiteSession

    from datus.cli.execution_state import PendingInputQueue
    from datus.schemas.action_history import ActionHistoryManager
    from datus.schemas.base import BaseInput


@dataclass
class StreamRunContext:
    """State threaded through one ``AgenticNode.execute_stream`` invocation.

    Attributes are populated in order by the template method:

    1. ``user_input`` / ``action_history_manager`` — set at construction.
    2. ``session`` — set after session setup; always populated so SDK
       receives prior items regardless of ``execution_mode``.
    3. ``system_instruction`` / ``user_prompt`` — set during prompt assembly.
    4. ``response_content`` / ``last_successful_output`` / ``last_tool_summary``
       — populated incrementally as actions flow through ``_stream_once``.
    5. ``attempt`` — incremented per retry iteration (1-based).
    """

    user_input: "BaseInput"
    action_history_manager: "ActionHistoryManager"

    session: Optional["AdvancedSQLiteSession"] = None

    system_instruction: str = ""
    user_prompt: str = ""

    # Allowed to be ``str`` (most nodes) or ``dict`` for structured outputs.
    response_content: Any = ""
    last_successful_output: Optional[Dict[str, Any]] = None
    last_tool_summary: str = ""

    attempt: int = 1

    # When set by ``_before_stream``, the template replaces
    # ``user_input.user_message`` for the duration of ``_build_enhanced_message``
    # and restores it afterwards. Used by Compare.
    user_message_override: Optional[str] = None

    # Free-form scratchpad for subclass hooks to share state.
    extras: Dict[str, Any] = field(default_factory=dict)

    # Per-run queue of free-text user messages staged during an agent run.
    # When set (typically by interactive callers like CLI/TUI and the API
    # ``/insert`` endpoint), the model layer attaches a
    # ``call_model_input_filter`` that drains this queue before each LLM
    # turn so injected text is seen on the very next request within the
    # same run. ``None`` for non-interactive callers (regression tests,
    # batch workflows) — those keep current behavior.
    pending_input_queue: Optional["PendingInputQueue"] = None
