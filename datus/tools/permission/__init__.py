# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified permission system for tools, MCP, and skills.

This module provides pattern-based permission control (allow/deny/ask)
for all tool types in Datus-agent, following Claude Code and OpenCode patterns.
"""

from datus.tools.permission.bash_classifier import (
    BashClassifierContext,
    BashCommandClassifier,
    ClassifierVerdict,
    create_bash_classifier,
)
from datus.tools.permission.bash_rules import (
    BashCommandRules,
    BashRuleDecision,
    evaluate_bash_command,
)
from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)
from datus.tools.permission.permission_hooks import (
    CompositeHooks,
    PermissionDeniedException,
    PermissionHooks,
)
from datus.tools.permission.permission_manager import PermissionManager

__all__ = [
    "PermissionLevel",
    "PermissionRule",
    "PermissionConfig",
    "PermissionManager",
    "PermissionHooks",
    "PermissionDeniedException",
    "CompositeHooks",
    "BashCommandRules",
    "BashRuleDecision",
    "evaluate_bash_command",
    "BashCommandClassifier",
    "BashClassifierContext",
    "ClassifierVerdict",
    "create_bash_classifier",
]
