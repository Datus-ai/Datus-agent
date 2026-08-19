# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified permission system for tools, MCP, and skills.

This module provides pattern-based permission control (allow/deny/ask)
for all tool types in Datus-agent, following Claude Code and OpenCode patterns.
"""

from datus.schemas.action_history import register_action_enricher
from datus.tools.permission.bash_classifier import (
    BashClassifierContext,
    BashCommandClassifier,
    ClassifierVerdict,
    create_bash_classifier,
)
from datus.tools.permission.bash_rules import (
    BashCommandRules,
    BashRuleDecision,
    BashSegmentDecision,
    evaluate_bash_command,
)
from datus.tools.permission.permission_config import (
    AutoReviewConfig,
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
    SqlStatementRules,
    classify_sql_kind,
)
from datus.tools.permission.permission_hooks import (
    CompositeHooks,
    PermissionDeniedException,
    PermissionHooks,
)
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.review_registry import (
    PERMISSION_REVIEW_OUTPUT_KEY,
    enrich_action,
)

# Wire the AI-review annotation into every action history at import time. The
# permission package owns the dependency direction: ``datus.schemas`` stays
# unaware of reviews, and any consumer that imports the permission system gets
# reviewed tool actions annotated for display.
register_action_enricher(enrich_action)

__all__ = [
    "PERMISSION_REVIEW_OUTPUT_KEY",
    "PermissionLevel",
    "AutoReviewConfig",
    "PermissionRule",
    "PermissionConfig",
    "PermissionManager",
    "PermissionHooks",
    "PermissionDeniedException",
    "CompositeHooks",
    "BashCommandRules",
    "BashRuleDecision",
    "BashSegmentDecision",
    "evaluate_bash_command",
    "SqlStatementRules",
    "classify_sql_kind",
    "BashCommandClassifier",
    "BashClassifierContext",
    "ClassifierVerdict",
    "create_bash_classifier",
]
