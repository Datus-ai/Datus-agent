# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""LLM-based bash command classifier — interface seam only, no implementation.

Reserved extension point for auto-resolving bash permission prompts with an
LLM judgment (analogous to Claude Code's bash prompt-rule classifier). The
static rules in ``bash_rules.py`` decide first; the classifier is a second
opinion that may upgrade an ASK to an auto-allow.

Consultation contract (enforced by ``PermissionHooks._handle_bash_permission``):

- Consulted ONLY when the static decision is ASK with ``safety_forced=False``
  and the session is interactive. Deny decisions and safety-ceiling asks
  (shell wrappers, metacharacters, unparseable commands) are never sent to
  the classifier — those must stay with the user.
- An ``ALLOW`` verdict with ``confidence >= BashClassifierConfig.
  confidence_threshold`` auto-allows the call (logged, NOT session-cached —
  every invocation is judged on its own).
- Anything else — ``None``, low confidence, a DENY/ASK verdict, or an
  exception — falls through to the normal confirmation prompt. Fail closed.

Future implementation sketch (TODO(llm-classifier)):

    model = LLMBaseModel.create_model(agent_config, model_name=config.model)
    verdict_json = model.generate_with_json_output(prompt)  # datus/models/base.py

with a prompt built from the command, cwd, and the natural-language rule
descriptions in ``BashClassifierContext.rule_descriptions``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from datus.tools.permission.bash_rules import BashCommandRules
from datus.tools.permission.permission_config import PermissionLevel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BashClassifierContext:
    """Context handed to the classifier alongside the command."""

    cwd: str
    node_name: str
    rule_descriptions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClassifierVerdict:
    """A classifier's judgment on one command."""

    permission: PermissionLevel
    confidence: float
    reason: str


class BashCommandClassifier(ABC):
    """Interface for LLM-based bash command classification."""

    @abstractmethod
    async def classify(self, command: str, context: BashClassifierContext) -> Optional[ClassifierVerdict]:
        """Judge a command; return None when no judgment can be made.

        Implementations must never raise for routine conditions (timeouts,
        malformed model output) — return None instead so the caller falls
        through to the confirmation prompt.
        """


class NoopBashCommandClassifier(BashCommandClassifier):
    """Default classifier: never renders a verdict."""

    async def classify(self, command: str, context: BashClassifierContext) -> Optional[ClassifierVerdict]:
        return None


def create_bash_classifier(rules: Optional[BashCommandRules], agent_config: Any) -> Optional[BashCommandClassifier]:
    """Build a classifier from config; None when disabled (the default).

    Returning None (rather than a Noop instance) lets the hook skip the
    consultation branch entirely.
    """
    if rules is None or not rules.classifier.enabled:
        return None
    # TODO(llm-classifier): construct the real implementation here via
    #   LLMBaseModel.create_model(agent_config, model_name=rules.classifier.model)
    # and generate_with_json_output(). Until then, an enabled flag yields the
    # Noop so turning it on is harmless.
    logger.warning("bash_commands.classifier.enabled is set but no LLM classifier is implemented yet; using no-op")
    return NoopBashCommandClassifier()
