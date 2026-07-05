# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the bash classifier interface seam."""

import pytest

from datus.tools.permission.bash_classifier import (
    BashClassifierContext,
    NoopBashCommandClassifier,
    create_bash_classifier,
)
from datus.tools.permission.bash_rules import BashClassifierConfig, BashCommandRules


class TestCreateBashClassifier:
    """Factory behavior for the reserved classifier seam."""

    def test_none_rules_yields_none(self):
        assert create_bash_classifier(None, agent_config=None) is None

    def test_disabled_yields_none(self):
        rules = BashCommandRules(classifier=BashClassifierConfig(enabled=False))
        assert create_bash_classifier(rules, agent_config=None) is None

    def test_enabled_yields_noop_until_implemented(self):
        rules = BashCommandRules(classifier=BashClassifierConfig(enabled=True))
        classifier = create_bash_classifier(rules, agent_config=None)
        assert isinstance(classifier, NoopBashCommandClassifier)

    @pytest.mark.asyncio
    async def test_noop_never_renders_verdict(self):
        classifier = NoopBashCommandClassifier()
        context = BashClassifierContext(cwd="/tmp", node_name="chat")
        verdict = await classifier.classify("git status", context)
        assert verdict is None
