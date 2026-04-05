"""Tests for datus.api.services.kb_service — knowledge base bootstrap."""

import pytest

from datus.api.services.kb_service import KbService


class TestKbServiceInit:
    """Tests for KbService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """KbService initializes with real agent config."""
        svc = KbService(agent_config=real_agent_config)
        assert svc is not None
        assert svc.agent_config is real_agent_config
