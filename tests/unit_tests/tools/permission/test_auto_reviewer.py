# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.configuration.agent_config import AgentConfig, ModelConfig
from datus.tools.permission.auto_reviewer import (
    AutoReviewer,
    AutoReviewRequest,
    AutoReviewVerdict,
    LLMAutoReviewer,
)
from datus.tools.permission.bash_rules import BashCommandRules
from datus.tools.permission.permission_config import AutoReviewConfig, PermissionConfig
from datus.tools.permission.permission_hooks import PermissionDeniedException, PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import build_effective_config, get_profile
from datus.tools.registry.tool_registry import ToolRegistry


class StubReviewer(AutoReviewer):
    def __init__(self, verdict=None):
        self.verdict = verdict
        self.requests = []

    async def review(self, request, config):
        self.requests.append((request, config))
        return self.verdict


def verdict(risk="low", decision="allow", confidence=0.95):
    return AutoReviewVerdict(
        risk_level=risk,
        user_authorization="high",
        decision=decision,
        confidence=confidence,
        rationale=f"{risk} test action",
    )


def context(args, *, direct=False):
    ctx = MagicMock()
    ctx.tool_arguments = json.dumps(args)
    ctx.direct_user_invocation = direct
    return ctx


def tool(name):
    value = MagicMock()
    value.name = name
    return value


def hooks_for(profile, reviewer, broker, *, non_interactive=False, bash_rules=None, review_context=None):
    config = get_profile(profile)
    if bash_rules is not None:
        config = PermissionConfig(
            default_permission=config.default_permission,
            rules=config.rules,
            bash_commands=bash_rules,
            sql_statements=config.sql_statements,
            auto_review=config.auto_review,
        )
    manager = PermissionManager(global_config=config, active_profile=profile)
    registry = ToolRegistry({"bash": "bash_tools", "execute_sql": "db_tools"})
    return PermissionHooks(
        broker=broker,
        permission_manager=manager,
        node_name="chat",
        tool_registry=registry,
        non_interactive=non_interactive,
        project_root="/tmp/project",
        auto_reviewer=reviewer,
        review_context_provider=lambda: (
            review_context
            if review_context is not None
            else {
                "trusted_user_messages": ["delete the one test row"],
                "prior_actions": [{"tool": "bash", "arguments": {"command": "pwd"}}],
            }
        ),
    ), manager


class TestAutoReviewConfig:
    def test_profile_defaults_and_override(self):
        assert get_profile("normal").auto_review.enabled is False
        assert get_profile("auto").auto_review.enabled is True
        assert get_profile("dangerous").auto_review.enabled is False

        config = build_effective_config(
            "auto",
            {"auto_review": {"model": "openai/gpt-5-mini", "confidence_threshold": 0.9}},
        )
        assert config.auto_review.enabled is True
        assert config.auto_review.model == "openai/gpt-5-mini"
        assert config.auto_review.confidence_threshold == 0.9
        assert config.auto_review.timeout_seconds == 20.0

    def test_manager_copy_preserves_reviewer_config(self):
        manager = PermissionManager(global_config=get_profile("auto"), active_profile="auto")
        assert manager.global_config.auto_review.enabled is True

    def test_profile_switch_preserves_user_reviewer_settings(self):
        override = PermissionConfig.from_dict({"auto_review": {"model": "custom/security", "timeout_seconds": 7}})
        manager = PermissionManager(global_config=get_profile("normal"), active_profile="normal")

        manager.switch_profile("auto", user_overrides=override)
        assert manager.global_config.auto_review.enabled is True
        assert manager.global_config.auto_review.model == "custom/security"
        assert manager.global_config.auto_review.timeout_seconds == 7

        manager.switch_profile("normal", user_overrides=override)
        assert manager.global_config.auto_review.enabled is False
        assert manager.global_config.auto_review.model == "custom/security"

    def test_medium_is_auto_allowable_but_high_is_not(self):
        config = AutoReviewConfig(enabled=True, confidence_threshold=0.8)
        assert verdict("low").can_auto_allow(config)
        assert verdict("medium").can_auto_allow(config)
        assert not verdict("high").can_auto_allow(config)
        assert not verdict("medium", confidence=0.5).can_auto_allow(config)


class TestReviewerRequest:
    @staticmethod
    def _history_bytes(payload):
        return len(
            json.dumps(
                {
                    "trusted_user_messages": payload["trusted_user_messages"],
                    "prior_planned_actions": payload["untrusted_prior_planned_actions"],
                }
            ).encode("utf-8")
        )

    @staticmethod
    def _oversized_request(command):
        return AutoReviewRequest(
            action_type="bash",
            action={"command": command},
            environment={},
            static_assessment={},
            trusted_user_messages=["u" * 5000 for _ in range(12)],
            prior_actions=[{"command": "p" * 5000} for _ in range(30)],
        )

    def test_planned_action_is_exact_and_history_is_bounded(self):
        command = "python -c '" + ("x" * 30000) + "'"
        request = self._oversized_request(command)

        payload = request.prompt_payload(AutoReviewConfig(enabled=True))

        assert payload["planned_action"]["command"] == command
        assert self._history_bytes(payload) <= 24 * 1024

    def test_history_bounds_follow_configuration(self):
        request = self._oversized_request("make")

        payload = request.prompt_payload(
            AutoReviewConfig(
                enabled=True,
                max_history_bytes=2048,
                max_user_messages=3,
                max_prior_actions=2,
            )
        )

        assert len(payload["trusted_user_messages"]) <= 3
        assert len(payload["untrusted_prior_planned_actions"]) <= 2
        assert self._history_bytes(payload) <= 2048
        assert payload["planned_action"] == {"command": "make", "type": "bash"}

    def test_action_payload_cannot_relabel_the_request_type(self):
        request = AutoReviewRequest(
            action_type="bash",
            action={"command": "make", "type": "sql"},
            environment={},
            static_assessment={},
        )

        payload = request.prompt_payload(AutoReviewConfig(enabled=True))

        assert payload["planned_action"]["type"] == "bash"

    @pytest.mark.asyncio
    async def test_invalid_model_output_fails_closed(self):
        fake_model = MagicMock()
        fake_model.generate_with_json_output.return_value = {"decision": "allow"}
        reviewer = LLMAutoReviewer(agent_config=MagicMock())
        request = AutoReviewRequest("bash", {"command": "make"}, {}, {})
        with patch("datus.models.base.LLMBaseModel.create_model", return_value=fake_model):
            assert await reviewer.review(request, AutoReviewConfig(enabled=True)) is None

    @pytest.mark.asyncio
    async def test_schema_is_passed_to_adapter_and_included_in_prompt(self):
        fake_model = MagicMock()
        fake_model.generate_with_json_output.return_value = {
            "risk_level": "low",
            "user_authorization": "high",
            "decision": "allow",
            "confidence": 0.95,
            "rationale": "Routine local build.",
        }
        reviewer = LLMAutoReviewer(agent_config=MagicMock())
        request = AutoReviewRequest("bash", {"command": "make"}, {}, {})

        with patch("datus.models.base.LLMBaseModel.create_model", return_value=fake_model):
            result = await reviewer.review(request, AutoReviewConfig(enabled=True))

        assert result == AutoReviewVerdict(
            risk_level="low",
            user_authorization="high",
            decision="allow",
            confidence=0.95,
            rationale="Routine local build.",
        )
        messages = fake_model.generate_with_json_output.call_args.args[0]
        prompt_payload = json.loads(messages[1]["content"])
        call_kwargs = fake_model.generate_with_json_output.call_args.kwargs
        schema = call_kwargs["output_schema"]
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["enable_thinking"] is False
        assert call_kwargs["timeout"] == 20.0
        assert prompt_payload["review_request"]["planned_action"]["command"] == "make"
        assert prompt_payload["required_response_schema"] == schema
        assert set(schema["required"]) == {
            "risk_level",
            "user_authorization",
            "decision",
            "confidence",
            "rationale",
        }

    @pytest.mark.asyncio
    async def test_configured_budgets_reach_the_adapter(self):
        fake_model = MagicMock()
        fake_model.generate_with_json_output.return_value = {
            "risk_level": "low",
            "user_authorization": "high",
            "decision": "allow",
            "confidence": 0.95,
            "rationale": "Routine local build.",
        }
        reviewer = LLMAutoReviewer(agent_config=MagicMock())
        request = AutoReviewRequest("bash", {"command": "make"}, {}, {})
        config = AutoReviewConfig(enabled=True, max_completion_tokens=256, timeout_seconds=5)

        with patch("datus.models.base.LLMBaseModel.create_model", return_value=fake_model):
            result = await reviewer.review(request, config)

        assert result == AutoReviewVerdict(
            risk_level="low",
            user_authorization="high",
            decision="allow",
            confidence=0.95,
            rationale="Routine local build.",
        )
        call_kwargs = fake_model.generate_with_json_output.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["timeout"] == 5

    @pytest.mark.asyncio
    async def test_timeout_fails_closed(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking_generate(*args, **kwargs):
            entered.set()
            # Held until the assertions below release it, so the review can only
            # return through the timeout path. The bound is a safety net against
            # a hung test, never a wait the passing case relies on.
            release.wait(timeout=30)
            return {
                "risk_level": "low",
                "user_authorization": "high",
                "decision": "allow",
                "confidence": 0.95,
                "rationale": "Routine local build.",
            }

        fake_model = MagicMock()
        fake_model.generate_with_json_output.side_effect = blocking_generate
        reviewer = LLMAutoReviewer(agent_config=MagicMock())
        request = AutoReviewRequest("bash", {"command": "make"}, {}, {})
        config = AutoReviewConfig(enabled=True, timeout_seconds=0.01)

        try:
            with patch("datus.models.base.LLMBaseModel.create_model", return_value=fake_model):
                assert await reviewer.review(request, config) is None
            # The model really was called; the verdict was dropped by the
            # timeout rather than never requested.
            assert entered.is_set()
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_explicit_model_failure_does_not_fallback(self):
        agent_config = MagicMock()
        reviewer = LLMAutoReviewer(agent_config=agent_config)
        request = AutoReviewRequest("bash", {"command": "make"}, {}, {})
        config = AutoReviewConfig(enabled=True, model="openai/missing-reviewer")

        with patch(
            "datus.models.base.LLMBaseModel.create_model",
            side_effect=KeyError("missing reviewer"),
        ) as create_model:
            assert await reviewer.review(request, config) is None

        create_model.assert_called_once_with(agent_config, model_name="openai/missing-reviewer")
        agent_config.active_model.assert_not_called()


class TestReviewerModelResolution:
    def test_custom_alias_and_provider_model_refs(self):
        custom = ModelConfig(type="openai", api_key="key", model="security-model")
        config = MagicMock()
        config.models = {"security": custom}
        config.provider_catalog = {"providers": {"openrouter": {}}}
        config.providers = {}
        provider_model = ModelConfig(type="openrouter", api_key="", model="vendor/security/model")
        config._synthesize_model.return_value = provider_model

        assert AgentConfig.resolve_model_ref(config, "custom/security") is custom
        assert AgentConfig.resolve_model_ref(config, "openrouter/vendor/security/model") is provider_model
        config._synthesize_model.assert_called_once_with("openrouter", "vendor/security/model")

    def test_unknown_explicit_provider_does_not_use_active_model(self):
        config = MagicMock()
        config.models = {}
        config.provider_catalog = {"providers": {}}
        config.providers = {}

        with pytest.raises(ValueError, match="Unknown model provider"):
            AgentConfig.resolve_model_ref(config, "missing/reviewer")
        config.active_model.assert_not_called()


class TestBashAutoReview:
    @pytest.mark.asyncio
    async def test_medium_action_auto_allows_with_minimal_context(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("medium"))
        hooks, _ = hooks_for("auto", reviewer, broker)

        await hooks.on_tool_start(context({"command": "cargo build"}), MagicMock(), tool("bash"))

        broker.request.assert_not_called()
        request = reviewer.requests[0][0]
        assert request.action == {"command": "cargo build", "timeout": None}
        assert request.trusted_user_messages == ["delete the one test row"]

    @pytest.mark.asyncio
    async def test_context_provider_cannot_override_hook_owned_environment(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("low"))
        hooks, _ = hooks_for(
            "auto",
            reviewer,
            broker,
            review_context={
                "environment": {
                    "profile": "dangerous",
                    "non_interactive": True,
                    "node_name": "spoofed",
                    "cwd": "/etc",
                    "project_root": "/etc",
                    "sandbox_enabled": True,
                }
            },
        )

        await hooks.on_tool_start(context({"command": "cargo build"}), MagicMock(), tool("bash"))

        environment = reviewer.requests[0][0].environment
        assert environment["profile"] == "auto"
        assert environment["non_interactive"] is False
        assert environment["node_name"] == "chat"
        assert environment["cwd"] == "/tmp/project"
        assert environment["project_root"] == "/tmp/project"
        # Keys the hook does not own are still forwarded.
        assert environment["sandbox_enabled"] is True

    @pytest.mark.asyncio
    async def test_safety_forced_command_is_reviewed(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("low"))
        hooks, _ = hooks_for("auto", reviewer, broker)

        await hooks.on_tool_start(context({"command": "git status && git diff"}), MagicMock(), tool("bash"))

        assert reviewer.requests[0][0].static_assessment["safety_forced"] is True
        broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_high_risk_prompts_with_rationale(self):
        broker = MagicMock()
        broker.request = AsyncMock(return_value=[["y"]])
        reviewer = StubReviewer(verdict("high", decision="ask"))
        hooks, _ = hooks_for("auto", reviewer, broker)

        await hooks.on_tool_start(context({"command": "git reset --hard HEAD"}), MagicMock(), tool("bash"))

        event = broker.request.await_args.args[0][0]
        assert "`high` risk" in event.content
        assert "high test action" in event.content

    @pytest.mark.asyncio
    async def test_failed_review_is_visible_in_confirmation(self):
        broker = MagicMock()
        broker.request = AsyncMock(return_value=[["y"]])
        reviewer = StubReviewer(None)
        hooks, _ = hooks_for("auto", reviewer, broker)

        await hooks.on_tool_start(context({"command": "cargo build"}), MagicMock(), tool("bash"))

        event = broker.request.await_args.args[0][0]
        assert "AI review:** unavailable or inconclusive" in event.content

    @pytest.mark.asyncio
    async def test_high_risk_non_interactive_denies(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("critical", decision="ask"))
        hooks, _ = hooks_for("auto", reviewer, broker, non_interactive=True)

        with pytest.raises(PermissionDeniedException, match="critical risk"):
            await hooks.on_tool_start(context({"command": "git reset --hard HEAD"}), MagicMock(), tool("bash"))
        broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_static_deny_and_session_grant_skip_reviewer(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("low"))
        rules = BashCommandRules(deny=["rm:*"])
        hooks, manager = hooks_for("auto", reviewer, broker, bash_rules=rules)
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(context({"command": "rm -rf data"}), MagicMock(), tool("bash"))
        assert not reviewer.requests

        # Keep fine-grained bash gating active so the session grant is checked
        # before the fallback tool-level ASK decision.
        manager.global_config.bash_commands = BashCommandRules(allow=["git log:*"])
        manager.approve_for_session("bash_tools", "bash::cargo build")
        await hooks.on_tool_start(context({"command": "cargo build"}), MagicMock(), tool("bash"))
        assert not reviewer.requests

    @pytest.mark.asyncio
    async def test_normal_profile_keeps_manual_confirmation(self):
        broker = MagicMock()
        broker.request = AsyncMock(return_value=[["y"]])
        reviewer = StubReviewer(verdict("low"))
        hooks, _ = hooks_for("normal", reviewer, broker)
        await hooks.on_tool_start(context({"command": "cargo build"}), MagicMock(), tool("bash"))
        assert not reviewer.requests
        assert broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_deprecated_classifier_config_overrides_bash_reviewer(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("low"))
        rules = BashCommandRules(
            allow=["git log:*"],
            classifier={
                "enabled": True,
                "model": "custom/security",
                "confidence_threshold": 0.9,
            },
        )
        hooks, _ = hooks_for("normal", reviewer, broker, bash_rules=rules)

        await hooks.on_tool_start(context({"command": "cargo build"}), MagicMock(), tool("bash"))

        assert len(reviewer.requests) == 1
        review_config = reviewer.requests[0][1]
        assert review_config.enabled is True
        assert review_config.model == "custom/security"
        assert review_config.confidence_threshold == 0.9
        broker.request.assert_not_called()


class TestSqlAutoReview:
    @pytest.mark.asyncio
    async def test_bounded_delete_can_auto_allow(self):
        broker = MagicMock()
        broker.request = AsyncMock()
        reviewer = StubReviewer(verdict("medium"))
        hooks, _ = hooks_for("auto", reviewer, broker)

        await hooks.on_tool_start(
            context({"sql": "DELETE FROM orders WHERE id = 7", "datasource": "warehouse"}),
            MagicMock(),
            tool("execute_sql"),
        )

        broker.request.assert_not_called()
        action = reviewer.requests[0][0].action
        assert action["statement_kind"] == "delete"
        assert action["datasource"] == "warehouse"

    @pytest.mark.asyncio
    async def test_high_sql_risk_prompts(self):
        broker = MagicMock()
        broker.request = AsyncMock(return_value=[["y"]])
        reviewer = StubReviewer(verdict("high", decision="ask"))
        hooks, _ = hooks_for("auto", reviewer, broker)

        await hooks.on_tool_start(context({"sql": "DROP DATABASE production"}), MagicMock(), tool("execute_sql"))

        event = broker.request.await_args.args[0][0]
        assert "`high` risk" in event.content
