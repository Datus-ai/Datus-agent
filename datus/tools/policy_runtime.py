# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Generic policy lifecycle composed from active Datus plugins.

The agent owns execution order and fail-closed validation. Policy plugins own
the meaning of ``policy_context`` and the concrete read transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from datus.plugins.registry import collect_plugin_policy_runtime_factories
from datus.utils.exceptions import DatusException, ErrorCode


@dataclass(frozen=True)
class PolicyValidationResult:
    allowed: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class SqlReadDecision:
    allowed: bool
    sql: Optional[str] = None
    reason: Optional[str] = None
    applied_policies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReadResultDecision:
    allowed: bool
    result: Any = None
    reason: Optional[str] = None
    applied_policies: list[str] = field(default_factory=list)


class PolicyRuntime:
    """Compose policy runtimes declared by active plugin manifests."""

    def __init__(self, agent_config: Any) -> None:
        self._runtimes: list[tuple[str, Any]] = []
        if agent_config is None:
            return
        try:
            active_names_getter = getattr(agent_config, "active_plugin_names", None)
            active_names = active_names_getter() if callable(active_names_getter) else None
            factories = collect_plugin_policy_runtime_factories(active_names)
            for plugin_name, factory in factories.items():
                profile_getter = getattr(agent_config, "get_plugin_profile", None)
                profile = profile_getter(plugin_name) if callable(profile_getter) else {}
                runtime = factory(dict(profile))
                if runtime is None:
                    raise TypeError("factory returned None")
                if not any(
                    callable(getattr(runtime, hook_name, None))
                    for hook_name in (
                        "validate_context",
                        "before_sql_read",
                        "before_metric_read",
                        "after_read_result",
                    )
                ):
                    raise TypeError("factory returned an object without policy lifecycle hooks")
                self._runtimes.append((plugin_name, runtime))
        except DatusException:
            raise
        except Exception as exc:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Failed to initialize policy runtime: {exc}",
            ) from exc

    def validate_context(self, policy_context: Optional[Dict[str, Any]]) -> PolicyValidationResult:
        context = self._normalize_context(policy_context)
        for plugin_name, runtime in self._runtimes:
            hook = getattr(runtime, "validate_context", None)
            if not callable(hook):
                continue
            raw = self._invoke(plugin_name, "validate_context", hook, context)
            decision = self._validation_decision(plugin_name, raw)
            if not decision.allowed:
                return decision
        return PolicyValidationResult(allowed=True)

    def before_sql_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        policy_context: Optional[Dict[str, Any]],
    ) -> SqlReadDecision:
        context = self._normalize_context(policy_context)
        current_sql = sql
        applied: list[str] = []
        for plugin_name, runtime in self._runtimes:
            hook = getattr(runtime, "before_sql_read", None)
            if not callable(hook):
                continue
            raw = self._invoke(
                plugin_name,
                "before_sql_read",
                hook,
                current_sql,
                datasource=datasource,
                dialect=dialect,
                policy_context=context,
            )
            decision = self._sql_decision(plugin_name, raw, current_sql)
            if not decision.allowed:
                return decision
            current_sql = decision.sql if decision.sql is not None else current_sql
            applied.extend(decision.applied_policies)
        return SqlReadDecision(allowed=True, sql=current_sql, applied_policies=applied)

    def after_read_result(
        self,
        result: Any,
        *,
        sql: str,
        datasource: str,
        dialect: str,
        policy_context: Optional[Dict[str, Any]],
    ) -> ReadResultDecision:
        context = self._normalize_context(policy_context)
        current_result = result
        applied: list[str] = []
        for plugin_name, runtime in self._runtimes:
            hook = getattr(runtime, "after_read_result", None)
            if not callable(hook):
                continue
            raw = self._invoke(
                plugin_name,
                "after_read_result",
                hook,
                current_result,
                sql=sql,
                datasource=datasource,
                dialect=dialect,
                policy_context=context,
            )
            decision = self._result_decision(plugin_name, raw, current_result)
            if not decision.allowed:
                return decision
            current_result = decision.result
            applied.extend(decision.applied_policies)
        return ReadResultDecision(allowed=True, result=current_result, applied_policies=applied)

    @staticmethod
    def _normalize_context(policy_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if policy_context is None:
            return {}
        if not isinstance(policy_context, dict):
            raise DatusException(ErrorCode.TOOL_INVALID_INPUT, message="policy_context must be a JSON object")
        return dict(policy_context)

    @staticmethod
    def _invoke(plugin_name: str, hook_name: str, hook, *args, **kwargs):
        try:
            return hook(*args, **kwargs)
        except DatusException:
            raise
        except Exception as exc:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message=f"Policy runtime {plugin_name!r} {hook_name} failed: {exc}",
            ) from exc

    @staticmethod
    def _allowed(plugin_name: str, raw: Any) -> bool:
        allowed = getattr(raw, "allowed", None)
        if not isinstance(allowed, bool):
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Policy runtime {plugin_name!r} returned a decision without boolean allowed",
            )
        return allowed

    @classmethod
    def _validation_decision(cls, plugin_name: str, raw: Any) -> PolicyValidationResult:
        return PolicyValidationResult(
            allowed=cls._allowed(plugin_name, raw),
            reason=getattr(raw, "reason", None),
        )

    @classmethod
    def _sql_decision(cls, plugin_name: str, raw: Any, current_sql: str) -> SqlReadDecision:
        allowed = cls._allowed(plugin_name, raw)
        sql = getattr(raw, "sql", None)
        if sql is not None and not isinstance(sql, str):
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Policy runtime {plugin_name!r} returned a non-string SQL rewrite",
            )
        return SqlReadDecision(
            allowed=allowed,
            sql=current_sql if sql is None else sql,
            reason=getattr(raw, "reason", None),
            applied_policies=cls._policy_names(plugin_name, raw),
        )

    @classmethod
    def _result_decision(cls, plugin_name: str, raw: Any, current_result: Any) -> ReadResultDecision:
        allowed = cls._allowed(plugin_name, raw)
        result = getattr(raw, "result", current_result)
        return ReadResultDecision(
            allowed=allowed,
            result=result,
            reason=getattr(raw, "reason", None),
            applied_policies=cls._policy_names(plugin_name, raw),
        )

    @staticmethod
    def _policy_names(plugin_name: str, raw: Any) -> list[str]:
        names = getattr(raw, "applied_policies", None) or []
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Policy runtime {plugin_name!r} returned invalid applied_policies",
            )
        return list(names)
