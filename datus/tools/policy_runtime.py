# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Generic policy lifecycle composed from active Datus plugins.

The agent owns execution order and fail-closed validation. Policy plugins own
the meaning of ``policy_context`` and the concrete read transformations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional

from datus.plugins.registry import collect_plugin_policy_runtime_factories
from datus.utils.bounded_tool_output import bounded_text
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _plain_metadata_read(sql: str, dialect: str) -> bool:
    """Only schema inspection may bypass the source rewrite, never query bodies."""
    if sql.strip().rstrip(";").strip().upper() in {"SHOW TABLES", "SHOW DATABASES", "SHOW SCHEMAS", "SHOW CATALOGS"}:
        return True
    import sqlglot
    from sqlglot import expressions as exp

    from datus.utils.sql_utils import parse_dialect

    try:
        statements = sqlglot.parse(sql, read=parse_dialect(dialect))
        if len(statements) != 1 or not isinstance(statements[0], exp.Describe):
            return False
        statement = statements[0]
        table = statement.this
        return (
            isinstance(table, exp.Table)
            and not any(statement.find_all(exp.Query, exp.Func))
            and all(isinstance(part, exp.Identifier) for part in table.parts)
        )
    except Exception:
        return False


#: Sibling key on ``policy_context`` carrying *why* SaaS denied this caller.
#:
#: Written by Datus-backend when it assembles the context; policy plugins read
#: only ``row_filter`` and ignore it. See ``saas_provider._DENIAL_KEY``.
_DENIAL_KEY = "x_saas_denial"


def _explain(context: Dict[str, Any], reason: Optional[str]) -> Optional[str]:
    """Append SaaS' explanation to a plugin's refusal.

    A plugin can only report that context validation failed. It does not know
    whose attributes are missing — or that attributes are involved at all — so
    the person reading "Policy context denies all data reads" has nothing to
    act on and no way to tell a gap in their own profile from a broken policy.
    SaaS decides ``access_mode`` precisely so the refusal can be explained;
    this is where that explanation reaches the caller.
    """
    detail = context.get(_DENIAL_KEY)
    if not isinstance(detail, str) or not detail.strip():
        return reason
    return f"{(reason or 'Policy denied the query').rstrip('. ')}: {detail.strip()}"


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
    verify_context: Optional[Callable[[], None]] = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ReadResultDecision:
    allowed: bool
    result: Any = None
    reason: Optional[str] = None
    applied_policies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetricReadDecision:
    allowed: bool
    where: Optional[str] = None
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
            configured = getattr(agent_config, "get_plugin_profile", None)
            expected = (
                configured("sql-policy")
                if callable(configured) and (active_names is None or "sql-policy" in active_names)
                else {}
            )
            if (
                getattr(agent_config, "sql_policy_required", False) is True
                or isinstance(expected, dict)
                and (expected.get("policies") or expected.get("column_masking") or expected.get("masking_profiles"))
            ) and "sql-policy" not in {name for name, _ in self._runtimes}:
                raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message="Required SQL policy plugin is unavailable")
        except DatusException:
            raise
        except Exception as exc:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Failed to initialize policy runtime: {exc}",
            ) from exc

    @property
    def runtime_names(self) -> tuple[str, ...]:
        """Names of runtimes actually instantiated, including no-policy reads."""
        return tuple(name for name, _runtime in self._runtimes)

    @property
    def has_metric_policies(self) -> bool:
        return any(callable(getattr(runtime, "before_metric_read", None)) for _, runtime in self._runtimes)

    @property
    def has_bound_read_provider(self) -> bool:
        return any(callable(getattr(runtime, "bind_read", None)) for _, runtime in self._runtimes)

    def requires_request_context(self) -> bool:
        """Ask plugins whether reads need trusted request-local authorization."""
        for plugin_name, runtime in self._runtimes:
            requirement = getattr(runtime, "requires_request_context", None)
            if callable(requirement):
                required = self._invoke(plugin_name, "requires_request_context", requirement)
                if not isinstance(required, bool):
                    raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message="Invalid policy context requirement")
                if required:
                    return True
            else:
                # Preserve existing providers which express required inputs
                # through their context validation hook.
                hook = getattr(runtime, "validate_context", None)
                if callable(hook):
                    result = self._invoke(plugin_name, "validate_context", hook, {})
                    if not self._validation_decision(plugin_name, result).allowed:
                        return True
        return False

    def before_metric_read(
        self, *, datasets: set[str], where: Optional[str], policy_context: Optional[Dict[str, Any]]
    ) -> MetricReadDecision:
        """Apply the same plugin metric hooks as interactive semantic tools."""
        context = self._normalize_context(policy_context)
        applied: list[str] = []
        current_where = where
        for name, runtime in self._runtimes:
            hook = getattr(runtime, "before_metric_read", None)
            if not callable(hook):
                continue
            raw = self._invoke(
                name, "before_metric_read", hook, datasets=datasets, where=current_where, policy_context=context
            )
            if not self._allowed(name, raw):
                return MetricReadDecision(allowed=False, reason=_explain(context, getattr(raw, "reason", None)))
            rewritten_where = getattr(raw, "where", None)
            if rewritten_where is not None:
                if not isinstance(rewritten_where, str):
                    raise DatusException(
                        ErrorCode.COMMON_CONFIG_ERROR, message="Metric policy returned a non-string filter"
                    )
                current_where = rewritten_where
            applied.extend(self._policy_names(name, raw))
        return MetricReadDecision(allowed=True, where=current_where, applied_policies=applied)

    def validate_context(self, policy_context: Optional[Dict[str, Any]]) -> PolicyValidationResult:
        context = self._normalize_context(policy_context)
        if not self._runtimes and any(
            key in context for key in ("row_filter", "column_masking", "column_mask", "sql_policy", "masking_profile")
        ):
            raise DatusException(
                ErrorCode.POLICY_DENIED, message="Request requires SQL policies but no implementation is loaded"
            )
        if any(key in context for key in ("sql_policy", "masking_profile")) and not any(
            callable(getattr(runtime, "bind_read", None)) for _, runtime in self._runtimes
        ):
            raise DatusException(
                ErrorCode.POLICY_DENIED,
                message="Effective SQL policies require a native source-policy provider",
            )
        for plugin_name, runtime in self._runtimes:
            hook = getattr(runtime, "validate_context", None)
            if not callable(hook):
                continue
            raw = self._invoke(plugin_name, "validate_context", hook, context)
            decision = self._validation_decision(plugin_name, raw)
            if not decision.allowed:
                return replace(decision, reason=_explain(context, decision.reason))
        return PolicyValidationResult(allowed=True)

    def requires_bound_read(self, policy_context: Optional[Dict[str, Any]]) -> bool:
        context = self._normalize_context(policy_context)
        validation = self.validate_context(context)
        if not validation.allowed:
            raise DatusException(ErrorCode.POLICY_DENIED, message=validation.reason or "Policy denied")
        for _, runtime in self._runtimes:
            requirement = getattr(runtime, "requires_bound_read", None)
            if callable(requirement):
                if requirement(context):
                    return True
            elif hasattr(runtime, "policies"):
                if runtime.policies:
                    return True
            elif callable(getattr(runtime, "before_metric_read", None)):
                return True
        return False

    def bind_read(self, *, connector: Any, datasource: str, policy_context: Optional[Dict[str, Any]]):
        context = self._normalize_context(policy_context)
        validation = self.validate_context(context)
        if not validation.allowed:
            raise DatusException(ErrorCode.POLICY_DENIED, message=validation.reason or "Policy denied")
        providers = [
            (name, runtime) for name, runtime in self._runtimes if callable(getattr(runtime, "bind_read", None))
        ]
        if len(providers) > 1:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR, message="Only one source-level SQL policy provider can be bound"
            )
        if not providers:
            if self.requires_bound_read(context):
                raise DatusException(
                    ErrorCode.POLICY_DENIED,
                    message="Installed metric policy provider does not support native Dosi protection",
                )
            return None
        from datus.tools.sql_read_context import SqlReadContext

        name, runtime = providers[0]
        # Empty policies need no connection or metadata. The provider owns this
        # decision, including a trusted effective snapshot supplied per request.
        required = getattr(runtime, "requires_bound_read", lambda context: True)(context)
        if not required:
            return None
        if connector is None:
            raise DatusException(ErrorCode.POLICY_DENIED, message="A live connector is required to bind SQL policies")
        read_context = SqlReadContext(connector, datasource, capture_metadata=False)
        bound = self._invoke(name, "bind_read", runtime.bind_read, read_context=read_context, policy_context=context)
        if bound is None:
            raise DatusException(
                ErrorCode.POLICY_DENIED, message="Required SQL policy provider returned no bound policy"
            )
        return bound

    def before_sql_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        policy_context: Optional[Dict[str, Any]],
        connector: Any = None,
    ) -> SqlReadDecision:
        context = self._normalize_context(policy_context)
        validation = self.validate_context(context)
        if not validation.allowed:
            return SqlReadDecision(allowed=False, reason=validation.reason)
        current_sql = sql
        verifier = None
        applied: list[str] = []
        for plugin_name, runtime in self._runtimes:
            if callable(getattr(runtime, "bind_read", None)):
                bound = self.bind_read(connector=connector, datasource=datasource, policy_context=context)
                if bound is not None:
                    if not _plain_metadata_read(current_sql, dialect):
                        current_sql = self._invoke(
                            plugin_name, "rewrite_sql", bound.rewrite_sql, current_sql, bound.target
                        )
                    verifier = bound.verify_context
                continue
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
                return replace(decision, reason=_explain(context, decision.reason))
            current_sql = decision.sql if decision.sql is not None else current_sql
            applied.extend(decision.applied_policies)
        logger.debug(
            "sql.policy.rewrite",
            datasource=datasource,
            dialect=dialect,
            original_sql_sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            rewritten_sql_sha256=hashlib.sha256(current_sql.encode("utf-8")).hexdigest(),
            original_sql=bounded_text(sql, 1024),
            rewritten_sql=bounded_text(current_sql, 1024),
            rewritten=current_sql != sql,
            policy_runtimes=self.runtime_names,
            masking_profile=context.get("masking_profile"),
            policy_revision=context.get("policy_revision"),
        )
        return SqlReadDecision(allowed=True, sql=current_sql, applied_policies=applied, verify_context=verifier)

    @property
    def supports_streaming_results(self) -> bool:
        """Whether every result hook explicitly supports batch semantics."""
        return all(
            not (
                callable(getattr(runtime, "after_read_result", None))
                and not callable(getattr(runtime, "after_read_batch", None))
                and getattr(runtime, "supports_batch_results", False) is not True
            )
            for _, runtime in self._runtimes
        )

    def validate_streaming_results(self) -> None:
        """Require explicit batch semantics for plugins with result hooks."""
        if not self.supports_streaming_results:
            raise ValueError("Policy runtime does not support streaming result enforcement")

    def after_read_batch(self, result: Any, **context) -> ReadResultDecision:
        """Apply only result hooks whose batch behavior was explicitly declared."""
        self.validate_streaming_results()
        context["policy_context"] = self._normalize_context(context.get("policy_context"))
        current = result
        applied = []
        for name, runtime in self._runtimes:
            hook = getattr(runtime, "after_read_batch", None)
            hook_name = "after_read_batch"
            if not callable(hook):
                hook = getattr(runtime, "after_read_result", None)
                hook_name = "after_read_result"
            if not callable(hook):
                continue
            decision = self._result_decision(name, self._invoke(name, hook_name, hook, current, **context), current)
            if not decision.allowed:
                return decision
            current = decision.result
            applied.extend(decision.applied_policies)
        return ReadResultDecision(allowed=True, result=current, applied_policies=applied)

    def after_read_result(
        self,
        result: Any,
        *,
        sql: str,
        datasource: str,
        dialect: str,
        policy_context: Optional[Dict[str, Any]],
        include_batch_hooks: bool = False,
    ) -> ReadResultDecision:
        """Apply complete-result policies, optionally including batch-only export hooks."""
        context = self._normalize_context(policy_context)
        current_result = result
        applied: list[str] = []
        for plugin_name, runtime in self._runtimes:
            hook = getattr(runtime, "after_read_result", None)
            hook_name = "after_read_result"
            if not callable(hook) and include_batch_hooks:
                hook = getattr(runtime, "after_read_batch", None)
                hook_name = "after_read_batch"
            if not callable(hook):
                continue
            raw = self._invoke(
                plugin_name,
                hook_name,
                hook,
                current_result,
                sql=sql,
                datasource=datasource,
                dialect=dialect,
                policy_context=context,
            )
            decision = self._result_decision(plugin_name, raw, current_result)
            if not decision.allowed:
                return replace(decision, reason=_explain(context, decision.reason))
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
    def _invoke(
        plugin_name: str,
        hook_name: str,
        hook: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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
