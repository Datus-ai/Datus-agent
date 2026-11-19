# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Trusted metric execution through the Agent policy lifecycle.

SQL policies run before reads and metric policies before compilation. Native
attribution currently has no per-query callback: its row-only compatibility
adapter scopes every dataset in a private model. Unsupported policies and
ontology models fail closed instead of bypassing policy hooks. Hosts never
need engine handles or model reconstruction details.
"""

import asyncio
import json
import warnings
from pathlib import Path
from typing import Any

import yaml
from sqlglot import exp, parse_one

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.policy_runtime import PolicyRuntime
from datus.tools.semantic_tools.dosi.engine import load_binding
from datus.tools.semantic_tools.dosi.runtime import DosiRuntime
from datus.tools.semantic_tools.models import AttributionRequest, AttributionResult, QueryResult
from datus.utils.exceptions import DatusException, ErrorCode


def _denied(reason: str) -> DatusException:
    """Build a policy-denial error without returning query results."""
    return DatusException(ErrorCode.POLICY_DENIED, message=reason)


class PolicyMetricExecutor:
    """One executor shared by host tools; context is supplied fresh on each call.

    Construct with the same runtime/config used to plan the metric. Do not
    expose raw SQL metadata or policy exceptions directly to an untrusted model.
    """

    def __init__(self, config: AgentConfig, runtime: DosiRuntime):
        """Bind the trusted planning config and runtime; supply policy context per call."""
        self.config = config
        self.runtime = runtime

    @staticmethod
    def validate_attribution_support(config: AgentConfig, policy: PolicyRuntime | None = None) -> None:
        """Reject policies that native attribution cannot faithfully enforce."""
        policy = policy if policy is not None else PolicyRuntime(config)
        for name in policy.runtime_names:
            if name != "sql-policy":
                raise _denied("analysis_runtime_unavailable")
            profile = config.get_plugin_profile(name)
            if any(item.get("type") not in {"row_filter", "metric_row_filter"} for item in profile.get("policies", [])):
                raise _denied("analysis_runtime_unavailable")

    async def _scope(self, metric: str, where: str | None, context: dict[str, Any]) -> tuple[str | None, PolicyRuntime]:
        """Validate current context and resolve the policy-constrained metric filter."""
        policy = PolicyRuntime(self.config)
        if not policy.validate_context(context).allowed:
            raise _denied("Current context does not allow metric reads")
        binding = await asyncio.to_thread(self.runtime.artifact_metric_binding, metric)
        datasets = binding["datasets"]
        if not datasets or any(not isinstance(item, str) for item in datasets):
            raise _denied("Metric datasets are unavailable")
        decision = policy.before_metric_read(datasets=set(datasets), where=where, policy_context=context)
        if not decision.allowed:
            raise _denied("Metric read denied")
        return decision.where, policy

    async def query_metrics(
        self, metric: str, *, policy_context: dict[str, Any], where: str | None = None, **query: Any
    ) -> QueryResult:
        """Compile under metric policy, then execute through SQL/result policies."""
        if "dry_run" in query:
            raise _denied("Policy execution does not expose dry-run SQL")
        where, _ = await self._scope(metric, where, policy_context)
        compiled = await self.runtime.query_metrics([metric], where=where, dry_run=True, **query)
        rows = await asyncio.to_thread(self._query_read, compiled.metadata["sql"], policy_context)
        return QueryResult(columns=list(rows[0]) if rows else [], data=rows, metadata={"row_count": len(rows)})

    def _query_read(self, sql: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Read compiled SQL through the standard SQL and result-policy pipeline."""
        tool = DBFuncTool(agent_config=self.config, read_only=True)
        datasource = self.runtime.datasource or self.config.current_datasource
        database = (self.runtime.config.db_config or {}).get("database") or self.config.default_database_name(
            datasource
        )
        connector = tool._get_connector(datasource, database=database or "")
        result = tool.execute_read_enforced(sql, connector, datasource=datasource, policy_context=context)
        if not result.success or not isinstance(result.sql_return, list):
            raise _denied("Query unavailable")
        return result.sql_return

    async def attribute_metrics(self, request: dict[str, Any], *, policy_context: dict[str, Any]) -> dict[str, Any]:
        """Dosi ``attribute_metrics`` under the metric's read policy.

        ``request`` is Dosi's (``metric``, ``baseline``, ``current`` and the
        metric query's fields). Its ``where_sql`` goes through the metric
        policy, whose condition is what runs, and every dataset the model reads
        is scoped by the row policies, so both periods and every query the
        attribution sends see only allowed rows. Returns Dosi's result as is;
        the caller decides what to show.
        """
        return await self._attribute_native("attribute_metrics", request, policy_context)

    async def attribute_dimensions(self, request: dict[str, Any], *, policy_context: dict[str, Any]) -> dict[str, Any]:
        """Dosi ``attribute_dimensions`` under the metric's read policy; as
        ``attribute_metrics``, with ``dimensions`` in the request."""
        return await self._attribute_native("attribute_dimensions", request, policy_context)

    async def _attribute_native(self, method: str, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Scope the request and the model, then run one Dosi attribution tool."""
        where, policy = await self._scope(str(request.get("metric") or ""), request.get("where_sql"), context)
        self.validate_attribution_support(self.config, policy)
        handle = await asyncio.to_thread(self.runtime._handle_for_metric, request["metric"])
        scoped = {key: value for key, value in request.items() if key != "where_sql"}
        if where:
            scoped["where_sql"] = where
        return await asyncio.to_thread(self._scoped_call, handle, policy, context, method, scoped)

    async def attribute(self, request: AttributionRequest, *, policy_context: dict[str, Any]) -> AttributionResult:
        """Deprecated: Dosi's shipped attribution. Use ``attribute_metrics`` or
        ``attribute_dimensions``; this goes when Dosi removes ``Engine.attribute``.

        Enforces the same metric scope for both periods and every native query."""
        warnings.warn(
            "PolicyMetricExecutor.attribute is deprecated; use attribute_metrics or attribute_dimensions",
            DeprecationWarning,
            stacklevel=2,
        )
        where, policy = await self._scope(request.metric, request.where_sql, policy_context)
        self.validate_attribution_support(self.config, policy)
        handle = await asyncio.to_thread(self.runtime._handle_for_metric, request.metric)
        result = await asyncio.to_thread(
            self._scoped_call,
            handle,
            policy,
            policy_context,
            "attribute",
            request.model_copy(update={"where_sql": where}).model_dump(mode="python", exclude_none=True),
        )
        return AttributionResult.model_validate({**result, "implementation": "dosi"})

    def _scoped_call(
        self, handle: Any, policy: PolicyRuntime, context: dict[str, Any], method: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Run ``Engine.<method>`` on a private model whose datasets read through the row policies."""
        engine = self._scoped_engine(handle, policy, context)
        return getattr(engine, method)(
            request,
            connection=handle.profile_name,
            timeout_secs=float(self.runtime.config.timeout_seconds),
        )

    def _scoped_engine(self, handle: Any, policy: PolicyRuntime, context: dict[str, Any]) -> Any:
        # Native attribution executes multiple queries internally, without a Python
        # SQL callback. Apply row policies to every dataset source in a private,
        # in-memory model before native compilation. Never alter the shared model.
        # Result/column policies and unknown plugins cannot be represented this way.
        """A private engine over the metric's model, each dataset's source rewritten through the row policies."""
        if handle.has_ontology:
            raise _denied("MAPPING_UNSUPPORTED")
        handle.get()
        document = yaml.safe_load(Path(handle._config.semantic_model_path).read_text())
        tool = DBFuncTool(agent_config=self.config, read_only=True)
        datasource = self.runtime.datasource or self.config.current_datasource
        database = (self.runtime.config.db_config or {}).get("database") or self.config.default_database_name(
            datasource
        )
        connector = tool._get_connector(datasource, database=database or "")
        for model in [document]:
            for dataset in model["datasets"]:
                source = dataset["source"]
                parsed = parse_one(source, read=connector.dialect)
                sql = (
                    source
                    if isinstance(parsed, exp.Query)
                    else exp.select("*").from_(exp.to_table(source)).sql(dialect=connector.dialect)
                )
                decision = policy.before_sql_read(
                    sql, datasource=datasource, dialect=connector.dialect, policy_context=context, connector=connector
                )
                if not decision.allowed:
                    raise _denied("ACCESS_DENIED")
                enforced = decision.sql if decision.sql is not None else sql
                error, _ = tool._validate_read_sql(enforced, connector)
                if error:
                    raise _denied("QUERY_UNAVAILABLE")
                dataset["source"] = enforced
                extensions = dataset.setdefault("custom_extensions", [])
                extension = next((item for item in extensions if item.get("vendor_name") == "DATUS"), None)
                if extension is None:
                    extension = {"vendor_name": "DATUS", "data": "{}"}
                    extensions.append(extension)
                data = json.loads(extension["data"])
                data["source_type"] = "query"
                extension["data"] = json.dumps(data)
        return load_binding().Engine(
            model_text=yaml.safe_dump(document), connections=handle._runtime_connections(handle._config), pool_size=1
        )
