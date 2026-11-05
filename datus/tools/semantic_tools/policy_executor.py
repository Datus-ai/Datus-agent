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
    return DatusException(ErrorCode.POLICY_DENIED, message=reason)


class PolicyMetricExecutor:
    """One executor shared by host tools; context is supplied fresh on each call.

    Construct with the same runtime/config used to plan the metric. Do not
    expose raw SQL metadata or policy exceptions directly to an untrusted model.
    """

    def __init__(self, config: AgentConfig, runtime: DosiRuntime):
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
        tool = DBFuncTool(agent_config=self.config, read_only=True)
        datasource = self.config.current_datasource
        connector = tool._get_connector(datasource, database=self.config.default_database_name(datasource) or "")
        result = tool.execute_read_enforced(sql, connector, datasource=datasource, policy_context=context)
        if not result.success or not isinstance(result.sql_return, list):
            raise _denied("Query unavailable")
        return result.sql_return

    async def attribute(self, request: AttributionRequest, *, policy_context: dict[str, Any]) -> AttributionResult:
        """Enforce the same metric scope for both periods and every native query."""
        where, policy = await self._scope(request.metric, request.where_sql, policy_context)
        self.validate_attribution_support(self.config, policy)
        handle = await asyncio.to_thread(self.runtime._handle_for_metric, request.metric)
        return await asyncio.to_thread(
            self._scoped_attribution, handle, policy, policy_context, request.model_copy(update={"where_sql": where})
        )

    def _scoped_attribution(
        self, handle: Any, policy: PolicyRuntime, context: dict[str, Any], request: AttributionRequest
    ) -> AttributionResult:
        # Native attribution executes multiple queries internally, without a Python
        # SQL callback. Apply row policies to every dataset source in a private,
        # in-memory model before native compilation. Never alter the shared model.
        # Result/column policies and unknown plugins cannot be represented this way.
        if handle.has_ontology:
            raise _denied("MAPPING_UNSUPPORTED")
        document = yaml.safe_load(Path(handle._config.semantic_model_path).read_text())
        tool = DBFuncTool(agent_config=self.config, read_only=True)
        datasource = self.config.current_datasource
        connector = tool._get_connector(datasource, database=self.config.default_database_name(datasource) or "")
        for model in document["semantic_model"]:
            for dataset in model["datasets"]:
                source = dataset["source"]
                parsed = parse_one(source, read=connector.dialect)
                sql = (
                    source
                    if isinstance(parsed, exp.Query)
                    else exp.select("*").from_(exp.to_table(source)).sql(dialect=connector.dialect)
                )
                decision = policy.before_sql_read(
                    sql, datasource=datasource, dialect=connector.dialect, policy_context=context
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
        engine = load_binding().Engine(
            model_text=yaml.safe_dump(document), connections=handle._runtime_connections(handle._config), pool_size=1
        )
        result = engine.attribute(
            request.model_dump(mode="python", exclude_none=True),
            connection=handle.profile_name,
            timeout_secs=float(self.runtime.config.timeout_seconds),
        )
        return AttributionResult.model_validate({**result, "implementation": "dosi"})
