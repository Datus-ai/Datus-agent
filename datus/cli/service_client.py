# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CLI service client registry.

Exposes read-only tool methods from ``services.bi_tools`` /
``services.schedulers`` / ``services.semantic_layer`` to the CLI via
``ServiceClientRegistry``. Write methods are never registered here — the CLI
is a read surface; mutating operations belong to the agent.

Keyed by the service name the user configured in ``agent.yml``. Multiple BI
services (e.g. ``superset`` and ``superset_prod``) are supported: each gets its
own ``ServiceClient`` entry, and the CLI addresses them by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents import FunctionTool

    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)


# Explicit per-service-type allow-list of read methods (Option C).
# Write methods (create_*, update_*, delete_*, submit_*, write_*, add_*)
# never appear here — so even if a new one is added to a *FuncTool class it
# will not accidentally surface in the CLI.
READ_METHODS: Dict[str, Set[str]] = {
    "bi_tools": {
        "list_dashboards",
        "get_dashboard",
        "list_charts",
        "get_chart",
        "get_chart_data",
        "list_datasets",
        "list_bi_databases",
    },
    "schedulers": {
        "list_scheduler_jobs",
        "get_scheduler_job",
        "list_job_runs",
        "get_run_log",
        "list_scheduler_connections",
    },
    "semantic_layer": {
        "list_metrics",
        "get_dimensions",
        "query_metrics",
        "validate_semantic",
        "attribution_analyze",
    },
}


class ServiceClient:
    """A single configured service, with its read-only methods filter-exposed.

    Exposed methods are the intersection of:

    1. The per-service-type ``READ_METHODS`` allow-list (blocks writes).
    2. The service's own ``available_tools()`` output — adapters like
       ``BIFuncTool`` and ``SemanticTools`` dynamically omit capability-less
       methods (e.g. read-only BI adapter hides ``get_chart_data``; semantic
       tool with no adapter hides ``validate_semantic`` /
       ``attribution_analyze``). Relying on this prevents the CLI from
       advertising commands that would always fail at runtime.

    If the tool instance does not expose ``available_tools`` (rare), the
    allow-list is treated as authoritative.
    """

    def __init__(
        self,
        service_type: str,
        service_name: str,
        tool_instance: Any,
        method_names: Set[str],
    ):
        self.service_type = service_type
        self.service_name = service_name
        self.tool_instance = tool_instance
        self._method_names = method_names
        self._tool_cache: Dict[str, "FunctionTool"] = {}
        self._exposed_cache: Optional[Set[str]] = None

    def _exposed(self) -> Set[str]:
        """Intersect the allow-list with the tool's ``available_tools()`` names."""
        if self._exposed_cache is not None:
            return self._exposed_cache
        available_fn = getattr(self.tool_instance, "available_tools", None)
        if not callable(available_fn):
            # No capability gating on this tool — trust the allow-list.
            self._exposed_cache = {m for m in self._method_names if hasattr(self.tool_instance, m)}
            return self._exposed_cache
        try:
            tools = available_fn()
        except Exception as exc:
            logger.warning(
                f"available_tools() failed on '{self.service_name}': {exc}. Falling back to static allow-list."
            )
            self._exposed_cache = {m for m in self._method_names if hasattr(self.tool_instance, m)}
            return self._exposed_cache
        advertised = {getattr(t, "name", "") for t in (tools or [])}
        self._exposed_cache = self._method_names & advertised
        return self._exposed_cache

    def list_methods(self) -> List[Tuple[str, str]]:
        """Return ``[(method_name, first_line_of_docstring), ...]`` sorted by name."""
        out: List[Tuple[str, str]] = []
        for name in sorted(self._exposed()):
            method = getattr(self.tool_instance, name, None)
            if method is None:
                continue
            doc = (method.__doc__ or "").strip().split("\n", 1)[0]
            out.append((name, doc))
        return out

    def has_method(self, method_name: str) -> bool:
        return method_name in self._exposed()

    def get_tool(self, method_name: str) -> Optional["FunctionTool"]:
        """Return the ``FunctionTool`` wrapper, or ``None`` if the method is blocked."""
        if method_name not in self._exposed():
            return None
        cached = self._tool_cache.get(method_name)
        if cached is not None:
            return cached
        method = getattr(self.tool_instance, method_name, None)
        if method is None:
            return None
        from datus.tools.func_tool.base import trans_to_function_tool

        tool = trans_to_function_tool(method)
        self._tool_cache[method_name] = tool
        return tool


_FactoryFn = Callable[["AgentConfig", str], Any]


def _build_bi_tool(agent_config: "AgentConfig", service_name: str) -> Any:
    from datus.tools.func_tool.bi_tools import BIFuncTool

    return BIFuncTool(agent_config, bi_service=service_name)


def _build_scheduler_tool(agent_config: "AgentConfig", service_name: str) -> Any:
    from datus.tools.func_tool.scheduler_tools import SchedulerTools

    return SchedulerTools(agent_config, scheduler_service=service_name)


def _build_semantic_tool(agent_config: "AgentConfig", service_name: str) -> Any:
    from datus.tools.func_tool.semantic_tools import SemanticTools

    # The YAML key under ``services.semantic_layer`` is the adapter type
    # (e.g. ``metricflow``). Passing it as ``adapter_type`` mirrors how
    # ``SemanticTools`` is used elsewhere.
    return SemanticTools(agent_config, adapter_type=service_name)


# Section-name → (factory, READ_METHODS key). Order is deterministic so
# ``list_services`` output is stable.
_FACTORIES: Dict[str, _FactoryFn] = {
    "bi_tools": _build_bi_tool,
    "schedulers": _build_scheduler_tool,
    "semantic_layer": _build_semantic_tool,
}


class ServiceClientRegistry:
    """Lazily-instantiated registry of CLI-exposed service clients.

    Discovery scans ``agent_config.services`` for configured names under each
    supported section. A service's underlying ``*FuncTool`` is not constructed
    until its first ``get()`` call — merely listing services is free.

    Service names are lowercased on registration so they line up with the
    CLI's lowercased command tokens (see ``DatusCLI._parse_command``).
    """

    def __init__(self, agent_config: "AgentConfig"):
        self._agent_config = agent_config
        # lowered_name → (service_type, original_name)
        # Factory is resolved on each ``get()`` via ``_FACTORIES`` so tests can
        # monkey-patch the module-level helper functions.
        self._entries: Dict[str, Tuple[str, str]] = {}
        self._clients: Dict[str, ServiceClient] = {}
        self._fingerprint: Optional[Tuple[Any, ...]] = None
        self._discover()

    def _discover(self) -> None:
        services = getattr(self._agent_config, "services", None)
        if services is None:
            return

        for section in _FACTORIES:
            entries = getattr(services, section, {}) or {}
            for service_name in entries.keys():
                key = service_name.lower()
                if key in self._entries:
                    existing_section, existing_name = self._entries[key]
                    logger.warning(
                        f"Duplicate CLI service name '{service_name}' in '{section}' "
                        f"collides with '{existing_name}' in '{existing_section}'; "
                        f"ignoring the second entry."
                    )
                    continue
                self._entries[key] = (section, service_name)

    def _namespace_fingerprint(self) -> Tuple[Any, ...]:
        """Capture the agent_config state that affects built tool instances.

        ``SemanticTools`` bakes ``current_database`` into ``MetricRAG`` /
        ``SemanticModelRAG`` at init time and resolves the adapter against
        the active namespace. ``BIFuncTool.read_connector`` is similarly
        pulled via ``db_manager.get_conn(current_database, current_database)``
        on first use. If any of those change, cached instances must be
        rebuilt — a session-scoped cache would otherwise keep executing
        queries against the pre-switch namespace after ``.database ...`` /
        ``.namespace ...``.
        """
        cfg = self._agent_config
        return (
            getattr(cfg, "current_database", None),
            getattr(cfg, "namespace", None),
        )

    def _invalidate_if_stale(self) -> None:
        """Drop cached clients when the namespace fingerprint has changed."""
        fp = self._namespace_fingerprint()
        if self._fingerprint is None:
            self._fingerprint = fp
            return
        if fp != self._fingerprint:
            self._clients.clear()
            self._fingerprint = fp

    def list_services(self) -> List[Tuple[str, str, str]]:
        """Return ``[(service_name, service_type, status), ...]`` sorted by name.

        ``status`` is ``"ready"`` if the client has been constructed under the
        current namespace fingerprint, ``"lazy"`` otherwise.
        """
        self._invalidate_if_stale()
        out: List[Tuple[str, str, str]] = []
        for key, (section, original_name) in sorted(self._entries.items()):
            status = "ready" if key in self._clients else "lazy"
            out.append((original_name, section, status))
        return out

    def has(self, service_name: str) -> bool:
        return service_name.lower() in self._entries

    def get(self, service_name: str) -> Optional[ServiceClient]:
        """Return the ``ServiceClient`` for ``service_name`` (lazy construct)."""
        self._invalidate_if_stale()
        key = service_name.lower()
        cached = self._clients.get(key)
        if cached is not None:
            return cached
        entry = self._entries.get(key)
        if entry is None:
            return None
        service_type, original_name = entry
        factory = _FACTORIES.get(service_type)
        if factory is None:
            return None
        try:
            instance = factory(self._agent_config, original_name)
        except Exception as exc:
            logger.error(f"Failed to build service client '{original_name}': {exc}")
            return None
        client = ServiceClient(
            service_type=service_type,
            service_name=original_name,
            tool_instance=instance,
            method_names=READ_METHODS.get(service_type, set()),
        )
        self._clients[key] = client
        return client
