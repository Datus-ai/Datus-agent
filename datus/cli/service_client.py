# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CLI service client registry.

Exposes read-only tool methods from ``services.bi_platforms`` /
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
    "bi_platforms": {
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
    "bi_platforms": _build_bi_tool,
    "schedulers": _build_scheduler_tool,
    "semantic_layer": _build_semantic_tool,
}


# ---------------------------------------------------------------------------
# Adapter-availability probes
#
# An entry in ``services.*`` only means "the user listed this in agent.yml" —
# not that the corresponding adapter package is installed. We probe each
# section's registry so listings / completion / dispatch can distinguish
# ``configured`` (usable) from ``missing adapter`` (package not installed
# or adapter not registered). Probes are intentionally cheap and defensive:
# any ImportError / lookup failure is treated as "unavailable" rather than
# raising.
# ---------------------------------------------------------------------------


# Human-readable label for the YAML section name. The section key remains
# the source of truth for config lookups; this mapping is display-only.
TYPE_LABELS: Dict[str, str] = {
    "bi_platforms": "BI platform",
    "schedulers": "scheduler",
    "semantic_layer": "semantic layer",
}


def service_type_label(service_type: str) -> str:
    """Return a user-facing label for an ``agent.yml`` services section key."""
    return TYPE_LABELS.get(service_type, service_type)


_ProbeFn = Callable[["AgentConfig", str], bool]


def _probe_bi_adapter(agent_config: "AgentConfig", service_name: str) -> bool:
    try:
        from datus_bi_core import adapter_registry
    except ImportError:
        return False
    try:
        adapter_registry.discover_adapters()
        # Resolve the actual adapter kind — ``DashboardConfig.adapter_type``
        # is set from the ``type`` field in agent.yml (falls back to the
        # service alias when omitted). This matches what BIFuncTool uses at
        # invocation time, so the probe and the real lookup agree on what
        # "installed" means.
        adapter_type = service_name
        dashboards = getattr(agent_config, "dashboard_config", None) if agent_config else None
        if isinstance(dashboards, dict):
            dash_cfg = dashboards.get(service_name)
            if dash_cfg is not None:
                adapter_type = getattr(dash_cfg, "adapter_type", "") or service_name
        return adapter_registry.get(adapter_type) is not None
    except Exception as exc:
        logger.debug(f"BI adapter probe failed for '{service_name}': {exc}")
        return False


def _probe_semantic_adapter(agent_config: "AgentConfig", service_name: str) -> bool:
    try:
        from datus.tools.semantic_tools.registry import semantic_adapter_registry
    except ImportError:
        return False
    try:
        return semantic_adapter_registry.get_metadata(service_name) is not None
    except Exception as exc:
        logger.debug(f"Semantic adapter probe failed for '{service_name}': {exc}")
        return False


def _probe_scheduler_adapter(agent_config: "AgentConfig", service_name: str) -> bool:
    """True when both ``datus-scheduler-core`` and the platform adapter
    package are importable / registered.

    A scheduler service entry has a ``type`` field (e.g. ``airflow``)
    that points at a platform-specific adapter. Checking only that
    ``datus_scheduler_core`` imports is not enough — core is the base
    framework; the per-platform adapter (``datus-scheduler-airflow`` /
    ``datus-scheduler-dolphinscheduler`` / ...) is what actually
    registers ``SchedulerAdapterRegistry`` entries.
    """
    try:
        from datus_scheduler_core.registry import SchedulerAdapterRegistry
    except ImportError:
        return False

    # Read the platform from the service's config.
    platform = "airflow"
    try:
        cfg_fn = getattr(agent_config, "get_scheduler_config", None)
        if callable(cfg_fn):
            scheduler_cfg = cfg_fn(service_name) or {}
            platform = scheduler_cfg.get("type", platform) or platform
    except Exception as exc:
        logger.debug(f"Could not resolve scheduler platform for '{service_name}': {exc}")

    # Registry may expose one of several getter names depending on
    # ``datus-scheduler-core`` version. Try them in order before giving up.
    for attr in ("get_adapter_class", "has_adapter", "get"):
        getter = getattr(SchedulerAdapterRegistry, attr, None)
        if not callable(getter):
            continue
        try:
            result = getter(platform)
        except Exception:
            continue
        return bool(result)

    # Fallback: core is importable but neither getter API is present.
    # Assume the platform adapter may or may not be there; surface as
    # available so invocation isn't preemptively blocked.
    return True


_PROBES: Dict[str, _ProbeFn] = {
    "bi_platforms": _probe_bi_adapter,
    "schedulers": _probe_scheduler_adapter,
    "semantic_layer": _probe_semantic_adapter,
}


def _probe(agent_config: "AgentConfig", section: str, service_name: str) -> bool:
    fn = _PROBES.get(section)
    if fn is None:
        return True
    try:
        return fn(agent_config, service_name)
    except Exception as exc:
        logger.debug(f"adapter probe raised for {section}/{service_name}: {exc}")
        return False


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
        # Probe results — cached because ``adapter_registry.discover_adapters``
        # (BI) walks entry points, and repeating it on every tab-complete is
        # wasteful. Invalidated together with the client cache when the
        # datasource fingerprint changes.
        self._adapter_available: Dict[str, bool] = {}
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

    def _datasource_fingerprint(self) -> Tuple[Any, ...]:
        """Capture the agent_config state that affects built tool instances.

        ``SemanticTools`` bakes ``current_datasource`` into ``MetricRAG`` /
        ``SemanticModelRAG`` at init time and resolves the adapter against
        the active datasource. ``BIFuncTool.read_connector`` is similarly
        pulled via ``db_manager.get_conn(current_datasource, current_datasource)``
        on first use. If any of those change, cached instances must be
        rebuilt — a session-scoped cache would otherwise keep executing
        queries against the pre-switch datasource after ``.database ...`` /
        ``.datasource ...``.
        """
        cfg = self._agent_config
        return (
            getattr(cfg, "current_datasource", None),
            getattr(cfg, "datasource", None),
        )

    def _invalidate_if_stale(self) -> None:
        """Drop cached clients when the datasource fingerprint has changed."""
        fp = self._datasource_fingerprint()
        if self._fingerprint is None:
            self._fingerprint = fp
            return
        if fp != self._fingerprint:
            self._clients.clear()
            # Adapter availability can, in principle, change with datasource
            # (different registered providers per tenant). Drop the probe
            # cache too so the next listing re-checks.
            self._adapter_available.clear()
            self._fingerprint = fp

    def adapter_available(self, service_name: str) -> bool:
        """Report whether the adapter backing ``service_name`` is installed.

        ``ServiceClientRegistry`` discovers from ``agent.yml``; that tells us
        the service is *configured*, not that its adapter package
        (``datus-bi-<platform>``, ``datus-scheduler-core``, ``datus-semantic-<type>``)
        is installed. The result is cached until the datasource fingerprint
        changes.
        """
        self._invalidate_if_stale()
        key = service_name.lower()
        if key not in self._entries:
            return False
        if key in self._adapter_available:
            return self._adapter_available[key]
        section, original_name = self._entries[key]
        available = _probe(self._agent_config, section, original_name)
        self._adapter_available[key] = available
        return available

    def list_services(self) -> List[Tuple[str, str, str]]:
        """Return ``[(service_name, service_type, status), ...]`` sorted by name.

        ``status`` is user-facing:

        - ``active`` — client already constructed under the current datasource.
        - ``configured`` — adapter package installed; first use will build.
        - ``missing adapter`` — configured in ``agent.yml`` but the adapter
          package isn't installed (or its platform isn't registered).
        """
        self._invalidate_if_stale()
        out: List[Tuple[str, str, str]] = []
        for key, (section, original_name) in sorted(self._entries.items()):
            # Adapter availability is the source of truth for "is this
            # usable?". A ServiceClient in ``_clients`` only means the
            # lightweight wrapper has been constructed — the *FuncTool
            # constructors are lazy, so a client can exist for a service
            # whose platform adapter is missing (e.g. ``dispatch`` built
            # one before the preflight hint fired). Check probe first so
            # we never call such services "active".
            if not self.adapter_available(original_name):
                status = "missing adapter"
            elif key in self._clients:
                status = "active"
            else:
                status = "configured"
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
