# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.autocomplete.ServiceCommandCompleter``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from datus.cli.autocomplete import ServiceCommandCompleter
from datus.cli.service_client import READ_METHODS, ServiceClient, ServiceClientRegistry


class _Tool:
    """A fake tool with a few read methods + advertised_tools capability gate."""

    def __init__(self, advertised):
        self._advertised = advertised

    def list_dashboards(self):
        """List all dashboards."""

    def get_dashboard(self):
        """Get one dashboard."""

    def get_chart_data(self):
        """Get chart data (capability-gated)."""

    def available_tools(self):
        return [SimpleNamespace(name=n) for n in self._advertised]


def _make_completer(advertised=("list_dashboards", "get_dashboard")):
    """Construct a ServiceCommandCompleter wired to an in-memory registry."""
    registry = ServiceClientRegistry.__new__(ServiceClientRegistry)
    registry._agent_config = None
    registry._entries = {"superset": ("bi_platforms", "superset")}
    registry._fingerprint = None
    registry._adapter_available = {"superset": True}
    registry._clients = {
        "superset": ServiceClient(
            service_type="bi_platforms",
            service_name="superset",
            tool_instance=_Tool(advertised=set(advertised)),
            method_names=READ_METHODS["bi_platforms"],
        ),
    }
    service_commands = MagicMock()
    service_commands.registry = registry
    cli = MagicMock()
    cli.service_commands = service_commands
    return ServiceCommandCompleter(cli), registry


def _completions(completer, text):
    doc = Document(text=text, cursor_position=len(text))
    return list(completer.get_completions(doc))


class TestNonSlashInput:
    def test_plain_text_yields_nothing(self):
        completer, _ = _make_completer()
        assert _completions(completer, "select * from t") == []

    def test_at_reference_yields_nothing(self):
        completer, _ = _make_completer()
        assert _completions(completer, "@Table ") == []

    def test_dot_prefix_yields_nothing(self):
        """Legacy ``.`` prefix is no longer recognised — the completer now
        owns the ``/<service>.<method>`` namespace exclusively."""
        completer, _ = _make_completer()
        assert _completions(completer, ".superset") == []


class TestServiceNameCompletion:
    def test_bare_slash_lists_services(self):
        """``/`` on its own offers the registered service names. ``/services``
        itself is registered in the slash registry, not here, so this
        completer does not emit it — tested separately in the slash
        completer's own coverage."""
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/")]
        assert "/superset" in texts
        assert "/services" not in texts

    def test_partial_service_name_filters(self):
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/super")]
        assert "/superset" in texts

    def test_exact_service_shows_itself(self):
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/superset")]
        assert texts == ["/superset"]

    def test_display_includes_service_type(self):
        completer, _ = _make_completer()
        completions = _completions(completer, "/sup")
        displays = [str(c.display) for c in completions]
        # Completions show the human-readable label (not the raw agent.yml key).
        assert any("BI platform" in d for d in displays)

    def test_missing_adapter_annotated_in_display(self):
        """Completions for services whose adapter isn't installed show a
        "(missing adapter)" annotation so the user isn't surprised when
        pressing Enter."""
        completer, registry = _make_completer()
        # Pretend the client was never built AND the probe failed — this is
        # the real-world "datus-bi-<platform> not installed" scenario. (The
        # default helper pre-caches a client for happy-path tests; drop it
        # so ``list_services`` evaluates the probe path.)
        registry._clients = {}
        registry._adapter_available = {"superset": False}
        completions = _completions(completer, "/sup")
        displays = [str(c.display) for c in completions]
        assert any("missing adapter" in d for d in displays)


class TestMethodNameCompletion:
    def test_dot_after_service_lists_methods(self):
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/superset.")]
        # Both advertised read methods should be offered; unadvertised ones skipped.
        assert "/superset.list_dashboards " in texts
        assert "/superset.get_dashboard " in texts
        # get_chart_data isn't advertised in the stub → should NOT appear.
        assert not any("get_chart_data" in t for t in texts)

    def test_partial_method_filters(self):
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/superset.get")]
        assert "/superset.get_dashboard " in texts
        assert "/superset.list_dashboards " not in texts

    def test_unknown_service_yields_nothing(self):
        completer, _ = _make_completer()
        assert _completions(completer, "/mystery.") == []

    def test_capability_gating_hides_method(self):
        """Methods in the allow-list but not advertised by available_tools() are hidden."""
        completer, _ = _make_completer(advertised=("list_dashboards",))
        texts = [c.text for c in _completions(completer, "/superset.")]
        assert texts == ["/superset.list_dashboards "]

    def test_missing_adapter_hides_methods(self):
        """When the adapter package isn't installed, completing
        ``/<service>.`` must not offer method names that would only fail
        on Enter — tab complete should stay honest with what's
        executable."""
        completer, registry = _make_completer()
        registry._clients = {}
        registry._adapter_available = {"superset": False}
        texts = [c.text for c in _completions(completer, "/superset.")]
        # No method completions — only a single informational placeholder.
        assert not any(t.startswith("/superset.") for t in texts)


class TestArgFlagCompletion:
    def test_double_dash_suggests_help_and_params(self):
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/superset.get_dashboard --")]
        assert "--help" in texts
        # get_dashboard stub takes no params, but --help should always be there.

    def test_partial_flag_filters(self):
        """A partial flag prefix narrows the suggestions."""
        completer, _ = _make_completer()
        texts = [c.text for c in _completions(completer, "/superset.get_dashboard --h")]
        assert texts == ["--help"]

    def test_positional_arg_yields_nothing(self):
        completer, _ = _make_completer()
        assert _completions(completer, "/superset.get_dashboard abc") == []

    def test_flag_with_equals_yields_nothing(self):
        """Once the user has typed ``=``, we don't complete the value."""
        completer, _ = _make_completer()
        assert _completions(completer, "/superset.get_dashboard --help=") == []

    def test_unknown_service_in_args_yields_nothing(self):
        completer, _ = _make_completer()
        assert _completions(completer, "/mystery.foo --") == []


class TestDefensiveBranches:
    def test_no_service_commands_attribute_yields_nothing(self):
        cli = SimpleNamespace()  # no ``service_commands`` attribute
        completer = ServiceCommandCompleter(cli)
        assert _completions(completer, "/superset.") == []

    def test_registry_exception_yields_nothing(self):
        cli = MagicMock()
        cli.service_commands = MagicMock()
        # Accessing ``registry`` raises.
        type(cli.service_commands).registry = property(fget=lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        completer = ServiceCommandCompleter(cli)
        assert _completions(completer, "/superset.") == []
