# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the ``datus.cli_commands`` entry-point dispatch in main()."""

import importlib.metadata as importlib_metadata
from collections.abc import Callable
from typing import Any, Optional

from datus.cli.main import _dispatch_external_command


class _FakeEntryPoint:
    def __init__(self, name: str, handler: Optional[Callable[[list[str]], object]], *, raises: bool = False) -> None:
        self.name = name
        self.group = "datus.cli_commands"
        self._handler = handler
        self._raises = raises

    def load(self) -> Optional[Callable[[list[str]], object]]:
        if self._raises:
            raise ImportError("cannot import adapter cli")
        return self._handler


class _FakeEntryPoints:
    def __init__(self, eps: list["_FakeEntryPoint"]) -> None:
        self._eps = eps

    def select(self, *, group: str, name: Optional[str] = None) -> list["_FakeEntryPoint"]:
        out = [ep for ep in self._eps if ep.group == group]
        if name is not None:
            out = [ep for ep in out if ep.name == name]
        return out


def _patch(monkeypatch: Any, eps: list["_FakeEntryPoint"]) -> None:
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: _FakeEntryPoints(eps))


def test_matching_command_invoked_with_remaining_argv(monkeypatch):
    """A registered command is invoked with argv after the subcommand name and its rc returned."""
    captured = {}

    def handler(argv):
        captured["argv"] = argv
        return 0

    _patch(monkeypatch, [_FakeEntryPoint("hello", handler)])
    rc = _dispatch_external_command(["hello", "dags", "list", "--json"])
    assert rc == 0
    assert captured["argv"] == ["dags", "list", "--json"]


def test_handler_returning_none_maps_to_zero(monkeypatch):
    """A handler returning ``None`` is normalized to exit code 0."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: None)])
    assert _dispatch_external_command(["hello", "version"]) == 0


def test_handler_nonzero_rc_propagates(monkeypatch):
    """A non-zero handler return code propagates unchanged."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: 5)])
    assert _dispatch_external_command(["hello", "dags", "trigger", "x"]) == 5


def test_unknown_command_returns_none(monkeypatch):
    """No matching entry point → None (caller falls through to REPL/argparse)."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: 0)])
    assert _dispatch_external_command(["definitely_unknown"]) is None


def test_flag_only_invocation_returns_none(monkeypatch):
    """A leading flag (datus --web / -p) is never hijacked."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: 0)])
    assert _dispatch_external_command(["--web"]) is None
    assert _dispatch_external_command(["-p", "hi"]) is None


def test_empty_argv_returns_none(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: 0)])
    assert _dispatch_external_command([]) is None


def test_reserved_subcommands_return_none(monkeypatch):
    """Reserved built-in tokens are never dispatched to adapters."""
    _patch(monkeypatch, [_FakeEntryPoint("upgrade", lambda argv: 99), _FakeEntryPoint("skill", lambda argv: 99)])
    assert _dispatch_external_command(["upgrade"]) is None
    assert _dispatch_external_command(["skill", "list"]) is None


def test_load_failure_returns_one(monkeypatch):
    """A broken adapter entry point returns rc=1 instead of crashing."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", None, raises=True)])
    assert _dispatch_external_command(["hello", "dags", "list"]) == 1


def test_handler_exception_returns_one_instead_of_crashing(monkeypatch, capsys):
    """A handler that raises must not crash the CLI with a raw traceback."""

    def handler(argv):
        raise KeyError("bad config")

    _patch(monkeypatch, [_FakeEntryPoint("hello", handler)])
    assert _dispatch_external_command(["hello", "dags", "list"]) == 1
    assert "datus hello" in capsys.readouterr().err


def test_handler_non_int_rc_treated_as_success(monkeypatch):
    """Legacy handlers returning non-int values (e.g. 'ok') must not raise
    ValueError after a successful run."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: "ok")])
    assert _dispatch_external_command(["hello", "version"]) == 0


def test_handler_bool_rc_maps_to_exit_semantics(monkeypatch):
    """True means success (exit 0); False means failure (exit 1)."""
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: True)])
    assert _dispatch_external_command(["hello", "version"]) == 0
    _patch(monkeypatch, [_FakeEntryPoint("hello", lambda argv: False)])
    assert _dispatch_external_command(["hello", "version"]) == 1


def test_multiple_candidates_first_wins(monkeypatch):
    """When two adapters register the same name, the first is used."""
    _patch(
        monkeypatch,
        [
            _FakeEntryPoint("hello", lambda argv: 1),
            _FakeEntryPoint("hello", lambda argv: 2),
        ],
    )
    assert _dispatch_external_command(["hello", "x"]) == 1


def test_lookup_failure_returns_none(monkeypatch):
    """A failure enumerating entry points falls through (None), never raises."""

    def _boom():
        raise RuntimeError("metadata broken")

    monkeypatch.setattr(importlib_metadata, "entry_points", _boom)
    assert _dispatch_external_command(["hello", "dags", "list"]) is None
