# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integrity tests for ``datus.cli.slash_registry``.

These tests protect the invariants consumed by :class:`DatusCLI`:

- every canonical name resolves to a registry entry
- aliases never collide with a different canonical
- ``iter_visible`` respects the ``hidden`` flag
- the registry matches the handler map wired in ``DatusCLI``

Registry integrity is a pure-data concern, so no ``DatusCLI`` construction is
needed here.
"""

from __future__ import annotations

from datus.cli.slash_registry import (
    GROUP_ORDER,
    GROUP_TITLES,
    SLASH_COMMANDS,
    all_tokens,
    iter_visible,
    lookup,
)


class TestSpecIntegrity:
    def test_names_are_unique(self):
        names = [spec.name for spec in SLASH_COMMANDS]
        assert len(names) == len(set(names)), "Duplicate canonical names in SLASH_COMMANDS"

    def test_aliases_are_unique(self):
        aliases: list[str] = []
        for spec in SLASH_COMMANDS:
            aliases.extend(spec.aliases)
        assert len(aliases) == len(set(aliases)), "Duplicate aliases across SLASH_COMMANDS"

    def test_aliases_do_not_collide_with_canonical_names(self):
        canonical = {spec.name for spec in SLASH_COMMANDS}
        for spec in SLASH_COMMANDS:
            for alias in spec.aliases:
                assert alias not in canonical, f"Alias '{alias}' collides with a canonical name"

    def test_every_spec_has_non_empty_summary(self):
        for spec in SLASH_COMMANDS:
            assert spec.summary, f"Spec '{spec.name}' is missing a summary"
            assert not spec.summary.startswith(" ")

    def test_every_group_is_known(self):
        for spec in SLASH_COMMANDS:
            assert spec.group in GROUP_ORDER, f"Spec '{spec.name}' has unknown group '{spec.group}'"
            assert spec.group in GROUP_TITLES


class TestLookup:
    def test_canonical_name_resolves(self):
        spec = lookup("help")
        assert spec.name == "help"

    def test_alias_resolves_to_canonical(self):
        spec = lookup("quit")
        assert spec.name == "exit"

    def test_unknown_token_returns_none(self):
        assert lookup("not-a-real-command") is None


class TestIterVisible:
    def test_iter_visible_skips_hidden(self):
        visible = {spec.name for spec in iter_visible()}
        hidden = {spec.name for spec in SLASH_COMMANDS if spec.hidden}
        assert visible.isdisjoint(hidden)

    def test_all_tokens_covers_names_and_aliases(self):
        tokens = set(all_tokens())
        for spec in SLASH_COMMANDS:
            assert spec.name in tokens
            for alias in spec.aliases:
                assert alias in tokens


class TestHandlerMapAlignment:
    """Handlers are wired in :meth:`DatusCLI._build_slash_handler_map`.

    The registry and the map must agree on canonical names so the commands
    dict covers every visible spec. This test inspects the method directly
    — no DatusCLI instance needed.
    """

    def test_handler_map_covers_every_canonical_name(self):
        import inspect

        from datus.cli.repl import DatusCLI

        source = inspect.getsource(DatusCLI._build_slash_handler_map)
        for spec in SLASH_COMMANDS:
            needle = f'"{spec.name}"'
            assert needle in source, f"Handler for '{spec.name}' missing in _build_slash_handler_map"


def test_profile_command_registered():
    from datus.cli.slash_registry import SLASH_COMMANDS

    names = {spec.name for spec in SLASH_COMMANDS}
    assert "profile" in names


def test_profile_command_is_in_system_group():
    from datus.cli.slash_registry import SLASH_COMMANDS

    spec = next(s for s in SLASH_COMMANDS if s.name == "profile")
    assert spec.group == "system"
    assert "profile" in spec.summary.lower()  # non-empty and autocomplete-relevant
