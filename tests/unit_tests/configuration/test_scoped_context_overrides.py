# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.configuration.scoped_context_overrides``."""

from __future__ import annotations

import asyncio

import pytest

from datus.configuration.scoped_context_overrides import effective_subagent, get_override
from datus.schemas.agent_models import ScopedContext, SubAgentConfig


def _cfg(tables: str) -> SubAgentConfig:
    return SubAgentConfig(system_prompt="x", scoped_context=ScopedContext(tables=tables))


@pytest.mark.ci
class TestScopedContextOverrides:
    def test_no_override_returns_none(self):
        assert get_override("missing") is None

    def test_push_then_pop_via_context_manager(self):
        cfg = _cfg("public.users")
        with effective_subagent("agent_a", cfg):
            got = get_override("agent_a")
            assert got is not None
            assert got.scoped_context.tables == "public.users"
        # After exit, override is gone.
        assert get_override("agent_a") is None

    def test_nested_overrides_stack_and_unwind(self):
        outer = _cfg("public.users")
        inner = _cfg("public.orders")
        with effective_subagent("agent_a", outer):
            assert get_override("agent_a").scoped_context.tables == "public.users"
            with effective_subagent("agent_a", inner):
                assert get_override("agent_a").scoped_context.tables == "public.orders"
            # Pop restores outer.
            assert get_override("agent_a").scoped_context.tables == "public.users"
        assert get_override("agent_a") is None

    def test_exception_inside_block_still_resets(self):
        cfg = _cfg("public.users")
        with pytest.raises(RuntimeError):
            with effective_subagent("agent_a", cfg):
                raise RuntimeError("boom")
        assert get_override("agent_a") is None

    def test_multiple_distinct_names_coexist(self):
        a = _cfg("t_a")
        b = _cfg("t_b")
        with effective_subagent("agent_a", a):
            with effective_subagent("agent_b", b):
                assert get_override("agent_a").scoped_context.tables == "t_a"
                assert get_override("agent_b").scoped_context.tables == "t_b"

    @pytest.mark.asyncio
    async def test_override_isolated_across_asyncio_tasks(self):
        """Sibling ``asyncio.Task`` that captures context at creation time must not see
        an override pushed afterwards in the parent task."""
        captured = {}

        async def reader(name: str):
            # No override pushed in this task → parent's later push must not leak in.
            await asyncio.sleep(0.01)
            captured[name] = get_override("agent_a")

        async def pusher():
            with effective_subagent("agent_a", _cfg("inside_task")):
                await asyncio.sleep(0.02)
                captured["inside_pusher"] = get_override("agent_a")

        sibling = asyncio.create_task(reader("sibling"))
        pusher_task = asyncio.create_task(pusher())
        await asyncio.gather(sibling, pusher_task)

        assert captured["sibling"] is None
        assert captured["inside_pusher"].scoped_context.tables == "inside_task"
        # Outer scope sees no override either.
        assert get_override("agent_a") is None

    @pytest.mark.asyncio
    async def test_parallel_pushers_do_not_pollute_each_other(self):
        """Two sibling tasks pushing the same name with different cfgs must each see only their own."""
        observed: dict[str, str] = {}

        async def run(label: str, tables: str):
            with effective_subagent("agent_a", _cfg(tables)):
                # Yield to scheduler so the other task runs concurrently.
                await asyncio.sleep(0.01)
                observed[label] = get_override("agent_a").scoped_context.tables

        await asyncio.gather(run("p1", "table_one"), run("p2", "table_two"))
        assert observed == {"p1": "table_one", "p2": "table_two"}
