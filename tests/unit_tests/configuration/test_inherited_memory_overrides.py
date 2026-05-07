# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.configuration.inherited_memory_overrides``."""

from __future__ import annotations

import asyncio

import pytest

from datus.configuration.inherited_memory_overrides import (
    get_inherited_memory,
    inherited_memory,
)


@pytest.mark.ci
class TestInheritedMemoryOverrides:
    def test_default_returns_none(self):
        assert get_inherited_memory("missing") is None

    def test_push_then_pop_via_context_manager(self):
        with inherited_memory("gen_sql", "chat"):
            assert get_inherited_memory("gen_sql") == "chat"
        assert get_inherited_memory("gen_sql") is None

    def test_nested_overrides_stack_and_unwind(self):
        with inherited_memory("gen_sql", "chat"):
            assert get_inherited_memory("gen_sql") == "chat"
            with inherited_memory("gen_sql", "custom_agent"):
                assert get_inherited_memory("gen_sql") == "custom_agent"
            assert get_inherited_memory("gen_sql") == "chat"
        assert get_inherited_memory("gen_sql") is None

    def test_exception_inside_block_still_resets_token(self):
        with pytest.raises(RuntimeError):
            with inherited_memory("gen_sql", "chat"):
                raise RuntimeError("boom")
        assert get_inherited_memory("gen_sql") is None

    def test_distinct_subagent_names_coexist(self):
        with inherited_memory("gen_sql", "chat"):
            with inherited_memory("gen_report", "custom_agent"):
                assert get_inherited_memory("gen_sql") == "chat"
                assert get_inherited_memory("gen_report") == "custom_agent"
                assert get_inherited_memory("explore") is None

    @pytest.mark.asyncio
    async def test_isolated_across_asyncio_tasks(self):
        """A sibling Task that captures context at creation must not see a later push."""
        captured: dict[str, str | None] = {}

        async def reader():
            await asyncio.sleep(0.01)
            captured["sibling"] = get_inherited_memory("gen_sql")

        async def pusher():
            with inherited_memory("gen_sql", "chat"):
                await asyncio.sleep(0.02)
                captured["inside"] = get_inherited_memory("gen_sql")

        sibling = asyncio.create_task(reader())
        pusher_task = asyncio.create_task(pusher())
        await asyncio.gather(sibling, pusher_task)

        assert captured["sibling"] is None
        assert captured["inside"] == "chat"
        assert get_inherited_memory("gen_sql") is None

    @pytest.mark.asyncio
    async def test_parallel_pushers_do_not_pollute_each_other(self):
        observed: dict[str, str] = {}

        async def run(label: str, parent: str):
            with inherited_memory("gen_sql", parent):
                await asyncio.sleep(0.01)
                observed[label] = get_inherited_memory("gen_sql")

        await asyncio.gather(run("p1", "chat"), run("p2", "custom_agent"))
        assert observed == {"p1": "chat", "p2": "custom_agent"}
