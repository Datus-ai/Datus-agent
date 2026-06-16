"""Tests for datus.api.services.background_drain — shared background task registry."""

import asyncio

import pytest

import datus.api.services.background_drain as mod
from datus.api.services.background_drain import drain_background_tasks, track_background_task


@pytest.mark.asyncio
class TestDrainBackgroundTasks:
    """Tests for drain_background_tasks / track_background_task lifecycle."""

    async def test_drain_awaits_registered_tasks(self):
        """drain_background_tasks() awaits all tasks added to _background_tasks."""
        completed = []

        async def _work():
            await asyncio.sleep(0.01)
            completed.append(1)

        original = mod._background_tasks
        task = asyncio.create_task(_work())
        mod._background_tasks = {task}
        task.add_done_callback(mod._background_tasks.discard)

        try:
            await drain_background_tasks()
            assert completed == [1]
            assert len(mod._background_tasks) == 0
        finally:
            mod._background_tasks = original

    async def test_drain_is_noop_when_empty(self):
        """drain_background_tasks() leaves the set empty and does not raise."""
        original = mod._background_tasks
        mod._background_tasks = set()
        try:
            await drain_background_tasks()
            assert len(mod._background_tasks) == 0
        finally:
            mod._background_tasks = original

    async def test_drain_tolerates_task_exception(self):
        """drain_background_tasks() swallows task exceptions and empties the set."""
        original = mod._background_tasks
        mod._background_tasks = set()

        async def _boom():
            raise RuntimeError("hook failure")

        task = asyncio.create_task(_boom())
        mod._background_tasks.add(task)
        task.add_done_callback(mod._background_tasks.discard)

        try:
            await drain_background_tasks()
            assert len(mod._background_tasks) == 0
        finally:
            mod._background_tasks = original

    async def test_track_registers_and_auto_removes_on_completion(self):
        """track_background_task() adds the task and removes it via done-callback."""
        pause = asyncio.Event()

        async def _work():
            await pause.wait()

        original = mod._background_tasks
        mod._background_tasks = set()
        try:
            task = asyncio.create_task(_work())
            track_background_task(task)
            assert len(mod._background_tasks) == 1

            pause.set()
            await task  # wait for completion; done-callback runs before await returns
            assert len(mod._background_tasks) == 0
        finally:
            mod._background_tasks = original
