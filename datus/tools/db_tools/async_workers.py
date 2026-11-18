# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Own blocking database work without joining native drivers at process exit."""

import asyncio
import contextvars
import queue
import threading
from concurrent.futures import Executor, Future
from typing import Any, Callable

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DatabaseWorkers(Executor):
    """Bound worker count and retain jobs until completion, even after a waiter leaves.

    Native drivers cannot reliably be interrupted. These daemon threads are
    deliberately outside ThreadPoolExecutor's interpreter-exit join registry.
    Session persistence must never be submitted here.
    """

    def __init__(self, max_workers: int = 8) -> None:
        if max_workers < 1:
            raise ValueError("Database workers must have at least one worker")
        self._max_workers = max_workers
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._threads = []
        self._jobs = set()
        self._closed = False

    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any) -> Future:
        future = Future()
        context = contextvars.copy_context()
        with self._lock:
            if self._closed:
                raise RuntimeError("Database workers are shut down")
            self._jobs.add(future)
            self._queue.put((future, context, fn, args, kwargs))
            if len(self._threads) < min(self._max_workers, len(self._jobs)):
                worker = threading.Thread(target=self._work, name="datus-database", daemon=True)
                self._threads.append(worker)
                worker.start()
        future.add_done_callback(self._observe)
        return future

    def _observe(self, future: Future) -> None:
        with self._lock:
            self._jobs.discard(future)
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                # Driver errors can include queries/credentials; log only the type.
                logger.warning("Database worker failed (%s)", type(error).__name__)

    @staticmethod
    def _execute(work: tuple) -> None:
        future, context, fn, args, kwargs = work
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(context.run(fn, *args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def _work(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work is None:
                    return
                self._execute(work)
            finally:
                del work
                self._queue.task_done()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                for _ in self._threads:
                    self._queue.put(None)
            jobs, threads = list(self._jobs), list(self._threads)
        if cancel_futures:
            for job in jobs:
                job.cancel()
        if wait:
            for thread in threads:
                thread.join()


# Separate queues keep close jobs from consuming the database-operation slots
# whose completion they are waiting for. Neither queue owns session writes.
DATABASE_OPERATIONS = DatabaseWorkers()
DATABASE_CLEANUP = DatabaseWorkers(max_workers=2)


async def run_database_operation(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.wrap_future(DATABASE_OPERATIONS.submit(fn, *args, **kwargs))
