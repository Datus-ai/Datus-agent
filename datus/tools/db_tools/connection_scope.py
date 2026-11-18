"""Keep database borrows alive for a complete synchronous tool operation."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import Condition

_scope = ContextVar("database_connection_scope", default=None)


class _Lifetime:
    def __init__(self):
        self.condition = Condition()
        self.users = 0
        self.closed = False


def _lifetime(tool):
    state = tool.__dict__.get("_connection_lifetime")
    if state is None:
        state = tool.__dict__.setdefault("_connection_lifetime", _Lifetime())
    return state


@contextmanager
def _operation(tool):
    state = _lifetime(tool)
    with state.condition:
        if state.closed:
            raise RuntimeError("Database tool is closed")
        state.users += 1
    try:
        yield
    finally:
        with state.condition:
            state.users -= 1
            state.condition.notify_all()


def drain_connections(tool):
    """Wait for actual worker operations before detaching the tool's bindings."""
    state = _lifetime(tool)
    with state.condition:
        state.closed = True
        while state.users:
            state.condition.wait()


def connection_scope(method=None, *, owner_attr=None):
    if method is None:
        return lambda function: connection_scope(function, owner_attr=owner_attr)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        tool = getattr(self, owner_attr) if owner_attr is not None else self
        current = _scope.get()
        if tool is None or (current is not None and current[0] is tool):
            return method(self, *args, **kwargs)
        with _operation(tool), ExitStack() as stack:
            token = _scope.set((tool, stack, {}))
            try:
                return method(self, *args, **kwargs)
            finally:
                _scope.reset(token)

    wrapped.__datus_database_operation__ = True
    return wrapped


def borrow_connector(tool, datasource, database, **kwargs):
    manager = tool._db_manager
    # Some integrations implement only the historical get_conn protocol.
    if not callable(getattr(type(manager), "acquire_conn", None)):
        return manager.get_conn(datasource, database, **kwargs)
    context = kwargs.get("context")
    key = (datasource, database, getattr(context, "catalog", None), getattr(context, "schema", None))
    scope = _scope.get()
    if scope is not None and scope[0] is tool:
        _, stack, leases = scope
        if key not in leases:
            lease = manager.acquire_conn(datasource, database, **kwargs)
            leases[key] = stack.enter_context(lease)
        return leases[key]
    # Returning a naked connector outside an operation must pin it until the
    # tool closes. This includes the primary connector used by legacy helpers.
    with getattr(tool, "_connection_lock", manager._condition):
        if _lifetime(tool).closed:
            raise RuntimeError("Database tool is closed")
        leases = tool.__dict__.setdefault("_borrowed_connections", {})
        if key not in leases:
            leases[key] = manager.acquire_conn(datasource, database, **kwargs)
        return leases[key].connector
