import importlib
import os

import pytest

STRICT_NIGHTLY_ENV = "DATUS_STRICT_NIGHTLY_REQUIREMENTS"


def _strict_nightly_enabled() -> bool:
    return os.getenv(STRICT_NIGHTLY_ENV) == "1"


def require_opt_in_env(var_name: str, docs_path: str) -> None:
    """Gate expensive opt-in tests locally, but fail loudly in strict nightly mode."""
    if os.getenv(var_name) == "1":
        return

    message = f"{var_name}=1 not set; see {docs_path}"
    if _strict_nightly_enabled():
        raise RuntimeError(f"Required nightly test misconfigured: {message}")
    pytest.skip(message, allow_module_level=True)


def import_required(module_name: str, *, reason: str):
    """Import an optional test dependency.

    Local runs may skip when the dependency is missing. Strict nightly mode
    turns that missing dependency into a hard failure so coverage cannot hide
    behind importorskip.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        if _strict_nightly_enabled():
            raise RuntimeError(reason) from exc
        pytest.skip(reason, allow_module_level=True)
