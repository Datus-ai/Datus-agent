# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Contract tests for ``datus.storage.vector`` package-level backend registration.

Protects the fix for issue #1308: importing ``datus.storage.vector`` must not
import ``lancedb``. lancedb's wheel is built at the x86-64-haswell baseline
(AVX2/FMA/F16C), so on a CPU without those instructions the import itself dies
with SIGILL — a signal, not an exception, which no ``try/except`` can contain.
Deployments configured with a different vector backend (pgvector) crash-loop
with exit code 132 if that import ever runs, so the absence of the import is an
external contract, not an implementation detail.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import datus.storage.vector as vector_pkg
from datus.storage.vector import VectorRegistry, _LazyLanceVectorBackend
from datus.storage.vector.lance_backend import LanceVectorBackend

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Runs in a clean interpreter: ``sys.modules`` is process-global and sibling
# tests import lancedb directly, so the contract is unobservable in-process.
_IMPORT_PROBE = textwrap.dedent(
    """
    import sys

    import datus.storage.vector as vector_pkg

    leaked = sorted(name for name in sys.modules if name == "lancedb" or name.startswith("lancedb."))
    print("leaked=" + ",".join(leaked))
    print("registered=" + ",".join(sorted(vector_pkg.VectorRegistry.registered_types())))
    """
)


def _run_import_probe() -> dict[str, str]:
    """Import the vector package in a fresh interpreter and report what it loaded."""
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    # Ignore anything the interpreter or logging config may print around the markers.
    fields = {}
    for line in result.stdout.splitlines():
        for key in ("leaked", "registered"):
            if line.startswith(f"{key}="):
                fields[key] = line.split("=", 1)[1]
    assert set(fields) == {"leaked", "registered"}, f"unexpected probe output: {result.stdout!r}"
    return fields


@pytest.fixture(scope="module")
def import_probe() -> dict[str, str]:
    return _run_import_probe()


class TestLazyLanceImport:
    """``import datus.storage.vector`` must not pull in lancedb (issue #1308)."""

    def test_importing_package_does_not_import_lancedb(self, import_probe):
        assert import_probe["leaked"] == ""

    def test_lance_is_registered_without_importing_lancedb(self, import_probe):
        assert "lance" in import_probe["registered"].split(",")


class TestLanceRegistration:
    """The lazy proxy must be indistinguishable from eager registration in use."""

    def test_lance_backend_type_is_registered(self):
        assert VectorRegistry.is_registered("lance") is True
        assert "lance" in VectorRegistry.registered_types()

    def test_create_backend_returns_initialized_lance_backend(self, tmp_path):
        backend = VectorRegistry.create_backend("lance", {"data_dir": str(tmp_path)})

        assert type(backend) is LanceVectorBackend
        # ``initialize(config)`` must have reached the real instance, not the proxy.
        assert backend._data_dir == str(tmp_path)

    def test_created_backend_connects_to_a_project_directory(self, tmp_path):
        backend = VectorRegistry.create_backend("lance", {"data_dir": str(tmp_path)})

        database = backend.connect("proj")

        assert database.table_names() == []
        assert (tmp_path / "proj" / "datus_db").is_dir()

    def test_instantiating_the_proxy_yields_the_real_backend(self):
        backend = _LazyLanceVectorBackend()

        assert type(backend) is LanceVectorBackend
        assert not isinstance(backend, _LazyLanceVectorBackend)


class TestModuleAttributeAccess:
    """PEP 562 ``__getattr__`` keeps the public ``__all__`` surface intact."""

    def test_lance_vector_backend_attribute_resolves_to_the_real_class(self):
        assert vector_pkg.LanceVectorBackend is LanceVectorBackend

    def test_lance_vector_backend_is_exported(self):
        assert "LanceVectorBackend" in vector_pkg.__all__

    def test_unknown_attribute_raises_attribute_error(self):
        missing_name = "does_not_exist"

        with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
            getattr(vector_pkg, missing_name)
