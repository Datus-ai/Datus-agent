# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared fixtures for manifest-plugin tests.

``plugin_env`` builds REAL importable packages on disk (with a
``datus-plugin.yml``) and registers fake ``datus.plugins`` entry points whose
values are module refs — so tests exercise the actual
``find_spec``-without-import discovery path instead of stubbing it.
"""

import importlib.metadata as importlib_metadata
import itertools
import re

import pytest

from datus.plugins import registry

_PKG_COUNTER = itertools.count()


class FakeEntryPoint:
    """Metadata-shaped entry point: ``name`` + ``value`` (module ref), no load()."""

    def __init__(self, name, value, group="datus.plugins"):
        self.name = name
        self.value = value
        self.group = group


class FakeEntryPoints:
    def __init__(self, eps):
        self._eps = eps

    def select(self, *, group, name=None):
        out = [ep for ep in self._eps if ep.group == group]
        if name is not None:
            out = [ep for ep in out if ep.name == name]
        return out


@pytest.fixture
def plugin_env(tmp_path, monkeypatch):
    """Return a ``register(name, manifest_yaml, ...)`` factory.

    Each call creates a uniquely-named package directory (so imports from one
    test never leak into another), prepends its site dir to ``sys.path``, and
    registers an entry point ``name = <pkg>``. ``{pkg}`` inside
    ``manifest_yaml`` / ``files`` values is replaced with the generated package
    name so manifests can reference their own modules. ``manifest_yaml=None``
    creates a package without a manifest; ``value`` overrides the entry-point
    value (e.g. to simulate a legacy ``pkg:Class`` ref).
    """
    eps = []
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: FakeEntryPoints(eps))

    def register(name, manifest_yaml=None, *, init_body="", files=None, value=None):
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        pkg_name = f"datus_plugin_{safe}_{next(_PKG_COUNTER)}"
        site_dir = tmp_path / f"site_{pkg_name}"
        pkg = site_dir / pkg_name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(init_body, encoding="utf-8")
        if manifest_yaml is not None:
            (pkg / "datus-plugin.yml").write_text(manifest_yaml.replace("{pkg}", pkg_name), encoding="utf-8")
        for rel, content in (files or {}).items():
            target = pkg / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content.replace("{pkg}", pkg_name), encoding="utf-8")
        monkeypatch.syspath_prepend(str(site_dir))
        eps.append(FakeEntryPoint(name, value if value is not None else pkg_name))
        registry.invalidate_plugin_cache()
        return pkg

    return register
