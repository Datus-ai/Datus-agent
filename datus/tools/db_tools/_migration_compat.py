# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Compatibility shim for ``MigrationTargetMixin``.

Older releases of ``datus-db-core`` (<= 0.1.2) do not expose the
``MigrationTargetMixin`` symbol. The cross-repo release of the adapter
package that adds the Mixin is sequenced after this agent branch, so we
fall back to a no-op base class when the import fails.

Downstream connectors override every Mixin method they implement, so the
fallback class body is empty — the runtime behavior is unchanged.
"""

try:  # pragma: no cover - exercised implicitly by either branch
    from datus_db_core import MigrationTargetMixin  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover

    class MigrationTargetMixin:
        """Fallback shim until datus-db-core ships the real Mixin."""


__all__ = ["MigrationTargetMixin"]
