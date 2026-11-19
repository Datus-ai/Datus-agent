# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plain-data semantic authoring capabilities injected into private runtimes."""

from collections.abc import Awaitable, Callable
from typing import Any


class SemanticAuthoringHost:
    """A plugin-facing contract, not a model tool or a shared plugin SDK.

    Plugins use this object by duck typing. The host retains the session and
    applies the current tool permissions and resource scope on every operation.
    """

    def __init__(self, invoke: Callable[..., Awaitable[dict[str, Any]]]):
        self._invoke = invoke

    async def state(self) -> dict[str, Any]:
        """Report selection and mutation state even before a target is bound."""
        return await self._invoke("state")

    async def snapshot(self) -> dict[str, Any]:
        """Read the complete selected document and its revision, without preview truncation."""
        return await self._invoke("snapshot")

    async def prepare(self) -> dict[str, Any]:
        """Apply host-managed identities before validating the exact final bytes."""
        return await self._invoke("prepare_artifact")

    async def validate(self) -> dict[str, Any]:
        """Validate the current selected revision with the existing semantic engine."""
        return await self._invoke("validate_artifact")

    async def reconcile(self) -> dict[str, Any]:
        """Prepare, validate and reconcile the current revision; retain rollback state."""
        return await self._invoke("reconcile_artifact")

    async def complete(self) -> dict[str, Any]:
        """Release rollback state once the exact current revision has reached the KB."""
        return await self._invoke("complete")

    async def begin_draft(self) -> dict[str, Any]:
        """Start an isolated per-turn draft and return authorized artifact inputs."""
        return await self._invoke("begin_draft")

    async def inspect_bundle(
        self, files: dict[str, str], mapping_file: str, *, operation: str = "validate", arguments: dict | None = None
    ) -> dict[str, Any]:
        """Inspect exact draft inputs with production discovery and native lowering."""
        return await self._invoke(
            "inspect_bundle",
            {
                "files": files,
                "mapping_file": mapping_file,
                "inspection": operation,
                "arguments": arguments or {},
            },
        )

    async def publish(self, files: dict[str, str], *, expected_revision: str) -> dict[str, Any]:
        """Publish the selected model and validated private artifacts as one bundle."""
        return await self._invoke("publish", {"files": files, "expected_revision": expected_revision})
