# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract base classes for storage backend test environment providers.

Test environments are discovered via entry points and manage the lifecycle
of external resources (containers, databases, etc.) needed for testing.

Entry point groups:
    datus.storage.rdb.testing    -- RDB test environment providers
    datus.storage.vector.testing -- Vector test environment providers

Each entry point should reference a factory function returning a TestEnv instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TestEnvConfig:
    """Configuration returned by a test environment provider.

    Attributes:
        backend_type: Backend type name (e.g. "postgresql") matching the registry key.
        params: Parameters passed to BaseRdbBackend.initialize() / BaseVectorBackend.initialize().
    """

    __test__ = False  # prevent pytest collection

    backend_type: str
    params: Dict[str, Any] = field(default_factory=dict)


class RdbTestEnv(ABC):
    """RDB backend test environment provider.

    Lifecycle:
        setup()      -- session-level, start containers / prepare resources
        get_config() -- return connection parameters
        clear_data() -- test-level, clean data for isolation
        teardown()   -- session-level, destroy resources
    """

    @abstractmethod
    def setup(self) -> None:
        """Initialize the test environment (start containers, create databases, etc.)."""

    @abstractmethod
    def teardown(self) -> None:
        """Destroy the test environment (stop containers, etc.)."""

    @abstractmethod
    def clear_data(self, namespace: str) -> None:
        """Clear all data for the given namespace.

        File-based backends (e.g. SQLite) may no-op since tmp_path provides isolation.
        Server-based backends should DROP SCHEMA or TRUNCATE all tables.
        """

    @abstractmethod
    def get_config(self) -> TestEnvConfig:
        """Return configuration for the current test environment."""


class VectorTestEnv(ABC):
    """Vector backend test environment provider.

    Lifecycle is identical to RdbTestEnv.
    """

    @abstractmethod
    def setup(self) -> None:
        """Initialize the test environment."""

    @abstractmethod
    def teardown(self) -> None:
        """Destroy the test environment."""

    @abstractmethod
    def clear_data(self, namespace: str) -> None:
        """Clear all data for the given namespace."""

    @abstractmethod
    def get_config(self) -> TestEnvConfig:
        """Return configuration for the current test environment."""
