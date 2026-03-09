# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level tests for datus.storage.testing module."""

import pytest

from datus.storage.testing import RdbTestEnv, TestEnvConfig, VectorTestEnv


class TestTestEnvConfig:
    """Tests for the TestEnvConfig dataclass."""

    def test_defaults(self):
        """TestEnvConfig has empty params by default."""
        cfg = TestEnvConfig(backend_type="sqlite")
        assert cfg.backend_type == "sqlite"
        assert cfg.params == {}

    def test_with_params(self):
        """TestEnvConfig stores arbitrary params."""
        params = {"host": "localhost", "port": 5432}
        cfg = TestEnvConfig(backend_type="postgresql", params=params)
        assert cfg.backend_type == "postgresql"
        assert cfg.params == params

    def test_equality(self):
        """Two TestEnvConfig with same values are equal."""
        cfg1 = TestEnvConfig(backend_type="postgresql", params={"host": "localhost"})
        cfg2 = TestEnvConfig(backend_type="postgresql", params={"host": "localhost"})
        assert cfg1 == cfg2

    def test_params_default_is_independent(self):
        """Default params dict is independent across instances."""
        cfg1 = TestEnvConfig(backend_type="a")
        cfg2 = TestEnvConfig(backend_type="b")
        cfg1.params["key"] = "value"
        assert "key" not in cfg2.params


class TestRdbTestEnvABC:
    """Tests for the RdbTestEnv abstract base class."""

    def test_cannot_instantiate(self):
        """RdbTestEnv cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RdbTestEnv()

    def test_must_implement_all_methods(self):
        """Subclass missing abstract methods cannot be instantiated."""

        class PartialImpl(RdbTestEnv):
            def setup(self):
                pass

            def teardown(self):
                pass

        with pytest.raises(TypeError):
            PartialImpl()

    def test_concrete_subclass(self):
        """A complete concrete subclass can be instantiated."""

        class ConcreteRdbTestEnv(RdbTestEnv):
            def setup(self):
                pass

            def teardown(self):
                pass

            def clear_data(self, namespace):
                pass

            def get_config(self):
                return TestEnvConfig(backend_type="test")

        env = ConcreteRdbTestEnv()
        assert isinstance(env, RdbTestEnv)
        cfg = env.get_config()
        assert cfg.backend_type == "test"


class TestVectorTestEnvABC:
    """Tests for the VectorTestEnv abstract base class."""

    def test_cannot_instantiate(self):
        """VectorTestEnv cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorTestEnv()

    def test_must_implement_all_methods(self):
        """Subclass missing abstract methods cannot be instantiated."""

        class PartialImpl(VectorTestEnv):
            def setup(self):
                pass

            def teardown(self):
                pass

        with pytest.raises(TypeError):
            PartialImpl()

    def test_concrete_subclass(self):
        """A complete concrete subclass can be instantiated."""

        class ConcreteVectorTestEnv(VectorTestEnv):
            def setup(self):
                pass

            def teardown(self):
                pass

            def clear_data(self, namespace):
                pass

            def get_config(self):
                return TestEnvConfig(backend_type="test")

        env = ConcreteVectorTestEnv()
        assert isinstance(env, VectorTestEnv)
        cfg = env.get_config()
        assert cfg.backend_type == "test"
