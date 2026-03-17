# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import os
from unittest.mock import patch

import pytest

from datus.utils.env import get_env_int


class TestGetEnvInt:
    def test_returns_default_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            if "TEST_MY_VAR" in os.environ:
                del os.environ["TEST_MY_VAR"]
            result = get_env_int("TEST_MY_VAR", default=42)
        assert result == 42

    def test_returns_zero_as_default_when_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            if "TEST_MISSING_VAR" in os.environ:
                del os.environ["TEST_MISSING_VAR"]
            result = get_env_int("TEST_MISSING_VAR")
        assert result == 0

    def test_returns_integer_when_valid_env_set(self):
        with patch.dict(os.environ, {"TEST_INT_VAR": "123"}):
            result = get_env_int("TEST_INT_VAR")
        assert result == 123

    def test_returns_negative_integer(self):
        with patch.dict(os.environ, {"TEST_NEG_VAR": "-10"}):
            result = get_env_int("TEST_NEG_VAR")
        assert result == -10

    def test_returns_zero_from_env(self):
        with patch.dict(os.environ, {"TEST_ZERO_VAR": "0"}):
            result = get_env_int("TEST_ZERO_VAR")
        assert result == 0

    def test_returns_default_when_env_not_integer(self):
        with patch.dict(os.environ, {"TEST_INVALID_VAR": "not_an_int"}):
            result = get_env_int("TEST_INVALID_VAR", default=99)
        assert result == 99

    def test_returns_default_when_env_is_float_string(self):
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "3.14"}):
            result = get_env_int("TEST_FLOAT_VAR", default=7)
        assert result == 7

    def test_returns_default_when_env_is_empty_string(self):
        with patch.dict(os.environ, {"TEST_EMPTY_VAR": ""}):
            result = get_env_int("TEST_EMPTY_VAR", default=5)
        assert result == 5

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1", 1),
            ("100", 100),
            ("-5", -5),
            ("0", 0),
        ],
    )
    def test_various_valid_integers(self, value, expected):
        with patch.dict(os.environ, {"MY_VAR": value}):
            assert get_env_int("MY_VAR") == expected
