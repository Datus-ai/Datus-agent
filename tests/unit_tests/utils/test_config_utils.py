# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import Mock

import pytest

from datus.utils.config_utils import coerce_bool


class TestCoerceBool:
    """The shared YAML-boolean coercion behind every ``agent.yml`` toggle.

    ``bool("false")`` is ``True`` in Python, so a naive cast on a quoted YAML
    value silently inverts a switch. For ``sql_read_only`` that would mean a
    deployment believing it is read-only while every write path is open.
    """

    @pytest.mark.parametrize("value", [True, "true", "True", "TRUE", " true ", "yes", "on", "1"])
    def test_truthy_spellings(self, value):
        assert coerce_bool(value, False) is True

    @pytest.mark.parametrize("value", [False, "false", "False", "FALSE", " false ", "no", "off", "0", ""])
    def test_falsy_spellings(self, value):
        """``"false"`` is the one that matters — the whole reason this helper
        exists instead of a ``bool()`` cast."""
        assert coerce_bool(value, True) is False

    @pytest.mark.parametrize("default", [True, False])
    def test_none_yields_the_default(self, default):
        """``None`` means the key is absent, which is different from being set
        to a falsy value."""
        assert coerce_bool(None, default) is default

    @pytest.mark.parametrize("value", [1, 2, -1, [0], {"a": 1}])
    def test_unrecognised_truthy_values_read_as_on(self, value):
        """Fail-closed for security toggles: something we cannot interpret reads
        as "on" rather than silently off."""
        assert coerce_bool(value, False) is True

    @pytest.mark.parametrize("value", [0, [], {}])
    def test_unrecognised_falsy_values_read_as_off(self, value):
        assert coerce_bool(value, True) is False

    def test_a_bare_mock_reads_as_on(self):
        """Pins the behaviour that shapes how tests must build config doubles: a
        ``Mock`` misses the None/bool/str branches and falls through to
        ``bool(value)``. Test configs therefore have to pin security flags
        explicitly (see ``_mock_agent_config`` in the DBFuncTool tests) rather
        than relying on this helper to normalize them away.
        """
        assert coerce_bool(Mock(), False) is True

    def test_returns_real_bools_not_truthy_objects(self):
        """Callers store the result on config objects and compare with ``is``."""
        for value in ("yes", "no", 1, 0, None):
            assert type(coerce_bool(value, False)) is bool
