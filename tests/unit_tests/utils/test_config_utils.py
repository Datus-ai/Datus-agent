# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import Mock

import pytest

from datus.utils.config_utils import coerce_bool, coerce_positive_int, coerce_positive_seconds


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


class TestCoercePositiveInt:
    """Bounds read from ``agent.yml``, where a typo must not become a tiny bound.

    An absent bound is obvious; a bound of 1 looks deliberate and quietly
    rejects almost everything. These are the coercions Python would perform
    silently to get there.
    """

    @pytest.mark.parametrize("value", [20, 20.0, "20", " 20 "])
    def test_accepts_a_whole_number_however_written(self, value):
        assert coerce_positive_int(value, 500) == 20

    @pytest.mark.parametrize("value", [True, False])
    def test_a_yaml_boolean_is_a_typo_not_a_bound_of_one(self, value):
        """``bool`` is an ``int`` subclass and YAML reads ``true``/``yes``/``on``
        as ``True``, so ``int()`` would hand back a bound of 1."""
        assert coerce_positive_int(value, 500) == 500

    @pytest.mark.parametrize(("value", "expected"), [(3.5, 3), (1.9, 1), (20.5, 20)])
    def test_a_fractional_number_truncates(self, value, expected):
        """Deliberate: request-supplied page sizes come through here, and
        truncating still bounds the request where falling back would hand back a
        larger default than was asked for. Callers needing a whole number must
        check before calling."""
        assert coerce_positive_int(value, 500) == expected

    @pytest.mark.parametrize("value", [0.5, -0.5])
    def test_a_fraction_below_one_falls_back_for_being_non_positive(self, value):
        assert coerce_positive_int(value, 500) == 500

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), "inf", ".inf"])
    def test_infinity_falls_back(self, value):
        """YAML ``.inf`` is what an operator writes for "no limit"; ``int()``
        raises ``OverflowError`` on it rather than converting."""
        assert coerce_positive_int(value, 500) == 500

    @pytest.mark.parametrize("value", [float("nan"), None, "abc", "", [], {"a": 1}, Mock()])
    def test_uninterpretable_values_fall_back(self, value):
        assert coerce_positive_int(value, 500) == 500

    @pytest.mark.parametrize("value", [0, -1, -500])
    def test_non_positive_values_fall_back(self, value):
        assert coerce_positive_int(value, 500) == 500

    def test_returns_a_real_int(self):
        assert type(coerce_positive_int(20.0, 500)) is int


class TestCoercePositiveSeconds:
    """A timeout budget. Unlike a count it keeps fractions, since sub-second
    budgets are meaningful — but it must never resolve to *no* deadline."""

    @pytest.mark.parametrize("value", [0.5, 1.9, 3, "2.5"])
    def test_keeps_fractions(self, value):
        assert coerce_positive_seconds(value, 3.0) == float(value)

    @pytest.mark.parametrize("value", [float("inf"), "inf", ".inf"])
    def test_infinity_is_rejected_not_honoured(self, value):
        """It passes a bare ``> 0`` and would turn a deadline into no deadline —
        the opposite of what a timeout is for."""
        assert coerce_positive_seconds(value, 3.0) == 3.0

    @pytest.mark.parametrize("value", [True, False])
    def test_a_yaml_boolean_falls_back(self, value):
        """``true`` would otherwise buy a 1-second budget."""
        assert coerce_positive_seconds(value, 3.0) == 3.0

    @pytest.mark.parametrize("value", [float("nan"), None, "abc", 0, -1, []])
    def test_uninterpretable_or_non_positive_values_fall_back(self, value):
        assert coerce_positive_seconds(value, 3.0) == 3.0
