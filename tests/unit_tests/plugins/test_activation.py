# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins.activation.active_names_for_cwd``.

CI-level: zero external deps; all I/O is under tmp_path.
"""

import yaml

from datus.plugins.activation import active_names_for_cwd

_REL = ".datus/config.yml"


def _write(tmp_path, payload):
    path = tmp_path / _REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))


def test_absent_section_returns_none(tmp_path):
    """No ``plugins:`` key → ``None`` (no filter, every plugin active)."""
    _write(tmp_path, {"project_name": "p"})
    assert active_names_for_cwd(str(tmp_path)) is None


def test_missing_file_returns_none(tmp_path):
    assert active_names_for_cwd(str(tmp_path)) is None


def test_whitelist_returns_enabled_names(tmp_path):
    _write(
        tmp_path,
        {
            "plugins": {
                "alpha": {"enabled": True},
                "beta": {"enabled": False},
                "gamma": {"enabled": True, "active_profile": ["prod"]},
            }
        },
    )
    assert active_names_for_cwd(str(tmp_path)) == {"alpha", "gamma"}


def test_present_empty_section_returns_empty_set(tmp_path):
    """A present-but-empty ``plugins: {}`` deactivates all → empty set (not None)."""
    _write(tmp_path, {"plugins": {}})
    assert active_names_for_cwd(str(tmp_path)) == set()


def test_string_shorthand_counts_as_enabled(tmp_path):
    _write(tmp_path, {"plugins": {"alpha": "prod"}})
    assert active_names_for_cwd(str(tmp_path)) == {"alpha"}
