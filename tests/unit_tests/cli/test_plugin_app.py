# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_app.PluginApp`` logic.

CI-level: exercises the data-model / action methods without running the
prompt_toolkit Application (``_focus`` is a no-op with no live layout).
"""

import pytest
from rich.console import Console

from datus.cli import plugin_app as pa
from datus.cli.plugin_app import PluginApp, _View


class FakeConfig:
    """Records activation / profile mutations the app performs."""

    def __init__(self, plugin_services=None, active_profiles_map=None, active=True):
        self.plugin_services = plugin_services or {}
        self._active_profiles_map = active_profiles_map or {}
        self._active = active
        self.activation_calls = []
        self.saved = []
        self.deleted = []

    def active_plugin_names(self):
        return None

    def active_plugin_profiles(self, plugin):
        return self._active_profiles_map.get(plugin)

    def plugin_active(self, plugin):
        return self._active

    def set_plugin_activation(self, name, *, enabled=None, active_profiles=None, clear_profiles=False, persist=True):
        self.activation_calls.append(
            {"name": name, "enabled": enabled, "active_profiles": active_profiles, "clear_profiles": clear_profiles}
        )

    def save_plugin_profile(self, plugin, profile, config):
        self.saved.append((plugin, profile, dict(config)))
        self.plugin_services.setdefault(plugin, {})[profile] = dict(config)

    def delete_plugin_profile(self, plugin, profile):
        self.deleted.append((plugin, profile))
        return self.plugin_services.get(plugin, {}).pop(profile, None) is not None


@pytest.fixture
def app(monkeypatch):
    # list_plugins is called in __init__; stub it to a single plugin.
    from datus.cli.plugin_service import PluginInfo

    monkeypatch.setattr(
        pa,
        "list_plugins",
        lambda cfg: [PluginInfo(name="statsig", package="p", version="1.0", profiles=["dev", "prod"], active=True)],
    )
    cfg = FakeConfig(
        plugin_services={"statsig": {"dev": {"name": "dev"}, "prod": {"name": "prod"}}},
        active_profiles_map={"statsig": None},
    )
    return PluginApp(cfg, Console()), cfg


def test_plugin_items_render(app):
    application, _cfg = app
    items = application._plugin_items()
    assert len(items) == 1
    assert "statsig" in items[0][0]
    assert "2 profile(s)" in items[0][0]


def test_profile_items_all_active(app):
    application, _cfg = app
    application._selected_plugin = "statsig"
    application._view = _View.PROFILE_LIST
    items = application._profile_items()
    labels = [label for label, _ in items]
    # Both profiles active (active_profiles is None → all).
    assert all("[✓]" in label for label in labels)
    assert {"dev", "prod"} == {label.split()[-1] for label in labels}


def test_toggle_plugin_enabled_inverts(app):
    application, cfg = app
    application._toggle_plugin_enabled("statsig")  # currently active → disable
    assert cfg.activation_calls[-1] == {
        "name": "statsig",
        "enabled": False,
        "active_profiles": None,
        "clear_profiles": False,
    }


def test_toggle_profile_active_removes_one(app):
    application, cfg = app
    # All active (None) → toggling 'dev' off leaves ['prod'].
    application._toggle_profile_active("statsig", "dev")
    call = cfg.activation_calls[-1]
    assert call["active_profiles"] == ["prod"]
    assert call["clear_profiles"] is False


def test_toggle_profile_active_back_to_all(app):
    application, cfg = app
    # Start from only 'prod' active → toggling 'dev' on restores "all" (clear).
    cfg._active_profiles_map["statsig"] = ["prod"]
    application._toggle_profile_active("statsig", "dev")
    call = cfg.activation_calls[-1]
    assert call["clear_profiles"] is True


def test_toggle_profile_active_deselect_last_disables_plugin(app):
    application, cfg = app
    # Only 'dev' active → toggling it off leaves no active profile. That must
    # disable the plugin, not persist enabled=True with an empty pin (which
    # would read as active-with-no-narrowing → all profiles).
    cfg._active_profiles_map["statsig"] = ["dev"]
    application._toggle_profile_active("statsig", "dev")
    call = cfg.activation_calls[-1]
    assert call["enabled"] is False
    assert call["clear_profiles"] is True


def test_open_and_submit_new_profile(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [
            {"name": "api_key", "description": "", "required": True, "secret": True},
            {"name": "base_url", "description": "", "required": False, "secret": False, "default": "https://x"},
        ],
    )
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])

    application._open_profile_form("new")
    assert application._view == _View.PROFILE_FORM
    # Fill the name + fields.
    application._form_name_input.text = "staging"
    application._form_inputs[0].text = "${STATSIG_KEY}"  # api_key
    application._form_inputs[1].text = "https://custom"  # base_url
    application._submit_profile_form()

    assert cfg.saved == [("statsig", "staging", {"api_key": "${STATSIG_KEY}", "base_url": "https://custom"})]
    assert application._view == _View.PROFILE_LIST


def test_submit_new_profile_requires_name(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: [])
    application._open_profile_form("new")
    application._form_name_input.text = ""
    application._submit_profile_form()
    assert "name is required" in (application._error_message or "").lower()
    assert cfg.saved == []


def test_submit_missing_required_field_blocks(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [{"name": "api_key", "description": "", "required": True, "secret": True}],
    )
    application._open_profile_form("new")
    application._form_name_input.text = "staging"
    application._form_inputs[0].text = ""  # required, left blank
    application._submit_profile_form()
    assert "Missing required" in (application._error_message or "")
    assert cfg.saved == []


def test_submit_validate_profile_errors_block(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [{"name": "base_url", "description": "", "required": False, "secret": False}],
    )
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_validate_profile", lambda name, profile: ["base_url must be https"]
    )
    application._open_profile_form("new")
    application._form_name_input.text = "staging"
    application._form_inputs[0].text = "ftp://x"
    application._submit_profile_form()
    assert application._error_message == "base_url must be https"
    assert cfg.saved == []


def test_submit_rejects_literal_secret(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [{"name": "api_key", "description": "", "required": True, "secret": True}],
    )
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])
    application._open_profile_form("new")
    application._form_name_input.text = "staging"
    application._form_inputs[0].text = "sk-literal-secret"  # a literal, not a ${ENV_VAR} reference
    application._submit_profile_form()
    assert "${ENV_VAR}" in (application._error_message or "")
    assert cfg.saved == []


def test_submit_accepts_env_ref_secret(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [{"name": "api_key", "description": "", "required": True, "secret": True}],
    )
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])
    application._open_profile_form("new")
    application._form_name_input.text = "staging"
    application._form_inputs[0].text = "${STATSIG_KEY}"
    application._submit_profile_form()
    assert cfg.saved == [("statsig", "staging", {"api_key": "${STATSIG_KEY}"})]


def test_edit_blank_secret_keeps_existing(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    cfg.plugin_services["statsig"]["dev"] = {"name": "dev", "api_key": "${OLD}"}
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [{"name": "api_key", "description": "", "required": True, "secret": True}],
    )
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])

    application._open_profile_form("edit", "dev")
    # Secret field starts blank on edit; leaving it blank keeps the old value.
    assert application._form_inputs[0].text == ""
    application._submit_profile_form()
    assert cfg.saved[-1] == ("statsig", "dev", {"api_key": "${OLD}"})


_NESTED_SPECS = [
    {"name": "api_base_url", "description": "REST endpoint", "required": True, "secret": False},
    {"name": "s3.region", "description": "AWS region", "required": False, "secret": False, "default": "us-east-1"},
    {"name": "s3.secret_access_key", "description": "S3 secret key", "required": True, "secret": True},
]


def _placeholder_of(area):
    """Return the placeholder processor attached to a form input, or None."""
    from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor

    for proc in area.control.input_processors or []:
        if isinstance(proc, ConditionalProcessor) and isinstance(proc.processor, AfterInput):
            return proc
    return None


def test_nested_specs_submit_builds_nested_config(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: list(_NESTED_SPECS))
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])

    application._open_profile_form("new")
    # The default is pre-filled; the no-default and secret fields start blank.
    assert application._form_inputs[0].text == ""
    assert application._form_inputs[1].text == "us-east-1"
    assert application._form_inputs[2].text == ""
    application._form_name_input.text = "alpha"
    application._form_inputs[0].text = "http://127.0.0.1:8080"
    application._form_inputs[2].text = "${S3_SECRET}"
    application._submit_profile_form()

    assert cfg.saved == [
        (
            "statsig",
            "alpha",
            {
                "api_base_url": "http://127.0.0.1:8080",
                "s3": {"region": "us-east-1", "secret_access_key": "${S3_SECRET}"},
            },
        )
    ]


def test_nested_missing_required_blocks(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: list(_NESTED_SPECS))
    application._open_profile_form("new")
    application._form_name_input.text = "alpha"
    application._form_inputs[0].text = "http://127.0.0.1:8080"
    # s3.secret_access_key (required) left blank.
    application._submit_profile_form()
    assert "s3.secret_access_key" in (application._error_message or "")
    assert cfg.saved == []


def test_edit_nested_blank_secret_keeps_existing(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    cfg.plugin_services["statsig"]["dev"] = {
        "name": "dev",
        "api_base_url": "http://a",
        "s3": {"region": "eu-west-1", "secret_access_key": "${OLD_SECRET}"},
    }
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: list(_NESTED_SPECS))
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])

    application._open_profile_form("edit", "dev")
    # Nested non-secret values pre-fill from the stored profile; secrets stay blank.
    assert application._form_inputs[1].text == "eu-west-1"
    assert application._form_inputs[2].text == ""
    application._submit_profile_form()
    assert cfg.saved[-1] == (
        "statsig",
        "dev",
        {"api_base_url": "http://a", "s3": {"region": "eu-west-1", "secret_access_key": "${OLD_SECRET}"}},
    )


def test_freeform_fallback_flattens_nested_existing(app, monkeypatch):
    application, cfg = app
    application._selected_plugin = "statsig"
    cfg.plugin_services["statsig"]["dev"] = {"name": "dev", "url": "https://x", "s3": {"region": "us"}}
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: [])
    monkeypatch.setattr("datus.plugins.registry.plugin_validate_profile", lambda name, profile: [])

    application._open_profile_form("edit", "dev")
    assert [s["name"] for s in application._form_specs] == ["url", "s3.region"]
    assert [a.text for a in application._form_inputs] == ["https://x", "us"]
    application._submit_profile_form()
    assert cfg.saved[-1] == ("statsig", "dev", {"url": "https://x", "s3": {"region": "us"}})


def test_empty_field_shows_description_placeholder(app, monkeypatch):
    application, _cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [
            {"name": "api_base_url", "description": "REST endpoint", "required": True, "secret": False},
            {"name": "region", "description": "", "required": False, "secret": False},
        ],
    )
    application._open_profile_form("new")

    proc = _placeholder_of(application._form_inputs[0])
    assert proc.processor.text == "REST endpoint"
    # Shown while the field is empty, hidden as soon as the user types.
    assert proc.filter()
    application._form_inputs[0].text = "http://x"
    assert not proc.filter()
    # A field without a description gets no placeholder.
    assert _placeholder_of(application._form_inputs[1]) is None


def test_prefilled_default_hides_placeholder(app, monkeypatch):
    application, _cfg = app
    application._selected_plugin = "statsig"
    monkeypatch.setattr(
        "datus.plugins.registry.plugin_config_schema",
        lambda name: [
            {"name": "region", "description": "AWS region", "required": False, "secret": False, "default": "us-east-1"}
        ],
    )
    application._open_profile_form("new")
    proc = _placeholder_of(application._form_inputs[0])
    assert proc.processor.text == "AWS region"
    assert application._form_inputs[0].text == "us-east-1"
    assert not proc.filter()


def test_delete_profile_two_press_confirm(app):
    application, cfg = app
    application._selected_plugin = "statsig"
    application._delete_profile("statsig", "dev")  # first press → arm
    assert cfg.deleted == []
    assert "again" in (application._error_message or "")
    application._delete_profile("statsig", "dev")  # second press → delete
    assert cfg.deleted == [("statsig", "dev")]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
