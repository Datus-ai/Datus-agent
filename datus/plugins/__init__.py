# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""datus-plugin framework: discover ``datus.plugins`` manifest contributors."""

from datus.plugins.activation import active_names_for_cwd
from datus.plugins.base import MANIFEST_FILENAME, MANIFEST_VERSION, PluginManifest
from datus.plugins.registry import (
    PLUGIN_ENTRY_POINT_GROUP,
    iter_plugin_entry_points,
    load_plugin_manifest,
    plugin_config_schema,
    plugin_entry_point_exists,
    plugin_skill_directories,
    plugin_system_prompt_sections,
    plugin_validate_profile,
    resolve_code_ref,
)

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "PLUGIN_ENTRY_POINT_GROUP",
    "PluginManifest",
    "active_names_for_cwd",
    "iter_plugin_entry_points",
    "load_plugin_manifest",
    "plugin_config_schema",
    "plugin_entry_point_exists",
    "plugin_skill_directories",
    "plugin_system_prompt_sections",
    "plugin_validate_profile",
    "resolve_code_ref",
]
