# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""datus-plugin framework: discover ``datus.plugins`` CLI + skill contributors."""

from datus.plugins.base import DatusPlugin
from datus.plugins.registry import (
    PLUGIN_ENTRY_POINT_GROUP,
    iter_plugin_entry_points,
    load_plugin_class,
    plugin_skill_directories,
    plugin_system_prompt_sections,
)

__all__ = [
    "DatusPlugin",
    "PLUGIN_ENTRY_POINT_GROUP",
    "iter_plugin_entry_points",
    "load_plugin_class",
    "plugin_skill_directories",
    "plugin_system_prompt_sections",
]
