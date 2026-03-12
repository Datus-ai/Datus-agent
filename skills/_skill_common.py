#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared utilities for skill scripts.

Provides build_agent_config() used by all skill prepare_context.py and save_to_db.py scripts.
Configuration is read from environment variables injected by SkillBashTool:
  - DATUS_CONFIG_PATH: Path to agent.yml
  - DATUS_HOME: Datus home directory
  - DATUS_NAMESPACE: Current namespace
"""

import json
import os
import sys


def build_agent_config():
    """Build AgentConfig from environment variables."""
    config_path = os.environ.get("DATUS_CONFIG_PATH", "")
    home = os.environ.get("DATUS_HOME", "")
    namespace = os.environ.get("DATUS_NAMESPACE", "")

    if not config_path:
        print(json.dumps({"success": False, "error": "DATUS_CONFIG_PATH environment variable not set"}))
        sys.exit(1)

    from datus.configuration.agent_config_loader import load_agent_config

    agent_config = load_agent_config(config=config_path)
    override_kwargs = {}
    if home:
        override_kwargs["home"] = home
    if namespace:
        override_kwargs["namespace"] = namespace
    if override_kwargs:
        agent_config.override_by_args(**override_kwargs)

    return agent_config
