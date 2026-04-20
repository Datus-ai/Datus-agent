# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Config migration tool: converts legacy namespace-based agent.yml to new services-based format.

Usage:
    python -m datus.configuration.config_migrator [--config path/to/agent.yml] [--dry-run]
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def migrate_namespace_to_services(config: dict) -> dict:
    """Convert legacy namespace config to new services.databases format.

    Old format:
        agent:
          namespace:
            my_ns:
              type: sqlite
              dbs:
                - name: db1
                  uri: path/to/db1.sqlite
              ...
            another_ns:
              type: snowflake
              ...

    New format:
        agent:
          services:
            databases:
              db1:
                type: sqlite
                uri: path/to/db1.sqlite
              another_ns:
                type: snowflake
                ...
            semantic_layer: {}
            bi_tools: {}
            schedulers: {}
    """
    config = copy.deepcopy(config)
    agent = config.get("agent", {})

    if "services" in agent:
        logger.info("Config already uses 'services' format, no migration needed.")
        return config

    namespace_config = agent.pop("namespace", {})
    if not namespace_config:
        logger.info("No 'namespace' section found, nothing to migrate.")
        return config

    databases = {}
    first_db_name = None

    for ns_name, ns_cfg in namespace_config.items():
        if not isinstance(ns_cfg, dict):
            continue
        db_type = ns_cfg.get("type", "")

        if "dbs" in ns_cfg:
            # Multi-database namespace: flatten each db as independent entry
            for item in ns_cfg["dbs"]:
                name = item.get("name", ns_name)
                entry = {k: v for k, v in item.items() if k != "name"}
                entry["type"] = db_type
                databases[name] = entry
                if first_db_name is None:
                    first_db_name = name
        elif "path_pattern" in ns_cfg:
            # Glob pattern: keep as-is under the namespace name
            databases[ns_name] = ns_cfg
            if first_db_name is None:
                first_db_name = ns_name
        else:
            # Single database or server-based (snowflake, starrocks, etc.)
            databases[ns_name] = ns_cfg
            if first_db_name is None:
                first_db_name = ns_name

    # Mark first database as default if only one entry or user had a single namespace
    if len(databases) == 1:
        only_key = next(iter(databases))
        databases[only_key]["default"] = True

    agent["services"] = {
        "databases": databases,
        "semantic_layer": {},
        "bi_tools": {},
        "schedulers": {},
    }

    # Update any --namespace references in agentic_nodes or workflow if present
    config["agent"] = agent
    return config


def migrate_file(config_path: str, dry_run: bool = False) -> bool:
    """Migrate a single agent.yml file.

    Returns True if migration was performed.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "agent" not in config:
        logger.error(f"Invalid config file (no 'agent' section): {config_path}")
        return False

    if "services" in config.get("agent", {}):
        logger.info(f"Already migrated: {config_path}")
        return False

    if "namespace" not in config.get("agent", {}):
        logger.info(f"No namespace section to migrate: {config_path}")
        return False

    migrated = migrate_namespace_to_services(config)

    if dry_run:
        print("--- Migrated config (dry run) ---")
        print(yaml.dump(migrated, default_flow_style=False, allow_unicode=True))
        return True

    # Backup original
    backup_path = path.with_suffix(".yml.bak")
    path.rename(backup_path)
    logger.info(f"Backed up original to: {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(migrated, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Migrated config saved to: {config_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate agent.yml from namespace to services format")
    parser.add_argument("--config", type=str, default="conf/agent.yml", help="Path to agent.yml")
    parser.add_argument("--dry-run", action="store_true", help="Print migrated config without writing")
    args = parser.parse_args()

    success = migrate_file(args.config, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
