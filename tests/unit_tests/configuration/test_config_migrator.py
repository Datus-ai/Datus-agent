# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/configuration/config_migrator.py"""

import yaml

from datus.configuration.config_migrator import migrate_file, migrate_namespace_to_services


class TestMigrateNamespaceToServices:
    """Tests for migrate_namespace_to_services()."""

    def test_single_db_namespace_produces_services_databases(self):
        """Old namespace with single DB entry becomes services.databases entry."""
        config = {
            "agent": {
                "namespace": {
                    "my_db": {
                        "type": "sqlite",
                        "uri": "path/to/my.sqlite",
                    }
                }
            }
        }
        result = migrate_namespace_to_services(config)
        services = result["agent"]["services"]
        assert "databases" in services
        assert "my_db" in services["databases"]
        db = services["databases"]["my_db"]
        assert db["type"] == "sqlite"
        assert db["uri"] == "path/to/my.sqlite"
        # Single-entry default mark
        assert db.get("default") is True
        assert services["semantic_layer"] == {}
        assert services["bi_tools"] == {}
        assert services["schedulers"] == {}

    def test_dbs_list_is_flattened_to_individual_entries(self):
        """Namespace with 'dbs' list produces one entry per db, type inherited from namespace."""
        config = {
            "agent": {
                "namespace": {
                    "my_ns": {
                        "type": "sqlite",
                        "dbs": [
                            {"name": "db1", "uri": "path/to/db1.sqlite"},
                            {"name": "db2", "uri": "path/to/db2.sqlite"},
                        ],
                    }
                }
            }
        }
        result = migrate_namespace_to_services(config)
        databases = result["agent"]["services"]["databases"]
        assert "db1" in databases
        assert "db2" in databases
        assert databases["db1"]["type"] == "sqlite"
        assert databases["db1"]["uri"] == "path/to/db1.sqlite"
        assert databases["db2"]["type"] == "sqlite"
        assert databases["db2"]["uri"] == "path/to/db2.sqlite"
        # 'name' key should be stripped from individual entries
        assert "name" not in databases["db1"]
        assert "name" not in databases["db2"]

    def test_path_pattern_preserved_under_namespace_name(self):
        """Namespace with path_pattern keeps the entry under namespace name."""
        config = {
            "agent": {
                "namespace": {
                    "glob_ns": {
                        "type": "sqlite",
                        "path_pattern": "data/*.sqlite",
                    }
                }
            }
        }
        result = migrate_namespace_to_services(config)
        databases = result["agent"]["services"]["databases"]
        assert "glob_ns" in databases
        assert databases["glob_ns"]["path_pattern"] == "data/*.sqlite"
        assert databases["glob_ns"]["type"] == "sqlite"

    def test_already_migrated_config_is_no_op(self):
        """Config with existing 'services' key is returned unchanged."""
        config = {
            "agent": {
                "services": {
                    "databases": {"existing_db": {"type": "duckdb"}},
                    "semantic_layer": {},
                    "bi_tools": {},
                    "schedulers": {},
                }
            }
        }
        result = migrate_namespace_to_services(config)
        # Namespace section should not have been touched; services intact
        assert "namespace" not in result["agent"]
        assert result["agent"]["services"]["databases"]["existing_db"]["type"] == "duckdb"

    def test_no_namespace_section_is_no_op(self):
        """Config with no 'namespace' section is returned unchanged (no 'services' added)."""
        config = {
            "agent": {
                "target": "openai",
                "models": {"openai": {"type": "openai", "model": "gpt-4o"}},
            }
        }
        result = migrate_namespace_to_services(config)
        # No services key added when there was no namespace to migrate
        assert "services" not in result["agent"]
        assert result["agent"]["target"] == "openai"

    def test_original_config_not_mutated(self):
        """migrate_namespace_to_services returns a deep copy; original dict unchanged."""
        config = {
            "agent": {
                "namespace": {
                    "my_db": {"type": "duckdb", "uri": "my.duckdb"},
                }
            }
        }
        original_copy = {
            "agent": {
                "namespace": {
                    "my_db": {"type": "duckdb", "uri": "my.duckdb"},
                }
            }
        }
        migrate_namespace_to_services(config)
        assert config == original_copy

    def test_multiple_namespaces_produce_multiple_databases(self):
        """Multiple namespaces without dbs list each become their own database entry."""
        config = {
            "agent": {
                "namespace": {
                    "snowflake_ns": {"type": "snowflake", "account": "myaccount"},
                    "duckdb_ns": {"type": "duckdb", "uri": "local.duckdb"},
                }
            }
        }
        result = migrate_namespace_to_services(config)
        databases = result["agent"]["services"]["databases"]
        assert "snowflake_ns" in databases
        assert "duckdb_ns" in databases
        # Multiple entries: no automatic default marking
        assert databases["snowflake_ns"].get("default") is None or databases["snowflake_ns"].get("default") is not True


class TestMigrateFile:
    """Tests for migrate_file()."""

    def test_dry_run_does_not_write_file(self, tmp_path, capsys):
        """dry_run=True prints migrated config but does not write or rename files."""
        config_data = {
            "agent": {
                "namespace": {
                    "test_db": {"type": "sqlite", "uri": "test.sqlite"},
                }
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        result = migrate_file(str(config_file), dry_run=True)

        assert result is True
        # Original file should still exist (not renamed to .bak)
        assert config_file.exists()
        backup = tmp_path / "agent.yml.bak"
        assert not backup.exists()

    def test_migrate_file_writes_new_format(self, tmp_path):
        """migrate_file writes the new services format and creates a backup."""
        config_data = {
            "agent": {
                "namespace": {
                    "db1": {"type": "duckdb", "uri": "db1.duckdb"},
                }
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        result = migrate_file(str(config_file), dry_run=False)

        assert result is True
        # Backup created
        backup = tmp_path / "agent.yml.bak"
        assert backup.exists()
        # New file has services format
        with open(config_file, encoding="utf-8") as f:
            new_config = yaml.safe_load(f)
        assert "services" in new_config["agent"]
        assert "db1" in new_config["agent"]["services"]["databases"]

    def test_migrate_file_nonexistent_path_returns_false(self, tmp_path):
        """migrate_file returns False when config file does not exist."""
        result = migrate_file(str(tmp_path / "nonexistent.yml"))
        assert result is False

    def test_migrate_file_already_migrated_returns_false(self, tmp_path):
        """migrate_file returns False when config already uses services format."""
        config_data = {
            "agent": {
                "services": {
                    "databases": {},
                    "semantic_layer": {},
                    "bi_tools": {},
                    "schedulers": {},
                }
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        result = migrate_file(str(config_file))
        assert result is False
