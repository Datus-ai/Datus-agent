# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.semantic_model.semantic_model_init."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.semantic_model.semantic_model_init import (
    _infer_semantic_yaml_authoring_format,
    _load_success_story_profile_entries,
    _metricflow_data_source_table,
    _semantic_yaml_profile_tables,
    init_semantic_yaml_semantic_model,
    process_semantic_yaml_file,
    refresh_semantic_yaml_profile_descriptions,
    refresh_success_story_semantic_model_profile,
)

# ---------------------------------------------------------------------------
# init_semantic_yaml_semantic_model
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSemanticYamlSemanticModel:
    """Tests for init_semantic_yaml_semantic_model function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when YAML file does not exist."""
        nonexistent = str(tmp_path / "missing.yaml")
        mock_config = MagicMock()

        success, error = init_semantic_yaml_semantic_model(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_existing_file_delegates_to_process(self, tmp_path):
        """When file exists, calls process_semantic_yaml_file with include_metrics=False."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: test\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_process:
            success, error = init_semantic_yaml_semantic_model(str(yaml_file), mock_config)

        assert success is True
        assert error == ""
        mock_process.assert_called_once_with(str(yaml_file), mock_config, include_metrics=False)


# ---------------------------------------------------------------------------
# process_semantic_yaml_file
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProcessSemanticYamlFile:
    """Tests for process_semantic_yaml_file function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when file does not exist."""
        nonexistent = str(tmp_path / "missing.yaml")
        mock_config = MagicMock()

        success, error = process_semantic_yaml_file(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_sync_success(self, tmp_path):
        """Returns (True, '') when sync succeeds."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "synced"},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is True
        assert error == ""

    def test_sync_failure(self, tmp_path):
        """Returns (False, error) when sync reports failure."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": False, "error": "validation failed"},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "validation failed" in error

    def test_sync_exception(self, tmp_path):
        """Returns (False, error) when sync raises an exception."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            side_effect=RuntimeError("connection error"),
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "connection error" in error

    def test_default_includes_both(self, tmp_path):
        """By default, include_semantic_objects and include_metrics are both True."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=True,
        )

    def test_exclude_metrics(self, tmp_path):
        """include_metrics=False is forwarded to _sync_semantic_to_db."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config, include_metrics=False)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=False,
        )

    def test_exclude_semantic_objects(self, tmp_path):
        """include_semantic_objects=False is forwarded to _sync_semantic_to_db."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config, include_semantic_objects=False)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=False,
            include_metrics=True,
        )

    def test_sync_unknown_error(self, tmp_path):
        """When sync returns failure with no error key, uses 'Unknown error'."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": False},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "Unknown error" in error


# ---------------------------------------------------------------------------
# refresh_semantic_yaml_profile_descriptions
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestRefreshSemanticYamlProfileDescriptions:
    def test_metricflow_refresh_writes_description_and_syncs(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  description: Orders table.
  sql_table: orders
  dimensions:
    - name: status
      expr: status
      type: CATEGORICAL
      description: Order status.
""",
            encoding="utf-8",
        )
        evidence = {
            "tables": {
                "orders": {
                    "query_count": 1,
                    "data_distribution_profile": {
                        "columns": {
                            "status": {
                                "kind": "categorical",
                                "stats": {"distinct_count": 2},
                                "top_values": [{"value": "paid"}],
                            }
                        }
                    },
                }
            }
        }
        mock_config = MagicMock()
        mock_config.path_manager = None

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_sync:
            success, error, changed = refresh_semantic_yaml_profile_descriptions(
                str(yaml_file),
                evidence,
                agent_config=mock_config,
                sync_to_storage=True,
            )

        assert success is True
        assert error == ""
        assert changed == 2
        assert "Observed profile:" in yaml_file.read_text(encoding="utf-8")
        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=True,
        )

    def test_sync_requires_agent_config(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("data_source:\n  name: orders\n", encoding="utf-8")

        success, error, changed = refresh_semantic_yaml_profile_descriptions(
            str(yaml_file),
            {"tables": {}},
            sync_to_storage=True,
        )

        assert success is False
        assert "agent_config is required" in error
        assert changed == 0

    def test_osi_refresh_syncs_via_generation_tools(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
semantic_model:
  - name: shop
    datasets:
      - name: orders
        description: Orders dataset.
        source: orders
        dimensions:
          - name: status
            description: Order status.
""",
            encoding="utf-8",
        )
        evidence = {
            "tables": {
                "orders": {
                    "query_count": 1,
                    "data_distribution_profile": {
                        "columns": {
                            "status": {
                                "kind": "categorical",
                                "stats": {"distinct_count": 2},
                                "top_values": [{"value": "paid"}],
                            }
                        }
                    },
                }
            }
        }
        mock_config = MagicMock()
        mock_config.path_manager = None

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as mock_tools_cls:
            mock_tools_cls.return_value.sync_osi_semantic_to_db.return_value = {"success": True}
            success, error, changed = refresh_semantic_yaml_profile_descriptions(
                str(yaml_file),
                evidence,
                authoring_format="osi",
                agent_config=mock_config,
                sync_to_storage=True,
            )

        assert success is True
        assert error == ""
        assert changed == 2
        mock_tools_cls.assert_called_once_with(agent_config=mock_config, authoring_format="osi")
        mock_tools_cls.return_value.sync_osi_semantic_to_db.assert_called_once_with(str(yaml_file))

    def test_osi_refresh_returns_sync_exception(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
semantic_model:
  - name: shop
    datasets:
      - name: orders
        description: Orders dataset.
        source: orders
        dimensions:
          - name: status
            description: Order status.
""",
            encoding="utf-8",
        )
        evidence = {
            "tables": {
                "orders": {
                    "query_count": 1,
                    "data_distribution_profile": {
                        "columns": {
                            "status": {
                                "kind": "categorical",
                                "stats": {"distinct_count": 2},
                                "top_values": [{"value": "paid"}],
                            }
                        }
                    },
                }
            }
        }
        mock_config = MagicMock()
        mock_config.path_manager = None

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as mock_tools_cls:
            mock_tools_cls.return_value.sync_osi_semantic_to_db.side_effect = RuntimeError("storage offline")
            success, error, changed = refresh_semantic_yaml_profile_descriptions(
                str(yaml_file),
                evidence,
                authoring_format="osi",
                agent_config=mock_config,
                sync_to_storage=True,
            )

        assert success is False
        assert "Failed to sync OSI semantic YAML file" in error
        assert "storage offline" in error
        assert changed == 2

    def test_unchanged_metricflow_yaml_still_retries_storage_sync(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  sql_table: orders
""",
            encoding="utf-8",
        )
        mock_config = MagicMock()
        mock_config.path_manager = None

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_sync:
            success, error, changed = refresh_semantic_yaml_profile_descriptions(
                str(yaml_file),
                {"tables": {}},
                agent_config=mock_config,
                sync_to_storage=True,
            )

        assert success is True
        assert error == ""
        assert changed == 0
        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=True,
        )

    def test_unchanged_metricflow_yaml_returns_sync_failure(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  sql_table: orders
""",
            encoding="utf-8",
        )
        mock_config = MagicMock()
        mock_config.path_manager = None

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(False, "storage unavailable"),
        ):
            success, error, changed = refresh_semantic_yaml_profile_descriptions(
                str(yaml_file),
                {"tables": {}},
                agent_config=mock_config,
                sync_to_storage=True,
            )

        assert success is False
        assert error == "storage unavailable"
        assert changed == 0

    def test_refresh_rejects_path_outside_semantic_sandbox(self, tmp_path):
        subject_dir = tmp_path / "subject"
        (subject_dir / "semantic_models").mkdir(parents=True)
        outside = tmp_path / "outside.yml"
        outside.write_text("data_source:\n  name: orders\n", encoding="utf-8")
        mock_config = MagicMock()
        mock_config.path_manager.subject_dir = subject_dir

        success, error, changed = refresh_semantic_yaml_profile_descriptions(
            str(outside),
            {"tables": {}},
            agent_config=mock_config,
        )

        assert success is False
        assert "rejected by sandbox" in error
        assert changed == 0

    def test_atomic_write_failure_preserves_existing_yaml(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        original = """
data_source:
  name: orders
  description: Orders table.
  sql_table: orders
  dimensions:
    - name: status
      expr: status
      type: CATEGORICAL
      description: Order status.
"""
        yaml_file.write_text(original, encoding="utf-8")
        evidence = {
            "tables": {
                "orders": {
                    "query_count": 1,
                    "data_distribution_profile": {
                        "columns": {
                            "status": {
                                "kind": "categorical",
                                "stats": {"distinct_count": 2},
                                "top_values": [{"value": "paid"}],
                            }
                        }
                    },
                }
            }
        }

        with patch("datus.storage.semantic_model.semantic_model_init.yaml.safe_dump_all") as mock_dump:
            mock_dump.side_effect = RuntimeError("dump failed")
            success, error, changed = refresh_semantic_yaml_profile_descriptions(str(yaml_file), evidence)

        assert success is False
        assert "dump failed" in error
        assert changed == 0
        assert yaml_file.read_text(encoding="utf-8") == original


@pytest.mark.ci
class TestRefreshSuccessStorySemanticModelProfile:
    def test_profiles_existing_metricflow_yaml_and_refreshes_descriptions(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text(
            """
data_source:
  name: orders
  sql_table: marts.orders
  description: Orders table.
""",
            encoding="utf-8",
        )
        success_story = tmp_path / "stories.csv"
        success_story.write_text(
            "source_context_id,question,sql\n"
            "q1,paid orders,\"SELECT status FROM marts.orders WHERE status = 'paid'\"\n",
            encoding="utf-8",
        )
        mock_config = MagicMock()
        mock_config.path_manager = None
        mock_config.runtime_db_context.return_value = {}
        current_db_config = MagicMock()
        current_db_config.catalog = ""
        current_db_config.database = "analytics"
        current_db_config.schema = ""
        mock_config.current_db_config.return_value = current_db_config

        from datus.tools.func_tool.base import FuncToolResult

        mock_discovery = MagicMock()
        mock_discovery.profile_semantic_model_evidence.return_value = FuncToolResult(
            result={"tables": {"orders": {"data_distribution_profile": {"columns": {}}}}}
        )

        events = []
        with (
            patch("datus.tools.func_tool.database.DBFuncTool") as mock_db_tool,
            patch(
                "datus.tools.func_tool.semantic_discovery_tools.SemanticDiscoveryTools",
                return_value=mock_discovery,
            ),
            patch(
                "datus.storage.semantic_model.semantic_model_init.refresh_semantic_yaml_profile_descriptions",
                return_value=(True, "", 2),
            ) as mock_refresh,
        ):
            success, error, changed = refresh_success_story_semantic_model_profile(
                mock_config,
                str(yaml_file),
                str(success_story),
                emit=events.append,
            )

        assert success is True
        assert error == ""
        assert changed == 2
        assert [event.stage for event in events] == ["task_started", "task_completed"]
        assert events[-1].payload == {"semantic_yaml": str(yaml_file), "changed_description_count": 2}
        mock_db_tool.assert_called_once_with(
            agent_config=mock_config,
            sub_agent_name="gen_semantic_model",
            read_only=True,
        )
        profile_kwargs = mock_discovery.profile_semantic_model_evidence.call_args.kwargs
        assert profile_kwargs["tables"] == ["marts.orders"]
        assert profile_kwargs["database"] == "analytics"
        assert profile_kwargs["profile_mode"] == "deep"
        assert "paid orders" in profile_kwargs["sql_entries_json"]
        mock_refresh.assert_called_once()
        assert mock_refresh.call_args.args[0] == str(yaml_file)
        assert mock_refresh.call_args.kwargs["authoring_format"] == "metricflow"
        assert mock_refresh.call_args.kwargs["sync_to_storage"] is True

    def test_refresh_profile_requires_semantic_yaml(self, tmp_path):
        success_story = tmp_path / "stories.csv"
        success_story.write_text("sql\nSELECT 1\n", encoding="utf-8")

        success, error, changed = refresh_success_story_semantic_model_profile(
            MagicMock(),
            "",
            str(success_story),
        )

        assert success is False
        assert "--semantic_yaml is required" in error
        assert changed == 0

    def test_refresh_profile_requires_success_story(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("data_source:\n  name: orders\n", encoding="utf-8")

        success, error, changed = refresh_success_story_semantic_model_profile(
            MagicMock(),
            str(yaml_file),
            "",
        )

        assert success is False
        assert "--success_story is required" in error
        assert changed == 0

    def test_refresh_profile_reports_missing_yaml_file(self, tmp_path):
        success_story = tmp_path / "stories.csv"
        success_story.write_text("sql\nSELECT 1\n", encoding="utf-8")

        success, error, changed = refresh_success_story_semantic_model_profile(
            MagicMock(path_manager=None),
            str(tmp_path / "missing.yml"),
            str(success_story),
        )

        assert success is False
        assert "Semantic YAML file not found" in error
        assert changed == 0

    def test_success_story_profile_entries_validate_required_sql(self, tmp_path):
        missing = tmp_path / "missing.csv"
        entries, error = _load_success_story_profile_entries(str(missing))
        assert entries == []
        assert "not found" in error

        no_sql = tmp_path / "no_sql.csv"
        no_sql.write_text("question\nhow many orders?\n", encoding="utf-8")
        entries, error = _load_success_story_profile_entries(str(no_sql))
        assert entries == []
        assert "missing required column: sql" in error

        blank_sql = tmp_path / "blank_sql.csv"
        blank_sql.write_text("source_context_id,name,question,sql\n,,ignored,\n", encoding="utf-8")
        entries, error = _load_success_story_profile_entries(str(blank_sql))
        assert entries == []
        assert "contains no SQL rows" in error

        valid = tmp_path / "valid.csv"
        valid.write_text("source_context_id,name,question,sql\n,story_name,,SELECT 1\n", encoding="utf-8")
        entries, error = _load_success_story_profile_entries(str(valid))
        assert error == ""
        assert entries == [{"name": "story_name", "question": "", "sql": "SELECT 1"}]

    def test_semantic_yaml_profile_table_helpers_cover_format_variants(self):
        docs = [
            None,
            {"data_source": "not-a-dict"},
            {
                "semantic_model": [
                    {
                        "datasets": [
                            {"source": {"table": "mart.orders"}},
                            {"source": "mart.customers"},
                            {"table": "fallback_table"},
                            {"name": "fallback_name"},
                        ]
                    },
                    "ignored",
                ],
                "datasets": [{"source": {"table": "mart.orders"}}],
            },
        ]

        assert _infer_semantic_yaml_authoring_format([], " OSI ") == "osi"
        assert _metricflow_data_source_table("not-a-dict") == ""
        assert _semantic_yaml_profile_tables(docs, "osi") == [
            "mart.orders",
            "mart.customers",
            "fallback_table",
            "fallback_name",
        ]

    def test_refresh_profile_rejects_unsupported_authoring_format(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("data_source:\n  name: orders\n", encoding="utf-8")
        success_story = tmp_path / "stories.csv"
        success_story.write_text("sql\nSELECT * FROM orders\n", encoding="utf-8")

        success, error, changed = refresh_success_story_semantic_model_profile(
            MagicMock(path_manager=None),
            str(yaml_file),
            str(success_story),
            authoring_format="unsupported",
        )

        assert success is False
        assert "Unsupported semantic YAML authoring format" in error
        assert changed == 0

    def test_refresh_profile_rejects_yaml_without_table_targets(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("data_source:\n  description: no table target\n", encoding="utf-8")
        success_story = tmp_path / "stories.csv"
        success_story.write_text("sql\nSELECT * FROM orders\n", encoding="utf-8")

        success, error, changed = refresh_success_story_semantic_model_profile(
            MagicMock(path_manager=None),
            str(yaml_file),
            str(success_story),
        )

        assert success is False
        assert "No table targets found" in error
        assert changed == 0

    def test_refresh_profile_reports_profiler_exception_and_emits_failure(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("data_source:\n  name: orders\n", encoding="utf-8")
        success_story = tmp_path / "stories.csv"
        success_story.write_text("sql\nSELECT * FROM orders\n", encoding="utf-8")
        current_db_config = MagicMock(catalog="", database="analytics", schema="")
        mock_config = MagicMock(path_manager=None)
        mock_config.current_db_config.return_value = current_db_config
        mock_config.runtime_db_context.return_value = {}
        events = []

        with patch("datus.tools.func_tool.database.DBFuncTool", side_effect=RuntimeError("db offline")):
            success, error, changed = refresh_success_story_semantic_model_profile(
                mock_config,
                str(yaml_file),
                str(success_story),
                emit=events.append,
            )

        assert success is False
        assert "Failed to profile semantic YAML" in error
        assert "db offline" in error
        assert changed == 0
        assert [event.stage for event in events] == ["task_started", "task_failed"]

    def test_refresh_profile_reports_profiler_failure_and_emits_failure(self, tmp_path):
        yaml_file = tmp_path / "orders.yml"
        yaml_file.write_text("data_source:\n  name: orders\n", encoding="utf-8")
        success_story = tmp_path / "stories.csv"
        success_story.write_text("sql\nSELECT * FROM orders\n", encoding="utf-8")
        current_db_config = MagicMock(catalog="", database="analytics", schema="")
        mock_config = MagicMock(path_manager=None)
        mock_config.current_db_config.return_value = current_db_config
        mock_config.runtime_db_context.return_value = {}
        mock_discovery = MagicMock()
        mock_discovery.profile_semantic_model_evidence.return_value = MagicMock(
            success=False,
            error="profile failed",
        )
        events = []

        with (
            patch("datus.tools.func_tool.database.DBFuncTool"),
            patch(
                "datus.tools.func_tool.semantic_discovery_tools.SemanticDiscoveryTools",
                return_value=mock_discovery,
            ),
        ):
            success, error, changed = refresh_success_story_semantic_model_profile(
                mock_config,
                str(yaml_file),
                str(success_story),
                emit=events.append,
            )

        assert success is False
        assert error == "profile failed"
        assert changed == 0
        assert [event.stage for event in events] == ["task_started", "task_failed"]


# ---------------------------------------------------------------------------
# init_success_story_semantic_model_async - importability and coroutine check
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStorySemanticModelAsync:
    """Tests for init_success_story_semantic_model_async importability and interface."""

    def test_async_function_is_importable(self):
        """init_success_story_semantic_model_async can be imported from the module."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        assert callable(init_success_story_semantic_model_async)

    def test_async_function_is_coroutine(self):
        """init_success_story_semantic_model_async is a coroutine function (async def)."""
        import inspect

        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        assert inspect.iscoroutinefunction(init_success_story_semantic_model_async)

    @pytest.mark.asyncio
    async def test_async_returns_false_for_missing_csv(self, tmp_path):
        """Awaiting init_success_story_semantic_model_async with a missing CSV returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        missing = str(tmp_path / "no_such_file.csv")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(mock_config, missing)

        assert success is False
        assert error == f"Success story CSV file not found: {missing}"

    @pytest.mark.asyncio
    async def test_async_returns_false_for_empty_csv(self, tmp_path):
        """Awaiting with an empty CSV (no rows) returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("sql,question\n")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert error != ""

    @pytest.mark.asyncio
    async def test_async_returns_false_for_missing_columns(self, tmp_path):
        """Awaiting with a CSV missing required columns returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "bad_cols.csv"
        csv_path.write_text("question\nWhat is revenue?\n")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert "missing required columns: ['sql']" in error

    @pytest.mark.asyncio
    async def test_async_rejects_unknown_build_mode(self, tmp_path):
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(
            mock_config,
            str(csv_path),
            build_mode="full",
        )

        assert success is False
        assert "Unsupported semantic model build_mode" in error

    @pytest.mark.asyncio
    async def test_check_mode_storage_init_failure_returns_error(self, tmp_path):
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.store.SemanticModelRAG",
            side_effect=RuntimeError("storage unavailable"),
        ):
            success, error = await init_success_story_semantic_model_async(
                mock_config,
                str(csv_path),
                build_mode="check",
            )

        assert success is False
        assert "Failed to initialize semantic model storage" in error
        assert "storage unavailable" in error

    @pytest.mark.asyncio
    async def test_overwrite_mode_truncate_failure_returns_error(self, tmp_path):
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")
        mock_config = MagicMock()
        mock_semantic_rag = MagicMock()
        mock_semantic_rag.datasource_id = "test_ds"
        mock_semantic_rag.truncate.side_effect = RuntimeError("truncate failed")
        mock_table_profile_rag = MagicMock()

        with (
            patch("datus.storage.semantic_model.store.SemanticModelRAG", return_value=mock_semantic_rag),
            patch(
                "datus.storage.table_semantic_profile.store.TableSemanticProfileRAG",
                return_value=mock_table_profile_rag,
            ),
        ):
            success, error = await init_success_story_semantic_model_async(
                mock_config,
                str(csv_path),
                build_mode="overwrite",
            )

        assert success is False
        assert "Failed to wipe semantic model storage" in error
        assert "truncate failed" in error
        mock_semantic_rag.truncate.assert_called_once()
        mock_table_profile_rag.truncate.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_mode_reports_existing_row_counts(self, tmp_path):
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")
        mock_config = MagicMock()
        mock_semantic_rag = MagicMock()
        mock_semantic_rag.get_size.return_value = 2
        mock_table_profile_rag = MagicMock()
        mock_table_profile_rag.get_size.return_value = 3

        with (
            patch("datus.storage.semantic_model.store.SemanticModelRAG", return_value=mock_semantic_rag),
            patch(
                "datus.storage.table_semantic_profile.store.TableSemanticProfileRAG",
                return_value=mock_table_profile_rag,
            ),
        ):
            success, error = await init_success_story_semantic_model_async(
                mock_config,
                str(csv_path),
                build_mode="check",
            )

        assert success is True
        assert error == ""
        mock_semantic_rag.get_size.assert_called_once()
        mock_table_profile_rag.get_size.assert_called_once()


# ---------------------------------------------------------------------------
# init_success_story_semantic_model sync wrapper - new signature
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStorySemanticModelSync:
    """Tests for init_success_story_semantic_model sync wrapper with decoupled signature."""

    def test_sync_function_is_importable(self):
        """init_success_story_semantic_model can be imported."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        assert callable(init_success_story_semantic_model)

    def test_sync_function_is_not_coroutine(self):
        """init_success_story_semantic_model is a plain sync function, not a coroutine."""
        import inspect

        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        assert not inspect.iscoroutinefunction(init_success_story_semantic_model)

    def test_sync_returns_tuple_for_missing_csv(self, tmp_path):
        """Sync wrapper returns (bool, str) tuple for a missing CSV path."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        missing = str(tmp_path / "no_file.csv")
        mock_config = MagicMock()

        result = init_success_story_semantic_model(mock_config, missing)

        assert isinstance(result, tuple)
        assert len(result) == 2
        success, error = result
        assert success is False
        assert isinstance(error, str)

    def test_sync_accepts_agent_config_and_success_story_args(self, tmp_path):
        """Sync wrapper accepts (agent_config, success_story) without argparse.Namespace."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\n")
        mock_config = MagicMock()

        # Should call without raising TypeError about unexpected args
        result = init_success_story_semantic_model(mock_config, str(csv_path))
        assert isinstance(result, tuple)

    def test_sync_accepts_optional_emit_kwarg(self, tmp_path):
        """Sync wrapper accepts optional emit keyword argument."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\n")
        mock_config = MagicMock()
        emit_calls = []

        result = init_success_story_semantic_model(mock_config, str(csv_path), emit=emit_calls.append)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# init_success_story_semantic_model_async - LLM execution path (lines 95-157)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStorySemanticModelAsyncLLMPath:
    """Tests for the async LLM execution path inside init_success_story_semantic_model_async."""

    @pytest.fixture(autouse=True)
    def _stub_semantic_rag(self, monkeypatch):
        """Stub the build_mode='overwrite' truncate path so it never reaches the Lance backend.

        These tests exercise the LLM execution flow only; the truncate behavior is
        covered separately by ``TestInitSuccessStorySemanticModelAsyncOverwriteTruncate``.
        """
        monkeypatch.setattr(
            "datus.storage.semantic_model.store.SemanticModelRAG",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "datus.storage.table_semantic_profile.store.TableSemanticProfileRAG",
            MagicMock(return_value=MagicMock()),
        )

    @pytest.mark.asyncio
    async def test_success_path_with_semantic_models_list(self, tmp_path, monkeypatch):
        """Success path: agentic node yields action with semantic_models list → returns (True, '')."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,What is one?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = "cat"
        mock_db_config.database = "db"
        mock_db_config.schema = "public"
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"semantic_models": ["model1.yaml", "model2.yaml"]},
                    messages="Generated models",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_success_path_emits_events(self, tmp_path, monkeypatch):
        """Success path: emit callback is called for ITEM_PROCESSING and TASK_COMPLETED stages."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchStage
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,What is one?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"semantic_models": ["model.yaml"]},
                    messages="Generated model",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        emitted_stages = []

        def capture_emit(event):
            emitted_stages.append(event.stage)

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path), emit=capture_emit)

        assert success is True
        assert BatchStage.TASK_STARTED in emitted_stages
        assert BatchStage.TASK_COMPLETED in emitted_stages

    @pytest.mark.asyncio
    async def test_success_path_single_model_string(self, tmp_path, monkeypatch):
        """Success path: semantic_models as a single string (not list) is also collected."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                # single string instead of list
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"semantic_models": "single_model.yaml"},
                    messages="Generated",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_recoverable_tool_failure_does_not_abort_success(self, tmp_path, monkeypatch):
        """A failed intermediate tool action should not abort a later successful semantic response."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                yield SimpleNamespace(
                    status=ActionStatus.FAILED,
                    action_type="validate_semantic",
                    output={"raw_output": {"success": 0, "error": "invalid yaml"}},
                    messages="validation failed",
                )
                yield SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"semantic_models": ["model.yaml"]},
                    messages="Generated",
                )

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_final_error_action_returns_failure(self, tmp_path, monkeypatch):
        """A terminal error action still fails the batch."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                yield SimpleNamespace(
                    status=ActionStatus.FAILED,
                    action_type="error",
                    output={"error": "Semantic model generation did not publish to Knowledge Base"},
                    messages="Semantic model generation did not publish to Knowledge Base",
                )

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert "did not publish" in error

    @pytest.mark.asyncio
    async def test_failed_final_response_action_returns_failure(self, tmp_path, monkeypatch):
        """A failed final response action still fails the batch."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                yield SimpleNamespace(
                    status=ActionStatus.FAILED,
                    action_type="gen_semantic_model_response",
                    output={"error": "failed final response"},
                    messages="failed final response",
                )

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert "failed final response" in error

    @pytest.mark.asyncio
    async def test_empty_result_path_returns_false(self, tmp_path, monkeypatch):
        """Empty result path: no generated files → returns (False, error)."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                # Yields an action with SUCCESS but no semantic_models key
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"other_key": "value"},
                    messages="Nothing useful",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert error != ""

    @pytest.mark.asyncio
    async def test_empty_result_emits_task_failed(self, tmp_path, monkeypatch):
        """Empty result path: emit callback receives TASK_FAILED event."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchStage
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={},
                    messages="",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        emitted_stages = []

        def capture_emit(event):
            emitted_stages.append(event.stage)

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path), emit=capture_emit)

        assert success is False
        assert BatchStage.TASK_FAILED in emitted_stages

    @pytest.mark.asyncio
    async def test_exception_path_returns_false(self, tmp_path, monkeypatch):
        """Exception path: execute_stream raises → returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                raise RuntimeError("LLM backend error")
                yield  # make it an async generator

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert "LLM backend error" in error

    @pytest.mark.asyncio
    async def test_exception_emits_task_failed(self, tmp_path, monkeypatch):
        """Exception path: emit receives TASK_FAILED when execute_stream raises."""
        from datus.schemas.batch_events import BatchStage
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                raise ValueError("unexpected error")
                yield  # async generator marker

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        emitted_stages = []

        def capture_emit(event):
            emitted_stages.append(event.stage)

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path), emit=capture_emit)

        assert success is False
        assert BatchStage.TASK_FAILED in emitted_stages

    @pytest.mark.asyncio
    async def test_action_with_none_output_skipped(self, tmp_path, monkeypatch):
        """Action with output=None should not cause error and counts as empty result."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output=None,
                    messages="",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        # No files generated → failure
        assert success is False


# ---------------------------------------------------------------------------
# init_success_story_semantic_model_async — overwrite truncate semantics
# ---------------------------------------------------------------------------


class TestInitSuccessStorySemanticModelAsyncOverwriteTruncate:
    """Verify build_mode='overwrite' wipes the semantic model store before LLM regeneration."""

    @pytest.mark.asyncio
    async def test_overwrite_calls_truncate_on_semantic_model_rag(self, tmp_path, monkeypatch):
        """build_mode='overwrite' must call SemanticModelRAG(...).truncate() exactly once."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_config.project_name = "unit-test-project"
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        fake_rag_instance = MagicMock()
        rag_factory = MagicMock(return_value=fake_rag_instance)
        fake_profile_rag_instance = MagicMock()
        profile_rag_factory = MagicMock(return_value=fake_profile_rag_instance)
        monkeypatch.setattr("datus.storage.semantic_model.store.SemanticModelRAG", rag_factory)
        monkeypatch.setattr(
            "datus.storage.table_semantic_profile.store.TableSemanticProfileRAG",
            profile_rag_factory,
        )

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                yield SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"semantic_models": ["m.yaml"]},
                    messages="ok",
                )

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, _error = await init_success_story_semantic_model_async(
            mock_config, str(csv_path), build_mode="overwrite"
        )

        assert success is True
        rag_factory.assert_called_once_with(mock_config)
        fake_rag_instance.truncate.assert_called_once_with()
        profile_rag_factory.assert_called_once_with(mock_config)
        fake_profile_rag_instance.truncate.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_incremental_does_not_call_truncate(self, tmp_path, monkeypatch):
        """build_mode='incremental' must NOT call truncate (no SemanticModelRAG instantiation)."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_config.project_name = "unit-test-project"
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        fake_rag_instance = MagicMock()
        rag_factory = MagicMock(return_value=fake_rag_instance)
        profile_rag_factory = MagicMock()
        monkeypatch.setattr("datus.storage.semantic_model.store.SemanticModelRAG", rag_factory)
        monkeypatch.setattr(
            "datus.storage.table_semantic_profile.store.TableSemanticProfileRAG",
            profile_rag_factory,
        )

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                yield SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    action_type="gen_semantic_model_response",
                    output={"semantic_models": ["m.yaml"]},
                    messages="ok",
                )

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, _error = await init_success_story_semantic_model_async(
            mock_config, str(csv_path), build_mode="incremental"
        )

        assert success is True
        fake_rag_instance.truncate.assert_not_called()
        profile_rag_factory.assert_not_called()
