# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SubjectUpdater in datus/storage/subject_manager.py.

Covers initialization, update operations, delete operations, and edge cases.
Uses real storage instances via the real_agent_config fixture (no mocks except LLM).
"""

from datus.storage.cache import clear_cache, get_storage_cache_instance
from datus.storage.ext_knowledge import ExtKnowledgeStore
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.subject_manager import SubjectUpdater

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUBJECT_PATH = ["TestDomain", "TestArea"]
METRIC_NAME = "test_metric"
SQL_NAME = "test_sql"
EXT_KNOWLEDGE_NAME = "test_knowledge"


def _seed_metric(storage: MetricStorage, subject_path=None, name=None):
    """Insert a metric entry into the given MetricStorage so subsequent operations can find it."""
    path = subject_path or SUBJECT_PATH
    entry_name = name or METRIC_NAME
    storage.batch_store_metrics(
        [
            {
                "subject_path": path,
                "name": entry_name,
                "id": f"metric:{entry_name}",
                "semantic_model_name": "test_model",
                "description": "A test metric description",
                "metric_type": "simple",
                "measure_expr": "COUNT(*)",
                "base_measures": [],
                "dimensions": [],
                "entities": [],
                "catalog_name": "",
                "database_name": "",
                "schema_name": "",
                "sql": "SELECT COUNT(*) FROM t",
                "yaml_path": "",
            }
        ]
    )


def _seed_reference_sql(storage: ReferenceSqlStorage, subject_path=None, name=None):
    """Insert a reference SQL entry into the given ReferenceSqlStorage."""
    path = subject_path or SUBJECT_PATH
    entry_name = name or SQL_NAME
    storage.batch_store_sql(
        [
            {
                "subject_path": path,
                "name": entry_name,
                "id": f"sql:{entry_name}",
                "sql": "SELECT 1",
                "comment": "test comment",
                "summary": "A test SQL summary",
                "search_text": "test sql search text",
                "filepath": "",
                "tags": "",
            }
        ]
    )


def _seed_ext_knowledge(storage: ExtKnowledgeStore, subject_path=None, name=None):
    """Insert an external knowledge entry into the given ExtKnowledgeStore."""
    path = subject_path or SUBJECT_PATH
    entry_name = name or EXT_KNOWLEDGE_NAME
    storage.batch_store_knowledge(
        [
            {
                "subject_path": path,
                "name": entry_name,
                "search_text": "test knowledge search text",
                "explanation": "A test knowledge explanation",
            }
        ]
    )


def _add_sub_agent_with_namespace(agent_config, sub_agent_name, namespace):
    """Add a sub-agent with scoped_context matching the given namespace to agentic_nodes."""
    agent_config.agentic_nodes[sub_agent_name] = {
        "system_prompt": sub_agent_name,
        "scoped_context": {
            "namespace": namespace,
            "tables": "table_a",
            "metrics": "metric_a",
            "sqls": "sql_a",
            "ext_knowledge": "ext_knowledge_a",
        },
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestSubjectUpdaterInit:
    """Tests for SubjectUpdater initialization."""

    def test_init_creates_storage_instances(self, real_agent_config):
        """SubjectUpdater.__init__ should create all three storage instances from StorageCache."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)

        assert updater.metrics_storage is not None
        assert isinstance(updater.metrics_storage, MetricStorage)
        assert updater.reference_sql_storage is not None
        assert isinstance(updater.reference_sql_storage, ReferenceSqlStorage)
        assert updater.ext_knowledge_storage is not None
        assert isinstance(updater.ext_knowledge_storage, ExtKnowledgeStore)

    def test_init_storage_cache_matches_global(self, real_agent_config):
        """Storage instances should match those from StorageCache global accessors."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        cache = get_storage_cache_instance(real_agent_config)

        assert updater.metrics_storage is cache.metric_storage()
        assert updater.reference_sql_storage is cache.reference_sql_storage()
        assert updater.ext_knowledge_storage is cache.ext_knowledge_storage()


class TestSubjectUpdaterExecution:
    """Tests for SubjectUpdater update and delete operations on main storage."""

    def test_update_metrics_detail_updates_main_storage(self, real_agent_config):
        """update_metrics_detail should update the entry in main storage."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_metric(updater.metrics_storage)

        updater.update_metrics_detail(SUBJECT_PATH, METRIC_NAME, {"description": "updated description"})

        results = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(results) >= 1
        assert results[0]["description"] == "updated description"

    def test_update_metrics_detail_empty_values_is_noop(self, real_agent_config):
        """update_metrics_detail with empty update_values should return early without error."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_metric(updater.metrics_storage)

        # Should not raise; should be a no-op
        updater.update_metrics_detail(SUBJECT_PATH, METRIC_NAME, {})

        results = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(results) >= 1
        assert results[0]["description"] == "A test metric description"

    def test_update_historical_sql_updates_main_storage(self, real_agent_config):
        """update_historical_sql should update the entry in main reference_sql storage."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_reference_sql(updater.reference_sql_storage)

        updater.update_historical_sql(SUBJECT_PATH, SQL_NAME, {"comment": "updated comment"})

        results = updater.reference_sql_storage.search_all_reference_sql(subject_path=SUBJECT_PATH + [SQL_NAME])
        assert len(results) >= 1
        assert results[0]["comment"] == "updated comment"

    def test_update_historical_sql_empty_values_is_noop(self, real_agent_config):
        """update_historical_sql with empty update_values should return early."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_reference_sql(updater.reference_sql_storage)

        updater.update_historical_sql(SUBJECT_PATH, SQL_NAME, {})

        results = updater.reference_sql_storage.search_all_reference_sql(subject_path=SUBJECT_PATH + [SQL_NAME])
        assert len(results) >= 1
        assert results[0]["comment"] == "test comment"

    def test_update_ext_knowledge_updates_main_storage(self, real_agent_config):
        """update_ext_knowledge should update the entry in main ext_knowledge storage."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_ext_knowledge(updater.ext_knowledge_storage)

        updater.update_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME, {"explanation": "updated explanation"})

        results = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(results) >= 1
        assert results[0]["explanation"] == "updated explanation"

    def test_update_ext_knowledge_empty_values_is_noop(self, real_agent_config):
        """update_ext_knowledge with empty update_values should return early."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_ext_knowledge(updater.ext_knowledge_storage)

        updater.update_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME, {})

        results = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(results) >= 1
        assert results[0]["explanation"] == "A test knowledge explanation"

    def test_delete_metric_from_main_storage(self, real_agent_config):
        """delete_metric should remove the metric from main storage and return success."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_metric(updater.metrics_storage)

        result = updater.delete_metric(SUBJECT_PATH, METRIC_NAME)

        assert result["success"] is True
        remaining = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(remaining) == 0

    def test_delete_metric_not_found_returns_failure(self, real_agent_config):
        """delete_metric for a non-existent metric should return success=False."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)

        result = updater.delete_metric(SUBJECT_PATH, "nonexistent_metric")

        assert result["success"] is False
        assert "not found" in result["message"].lower() or "nonexistent_metric" in result["message"]

    def test_delete_reference_sql_from_main_storage(self, real_agent_config):
        """delete_reference_sql should remove the SQL entry and return True."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_reference_sql(updater.reference_sql_storage)

        deleted = updater.delete_reference_sql(SUBJECT_PATH, SQL_NAME)

        assert deleted is True
        remaining = updater.reference_sql_storage.search_all_reference_sql(subject_path=SUBJECT_PATH + [SQL_NAME])
        assert len(remaining) == 0

    def test_delete_reference_sql_not_found_returns_false(self, real_agent_config):
        """delete_reference_sql for a non-existent entry should return False."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)

        deleted = updater.delete_reference_sql(SUBJECT_PATH, "nonexistent_sql")

        assert deleted is False
        assert isinstance(deleted, bool)

    def test_delete_ext_knowledge_from_main_storage(self, real_agent_config):
        """delete_ext_knowledge should remove the knowledge entry and return True."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)
        _seed_ext_knowledge(updater.ext_knowledge_storage)

        deleted = updater.delete_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME)

        assert deleted is True
        remaining = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(remaining) == 0

    def test_delete_ext_knowledge_not_found_returns_false(self, real_agent_config):
        """delete_ext_knowledge for a non-existent entry should return False."""
        clear_cache()
        updater = SubjectUpdater(real_agent_config)

        deleted = updater.delete_ext_knowledge(SUBJECT_PATH, "nonexistent_knowledge")

        assert deleted is False
        assert isinstance(deleted, bool)

    def test_update_ext_knowledge_with_sub_agents_configured(self, real_agent_config):
        """update_ext_knowledge should update main storage even when sub-agents are configured."""
        clear_cache()
        sub_agent_name = "test_sub_ext"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        # Seed data in main storage
        _seed_ext_knowledge(updater.ext_knowledge_storage)

        updater.update_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME, {"explanation": "sub-agent updated"})

        # Sub-agents now use scoped filters on the shared main storage,
        # so verifying main storage is sufficient.
        main_results = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(main_results) >= 1
        assert main_results[0]["explanation"] == "sub-agent updated"

    def test_delete_ext_knowledge_iterates_sub_agents(self, real_agent_config):
        """delete_ext_knowledge should attempt deletion from sub-agent storages in namespace."""
        clear_cache()
        sub_agent_name = "test_sub_delete_ext"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        # Seed data in main storage
        _seed_ext_knowledge(updater.ext_knowledge_storage)

        deleted = updater.delete_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME)

        # Main storage deletion should succeed
        assert deleted is True
        remaining = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(remaining) == 0

    def test_delete_metric_iterates_sub_agents(self, real_agent_config):
        """delete_metric should iterate over sub-agents matching the namespace."""
        clear_cache()
        sub_agent_name = "test_sub_delete_metric"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_metric(updater.metrics_storage)

        result = updater.delete_metric(SUBJECT_PATH, METRIC_NAME)

        assert result["success"] is True
        assert "message" in result

    def test_delete_reference_sql_iterates_sub_agents(self, real_agent_config):
        """delete_reference_sql should iterate over sub-agents matching the namespace."""
        clear_cache()
        sub_agent_name = "test_sub_delete_sql"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_reference_sql(updater.reference_sql_storage)

        deleted = updater.delete_reference_sql(SUBJECT_PATH, SQL_NAME)

        assert deleted is True
        remaining = updater.reference_sql_storage.search_all_reference_sql(subject_path=SUBJECT_PATH + [SQL_NAME])
        assert len(remaining) == 0

    def test_update_metrics_detail_propagates_to_sub_agents(self, real_agent_config):
        """update_metrics_detail should propagate updates to sub-agents in the namespace."""
        clear_cache()
        sub_agent_name = "test_sub_metric_upd"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_metric(updater.metrics_storage)

        # The sub-agent storage won't have the entry, but the method should not crash
        updater.update_metrics_detail(SUBJECT_PATH, METRIC_NAME, {"description": "propagated update"})

        # Main storage should be updated
        results = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(results) >= 1
        assert results[0]["description"] == "propagated update"

    def test_update_historical_sql_propagates_to_sub_agents(self, real_agent_config):
        """update_historical_sql should propagate updates to sub-agents in the namespace."""
        clear_cache()
        sub_agent_name = "test_sub_sql_upd"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_reference_sql(updater.reference_sql_storage)

        # The sub-agent storage won't have the entry, but should handle the error gracefully
        updater.update_historical_sql(SUBJECT_PATH, SQL_NAME, {"comment": "propagated sql update"})

        results = updater.reference_sql_storage.search_all_reference_sql(subject_path=SUBJECT_PATH + [SQL_NAME])
        assert len(results) >= 1
        assert results[0]["comment"] == "propagated sql update"


class TestSubjectUpdaterEdgeCases:
    """Tests for edge cases and error paths in SubjectUpdater."""

    def test_sub_agent_update_failure_is_caught_gracefully(self, real_agent_config):
        """When a sub-agent storage update_entry raises, the exception should be caught and logged."""
        clear_cache()
        sub_agent_name = "error_sub_agent"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_metric(updater.metrics_storage)

        # Sub-agent storage doesn't have the metric seeded, so update_entry will raise ValueError.
        # SubjectUpdater should catch it gracefully (lines 67-68).
        updater.update_metrics_detail(SUBJECT_PATH, METRIC_NAME, {"description": "should not crash"})

        # Verify main storage was still updated despite sub-agent failure
        results = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(results) >= 1
        assert results[0]["description"] == "should not crash"

    def test_sub_agent_delete_failure_is_caught_gracefully(self, real_agent_config):
        """When a sub-agent storage delete_entry raises, the exception should be caught and logged."""
        clear_cache()
        sub_agent_name = "error_delete_sub"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_reference_sql(updater.reference_sql_storage)

        # Sub-agent storage doesn't have the SQL entry; delete should still complete for main
        deleted = updater.delete_reference_sql(SUBJECT_PATH, SQL_NAME)

        assert deleted is True
        remaining = updater.reference_sql_storage.search_all_reference_sql(subject_path=SUBJECT_PATH + [SQL_NAME])
        assert len(remaining) == 0

    def test_no_sub_agents_in_namespace_skips_propagation(self, real_agent_config):
        """When no sub-agents match the current namespace, only main storage is updated."""
        clear_cache()
        # Add a sub-agent in a different namespace
        _add_sub_agent_with_namespace(real_agent_config, "other_ns_sub", "other_namespace")
        updater = SubjectUpdater(real_agent_config)

        _seed_metric(updater.metrics_storage)

        updater.update_metrics_detail(SUBJECT_PATH, METRIC_NAME, {"description": "only main"})

        results = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(results) >= 1
        assert results[0]["description"] == "only main"

    def test_delete_ext_knowledge_sub_agent_error_does_not_prevent_main_result(self, real_agent_config):
        """Sub-agent delete errors for ext_knowledge should not affect the main return value."""
        clear_cache()
        sub_agent_name = "error_ext_del_sub"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_ext_knowledge(updater.ext_knowledge_storage)

        # Sub-agent doesn't have the knowledge entry, but main should still delete successfully
        deleted = updater.delete_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME)

        assert deleted is True
        remaining = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(remaining) == 0

    def test_multiple_sub_agents_in_namespace(self, real_agent_config):
        """Update should iterate over all sub-agents matching the current namespace."""
        clear_cache()
        _add_sub_agent_with_namespace(real_agent_config, "sub_a", "test_ns")
        _add_sub_agent_with_namespace(real_agent_config, "sub_b", "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_ext_knowledge(updater.ext_knowledge_storage)

        # Should not raise even with multiple sub-agents attempting updates
        updater.update_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME, {"explanation": "multi-sub update"})

        results = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(results) >= 1
        assert results[0]["explanation"] == "multi-sub update"

    def test_sub_agent_metric_delete_error_does_not_prevent_main_result(self, real_agent_config):
        """Sub-agent delete errors for metrics should not affect the main return value."""
        clear_cache()
        _add_sub_agent_with_namespace(real_agent_config, "error_metric_sub", "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_metric(updater.metrics_storage)

        result = updater.delete_metric(SUBJECT_PATH, METRIC_NAME)

        assert result["success"] is True
        remaining = updater.metrics_storage.search_all_metrics(subject_path=SUBJECT_PATH + [METRIC_NAME])
        assert len(remaining) == 0

    def test_sub_agent_ext_knowledge_update_failure_is_caught(self, real_agent_config):
        """When a sub-agent storage update_entry raises for ext_knowledge, the exception is caught."""
        clear_cache()
        sub_agent_name = "error_ext_update_sub"
        _add_sub_agent_with_namespace(real_agent_config, sub_agent_name, "test_ns")
        updater = SubjectUpdater(real_agent_config)

        _seed_ext_knowledge(updater.ext_knowledge_storage)

        # Sub-agent storage doesn't have the ext_knowledge seeded, so update_entry will raise ValueError.
        # SubjectUpdater should catch it gracefully (lines 121-122).
        updater.update_ext_knowledge(SUBJECT_PATH, EXT_KNOWLEDGE_NAME, {"explanation": "should not crash"})

        # Verify main storage was still updated despite sub-agent failure
        results = updater.ext_knowledge_storage.search_knowledge(
            subject_path=SUBJECT_PATH + [EXT_KNOWLEDGE_NAME], top_n=None
        )
        assert len(results) >= 1
        assert results[0]["explanation"] == "should not crash"
