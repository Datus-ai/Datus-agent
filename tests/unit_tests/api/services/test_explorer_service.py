"""Tests for datus.api.services.explorer_service — catalog and subject tree."""

import pytest

from datus.api.models.explorer_models import (
    CreateDirectoryInput,
    CreateKnowledgeInput,
    DeleteSubjectInput,
    EditKnowledgeInput,
    ReferenceSQLInput,
    RenameSubjectInput,
    SubjectNodeType,
)
from datus.api.services.explorer_service import ExplorerService


class TestExplorerServiceInit:
    """Tests for ExplorerService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """ExplorerService initializes with real agent config."""
        svc = ExplorerService(agent_config=real_agent_config)
        assert svc is not None
        assert svc.agent_config is real_agent_config
        assert svc.datasource_id == real_agent_config.current_namespace

    def test_init_creates_rag_stores(self, real_agent_config):
        """ExplorerService creates metric, ref_sql, and knowledge RAG stores."""
        svc = ExplorerService(agent_config=real_agent_config)
        assert svc.metric_rag is not None
        assert svc.reference_sql_rag is not None
        assert svc.knowledge_rag is not None

    def test_init_creates_subject_tree_store(self, real_agent_config):
        """ExplorerService creates subject tree store."""
        svc = ExplorerService(agent_config=real_agent_config)
        assert svc.subject_tree_store is not None


@pytest.mark.asyncio
class TestExplorerServiceGetSubjectList:
    """Tests for get_subject_list — subject tree retrieval."""

    async def test_get_subject_list_returns_result(self, real_agent_config):
        """get_subject_list returns a Result object."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_subject_list()
        assert result.success is True
        assert result.data is not None

    async def test_get_subject_list_has_subjects_field(self, real_agent_config):
        """get_subject_list returns data with subjects field (possibly empty)."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_subject_list()
        assert hasattr(result.data, "subjects")


@pytest.mark.asyncio
class TestExplorerServiceCreateDirectory:
    """Tests for create_directory — subject tree directory creation."""

    async def test_create_directory_success(self, real_agent_config):
        """create_directory creates a new directory in subject tree."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = CreateDirectoryInput(subject_path=["test_dir"])
        result = await svc.create_directory(request)
        assert result.success is True

    async def test_create_nested_directory(self, real_agent_config):
        """create_directory creates nested directories."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = CreateDirectoryInput(subject_path=["parent", "child", "grandchild"])
        result = await svc.create_directory(request)
        assert result.success is True

    async def test_create_directory_empty_path_fails(self, real_agent_config):
        """create_directory with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = CreateDirectoryInput(subject_path=[])
        result = await svc.create_directory(request)
        assert result.success is False
        assert "empty" in result.errorMessage.lower()


@pytest.mark.asyncio
class TestExplorerServiceReferenceSql:
    """Tests for reference SQL CRUD operations."""

    async def test_create_reference_sql_success(self, real_agent_config):
        """create_reference_sql stores a new reference SQL entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        # Create parent directory first
        await svc.create_directory(CreateDirectoryInput(subject_path=["sql_test_dir"]))
        request = ReferenceSQLInput(
            subject_path=["sql_test_dir"],
            name="test_query",
            sql="SELECT COUNT(*) FROM schools",
            summary="Count all schools",
            search_text="count schools",
        )
        result = await svc.create_reference_sql(request)
        assert result.success is True

    async def test_create_reference_sql_empty_name_fails(self, real_agent_config):
        """create_reference_sql with empty name returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = ReferenceSQLInput(
            subject_path=[],
            name="",
            sql="SELECT 1",
            summary="test",
            search_text="test",
        )
        result = await svc.create_reference_sql(request)
        assert result.success is False

    async def test_get_reference_sql_nonexistent(self, real_agent_config):
        """get_reference_sql for nonexistent path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_reference_sql(["nonexistent", "path", "query"])
        # Should return not found
        assert result.success is False or (result.success and result.data is None)


@pytest.mark.asyncio
class TestExplorerServiceRenameSubject:
    """Tests for rename_subject operations."""

    async def test_rename_directory_success(self, real_agent_config):
        """rename_subject renames a directory."""
        svc = ExplorerService(agent_config=real_agent_config)
        # Create directory first
        await svc.create_directory(CreateDirectoryInput(subject_path=["rename_me"]))
        request = RenameSubjectInput(
            type=SubjectNodeType.DIRECTORY,
            subject_path=["rename_me"],
            new_subject_path=["renamed"],
        )
        result = await svc.rename_subject(request)
        assert result.success is True

    async def test_rename_empty_paths_fail(self, real_agent_config):
        """rename_subject with empty paths returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = RenameSubjectInput(
            type=SubjectNodeType.DIRECTORY,
            subject_path=[],
            new_subject_path=[],
        )
        result = await svc.rename_subject(request)
        assert result.success is False


@pytest.mark.asyncio
class TestExplorerServiceDeleteSubject:
    """Tests for delete_subject operations."""

    async def test_delete_directory(self, real_agent_config):
        """delete_subject removes a directory from tree."""
        svc = ExplorerService(agent_config=real_agent_config)
        # Create then delete
        await svc.create_directory(CreateDirectoryInput(subject_path=["to_delete"]))
        request = DeleteSubjectInput(
            type=SubjectNodeType.DIRECTORY,
            subject_path=["to_delete"],
        )
        result = await svc.delete_subject(request)
        assert result.success is True


@pytest.mark.asyncio
class TestExplorerServiceKnowledge:
    """Tests for knowledge CRUD operations."""

    async def test_create_knowledge_success(self, real_agent_config):
        """create_knowledge stores a new knowledge entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["kb_test"]))
        request = CreateKnowledgeInput(
            subject_path=["kb_test"],
            name="test_knowledge",
            search_text="california schools types",
            explanation="Schools in California have various types.",
        )
        result = await svc.create_knowledge(request)
        assert result.success is True

    async def test_get_knowledge_nonexistent(self, real_agent_config):
        """get_knowledge for nonexistent entry returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_knowledge(["nonexistent", "knowledge"])
        assert result.success is False or (result.success and result.data is None)


class TestMetricDbToYaml:
    """Tests for _metric_db_to_yaml — DB to YAML format conversion."""

    def test_simple_metric(self):
        """Simple metric with single measure."""
        data = {
            "name": "revenue",
            "description": "Total revenue",
            "metric_type": "simple",
            "base_measures": ["revenue_measure"],
            "measure_expr": "",
            "subject_path": ["finance"],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["name"] == "revenue"
        assert result["metric"]["description"] == "Total revenue"
        assert result["metric"]["type"] == "simple"
        assert result["metric"]["type_params"]["measure"] == "revenue_measure"
        assert "subject_tree: finance" in result["metric"]["locked_metadata"]["tags"][0]

    def test_ratio_metric(self):
        """Ratio metric with numerator and denominator."""
        data = {
            "name": "conversion_rate",
            "description": "Conversion rate",
            "metric_type": "ratio",
            "base_measures": ["conversions", "visits"],
            "measure_expr": "",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["type"] == "ratio"
        assert result["metric"]["type_params"]["numerator"]["name"] == "conversions"
        assert result["metric"]["type_params"]["denominator"]["name"] == "visits"

    def test_derived_metric(self):
        """Derived metric with expression."""
        data = {
            "name": "profit_margin",
            "description": "Profit margin",
            "metric_type": "derived",
            "base_measures": ["revenue", "cost"],
            "measure_expr": "revenue - cost",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["type"] == "derived"
        assert result["metric"]["type_params"]["metrics"] == ["revenue", "cost"]
        assert result["metric"]["type_params"]["expr"] == "revenue - cost"

    def test_measure_proxy_single(self):
        """Measure proxy metric with single measure."""
        data = {
            "name": "count_orders",
            "description": "",
            "metric_type": "measure_proxy",
            "base_measures": ["order_count"],
            "measure_expr": "",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["type_params"]["measure"] == "order_count"

    def test_measure_proxy_multiple(self):
        """Measure proxy metric with multiple measures."""
        data = {
            "name": "multi_measure",
            "description": "",
            "metric_type": "measure_proxy",
            "base_measures": ["m1", "m2"],
            "measure_expr": "",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["type_params"]["measures"] == ["m1", "m2"]

    def test_expr_metric(self):
        """Expression metric with measures and expr."""
        data = {
            "name": "custom_metric",
            "description": "Custom calc",
            "metric_type": "expr",
            "base_measures": ["base_m"],
            "measure_expr": "base_m * 100",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["type_params"]["measures"] == ["base_m"]
        assert result["metric"]["type_params"]["expr"] == "base_m * 100"

    def test_cumulative_metric(self):
        """Cumulative metric type."""
        data = {
            "name": "running_total",
            "description": "",
            "metric_type": "cumulative",
            "base_measures": ["daily_revenue"],
            "measure_expr": "",
            "subject_path": ["sales"],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert result["metric"]["type"] == "cumulative"
        assert result["metric"]["type_params"]["measures"] == ["daily_revenue"]

    def test_no_type_params_when_empty(self):
        """No type_params key when no measures or expression."""
        data = {
            "name": "empty_metric",
            "description": "",
            "metric_type": "unknown_type",
            "base_measures": [],
            "measure_expr": "",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert "type_params" not in result["metric"]

    def test_no_locked_metadata_when_no_path(self):
        """No locked_metadata when subject_path is empty."""
        data = {
            "name": "orphan",
            "description": "",
            "metric_type": "simple",
            "base_measures": [],
            "measure_expr": "",
            "subject_path": [],
        }
        result = ExplorerService._metric_db_to_yaml(data)
        assert "locked_metadata" not in result["metric"]


class TestUpdateMetricInYamlDocs:
    """Tests for _update_metric_in_yaml_docs helper."""

    def test_updates_existing_metric(self, real_agent_config):
        """Updates metric in document list when name matches."""
        svc = ExplorerService(agent_config=real_agent_config)
        docs = [
            {"metric": {"name": "revenue", "type": "simple"}},
            {"metric": {"name": "cost", "type": "simple"}},
        ]
        new_data = {"name": "revenue", "type": "derived", "description": "Updated"}
        updated, error = svc._update_metric_in_yaml_docs(docs, "revenue", new_data)
        assert error is None
        assert updated[0]["metric"]["type"] == "derived"
        assert updated[1]["metric"]["name"] == "cost"  # unchanged

    def test_metric_not_found_returns_error(self, real_agent_config):
        """Returns error message when metric name not found."""
        svc = ExplorerService(agent_config=real_agent_config)
        docs = [{"metric": {"name": "revenue"}}]
        updated, error = svc._update_metric_in_yaml_docs(docs, "nonexistent", {})
        assert error is not None
        assert "not found" in error

    def test_skips_none_documents(self, real_agent_config):
        """Skips None/empty documents without error."""
        svc = ExplorerService(agent_config=real_agent_config)
        docs = [None, {"metric": {"name": "target"}}, None]
        new_data = {"name": "target", "type": "updated"}
        updated, error = svc._update_metric_in_yaml_docs(docs, "target", new_data)
        assert error is None


class TestWriteYamlAtomic:
    """Tests for _write_yaml_atomic — atomic file writing."""

    def test_writes_yaml_documents(self, real_agent_config, tmp_path):
        """Successfully writes YAML documents atomically."""
        svc = ExplorerService(agent_config=real_agent_config)
        file_path = str(tmp_path / "test.yml")
        docs = [{"metric": {"name": "test", "type": "simple"}}]
        error = svc._write_yaml_atomic(file_path, docs)
        assert error is None
        # Verify file was written
        import yaml

        with open(file_path) as f:
            loaded = list(yaml.safe_load_all(f))
        assert loaded[0]["metric"]["name"] == "test"

    def test_writes_multiple_documents(self, real_agent_config, tmp_path):
        """Writes multiple YAML documents with separators."""
        svc = ExplorerService(agent_config=real_agent_config)
        file_path = str(tmp_path / "multi.yml")
        docs = [
            {"metric": {"name": "m1"}},
            {"metric": {"name": "m2"}},
        ]
        error = svc._write_yaml_atomic(file_path, docs)
        assert error is None
        import yaml

        with open(file_path) as f:
            loaded = list(yaml.safe_load_all(f))
        assert len(loaded) == 2

    def test_invalid_directory_returns_error(self, real_agent_config):
        """Writing to nonexistent directory returns error message."""
        svc = ExplorerService(agent_config=real_agent_config)
        error = svc._write_yaml_atomic("/nonexistent/path/file.yml", [{"a": 1}])
        assert error is not None
        assert "Failed to write" in error


class TestGetSemanticFilePath:
    """Tests for _get_semantic_file_path helper."""

    def test_no_semantic_model_returns_empty(self, real_agent_config):
        """Returns empty string when no semantic model found."""
        svc = ExplorerService(agent_config=real_agent_config)
        path, error = svc._get_semantic_file_path(None, None, None, "nonexistent_table")
        assert path == ""
        assert error is not None


class TestExplorerServiceHelpers:
    """Tests for ExplorerService helper methods."""

    def test_gen_reference_sql_id_deterministic(self, real_agent_config):
        """_gen_reference_sql_id returns stable ID for same SQL."""
        svc = ExplorerService(agent_config=real_agent_config)
        id1 = svc._gen_reference_sql_id("SELECT 1")
        id2 = svc._gen_reference_sql_id("SELECT 1")
        assert id1 == id2

    def test_gen_reference_sql_id_different_for_different_sql(self, real_agent_config):
        """_gen_reference_sql_id returns different IDs for different SQL."""
        svc = ExplorerService(agent_config=real_agent_config)
        id1 = svc._gen_reference_sql_id("SELECT 1")
        id2 = svc._gen_reference_sql_id("SELECT 2")
        assert id1 != id2

    def test_gen_subject_item_id_deterministic(self, real_agent_config):
        """_gen_subject_item_id returns stable ID for same inputs."""
        svc = ExplorerService(agent_config=real_agent_config)
        id1 = svc._gen_subject_item_id(["root", "child"], "item1")
        id2 = svc._gen_subject_item_id(["root", "child"], "item1")
        assert id1 == id2

    def test_gen_subject_item_id_different_for_different_path(self, real_agent_config):
        """_gen_subject_item_id returns different IDs for different paths."""
        svc = ExplorerService(agent_config=real_agent_config)
        id1 = svc._gen_subject_item_id(["root", "child"], "item1")
        id2 = svc._gen_subject_item_id(["root", "other"], "item1")
        assert id1 != id2
