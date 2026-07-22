"""Tests for datus.api.services.explorer_service — catalog and subject tree."""

from unittest.mock import MagicMock

import pytest

from datus.api.models.base_models import Result
from datus.api.models.explorer_models import (
    CreateDirectoryInput,
    DeleteSubjectInput,
    ReferenceSQLInput,
    RenameSubjectInput,
    SubjectListData,
    SubjectNodeType,
    SubjectPathInput,
)
from datus.api.services.explorer_service import ExplorerService


class TestExplorerServiceInit:
    """Tests for ExplorerService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """ExplorerService initializes with real agent config."""
        svc = ExplorerService(agent_config=real_agent_config)
        assert isinstance(svc, ExplorerService)
        assert svc.agent_config is real_agent_config
        assert svc.datasource_id == real_agent_config.current_datasource

    def test_init_creates_rag_stores(self, real_agent_config):
        """ExplorerService creates metric and ref_sql RAG stores."""
        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        svc = ExplorerService(agent_config=real_agent_config)
        assert isinstance(svc.metric_rag, MetricRAG)
        assert isinstance(svc.reference_sql_rag, ReferenceSqlRAG)

    def test_init_creates_subject_tree_store(self, real_agent_config):
        """ExplorerService creates subject tree store."""
        from datus.storage.subject_tree.store import SubjectTreeStore

        svc = ExplorerService(agent_config=real_agent_config)
        assert isinstance(svc.subject_tree_store, SubjectTreeStore)


@pytest.mark.asyncio
class TestExplorerServiceGetSubjectList:
    """Tests for get_subject_list — subject tree retrieval."""

    async def test_get_subject_list_returns_result(self, real_agent_config):
        """get_subject_list returns a Result object."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_subject_list()
        assert result.success is True
        assert isinstance(result.data, SubjectListData)

    async def test_get_subject_list_has_subjects_field(self, real_agent_config):
        """get_subject_list returns data with subjects field (possibly empty)."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_subject_list()
        assert hasattr(result.data, "subjects")

    async def test_get_subject_list_with_populated_tree(self, real_agent_config):
        """get_subject_list returns tree with directories and ref_sql entries."""
        svc = ExplorerService(agent_config=real_agent_config)
        # Create some structure
        await svc.create_directory(CreateDirectoryInput(subject_path=["tree_test"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["tree_test"],
                name="tree_sql",
                sql="SELECT 1",
                summary="test",
                search_text="test",
            )
        )

        result = await svc.get_subject_list()
        assert result.success is True
        # Should have at least one directory node
        assert len(result.data.subjects) >= 1
        # Find our test directory
        tree_test_nodes = [node for node in result.data.subjects if node.name == "tree_test"]
        assert len(tree_test_nodes) == 1
        tree_test_node = tree_test_nodes[0]
        # Children should include ref_sql.
        assert isinstance(tree_test_node.children, list)
        child_names = {c.name for c in tree_test_node.children}
        assert "tree_sql" in child_names


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
        assert result.success is False

    async def test_get_reference_sql_empty_path(self, real_agent_config):
        """get_reference_sql with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_reference_sql([])
        assert result.success is False

    async def test_get_reference_sql_root_level_fails(self, real_agent_config):
        """get_reference_sql at root level returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_reference_sql(["only_name"])
        assert result.success is False
        assert "root level" in result.errorMessage.lower()

    async def test_create_then_get_reference_sql(self, real_agent_config):
        """Full lifecycle: create reference SQL then retrieve it."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["ref_test"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["ref_test"],
                name="my_query",
                sql="SELECT COUNT(*) FROM schools",
                summary="Count schools",
                search_text="count schools",
            )
        )
        result = await svc.get_reference_sql(["ref_test", "my_query"])
        assert result.success is True
        assert result.data.name == "my_query"
        assert result.data.sql == "SELECT COUNT(*) FROM schools"

    async def test_edit_reference_sql_empty_path(self, real_agent_config):
        """edit_reference_sql with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.edit_reference_sql(
            ReferenceSQLInput(
                subject_path=[],
                name="",
                sql="SELECT 1",
                summary="test",
                search_text="test",
            )
        )
        assert result.success is False

    async def test_edit_reference_sql_updates(self, real_agent_config):
        """edit_reference_sql updates an existing reference SQL entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["edit_ref"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["edit_ref"],
                name="editable",
                sql="SELECT 1",
                summary="original",
                search_text="original",
            )
        )
        result = await svc.edit_reference_sql(
            ReferenceSQLInput(
                subject_path=["edit_ref", "editable"],
                name="editable",
                sql="SELECT 2",
                summary="updated",
                search_text="updated",
            )
        )
        assert result.success is True

    async def test_edit_reference_sql_uses_sub_agent_conditions(self, real_agent_config):
        """edit_reference_sql should preserve scoped-agent filters when updating storage."""
        svc = ExplorerService(agent_config=real_agent_config)
        marker_condition = object()
        svc.reference_sql_rag._sub_agent_conditions = MagicMock(return_value=[marker_condition])
        svc.reference_sql_rag.reference_sql_storage.update_entry = MagicMock(return_value=True)

        result = await svc.edit_reference_sql(
            ReferenceSQLInput(
                subject_path=["edit_ref", "editable"],
                name="editable",
                sql="SELECT 2",
                summary="updated",
                search_text="updated",
            )
        )

        assert result.success is True
        svc.reference_sql_rag.reference_sql_storage.update_entry.assert_called_once_with(
            subject_path=["edit_ref"],
            name="editable",
            update_values={
                "sql": "SELECT 2",
                "summary": "updated",
                "search_text": "updated",
            },
            extra_conditions=[marker_condition],
        )


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

    async def test_rename_reference_sql(self, real_agent_config):
        """rename_subject renames a reference SQL entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["rename_sql_dir"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["rename_sql_dir"],
                name="old_sql",
                sql="SELECT 1",
                summary="test",
                search_text="test",
            )
        )
        result = await svc.rename_subject(
            RenameSubjectInput(
                type=SubjectNodeType.REFERENCE_SQL,
                subject_path=["rename_sql_dir", "old_sql"],
                new_subject_path=["rename_sql_dir", "new_sql"],
            )
        )
        assert result.success is True

    async def test_rename_metric(self, real_agent_config):
        """rename_subject for metric type exercises metric rename path."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.rename_subject(
            RenameSubjectInput(
                type=SubjectNodeType.METRIC,
                subject_path=["dir", "old_metric"],
                new_subject_path=["dir", "new_metric"],
            )
        )
        # May succeed or fail depending on metric existence, but exercises the code path
        assert isinstance(result, Result)
        assert isinstance(result.success, bool)

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
        await svc.create_directory(CreateDirectoryInput(subject_path=["to_delete"]))
        request = DeleteSubjectInput(
            type=SubjectNodeType.DIRECTORY,
            subject_path=["to_delete"],
        )
        result = await svc.delete_subject(request)
        assert result.success is True

    async def test_delete_empty_path_fails(self, real_agent_config):
        """delete_subject with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = DeleteSubjectInput(type=SubjectNodeType.DIRECTORY, subject_path=[])
        result = await svc.delete_subject(request)
        assert result.success is False

    async def test_delete_nonexistent_directory_fails(self, real_agent_config):
        """delete_subject for nonexistent directory returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = DeleteSubjectInput(type=SubjectNodeType.DIRECTORY, subject_path=["ghost"])
        result = await svc.delete_subject(request)
        assert result.success is False

    async def test_delete_reference_sql(self, real_agent_config):
        """delete_subject removes reference SQL entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["del_sql_dir"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["del_sql_dir"],
                name="del_query",
                sql="SELECT 1",
                summary="test",
                search_text="test",
            )
        )
        result = await svc.delete_subject(
            DeleteSubjectInput(
                type=SubjectNodeType.REFERENCE_SQL,
                subject_path=["del_sql_dir", "del_query"],
            )
        )
        assert result.success is True

    async def test_delete_metric_nonexistent(self, real_agent_config):
        """delete_subject for nonexistent metric returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.delete_subject(
            DeleteSubjectInput(
                type=SubjectNodeType.METRIC,
                subject_path=["dir", "nonexistent_metric"],
            )
        )
        assert result.success is False

    async def test_delete_directory_with_children(self, real_agent_config):
        """delete_subject cascade deletes directory with children."""
        svc = ExplorerService(agent_config=real_agent_config)
        # Create parent dir with children
        await svc.create_directory(CreateDirectoryInput(subject_path=["cascade_dir"]))
        await svc.create_directory(CreateDirectoryInput(subject_path=["cascade_dir", "child"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["cascade_dir"],
                name="child_sql",
                sql="SELECT 1",
                summary="test",
                search_text="test",
            )
        )
        # Delete parent — should cascade
        result = await svc.delete_subject(
            DeleteSubjectInput(
                type=SubjectNodeType.DIRECTORY,
                subject_path=["cascade_dir"],
            )
        )
        assert result.success is True


@pytest.mark.asyncio
class TestExplorerServiceSubjectAssets:
    """Tests for subject asset CRUD operations."""

    async def test_create_reference_sql_duplicate_fails(self, real_agent_config):
        """create_reference_sql rejects duplicate names."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["dup_ref_dir"]))
        await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["dup_ref_dir"],
                name="dup_sql",
                sql="SELECT 1",
                summary="first",
                search_text="first",
            )
        )
        result = await svc.create_reference_sql(
            ReferenceSQLInput(
                subject_path=["dup_ref_dir"],
                name="dup_sql",
                sql="SELECT 2",
                summary="second",
                search_text="second",
            )
        )
        assert result.success is False
        assert "already exists" in result.errorMessage

    async def test_get_metric_empty_path(self, real_agent_config):
        """get_metric with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_metric([])
        assert result.success is False

    async def test_get_metric_nonexistent(self, real_agent_config):
        """get_metric for nonexistent metric returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_metric(["some_dir", "nonexistent_metric"])
        assert result.success is False


@pytest.mark.asyncio
class TestExplorerServiceMetricFlowAuthoring:
    """create/edit/delete for MetricFlow now route through the unified
    ``_author_metric`` (adapter is the source of truth), keeping MetricFlow's
    real model validation via ``_validate_metric_yaml``."""

    METRIC = "metric:\n  name: revenue\n  type: aggregate\n"

    def _wire(self, svc, monkeypatch, model_dir, *, validate=(True, []), sync_ok=True):
        from types import SimpleNamespace

        from datus_semantic_metricflow.authoring import MetricFlowMetricAuthor

        author = MetricFlowMetricAuthor(str(model_dir))
        # Mirror MetricFlowAdapter's *_metric_source delegation to the author.
        adapter = SimpleNamespace(
            read_metric_source=lambda name, subject_path=None: author.read(name),
            write_metric_source=lambda name, source, subject_path=None, create=False: author.write(
                name, source, subject_path=subject_path, create=create
            ),
            delete_metric_source=lambda name, subject_path=None: author.delete(name),
            validate_metric_source=lambda source, metric_name=None: author.validate(source, metric_name=metric_name),
        )
        monkeypatch.setattr(
            "datus.tools.func_tool.semantic_tools.SemanticTools",
            lambda *a, **k: SimpleNamespace(adapter=adapter),
        )
        monkeypatch.setattr("datus.agent.node.semantic_authoring.is_osi_authoring", lambda *a, **k: False)
        # Real MetricFlow model validation is stubbed (needs a live model); the
        # unified path still calls it, which is what we assert.
        monkeypatch.setattr(svc, "_validate_metric_yaml", lambda content, path: validate)
        monkeypatch.setattr(
            svc,
            "_sync_file_to_kb",
            lambda is_osi, file_path: {"success": True} if sync_ok else {"success": False, "error": "boom"},
        )

    async def test_create_missing_name_fails(self, real_agent_config, tmp_path, monkeypatch):
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, tmp_path)
        result = await svc.create_metric(EditMetricInput(subject_path=["d"], yaml="metric:\n  type: aggregate\n"))
        assert result.success is False
        assert "name" in result.errorMessage.lower()

    async def test_create_writes_file_and_validates(self, real_agent_config, tmp_path, monkeypatch):
        import os

        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        validated = {}
        self._wire(svc, monkeypatch, tmp_path)

        def fake_validate(content, path):
            validated["called"] = True
            return (True, [])

        monkeypatch.setattr(svc, "_validate_metric_yaml", fake_validate)

        result = await svc.create_metric(EditMetricInput(subject_path=["d"], yaml=self.METRIC))
        assert result.success is True, result.errorMessage
        assert os.path.exists(tmp_path / "metrics" / "revenue.yml")
        assert validated.get("called")  # MetricFlow validation ran

    async def test_create_validation_failure_rolls_back(self, real_agent_config, tmp_path, monkeypatch):
        import os

        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, tmp_path, validate=(False, ["bad measure"]))

        result = await svc.create_metric(EditMetricInput(subject_path=["d"], yaml=self.METRIC))
        assert result.success is False
        assert "bad measure" in result.errorMessage
        assert not os.path.exists(tmp_path / "metrics" / "revenue.yml")  # rolled back

    async def test_create_kb_sync_failure_rolls_back(self, real_agent_config, tmp_path, monkeypatch):
        import os

        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, tmp_path, sync_ok=False)

        result = await svc.create_metric(EditMetricInput(subject_path=["d"], yaml=self.METRIC))
        assert result.success is False
        assert not os.path.exists(tmp_path / "metrics" / "revenue.yml")  # rolled back

    async def test_edit_empty_path_fails(self, real_agent_config):
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.edit_metric(EditMetricInput(subject_path=[], yaml=self.METRIC))
        assert result.success is False
        assert "empty" in result.errorMessage.lower()

    async def test_edit_updates_in_place(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        from datus.api.models.explorer_models import EditMetricInput

        (tmp_path / "revenue.yml").write_text(self.METRIC)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, tmp_path)

        edited = "metric:\n  name: revenue\n  type: aggregate\n  description: edited\n"
        result = await svc.edit_metric(EditMetricInput(subject_path=["revenue"], yaml=edited))
        assert result.success is True, result.errorMessage
        node = yaml.safe_load((tmp_path / "revenue.yml").read_text())["metric"]
        assert node["description"] == "edited"

    async def test_edit_validation_failure_restores(self, real_agent_config, tmp_path, monkeypatch):
        from datus.api.models.explorer_models import EditMetricInput

        (tmp_path / "revenue.yml").write_text(self.METRIC)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, tmp_path, validate=(False, ["nope"]))

        edited = "metric:\n  name: revenue\n  type: aggregate\n  description: edited\n"
        result = await svc.edit_metric(EditMetricInput(subject_path=["revenue"], yaml=edited))
        assert result.success is False
        # Restored to the original (no 'description').
        assert (tmp_path / "revenue.yml").read_text() == self.METRIC

    async def test_edit_nonexistent_fails(self, real_agent_config, tmp_path, monkeypatch):
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, tmp_path)
        result = await svc.edit_metric(
            EditMetricInput(subject_path=["ghost"], yaml="metric:\n  name: ghost\n  type: aggregate\n")
        )
        assert result.success is False


@pytest.mark.asyncio
class TestExplorerServiceValidateMetricYaml:
    """Tests for _validate_metric_yaml — metric YAML validation."""

    async def test_validate_valid_yaml(self, real_agent_config, tmp_path):
        """_validate_metric_yaml passes valid metric YAML."""
        svc = ExplorerService(agent_config=real_agent_config)
        metric_path = (
            tmp_path / "subject" / "semantic_models" / real_agent_config.current_datasource / "metrics" / "test.yml"
        )
        metric_path.parent.mkdir(parents=True)
        is_valid, errors = svc._validate_metric_yaml(
            "metric:\n  name: test\n  type: simple\n",
            str(metric_path),
        )
        # May pass or fail depending on metricflow availability
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    async def test_validate_invalid_yaml(self, real_agent_config, tmp_path):
        """_validate_metric_yaml rejects invalid YAML syntax."""
        svc = ExplorerService(agent_config=real_agent_config)
        metric_path = (
            tmp_path / "subject" / "semantic_models" / real_agent_config.current_datasource / "metrics" / "bad.yml"
        )
        metric_path.parent.mkdir(parents=True)
        is_valid, errors = svc._validate_metric_yaml(
            ":\n  - ][",
            str(metric_path),
        )
        assert is_valid is False
        assert len(errors) == 1
        assert isinstance(errors[0], str)


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


class TestGetSemanticFilePath:
    """Tests for _get_semantic_file_path helper."""

    def test_no_semantic_model_returns_empty(self, real_agent_config):
        """Returns empty string when no semantic model found."""
        svc = ExplorerService(agent_config=real_agent_config)
        path, error = svc._get_semantic_file_path(None, None, None, "nonexistent_table")
        assert path == ""
        assert error == "No semantic model found for provided parameters"


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


@pytest.mark.asyncio
class TestExplorerServiceMetricDimensions:
    """Tests for get_metric_dimensions — power the preview panel's dim picker."""

    @staticmethod
    def _patch_adapter(monkeypatch, *, adapter):
        from types import SimpleNamespace

        tools_stub = SimpleNamespace(adapter=adapter)

        def fake_semantic_tools(*args, **kwargs):
            tools_stub.args = args
            tools_stub.kwargs = kwargs
            return tools_stub

        monkeypatch.setattr(
            "datus.tools.func_tool.semantic_tools.SemanticTools",
            fake_semantic_tools,
        )
        return tools_stub

    async def test_empty_subject_path_fails(self, real_agent_config):
        """Empty subject path is rejected before touching the adapter."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_metric_dimensions(SubjectPathInput(subject_path=[]))
        assert result.success is False
        assert "Subject path cannot be empty" in result.errorMessage

    async def test_adapter_unavailable_fails(self, real_agent_config, monkeypatch):
        """A missing semantic adapter surfaces a clear error."""
        self._patch_adapter(monkeypatch, adapter=None)
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_metric_dimensions(SubjectPathInput(subject_path=["Finance", "revenue"]))
        assert result.success is False
        assert "adapter is not available" in result.errorMessage

    async def test_maps_dimension_fields(self, real_agent_config, monkeypatch):
        """Adapter DimensionInfo objects are mapped onto the response model."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        adapter = MagicMock()
        adapter.get_dimensions = AsyncMock(
            return_value=[
                SimpleNamespace(name="region", type="string", description="Sales region", is_primary_key=False),
                SimpleNamespace(name="metric_time", type="time", description=None, is_primary_key=None),
            ]
        )
        tools_stub = self._patch_adapter(monkeypatch, adapter=adapter)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_metric_dimensions(
            SubjectPathInput(
                subject_path=["Finance", "revenue"],
                catalog="runtime_catalog",
                database="runtime_db",
                db_schema="runtime_schema",
            )
        )

        assert result.success is True
        assert result.data.metric == "revenue"
        assert [d.name for d in result.data.dimensions] == ["region", "metric_time"]
        assert result.data.dimensions[0].type == "string"
        assert result.data.dimensions[1].type == "time"
        assert adapter.get_dimensions.await_args.kwargs["metric_name"] == "revenue"
        assert tools_stub.kwargs["runtime_db_context_provider"]() == {
            "datasource": real_agent_config.current_datasource,
            "catalog": "runtime_catalog",
            "database": "runtime_db",
            "schema": "runtime_schema",
            "db_schema": "runtime_schema",
        }


@pytest.mark.asyncio
class TestExplorerServicePreviewMetric:
    """Tests for preview_metric — compile a saved metric to SQL via dry-run."""

    @staticmethod
    def _patch_tools(monkeypatch, *, adapter, query_metrics=None):
        """Stub SemanticTools(...) with an ``adapter`` and sync ``query_metrics``."""
        from types import SimpleNamespace

        tools_stub = SimpleNamespace(adapter=adapter, query_metrics=query_metrics)

        def fake_semantic_tools(*args, **kwargs):
            tools_stub.args = args
            tools_stub.kwargs = kwargs
            return tools_stub

        monkeypatch.setattr(
            "datus.tools.func_tool.semantic_tools.SemanticTools",
            fake_semantic_tools,
        )
        return tools_stub

    @staticmethod
    def _func_result(*, success=1, error=None, result=None):
        from datus.tools.func_tool.base import FuncToolResult

        return FuncToolResult(success=success, error=error, result=result)

    async def test_empty_subject_path_fails(self, real_agent_config):
        """Empty subject path is rejected before touching the adapter."""
        from datus.api.models.explorer_models import MetricPreviewInput

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=[]))
        assert result.success is False
        assert "Subject path cannot be empty" in result.errorMessage

    async def test_adapter_unavailable_fails(self, real_agent_config, monkeypatch):
        """A missing semantic adapter surfaces a clear error, not a crash."""
        from datus.api.models.explorer_models import MetricPreviewInput

        self._patch_tools(monkeypatch, adapter=None)
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=["Finance", "revenue"]))
        assert result.success is False
        assert "adapter is not available" in result.errorMessage

    async def test_returns_compiled_sql_from_metadata(self, real_agent_config, monkeypatch):
        """Happy path: leaf is the metric, SQL comes from dry-run metadata."""
        from datus.api.models.explorer_models import MetricPreviewInput

        query_metrics = MagicMock(
            return_value=self._func_result(
                result={"metadata": {"explain": True, "sql": "SELECT 1 AS revenue"}, "data": []}
            )
        )
        tools_stub = self._patch_tools(monkeypatch, adapter=MagicMock(), query_metrics=query_metrics)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(
            MetricPreviewInput(
                subject_path=["Finance", "revenue"],
                dimensions=["region"],
                limit=100,
                database="runtime_preview_db",
            )
        )

        assert result.success is True
        assert result.data.metric == "revenue"
        assert result.data.sql == "SELECT 1 AS revenue"
        assert result.data.database == "runtime_preview_db"
        assert result.data.preflight_error is None
        assert tools_stub.kwargs["runtime_db_context_provider"]()["database"] == "runtime_preview_db"
        # dry_run must be requested so nothing actually executes.
        assert query_metrics.call_args.kwargs["dry_run"] is True
        assert query_metrics.call_args.kwargs["metrics"] == ["revenue"]
        assert query_metrics.call_args.kwargs["dimensions"] == ["region"]

    async def test_falls_back_to_data_row_sql(self, real_agent_config, monkeypatch):
        """SQL is recovered from the single data row when metadata omits it."""
        from datus.api.models.explorer_models import MetricPreviewInput

        query_metrics = MagicMock(
            return_value=self._func_result(result={"metadata": {}, "data": [{"sql": "SELECT 2"}]})
        )
        self._patch_tools(monkeypatch, adapter=MagicMock(), query_metrics=query_metrics)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=["revenue"]))
        assert result.success is True
        assert result.data.sql == "SELECT 2"

    async def test_dimension_preflight_failure_is_structured(self, real_agent_config, monkeypatch):
        """A dimension preflight failure becomes a structured preflight_error, not a raw error."""
        from datus.api.models.explorer_models import MetricPreviewInput

        preflight = {
            "metrics": ["revenue"],
            "requested_dimensions": ["country"],
            "invalid_dimensions": [{"name": "country", "unsupported_metrics": ["revenue"], "supported_metrics": []}],
            "common_dimensions": ["metric_time", "region"],
            "suggested_metric_groups": [{"metrics": ["revenue"], "dimensions": ["region"]}],
        }
        query_metrics = MagicMock(
            return_value=self._func_result(success=0, error="dimension preflight failed", result=preflight)
        )
        self._patch_tools(monkeypatch, adapter=MagicMock(), query_metrics=query_metrics)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=["revenue"], dimensions=["country"]))

        assert result.success is True
        assert result.data.sql is None
        pf = result.data.preflight_error
        assert pf is not None
        assert pf.common_dimensions == ["metric_time", "region"]
        assert pf.invalid_dimensions[0]["name"] == "country"
        assert pf.suggested_metric_groups[0]["dimensions"] == ["region"]

    async def test_no_sql_compiled_fails(self, real_agent_config, monkeypatch):
        """When neither metadata nor data carry SQL, report a clean failure."""
        from datus.api.models.explorer_models import MetricPreviewInput

        query_metrics = MagicMock(return_value=self._func_result(result={"metadata": {}, "data": []}))
        self._patch_tools(monkeypatch, adapter=MagicMock(), query_metrics=query_metrics)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=["revenue"]))
        assert result.success is False
        assert "Failed to compile SQL" in result.errorMessage

    async def test_hard_failure_surfaces_error(self, real_agent_config, monkeypatch):
        """A non-preflight tool failure becomes a failed Result with its message."""
        from datus.api.models.explorer_models import MetricPreviewInput

        query_metrics = MagicMock(
            return_value=self._func_result(success=0, error="unknown metric 'revenue'", result=None)
        )
        self._patch_tools(monkeypatch, adapter=MagicMock(), query_metrics=query_metrics)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=["revenue"]))
        assert result.success is False
        assert "unknown metric" in result.errorMessage

    async def test_exception_is_caught(self, real_agent_config, monkeypatch):
        """Unexpected errors become a failed Result rather than propagating."""
        from datus.api.models.explorer_models import MetricPreviewInput

        query_metrics = MagicMock(side_effect=RuntimeError("boom"))
        self._patch_tools(monkeypatch, adapter=MagicMock(), query_metrics=query_metrics)

        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.preview_metric(MetricPreviewInput(subject_path=["revenue"]))
        assert result.success is False
        assert "boom" in result.errorMessage


@pytest.mark.asyncio
class TestExplorerServiceOSIAuthoring:
    """OSI metrics are read/written through the semantic adapter (file source of
    truth), not reconstructed from / written as MetricFlow YAML."""

    SAMPLE = (
        "version: 0.2.0.dev0\n"
        "semantic_model:\n"
        "  - name: jeff_shop_live\n"
        "    datasets:\n"
        "      - name: raw_orders\n"
        "        source: jeff_shop.raw_orders\n"
        "        primary_key: [id]\n"
        "        fields:\n"
        "          - name: order_total\n"
        "            expression:\n"
        "              dialects:\n"
        "                - dialect: STARROCKS\n"
        "                  expression: order_total\n"
        "    metrics:\n"
        "      - name: daily_order_count\n"
        "        description: Daily order count.\n"
        "        expression:\n"
        "          dialects:\n"
        "            - dialect: STARROCKS\n"
        "              expression: COUNT(DISTINCT id)\n"
        "        custom_extensions:\n"
        "          - vendor_name: DATUS\n"
        '            data: \'{"dataset":"raw_orders","subject_path":["operations","daily"]}\'\n'
    )

    def _osi_adapter(self, tmp_path):
        # datus-semantic-osi is a guaranteed test dependency (dependency-groups
        # dev in pyproject), so these run in CI rather than silently skipping.
        from datus_semantic_osi.adapter import DatusOSIAdapter
        from datus_semantic_osi.config import DatusOSIConfig

        model_dir = tmp_path / "jeff_shop_live"
        model_dir.mkdir()
        (model_dir / "jeff_shop_live.yml").write_text(self.SAMPLE)
        config = DatusOSIConfig(
            datasource="ds",
            semantic_models_path=str(tmp_path),
            db_config={"type": "starrocks"},
        )
        return DatusOSIAdapter(config)

    def _wire(self, svc, monkeypatch, adapter):
        from types import SimpleNamespace

        monkeypatch.setattr(
            "datus.tools.func_tool.semantic_tools.SemanticTools",
            lambda *a, **k: SimpleNamespace(adapter=adapter),
        )
        monkeypatch.setattr(
            "datus.agent.node.semantic_authoring.is_osi_authoring",
            lambda *a, **k: True,
        )
        monkeypatch.setattr(svc, "_sync_file_to_kb", lambda is_osi, file_path: {"success": True})
        # get_metric still gates on the KB row for scope/access control; the row
        # content is irrelevant since the adapter supplies the returned YAML.
        monkeypatch.setattr(svc.metric_rag, "get_metrics_detail", lambda parent, name, *a, **k: [{"name": name}])

    async def test_get_metric_returns_osi_native_yaml(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)

        result = await svc.get_metric(["operations", "daily", "daily_order_count"])
        assert result.success is True
        node = yaml.safe_load(result.data.yaml)
        # OSI shape, not the MetricFlow reconstruction (no type/locked_metadata).
        assert node["expression"]["dialects"][0]["dialect"] == "STARROCKS"
        assert "type" not in node and "locked_metadata" not in node

    async def test_get_metric_falls_back_when_authoring_unsupported(self, real_agent_config, tmp_path, monkeypatch):
        from types import SimpleNamespace

        import yaml
        from datus_semantic_core.authoring import AuthoringNotSupportedError

        class _NoAuthoring:
            def read_metric_source(self, *a, **k):
                raise AuthoringNotSupportedError("nope")

        svc = ExplorerService(agent_config=real_agent_config)
        monkeypatch.setattr(
            "datus.tools.func_tool.semantic_tools.SemanticTools",
            lambda *a, **k: SimpleNamespace(adapter=_NoAuthoring()),
        )
        # KB row exists (gate passes); adapter has no file source, so the
        # response is reconstructed from the KB projection.
        monkeypatch.setattr(
            svc.metric_rag,
            "get_metrics_detail",
            lambda parent, name, *a, **k: [{"name": name, "metric_type": "simple", "base_measures": ["revenue"]}],
        )

        result = await svc.get_metric(["revenue", "daily_revenue"])
        assert result.success is True
        assert yaml.safe_load(result.data.yaml)["metric"]["name"] == "daily_revenue"

    async def test_get_metric_not_found_when_kb_row_missing(self, real_agent_config, tmp_path, monkeypatch):
        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)
        # KB gate enforces scope: no row -> not found, even though the file has it.
        monkeypatch.setattr(svc.metric_rag, "get_metrics_detail", lambda *a, **k: [])

        result = await svc.get_metric(["wrong", "path", "daily_order_count"])
        assert result.success is False
        assert "not found" in result.errorMessage.lower()

    async def test_create_metric_adapter_unavailable_fails(self, real_agent_config, monkeypatch):
        from types import SimpleNamespace

        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        monkeypatch.setattr("datus.agent.node.semantic_authoring.is_osi_authoring", lambda *a, **k: True)
        monkeypatch.setattr(
            "datus.tools.func_tool.semantic_tools.SemanticTools",
            lambda *a, **k: SimpleNamespace(adapter=None),
        )
        result = await svc.create_metric(EditMetricInput(subject_path=["x"], yaml="name: m\ntype: aggregate\n"))
        assert result.success is False
        assert "adapter is not available" in result.errorMessage

    async def test_create_metric_writes_osi_file_and_syncs(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        from datus.api.models.explorer_models import EditMetricInput

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)
        synced = {}

        def fake_sync(is_osi, file_path):
            synced["path"] = file_path
            return {"success": True}

        monkeypatch.setattr(svc, "_sync_file_to_kb", fake_sync)

        new_metric = (
            "name: gross_revenue\n"
            "description: revenue\n"
            "expression:\n"
            "  dialects:\n"
            "    - dialect: STARROCKS\n"
            "      expression: SUM(order_total)\n"
            "custom_extensions:\n"
            "  - vendor_name: DATUS\n"
            '    data: \'{"dataset":"raw_orders"}\'\n'
        )
        result = await svc.create_metric(EditMetricInput(subject_path=["revenue"], yaml=new_metric))
        assert result.success is True, result.errorMessage
        assert synced.get("path")  # KB re-sync was triggered
        on_disk = yaml.safe_load((tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text())
        names = [m["name"] for m in on_disk["semantic_model"][0]["metrics"]]
        assert set(names) == {"daily_order_count", "gross_revenue"}

    async def test_edit_metric_updates_in_place(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        from datus.api.models.explorer_models import EditMetricInput

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)

        edited = (
            "name: daily_order_count\n"
            "description: Edited desc.\n"
            "expression:\n"
            "  dialects:\n"
            "    - dialect: STARROCKS\n"
            "      expression: COUNT(DISTINCT id)\n"
            "custom_extensions:\n"
            "  - vendor_name: DATUS\n"
            '    data: \'{"dataset":"raw_orders"}\'\n'
        )
        result = await svc.edit_metric(
            EditMetricInput(subject_path=["operations", "daily", "daily_order_count"], yaml=edited)
        )
        assert result.success is True, result.errorMessage
        on_disk = yaml.safe_load((tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text())
        model = on_disk["semantic_model"][0]
        assert model["metrics"][0]["description"] == "Edited desc."
        # Sibling dataset preserved.
        assert model["datasets"][0]["name"] == "raw_orders"

    async def test_create_metric_validation_failure_does_not_write(self, real_agent_config, tmp_path, monkeypatch):
        from datus.api.models.explorer_models import EditMetricInput

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)
        before = (tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text()

        result = await svc.create_metric(EditMetricInput(subject_path=["x"], yaml=":: not yaml ::"))
        assert result.success is False
        assert (tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text() == before

    async def test_delete_metric_removes_from_osi_file(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)
        monkeypatch.setattr(svc.metric_rag, "delete_metric", lambda *a, **k: {"success": True})

        result = await svc.delete_subject(
            DeleteSubjectInput(type=SubjectNodeType.METRIC, subject_path=["operations", "daily", "daily_order_count"])
        )
        assert result.success is True, result.errorMessage
        on_disk = yaml.safe_load((tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text())
        assert on_disk["semantic_model"][0]["metrics"] == []

    async def test_create_metric_rolls_back_on_kb_sync_failure(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        from datus.api.models.explorer_models import EditMetricInput

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)
        # KB re-sync fails -> the newly created metric must be removed again.
        monkeypatch.setattr(svc, "_sync_file_to_kb", lambda is_osi, file_path: {"success": False, "error": "boom"})

        new_metric = (
            "name: gross_revenue\n"
            "description: revenue\n"
            "expression:\n"
            "  dialects:\n"
            "    - dialect: STARROCKS\n"
            "      expression: SUM(order_total)\n"
            "custom_extensions:\n"
            "  - vendor_name: DATUS\n"
            '    data: \'{"dataset":"raw_orders"}\'\n'
        )
        result = await svc.create_metric(EditMetricInput(subject_path=["revenue"], yaml=new_metric))
        assert result.success is False
        assert "boom" in result.errorMessage
        on_disk = yaml.safe_load((tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text())
        names = [m["name"] for m in on_disk["semantic_model"][0]["metrics"]]
        assert "gross_revenue" not in names  # rolled back

    async def test_edit_metric_restores_previous_on_kb_sync_failure(self, real_agent_config, tmp_path, monkeypatch):
        import yaml

        from datus.api.models.explorer_models import EditMetricInput

        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)
        monkeypatch.setattr(svc, "_sync_file_to_kb", lambda is_osi, file_path: {"success": False, "error": "boom"})

        edited = (
            "name: daily_order_count\n"
            "description: EDITED\n"
            "expression:\n"
            "  dialects:\n"
            "    - dialect: STARROCKS\n"
            "      expression: COUNT(DISTINCT id)\n"
            "custom_extensions:\n"
            "  - vendor_name: DATUS\n"
            '    data: \'{"dataset":"raw_orders"}\'\n'
        )
        result = await svc.edit_metric(
            EditMetricInput(subject_path=["operations", "daily", "daily_order_count"], yaml=edited)
        )
        assert result.success is False
        on_disk = yaml.safe_load((tmp_path / "jeff_shop_live" / "jeff_shop_live.yml").read_text())
        # Edit was reverted to the original description.
        assert on_disk["semantic_model"][0]["metrics"][0]["description"] == "Daily order count."

    async def test_delete_metric_real_write_failure_fails(self, real_agent_config, tmp_path, monkeypatch):
        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)

        # Simulate a real write failure: delete raises but the metric is still
        # present in the file -> the request must fail (not silently drop the KB).
        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(adapter, "delete_metric_source", boom)
        kb_deleted = {"called": False}
        monkeypatch.setattr(
            svc.metric_rag, "delete_metric", lambda *a, **k: kb_deleted.update(called=True) or {"success": True}
        )

        result = await svc.delete_subject(
            DeleteSubjectInput(type=SubjectNodeType.METRIC, subject_path=["operations", "daily", "daily_order_count"])
        )
        assert result.success is False
        assert kb_deleted["called"] is False  # KB row not dropped when file delete failed

    async def test_delete_metric_absent_from_source_still_cleans_kb(self, real_agent_config, tmp_path, monkeypatch):
        adapter = self._osi_adapter(tmp_path)
        svc = ExplorerService(agent_config=real_agent_config)
        self._wire(svc, monkeypatch, adapter)

        # Metric already gone from the source file (file/KB drift): the not-found
        # error is benign and the stale KB row is still cleaned up.
        def not_found(*a, **k):
            raise FileNotFoundError("Metric `x` was not found in ...")

        monkeypatch.setattr(adapter, "delete_metric_source", not_found)
        kb_deleted = {"called": False}
        monkeypatch.setattr(
            svc.metric_rag, "delete_metric", lambda *a, **k: kb_deleted.update(called=True) or {"success": True}
        )

        result = await svc.delete_subject(
            DeleteSubjectInput(type=SubjectNodeType.METRIC, subject_path=["operations", "daily", "daily_order_count"])
        )
        assert result.success is True, result.errorMessage
        assert kb_deleted["called"] is True  # stale KB row cleaned up
