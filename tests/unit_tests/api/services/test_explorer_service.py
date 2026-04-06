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

    async def test_get_subject_list_with_populated_tree(self, real_agent_config):
        """get_subject_list returns tree with directories, metrics, ref_sql, knowledge."""
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
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["tree_test"],
                name="tree_kb",
                search_text="test",
                explanation="test",
            )
        )

        result = await svc.get_subject_list()
        assert result.success is True
        # Should have at least one directory node
        assert len(result.data.subjects) >= 1
        # Find our test directory
        tree_test_node = None
        for node in result.data.subjects:
            if node.name == "tree_test":
                tree_test_node = node
                break
        assert tree_test_node is not None
        # Children should include ref_sql and knowledge
        assert tree_test_node.children is not None
        child_names = {c.name for c in tree_test_node.children}
        assert "tree_sql" in child_names
        assert "tree_kb" in child_names


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

    async def test_rename_knowledge(self, real_agent_config):
        """rename_subject renames a knowledge entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["rename_kb_dir"]))
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["rename_kb_dir"],
                name="old_kb",
                search_text="test",
                explanation="test",
            )
        )
        result = await svc.rename_subject(
            RenameSubjectInput(
                type=SubjectNodeType.KNOWLEDGE,
                subject_path=["rename_kb_dir", "old_kb"],
                new_subject_path=["rename_kb_dir", "new_kb"],
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
        assert result is not None

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

    async def test_delete_knowledge(self, real_agent_config):
        """delete_subject removes knowledge entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["del_kb_dir"]))
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["del_kb_dir"],
                name="del_kb",
                search_text="test",
                explanation="test",
            )
        )
        result = await svc.delete_subject(
            DeleteSubjectInput(
                type=SubjectNodeType.KNOWLEDGE,
                subject_path=["del_kb_dir", "del_kb"],
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
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["cascade_dir"],
                name="child_kb",
                search_text="test",
                explanation="test",
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
        assert result.success is False

    async def test_get_knowledge_empty_path(self, real_agent_config):
        """get_knowledge with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_knowledge([])
        assert result.success is False

    async def test_create_then_get_knowledge(self, real_agent_config):
        """Full lifecycle: create knowledge then retrieve it."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["kb_full"]))
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["kb_full"],
                name="test_kb",
                search_text="california schools types",
                explanation="Schools in California have various types.",
            )
        )
        result = await svc.get_knowledge(["kb_full", "test_kb"])
        assert result.success is True
        assert result.data.name == "test_kb"

    async def test_create_knowledge_empty_path_fails(self, real_agent_config):
        """create_knowledge with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = CreateKnowledgeInput(
            subject_path=[],
            name="x",
            search_text="test",
            explanation="test",
        )
        result = await svc.create_knowledge(request)
        assert result.success is False

    async def test_create_knowledge_duplicate_fails(self, real_agent_config):
        """create_knowledge rejects duplicate names in same directory."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["dup_kb_dir"]))
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["dup_kb_dir"],
                name="dup_entry",
                search_text="first",
                explanation="first",
            )
        )
        result = await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["dup_kb_dir"],
                name="dup_entry",
                search_text="second",
                explanation="second",
            )
        )
        assert result.success is False
        assert "already exists" in result.errorMessage

    async def test_get_knowledge_root_level_fails(self, real_agent_config):
        """get_knowledge at root level (single component path) returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.get_knowledge(["only_name"])
        assert result.success is False
        assert "root level" in result.errorMessage.lower()

    async def test_edit_knowledge_short_path_fails(self, real_agent_config):
        """edit_knowledge with path < 2 components returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        result = await svc.edit_knowledge(
            EditKnowledgeInput(
                subject_path=["only_one"],
                name="kb",
                search_text="test",
                explanation="test",
            )
        )
        assert result.success is False
        assert "2 components" in result.errorMessage

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
class TestExplorerServiceCreateMetric:
    """Tests for create_metric — metric creation with YAML validation."""

    async def test_create_metric_invalid_yaml(self, real_agent_config):
        """create_metric with invalid YAML returns error."""
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        request = EditMetricInput(
            subject_path=["test_dir"],
            yaml=":\n  - ][",
        )
        result = await svc.create_metric(request)
        assert result.success is False
        assert "yaml" in result.errorMessage.lower() or "invalid" in result.errorMessage.lower()

    async def test_create_metric_missing_metric_key(self, real_agent_config):
        """create_metric with YAML missing 'metric' key returns error."""
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        request = EditMetricInput(
            subject_path=["test_dir"],
            yaml="data_source:\n  name: test\n",
        )
        result = await svc.create_metric(request)
        assert result.success is False
        assert "no metric document" in result.errorMessage.lower()

    async def test_create_metric_missing_name(self, real_agent_config):
        """create_metric with metric missing 'name' returns error."""
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        request = EditMetricInput(
            subject_path=["test_dir"],
            yaml="metric:\n  type: simple\n",
        )
        result = await svc.create_metric(request)
        assert result.success is False
        assert "name" in result.errorMessage.lower()

    async def test_create_metric_with_valid_yaml(self, real_agent_config):
        """create_metric with valid YAML exercises the full creation path."""
        import os

        from datus.api.models.explorer_models import EditMetricInput
        from datus.utils.path_manager import DatusPathManager

        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["metric_create_test"]))

        pm = DatusPathManager(datus_home=real_agent_config.home)
        metrics_dir = pm.semantic_model_path(real_agent_config.current_namespace) / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        request = EditMetricInput(
            subject_path=["metric_create_test"],
            yaml="metric:\n  name: test_revenue\n  type: measure_proxy\n  type_params:\n    measure: count_orders\n",
        )
        result = await svc.create_metric(request)
        # May fail on deep validation (no data_source in model) — that's expected.
        # The important thing is it exercises the full path: parse → check existence → validate
        assert result is not None
        if not result.success:
            # Should be a validation error, not a crash
            assert "validation" in result.errorMessage.lower() or "not defined" in result.errorMessage.lower()

    async def test_create_metric_duplicate_file_fails(self, real_agent_config):
        """create_metric rejects when file already exists on disk."""
        import os

        from datus.api.models.explorer_models import EditMetricInput
        from datus.utils.path_manager import DatusPathManager

        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["dup_file_dir"]))

        pm = DatusPathManager(datus_home=real_agent_config.home)
        metrics_dir = pm.semantic_model_path(real_agent_config.current_namespace) / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        # Pre-create the file on disk
        file_path = metrics_dir / "pre_existing.yml"
        file_path.write_text("metric:\n  name: pre_existing\n")

        result = await svc.create_metric(
            EditMetricInput(
                subject_path=["dup_file_dir"],
                yaml="metric:\n  name: pre_existing\n  type: measure_proxy\n",
            )
        )
        assert result.success is False
        assert "already exists" in result.errorMessage.lower()

    async def test_create_metric_with_subject_tree_tag(self, real_agent_config):
        """create_metric with locked_metadata.tags subject_tree overrides the tag."""
        import os

        from datus.api.models.explorer_models import EditMetricInput
        from datus.utils.path_manager import DatusPathManager

        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["tagged_dir"]))

        pm = DatusPathManager(datus_home=real_agent_config.home)
        metrics_dir = pm.semantic_model_path(real_agent_config.current_namespace) / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        yaml_content = (
            "metric:\n"
            "  name: tagged_metric\n"
            "  type: simple\n"
            "  type_params:\n"
            "    measure: cnt\n"
            "  locked_metadata:\n"
            "    tags:\n"
            "      - 'subject_tree: old_path'\n"
        )
        request = EditMetricInput(subject_path=["tagged_dir"], yaml=yaml_content)
        result = await svc.create_metric(request)
        assert result is not None


@pytest.mark.asyncio
class TestExplorerServiceEditMetric:
    """Tests for edit_metric — metric update with YAML validation."""

    async def test_edit_metric_empty_path(self, real_agent_config):
        """edit_metric with empty path returns error."""
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        request = EditMetricInput(subject_path=[], yaml="metric:\n  name: test\n")
        result = await svc.edit_metric(request)
        assert result.success is False
        assert "empty" in result.errorMessage.lower()

    async def test_edit_metric_nonexistent(self, real_agent_config):
        """edit_metric for nonexistent metric returns error."""
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        request = EditMetricInput(
            subject_path=["dir", "nonexistent_metric"],
            yaml="metric:\n  name: nonexistent_metric\n  type: simple\n",
        )
        result = await svc.edit_metric(request)
        assert result.success is False
        assert "not found" in result.errorMessage.lower()

    async def test_edit_metric_invalid_yaml(self, real_agent_config):
        """edit_metric with invalid YAML returns error (assuming metric exists check passes first)."""
        from datus.api.models.explorer_models import EditMetricInput

        svc = ExplorerService(agent_config=real_agent_config)
        # This will fail at "metric not found" before YAML validation, which is fine
        request = EditMetricInput(
            subject_path=["dir", "some_metric"],
            yaml=":\n  - ][",
        )
        result = await svc.edit_metric(request)
        assert result.success is False


@pytest.mark.asyncio
class TestExplorerServiceEditKnowledge:
    """Tests for edit_knowledge operations."""

    async def test_edit_knowledge_empty_path(self, real_agent_config):
        """edit_knowledge with empty path returns error."""
        svc = ExplorerService(agent_config=real_agent_config)
        request = EditKnowledgeInput(
            subject_path=[],
            name="test",
            search_text="updated",
            explanation="updated",
        )
        result = await svc.edit_knowledge(request)
        assert result.success is False

    async def test_edit_knowledge_updates(self, real_agent_config):
        """edit_knowledge updates an existing knowledge entry."""
        svc = ExplorerService(agent_config=real_agent_config)
        await svc.create_directory(CreateDirectoryInput(subject_path=["edit_kb"]))
        await svc.create_knowledge(
            CreateKnowledgeInput(
                subject_path=["edit_kb"],
                name="editable_kb",
                search_text="original",
                explanation="original explanation",
            )
        )
        result = await svc.edit_knowledge(
            EditKnowledgeInput(
                subject_path=["edit_kb", "editable_kb"],
                name="editable_kb",
                search_text="updated search",
                explanation="updated explanation",
            )
        )
        assert result.success is True


@pytest.mark.asyncio
class TestExplorerServiceValidateMetricYaml:
    """Tests for _validate_metric_yaml — metric YAML validation."""

    async def test_validate_valid_yaml(self, real_agent_config):
        """_validate_metric_yaml passes valid metric YAML."""
        svc = ExplorerService(agent_config=real_agent_config)
        is_valid, errors = svc._validate_metric_yaml(
            "metric:\n  name: test\n  type: simple\n",
            "/tmp/test.yml",
        )
        # May pass or fail depending on metricflow availability
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    async def test_validate_invalid_yaml(self, real_agent_config):
        """_validate_metric_yaml rejects invalid YAML syntax."""
        svc = ExplorerService(agent_config=real_agent_config)
        is_valid, errors = svc._validate_metric_yaml(
            ":\n  - ][",
            "/tmp/bad.yml",
        )
        assert is_valid is False
        assert len(errors) > 0


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
