"""Tests for datus.api.services.database_service — datasource management."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

from datus.api.models.base_models import Result
from datus.api.models.database_models import ListDatabasesInput
from datus.api.models.table_models import (
    SaveSemanticModelInput,
    ValidateSemanticModelData,
    ValidateSemanticModelInput,
)
from datus.api.services.database_service import DatasourceService
from datus.storage.semantic_model.artifact_file import artifact_revision, semantic_artifact_lock
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.func_tool.metric_filesystem_tools import MetricFilesystemFuncTool
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


def _service_with_semantic_adapter(
    adapter: str = "metricflow", *, models_root: Path | None = None
) -> DatasourceService:
    svc = DatasourceService.__new__(DatasourceService)
    svc.agent_config = SimpleNamespace(
        home="/datus-home",
        current_datasource="warehouse",
        resolve_semantic_adapter=lambda: adapter,
        path_manager=SimpleNamespace(semantic_models_dir=models_root),
    )
    return svc


def _write_model(
    models_root: Path,
    content: str,
    *,
    datasource: str = "warehouse",
    name: str = "orders.yml",
) -> tuple[Path, str]:
    """Create one artifact and return ``(absolute path, API selector)``."""

    target = models_root / datasource / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target, f"subject/semantic_models/{datasource}/{name}"


def _osi_yaml(*, metric_name: str = "") -> str:
    metrics = f"\n    metrics:\n      - name: {metric_name}" if metric_name else "\n    metrics: []"
    return (
        "version: 1.0.0\n"
        "semantic_model:\n"
        "  - name: orders\n"
        "    datasets:\n"
        "      - name: orders\n"
        "        source: orders"
        f"{metrics}\n"
    )


class TestDatasourceServiceInit:
    """Tests for DatasourceService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """DatasourceService initializes with real agent config."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert isinstance(svc, DatasourceService)
        assert svc.current_db_connector.get_type() == "sqlite"

    def test_init_sets_current_db_name(self, real_agent_config):
        """Init resolves the current database name from the datasource."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert svc.current_db_name == "california_schools"

    def test_init_sets_datasource(self, real_agent_config):
        """Init stores current_datasource from config."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert svc.current_datasource == real_agent_config.current_datasource

    def test_db_manager_created(self, real_agent_config):
        """Init creates DBManager."""
        svc = DatasourceService(agent_config=real_agent_config)
        assert isinstance(svc.db_manager, DBManager)

    def test_init_without_datasource(self, real_agent_config):
        """Init tolerates a config with no datasource selected."""
        real_agent_config.current_datasource = ""

        svc = DatasourceService(agent_config=real_agent_config)

        assert svc.current_datasource == ""


class TestDatabaseServiceGetDatabaseType:
    """Tests for _get_database_type helper."""

    def test_known_database_returns_type(self, real_agent_config):
        """Known database returns its type string."""
        svc = DatasourceService(agent_config=real_agent_config)
        db_type, ds_id = svc._get_database_type("california_schools")
        assert db_type == "sqlite"

    def test_current_db_name_used_as_default(self, real_agent_config):
        """Without database_name arg, uses current_db_name."""
        svc = DatasourceService(agent_config=real_agent_config)
        db_type, ds_id = svc._get_database_type()
        assert db_type == "sqlite"
        assert ds_id == svc.current_db_name


class TestSemanticLayerServiceBranches:
    def test_active_semantic_adapter_normalizes_resolved_name(self):
        svc = _service_with_semantic_adapter(" OSI ")

        assert svc._active_semantic_adapter() == "osi"
        # Plain OSI is query-only now; only Dosi counts as the authoring layer.
        assert svc._is_osi_semantic_layer() is False

    def test_dosi_is_classified_as_osi_semantic_layer(self):
        svc = _service_with_semantic_adapter(" DOSI ")

        assert svc._active_semantic_adapter() == "dosi"
        assert svc._is_osi_semantic_layer() is True

    def test_active_semantic_adapter_returns_empty_without_resolver(self):
        svc = DatasourceService.__new__(DatasourceService)
        svc.agent_config = SimpleNamespace()

        assert svc._active_semantic_adapter() == ""
        assert svc._is_osi_semantic_layer() is False

    @pytest.mark.asyncio
    async def test_validate_semantic_model_uses_dosi_validator(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        _yaml_file, selector = _write_model(tmp_path, _osi_yaml())
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(False, ["native Dosi validation failed"]))
        request = ValidateSemanticModelInput(
            semantic_model_file=selector,
            yaml=_osi_yaml(),
        )

        result = await svc.validate_semantic_model(request)

        assert result.success is True
        assert result.data == ValidateSemanticModelData(
            valid=False,
            invalid_message=["native Dosi validation failed"],
        )
        svc._validate_dosi_semantic_yaml.assert_called_once_with(request.yaml)

    @pytest.mark.asyncio
    async def test_validate_semantic_model_rejects_non_dosi_project(self, tmp_path):
        """Contract: semantic authoring is Dosi-only — the validate endpoint
        refuses non-Dosi projects with the query-only migration message."""
        svc = _service_with_semantic_adapter("metricflow", models_root=tmp_path)
        _yaml_file, selector = _write_model(tmp_path, "semantic_model:\n  name: orders\n")

        result = await svc.validate_semantic_model(
            ValidateSemanticModelInput(semantic_model_file=selector, yaml="semantic_model:\n  name: orders\n")
        )

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "query-only" in result.errorMessage

    @pytest.mark.asyncio
    async def test_validate_semantic_model_rejects_unknown_file(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)

        result = await svc.validate_semantic_model(
            ValidateSemanticModelInput(
                semantic_model_file="subject/semantic_models/warehouse/ghost.yml",
                yaml=_osi_yaml(),
            )
        )

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "not found" in result.errorMessage

    @pytest.mark.asyncio
    async def test_save_semantic_model_uses_osi_sync_tool(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        yaml_file, selector = _write_model(
            tmp_path,
            "version: 1.0.0\nsemantic_model:\n  - name: orders\n    datasets: []\n",
        )
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))
        request = SaveSemanticModelInput(
            semantic_model_file=selector,
            yaml=(
                "version: 1.0.0\nsemantic_model:\n  - name: orders\n    datasets:\n"
                "      - name: orders\n        source: orders\n    metrics: []\n"
            ),
        )

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {
                "success": True,
                "semantic_objects": 1,
                "metric_names": [],
            }
            result = await svc.save_semantic_model(request)

        assert result.success is True
        assert yaml_file.read_text(encoding="utf-8") == request.yaml
        assert result.data.status == "synced"
        assert result.data.revision == artifact_revision(request.yaml.encode())
        tools_cls.assert_called_once_with(agent_config=svc.agent_config, authoring_format="osi")
        tools_cls.return_value.sync_osi_to_db.assert_called_once_with(
            str(yaml_file),
            include_semantic_objects=True,
            include_metrics=True,
        )

    @pytest.mark.asyncio
    async def test_save_semantic_model_rejects_non_dosi_project_before_writing(self, tmp_path):
        """Contract: non-Dosi projects are query-only — save must fail before
        reading, validating, or writing anything, so the artifact on disk is
        untouched."""
        original = "semantic_model:\n  name: orders\n"
        svc = _service_with_semantic_adapter("metricflow", models_root=tmp_path)
        yaml_file, selector = _write_model(tmp_path, original)

        result = await svc.save_semantic_model(
            SaveSemanticModelInput(semantic_model_file=selector, yaml="semantic_model:\n  name: updated_orders\n")
        )

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "query-only" in result.errorMessage
        assert yaml_file.read_text(encoding="utf-8") == original

    @pytest.mark.asyncio
    async def test_save_semantic_model_rejects_stale_revision(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        original = _osi_yaml()
        yaml_file, selector = _write_model(tmp_path, original)

        result = await svc.save_semantic_model(
            SaveSemanticModelInput(
                semantic_model_file=selector,
                yaml=_osi_yaml(metric_name="order_count"),
                expected_revision="sha256:stale",
            )
        )

        assert result.success is False
        assert result.errorCode == "SEMANTIC_MODEL_REVISION_CONFLICT"
        assert result.data.status == "conflict"
        assert result.data.revision == artifact_revision(original.encode())
        assert yaml_file.read_text() == original

    def test_api_save_serializes_against_agent_metric_mutation(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        original = _osi_yaml()
        updated = _osi_yaml(metric_name="api_metric")
        yaml_file, selector = _write_model(tmp_path, original)
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(True, []))

        validation_started = threading.Event()
        release_validation = threading.Event()

        def validate_live_candidate(*_args, **_kwargs):
            validation_started.set()
            assert release_validation.wait(timeout=5)
            return True, {"valid": True, "issues": []}, ""

        svc._full_osi_validation = MagicMock(side_effect=validate_live_candidate)
        target_state = OsiSemanticModelTargetState()
        target_state.select(
            {
                "semantic_model_name": "orders",
                "semantic_model_file": "subject/semantic_models/warehouse/orders.yml",
                "absolute_path": str(yaml_file.resolve()),
                "artifact_sha256": artifact_revision(original.encode()).removeprefix("sha256:"),
            },
            mode="bound",
        )
        agent_tool = MetricFilesystemFuncTool(
            root_path=str(yaml_file.parent),
            current_node="gen_metrics",
            osi_target_state=target_state,
        )
        agent_lock_attempted = threading.Event()

        @contextmanager
        def observed_agent_lock(path):
            agent_lock_attempted.set()
            with semantic_artifact_lock(path):
                yield

        def mutate_from_agent():
            return agent_tool.upsert_osi_metrics("orders.yml", '[{"name":"agent_metric"}]')

        with (
            patch(
                "datus.tools.func_tool.metric_filesystem_tools.semantic_artifact_lock",
                observed_agent_lock,
            ),
            patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls,
        ):
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": True}
            with ThreadPoolExecutor(max_workers=2) as executor:
                api_future = executor.submit(
                    svc._save_semantic_model_sync,
                    SaveSemanticModelInput(
                        semantic_model_file=selector,
                        yaml=updated,
                        expected_revision=artifact_revision(original.encode()),
                    ),
                )
                assert validation_started.wait(timeout=5)
                agent_future = executor.submit(mutate_from_agent)
                assert agent_lock_attempted.wait(timeout=5)
                try:
                    assert not agent_future.done()
                finally:
                    release_validation.set()
                api_result = api_future.result(timeout=5)
                agent_result = agent_future.result(timeout=5)

        assert api_result.success is True
        assert agent_result.success == 0
        assert "changed after selection" in agent_result.error
        assert yaml_file.read_text() == updated

    @pytest.mark.asyncio
    async def test_save_semantic_model_restores_yaml_after_full_validation_failure(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        original = _osi_yaml()
        yaml_file, selector = _write_model(tmp_path, original)
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(
            return_value=(False, {"valid": False, "issues": [{"message": "bad metric"}]}, "bad metric")
        )

        result = await svc.save_semantic_model(
            SaveSemanticModelInput(semantic_model_file=selector, yaml=_osi_yaml(metric_name="bad_metric"))
        )

        assert result.success is False
        assert result.errorCode == "SEMANTIC_MODEL_INVALID"
        assert result.data.yaml_saved is False
        assert yaml_file.read_text() == original

    @pytest.mark.asyncio
    async def test_save_semantic_model_restores_yaml_when_full_validation_raises(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        original = _osi_yaml()
        yaml_file, selector = _write_model(tmp_path, original)
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(side_effect=RuntimeError("validator unavailable"))

        result = await svc.save_semantic_model(
            SaveSemanticModelInput(semantic_model_file=selector, yaml=_osi_yaml(metric_name="order_count"))
        )

        assert result.success is False
        assert result.errorCode == "INTERNAL_COMMAND_ERROR"
        assert result.data.retryable is True
        assert result.data.yaml_saved is False
        assert yaml_file.read_text() == original

    @pytest.mark.asyncio
    async def test_save_semantic_model_keeps_valid_yaml_when_sync_fails(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        yaml_file, selector = _write_model(tmp_path, _osi_yaml())
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))
        updated = _osi_yaml(metric_name="order_count")

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": False, "error": "storage down"}
            result = await svc.save_semantic_model(SaveSemanticModelInput(semantic_model_file=selector, yaml=updated))

        assert result.success is False
        assert result.errorCode == "SEMANTIC_MODEL_SYNC_FAILED"
        assert result.data.status == "saved_not_synced"
        assert result.data.retryable is True
        assert result.data.revision == artifact_revision(updated.encode())
        assert yaml_file.read_text() == updated

    @pytest.mark.asyncio
    async def test_save_semantic_model_unchanged_yaml_still_repairs_kb(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        content = _osi_yaml()
        _yaml_file, selector = _write_model(tmp_path, content)
        svc._validate_dosi_semantic_yaml = MagicMock(return_value=(True, []))
        svc._full_osi_validation = MagicMock(return_value=(True, {"valid": True, "issues": []}, ""))

        with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
            tools_cls.return_value.sync_osi_to_db.return_value = {"success": True}
            result = await svc.save_semantic_model(
                SaveSemanticModelInput(
                    semantic_model_file=selector,
                    yaml=content,
                    expected_revision=artifact_revision(content.encode()),
                )
            )

        assert result.success is True
        tools_cls.return_value.sync_osi_to_db.assert_called_once()


class TestGetSemanticModel:
    """Tests for the file-addressed get_semantic_model."""

    def test_get_semantic_model_returns_stable_file_identity_and_revision(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        content = _osi_yaml()
        _yaml_file, selector = _write_model(tmp_path, content)

        result = svc.get_semantic_model(selector)

        assert result.success is True
        assert result.data.yaml == content
        assert result.data.semantic_model_name == "orders"
        assert result.data.semantic_model_file == "subject/semantic_models/warehouse/orders.yml"
        assert result.data.revision == artifact_revision(content.encode())

    def test_get_semantic_model_reaches_a_non_active_datasource(self, tmp_path):
        """Resolution spans the whole tree, not just the first configured datasource."""
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        content = _osi_yaml()
        _yaml_file, selector = _write_model(tmp_path, content, datasource="lakehouse")

        result = svc.get_semantic_model(selector)

        assert result.success is True
        assert result.data.semantic_model_file == "subject/semantic_models/lakehouse/orders.yml"

    @pytest.mark.asyncio
    async def test_save_semantic_model_rejects_a_non_active_datasource(self, tmp_path):
        """Reads span every datasource; writes must stay on the active one.

        The validators and the knowledge-base sync are all scoped to
        ``current_datasource``, so saving another datasource's artifact would
        file its rows under the wrong datasource.
        """
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        _yaml_file, selector = _write_model(tmp_path, _osi_yaml(), datasource="lakehouse")

        result = await svc.save_semantic_model(SaveSemanticModelInput(semantic_model_file=selector, yaml=_osi_yaml()))

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "lakehouse" in result.errorMessage
        assert "warehouse" in result.errorMessage

    @pytest.mark.asyncio
    async def test_validate_semantic_model_rejects_a_non_active_datasource(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        _yaml_file, selector = _write_model(tmp_path, _osi_yaml(), datasource="lakehouse")

        result = await svc.validate_semantic_model(
            ValidateSemanticModelInput(semantic_model_file=selector, yaml=_osi_yaml())
        )

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"

    def test_get_semantic_model_accepts_a_selector_without_the_subject_prefix(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        _yaml_file, _selector = _write_model(tmp_path, _osi_yaml())

        result = svc.get_semantic_model("warehouse/orders.yml")

        assert result.success is True
        assert result.data.semantic_model_file == "subject/semantic_models/warehouse/orders.yml"

    def test_get_semantic_model_rejects_unknown_file(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)

        result = svc.get_semantic_model("subject/semantic_models/warehouse/ghost.yml")

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert isinstance(result, Result)

    def test_get_semantic_model_rejects_absolute_path(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        yaml_file, _selector = _write_model(tmp_path, _osi_yaml())

        result = svc.get_semantic_model(str(yaml_file))

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "project-relative" in result.errorMessage

    def test_get_semantic_model_rejects_non_yaml_suffix(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)
        (tmp_path / "warehouse").mkdir(parents=True, exist_ok=True)
        (tmp_path / "warehouse" / "orders.txt").write_text("nope")

        result = svc.get_semantic_model("subject/semantic_models/warehouse/orders.txt")

        assert result.success is False
        assert ".yml" in result.errorMessage

    @pytest.mark.asyncio
    async def test_save_semantic_model_rejects_file_escape(self, tmp_path):
        svc = _service_with_semantic_adapter("dosi", models_root=tmp_path)

        result = await svc.save_semantic_model(
            SaveSemanticModelInput(
                yaml=_osi_yaml(),
                semantic_model_file="../outside.yml",
                semantic_model_name="orders",
            )
        )

        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"
        assert "escapes" in result.errorMessage


class TestListDatabases:
    """Tests for list_databases with real SQLite connection."""

    def test_list_databases_returns_success(self, real_agent_config):
        """list_databases returns success with database info."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert result.success is True
        assert result.data.total_count == len(result.data.databases)
        assert result.data.total_count >= 1

    def test_list_databases_has_entries(self, real_agent_config):
        """list_databases returns at least one database entry."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert len(result.data.databases) >= 1

    def test_list_databases_connection_status(self, real_agent_config):
        """Databases are connected."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        for db in result.data.databases:
            assert db.connection_status == "connected"

    def test_list_databases_has_tables(self, real_agent_config):
        """Connected databases report table count > 0."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        connected_databases = [db for db in result.data.databases if db.connection_status == "connected"]
        assert connected_databases
        assert all(db.tables_count > 0 for db in connected_databases)

    def test_list_databases_with_datasource_filter(self, real_agent_config):
        """list_databases with datasource_id filter."""
        svc = DatasourceService(agent_config=real_agent_config)
        # datasource_id is a datasource name
        request = ListDatabasesInput(datasource_id="california_schools")
        result = svc.list_databases(request)
        assert result.success is True

    def test_list_databases_with_database_name_filter(self, real_agent_config):
        """list_databases with database_name filter narrows results."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput(database_name="main")
        result = svc.list_databases(request)
        assert result.success is True

    def test_list_databases_has_tables_list(self, real_agent_config):
        """list_databases includes tables list in database info."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        databases_with_tables = [db for db in result.data.databases if db.tables is not None]
        assert databases_with_tables
        assert all(isinstance(db.tables, list) for db in databases_with_tables)

    def test_list_databases_has_type_field(self, real_agent_config):
        """list_databases includes database type."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        for db in result.data.databases:
            assert db.type == "sqlite"

    def test_list_databases_has_current_database(self, real_agent_config):
        """list_databases data includes current_database field."""
        svc = DatasourceService(agent_config=real_agent_config)
        request = ListDatabasesInput()
        result = svc.list_databases(request)
        assert result.data.current_database == "california_schools"


class _FakeServerConnector:
    """No-schema (server-style) connector that distinguishes its configured
    database from every database reachable on the instance.

    ``get_databases`` mimics ``SHOW DATABASES`` (the whole server); a scoped
    listing must NOT call it when a database is configured.
    """

    dialect = "starrocks"
    catalog_name = "default_catalog"
    connection_string = "mysql+pymysql://u:p@host:9030/benchmark"

    def __init__(self, database_name: str):
        self.database_name = database_name
        self.get_databases_calls = 0

    def test_connection(self) -> bool:  # audit-noqa: zero_assert_test — connector API stub, not a test
        return True

    def get_databases(self, catalog_name: str = "", include_sys: bool = False):
        self.get_databases_calls += 1
        return ["benchmark", "ga4", "olist", "fund_poc"]

    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        return ["t2", "t1"]


@pytest.fixture
def _no_schema_dialect(monkeypatch):
    """Force the server-style (no per-database schema) code path."""
    from datus_db_core import connector_registry

    monkeypatch.setattr(connector_registry, "support_schema", lambda dialect: False)


class TestGetConnectionInfoScoping:
    """A datasource is a connection profile scoped to its configured database;
    listing must not leak every database on the server."""

    def test_configured_database_is_listed_without_enumerating_server(self, real_agent_config, _no_schema_dialect):
        """With a configured database, only that database is returned and the
        server-wide ``get_databases`` enumeration is never invoked."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="benchmark")

        infos = svc._get_connection_info(connector, "benchmark", ListDatabasesInput())

        assert [i.name for i in infos] == ["benchmark"]
        assert connector.get_databases_calls == 0
        assert infos[0].current is True
        # tables are surfaced (and sorted) for the scoped database
        assert infos[0].tables == ["t1", "t2"]

    def test_falls_back_to_server_enumeration_when_unconfigured(self, real_agent_config, _no_schema_dialect):
        """Only when no database is configured do we enumerate the server so the
        connection's reachable databases stay browsable."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="")

        infos = svc._get_connection_info(connector, "ds", ListDatabasesInput())

        assert connector.get_databases_calls == 1
        assert [i.name for i in infos] == ["benchmark", "ga4", "olist", "fund_poc"]

    def test_request_database_name_filter_takes_precedence(self, real_agent_config, _no_schema_dialect):
        """An explicit database_name filter wins over the configured database and
        still avoids the server-wide enumeration."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="benchmark")

        infos = svc._get_connection_info(connector, "benchmark", ListDatabasesInput(database_name="ga4"))

        assert [i.name for i in infos] == ["ga4"]
        assert connector.get_databases_calls == 0


class _FakeSchemaConnector:
    """Schema-capable connector whose per-database and per-schema listings can
    each be made to fail independently."""

    dialect = "postgresql"
    catalog_name = None
    connection_string = "postgresql://u:p@host:5432/warehouse"

    def __init__(self, database_name: str, failing_schemas_db: str = "", failing_tables_schema: str = ""):
        self.database_name = database_name
        self._failing_schemas_db = failing_schemas_db
        self._failing_tables_schema = failing_tables_schema

    def get_effective_capabilities(self) -> set[str]:
        return {"database", "schema"}

    def test_connection(self) -> bool:  # audit-noqa: zero_assert_test — connector API stub, not a test
        return True

    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> list[str]:
        return ["warehouse", "staging"]

    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> list[str]:
        if database_name == self._failing_schemas_db:
            raise RuntimeError("schema enumeration timed out")
        return ["public", "reporting"]

    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> list[str]:
        if schema_name == self._failing_tables_schema:
            raise RuntimeError("table enumeration timed out")
        return ["t2", "t1"]


class TestGetConnectionInfoListingFailure:
    """A reachable database whose objects cannot be listed is not a disconnected one."""

    def test_table_listing_failure_stays_connected_and_reports_the_error(self, real_agent_config, _no_schema_dialect):
        """Reporting ``disconnected`` hid the real cause and contradicted the agent,
        which keeps querying the same database successfully."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="benchmark")

        def _raise(catalog_name: str = "", database_name: str = "", schema_name: str = "") -> NoReturn:
            raise RuntimeError("THRIFT_EAGAIN (timed out)")

        connector.get_tables = _raise

        infos = svc._get_connection_info(connector, "benchmark", ListDatabasesInput())

        assert [i.name for i in infos] == ["benchmark"]
        assert infos[0].connection_status == "connected"
        assert infos[0].schema_name is None
        assert infos[0].tables is None
        assert infos[0].tables_count is None
        assert "THRIFT_EAGAIN" in infos[0].error

    def test_database_enumeration_failure_reports_the_error(self, real_agent_config, _no_schema_dialect):
        """Without a configured database there is nothing left to iterate, so the
        datasource is reported once — connected, with the reason attached."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="")

        def _raise(catalog_name: str = "", include_sys: bool = False) -> NoReturn:
            raise RuntimeError("SHOW DATABASES timed out")

        connector.get_databases = _raise

        infos = svc._get_connection_info(connector, "ds", ListDatabasesInput())

        assert len(infos) == 1
        assert infos[0].connection_status == "connected"
        assert infos[0].schema_name is None
        assert infos[0].tables is None
        assert infos[0].tables_count is None
        assert "SHOW DATABASES timed out" in infos[0].error

    def test_schema_resolution_failure_does_not_abort_sibling_databases(self, real_agent_config):
        """One database that cannot resolve its schemas must not cost the others
        their listing."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeSchemaConnector(database_name="", failing_schemas_db="staging")

        infos = svc._get_connection_info(connector, "ds", ListDatabasesInput())

        failed = [i for i in infos if i.name == "staging"]
        assert len(failed) == 1
        assert failed[0].connection_status == "connected"
        assert failed[0].schema_name is None
        assert failed[0].tables is None
        assert failed[0].tables_count is None
        assert "schema enumeration timed out" in failed[0].error

        healthy = [i for i in infos if i.name == "warehouse"]
        assert [i.schema_name for i in healthy] == ["public", "reporting"]
        assert all(i.connection_status == "connected" and i.error is None for i in healthy)
        assert all(i.tables == ["t1", "t2"] and i.tables_count == 2 for i in healthy)

    def test_table_fetch_failure_is_scoped_to_its_schema(self, real_agent_config):
        """A failing schema is reported with its own name; sibling schemas still list."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeSchemaConnector(database_name="warehouse", failing_tables_schema="reporting")

        infos = svc._get_connection_info(connector, "ds", ListDatabasesInput())

        assert [i.name for i in infos] == ["warehouse", "warehouse"]
        assert [i.schema_name for i in infos] == ["public", "reporting"]

        assert infos[0].error is None
        assert infos[0].tables == ["t1", "t2"]
        assert infos[0].tables_count == 2

        assert infos[1].connection_status == "connected"
        assert infos[1].tables is None
        assert infos[1].tables_count is None
        assert "table enumeration timed out" in infos[1].error

    def test_requested_schema_filter_is_honoured_when_its_tables_fail(self, real_agent_config):
        """An explicit schema_name skips resolution, so the filter is what fails."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeSchemaConnector(database_name="warehouse", failing_tables_schema="reporting")

        infos = svc._get_connection_info(connector, "ds", ListDatabasesInput(schema_name="reporting"))

        assert len(infos) == 1
        assert infos[0].name == "warehouse"
        assert infos[0].schema_name == "reporting"
        assert infos[0].connection_status == "connected"
        assert infos[0].tables is None
        assert "table enumeration timed out" in infos[0].error

    def test_failed_connection_test_is_still_disconnected(self, real_agent_config, _no_schema_dialect):
        """The disconnected status stays reserved for an unusable connection."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = _FakeServerConnector(database_name="benchmark")
        connector.test_connection = lambda: False

        infos = svc._get_connection_info(connector, "benchmark", ListDatabasesInput())

        assert infos[0].connection_status == "disconnected"
        assert infos[0].error is None


class TestGetTableSchema:
    """Tests for get_table_schema with real SQLite connection."""

    def test_get_table_schema_returns_columns(self, real_agent_config):
        """get_table_schema returns column info for existing table."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("schools")
        assert result.success is True
        assert result.data.table.name == "schools"
        assert [col.name for col in result.data.table.columns[:2]] == ["CDSCode", "NCESDist"]

    def test_get_table_schema_column_has_name_and_type(self, real_agent_config):
        """Each column has name and type fields."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("schools")
        for col in result.data.table.columns:
            assert col.name != ""
            assert col.type != ""

    def test_get_table_schema_uses_connector_nullable_contract(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        svc.current_db_connector.get_schema = MagicMock(
            return_value=[
                {
                    "name": "id",
                    "type": "BIGINT",
                    "nullable": False,
                    "default_value": None,
                    "pk": False,
                }
            ]
        )

        result = svc.get_table_schema("orders")

        assert result.success is True
        assert result.data.table.columns[0].nullable is False

    def test_get_table_schema_nonexistent_table(self, real_agent_config):
        """Nonexistent table returns failure."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_table_schema("totally_fake_table_xyz")
        assert result.success is False

    def test_get_table_schema_caches_columns(self, real_agent_config):
        """Second lookup is served from cache without re-hitting the connector."""
        svc = DatasourceService(agent_config=real_agent_config)
        spy = MagicMock(wraps=svc.current_db_connector.get_schema)
        svc.current_db_connector.get_schema = spy

        first = svc.get_table_schema("schools")
        second = svc.get_table_schema("schools")

        assert first.success is True and second.success is True
        assert [c.name for c in second.data.table.columns] == [c.name for c in first.data.table.columns]
        assert spy.call_count == 1


class TestGetTablesColumns:
    """Tests for the batch get_tables_columns (autocomplete prefetch)."""

    def test_returns_columns_for_known_tables(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_tables_columns(["schools"])
        assert result.success is True
        assert [t.table for t in result.data.tables] == ["schools"]
        col = result.data.tables[0].columns[0]
        assert col.name != "" and col.type != ""
        # Slim shape: no default_value in the prefetch payload.
        assert not hasattr(col, "default_value")

    def test_omits_unresolved_tables(self, real_agent_config):
        """A bad name is skipped rather than failing the whole batch."""
        svc = DatasourceService(agent_config=real_agent_config)
        result = svc.get_tables_columns(["schools", "totally_fake_table_xyz"])
        assert result.success is True
        assert [t.table for t in result.data.tables] == ["schools"]

    def test_populates_shared_cache(self, real_agent_config):
        """Columns fetched by the batch are reused by a later single-table detail."""
        svc = DatasourceService(agent_config=real_agent_config)
        spy = MagicMock(wraps=svc.current_db_connector.get_schema)
        svc.current_db_connector.get_schema = spy

        svc.get_tables_columns(["schools"])
        detail = svc.get_table_schema("schools")

        assert detail.success is True
        assert spy.call_count == 1

    def test_over_limit_returns_validation_error(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        svc.agent_config.api_config = {"max_prefetch_tables": 1}
        result = svc.get_tables_columns(["schools", "frpm"])
        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"


class TestSchemaOnlyDialectCatalog:
    """A dialect whose only namespace is the schema (Oracle) must still list.

    Such a connector reports no database and an empty ``get_databases()`` by
    design, which used to leave ``databases: []`` — the datasource vanished from
    the catalog while its tables were perfectly listable.
    """

    @staticmethod
    def _connector(schemas, tables, *, default_schema="APP", fail_on=None):
        connector = MagicMock()
        connector.dialect = "oracle"
        connector.connection_string = "oracle+oracledb://user:pw@host:1521/?service_name=FREEPDB1"
        connector.database_name = ""
        connector.schema_name = default_schema
        connector.catalog_name = None
        connector.test_connection.return_value = True
        connector.get_databases.return_value = []
        connector.get_schemas.return_value = schemas

        def _tables(catalog_name="", database_name="", schema_name=""):
            if fail_on and schema_name == fail_on:
                raise RuntimeError("ORA-00942: table or view does not exist")
            return list(tables.get(schema_name, []))

        connector.get_tables.side_effect = _tables
        return connector

    @staticmethod
    @contextmanager
    def _namespaces(*supported):
        with patch(
            "datus.api.services.database_service.supports_namespace",
            side_effect=lambda namespace, **_: namespace in supported,
        ):
            yield

    def test_schemas_become_the_catalog_roots(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        connector = self._connector(["APP", "REPORTING"], {"APP": ["ORDERS", "CUSTOMERS"], "REPORTING": ["KPI"]})

        with self._namespaces("schema"):
            infos = svc._get_connection_info(connector, "oracle_ds", ListDatabasesInput())

        assert [i.name for i in infos] == ["APP", "REPORTING"]
        # Already the root — nesting it under itself would print the name twice.
        assert all(i.schema_name is None for i in infos)
        assert infos[0].tables == ["CUSTOMERS", "ORDERS"]
        assert infos[0].tables_count == 2
        assert [i.current for i in infos] == [True, False]
        assert all(i.connection_status == "connected" for i in infos)

    def test_one_unreadable_schema_does_not_hide_the_others(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        connector = self._connector(["APP", "LOCKED"], {"APP": ["ORDERS"]}, fail_on="LOCKED")

        with self._namespaces("schema"):
            infos = svc._get_connection_info(connector, "oracle_ds", ListDatabasesInput())

        assert [i.name for i in infos] == ["APP", "LOCKED"]
        assert infos[0].tables == ["ORDERS"]
        # Reachable but unlistable: reported with its reason, not as disconnected.
        assert infos[1].connection_status == "connected"
        assert "ORA-00942" in infos[1].error
        assert infos[1].tables_count is None

    def test_requested_schema_narrows_the_listing(self, real_agent_config):
        svc = DatasourceService(agent_config=real_agent_config)
        connector = self._connector(["APP", "REPORTING"], {"REPORTING": ["KPI"]})

        with self._namespaces("schema"):
            infos = svc._get_connection_info(connector, "oracle_ds", ListDatabasesInput(schema_name="REPORTING"))

        assert [i.name for i in infos] == ["REPORTING"]
        connector.get_schemas.assert_not_called()

    def test_dialect_with_a_database_level_keeps_the_nested_shape(self, real_agent_config):
        """Regression: the fallback must not flatten PostgreSQL-shaped catalogs."""
        svc = DatasourceService(agent_config=real_agent_config)
        connector = self._connector(["public"], {"public": ["orders"]}, default_schema="public")
        connector.dialect = "postgresql"
        connector.database_name = "shop"

        with self._namespaces("database", "schema"):
            infos = svc._get_connection_info(connector, "pg_ds", ListDatabasesInput())

        assert [(i.name, i.schema_name) for i in infos] == [("shop", "public")]
