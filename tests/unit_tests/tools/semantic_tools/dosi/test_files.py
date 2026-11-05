"""Independent file consumers must load historical models before reading fields."""

from types import SimpleNamespace

import pytest
import yaml

from datus.agent.node.semantic_authoring import inspect_osi_semantic_model_inventory
from datus.api.services.database_service import DatasourceService
from datus.storage.semantic_model.artifact_file import artifact_revision
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.files import _file_config, file_engine
from datus.tools.semantic_tools.dosi.runtime import DosiRuntime
from datus.tools.semantic_tools.metric_author import MetricAuthor, MetricAuthoringError
from datus.tools.semantic_tools.osi_document import load_osi_document

MODEL = """version: 0.2.0.dev0
name: shop
datasets:
  - name: orders
    source: orders
    fields:
      - name: amount
        expression: {dialects: [{dialect: ANSI_SQL, expression: amount}]}
metrics:
  - name: revenue
    expression: {dialects: [{dialect: ANSI_SQL, expression: 'SUM(orders.amount)'}]}
    custom_extensions:
      - vendor_name: DATUS
        data: '{"uid":"tz4a98xxat96iws9zmbrgj3a"}'
"""


def historical_model():
    """Historical input is confined to file-upgrade regression tests."""
    model = yaml.safe_load(MODEL)
    version = model.pop("version")
    return yaml.safe_dump({"version": version, "semantic_model": [model]}, sort_keys=False)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "subject" / "semantic_models"
    model = root / "warehouse" / "shop.yml"
    model.parent.mkdir(parents=True)
    model.write_text(historical_model())
    config = SimpleNamespace(
        current_datasource="warehouse",
        project_root=str(tmp_path),
        datasource_configs={"warehouse": SimpleNamespace()},
        path_manager=SimpleNamespace(semantic_models_dir=root, semantic_model_path=lambda ds: root / ds),
    )
    return tmp_path, model, config


@pytest.mark.parametrize("entry", ["inventory", "api_get", "read", "grep", "source", "edit", "runtime", "package"])
def test_first_entry_loads_and_exposes_the_current_file(project, entry):
    root, path, config = project
    original = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    if entry == "inventory":
        result = inspect_osi_semantic_model_inventory(config)
        assert [item["semantic_model_name"] for item in result["models"]] == ["shop"]
    elif entry == "api_get":
        service = DatasourceService.__new__(DatasourceService)
        service.agent_config = config
        result = service.get_semantic_model(relative)
        assert result.success, result.errorMessage
        assert result.data.yaml == path.read_text()
        assert result.data.revision == artifact_revision(path.read_bytes())
    elif entry in {"read", "grep"}:
        tool = FilesystemFuncTool(root_path=str(root), current_node="semantic_modeling")
        result = tool.read_file(relative) if entry == "read" else tool.grep("^name: shop", "subject")
        assert result.success, result.error
        assert "name: shop" in str(result.result)
    elif entry == "package":
        from datus.cli.package_builder import StagedEntry

        content = StagedEntry(arcname=relative, source=path).read_bytes()
        assert yaml.safe_load(content)["name"] == "shop"
        assert content == path.read_bytes()
    elif entry == "source":
        result = load_osi_document(str(path), "shop")
        assert result.metrics[0].uid == "tz4a98xxat96iws9zmbrgj3a"
    elif entry == "edit":
        author = MetricAuthor(str(path))
        metric = yaml.safe_load(MODEL)["metrics"][0]
        metric["description"] = "Edited after file loading"
        author.write("revenue", yaml.safe_dump(metric))
        assert yaml.safe_load(author.read("revenue").text)["description"] == metric["description"]
    else:
        runtime = DosiRuntime(DosiConfig(semantic_models_path=str(path.parent)))
        assert runtime.initialize() == {}
        assert [item.name for item in runtime.list_semantic_models()] == ["orders"]
    document = yaml.safe_load(path.read_text())
    assert document["name"] == "shop"
    assert document["datasets"] == yaml.safe_load(MODEL)["datasets"]
    assert document["metrics"][0]["custom_extensions"] == yaml.safe_load(MODEL)["metrics"][0]["custom_extensions"]
    backups = list(path.parent.glob(".*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_engine_reuse_and_external_historical_replacement(project):
    _, path, _ = project
    first = file_engine(path)
    current, modified = path.read_bytes(), path.stat().st_mtime_ns
    assert file_engine(path) is first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == (current, modified)
    replacement = path.with_suffix(".tmp")
    replacement.write_text(historical_model().replace("name: revenue", "name: gross_revenue"))
    replacement.replace(path)
    second = file_engine(path)
    assert second is not first
    assert [metric["name"] for metric in second.metrics()] == ["gross_revenue"]
    assert yaml.safe_load(path.read_text())["metrics"][0]["name"] == "gross_revenue"


def test_broken_file_does_not_hide_healthy_catalog_and_remains_repairable(project):
    _, path, _ = project
    bad = path.with_name("broken.yml")
    bad.write_text(
        historical_model().replace("name: shop", "name: broken").replace("SUM(orders.amount)", "SUM(orders.missing)")
    )
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(path.parent)))
    errors = runtime.initialize()
    assert list(errors) == [str(bad)]
    assert "missing" in errors[str(bad)]
    assert yaml.safe_load(bad.read_text())["name"] == "broken"
    assert [item.name for item in runtime.list_semantic_models()] == ["orders"]
    bad.write_text(MODEL.replace("name: shop", "name: repaired").replace("name: revenue", "name: other_revenue"))
    assert runtime.initialize() == {}
    assert len(runtime.list_semantic_models()) == 2


def test_author_rejects_concurrent_edits(project):
    _, path, _ = project

    def concurrent_edit(document):
        path.write_text(path.read_text() + "description: Another writer\n")

    author = MetricAuthor(str(path), validate_document=concurrent_edit)
    with pytest.raises(MetricAuthoringError, match="changed during editing"):
        author.delete("revenue")
    document = yaml.safe_load(path.read_text())
    assert document["description"] == "Another writer"
    assert document["metrics"][0]["name"] == "revenue"


def test_file_context_uses_target_datasource_without_mutating_request(project):
    root, _, config = project
    seen = []
    config.runtime_db_context = lambda: {"datasource": "warehouse", "database": "active_db"}
    config.current_db_config = lambda datasource: seen.append(datasource)
    result = _file_config(root / "subject/semantic_models/other/model.yml", config)
    assert result.datasource == "other"
    assert seen == ["other"]
    assert config.current_datasource == "warehouse"
    assert config.runtime_db_context()["database"] == "active_db"
