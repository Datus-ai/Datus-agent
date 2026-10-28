"""Frozen model references are portable, verified, and backward compatible."""

import hashlib

import pytest

from datus.schemas.metric_artifact_query import MetricQueryFile
from datus.utils.metric_artifact_models import resolve_metric_query_snapshot


def recipe(content="semantic_model: []\r\n", **changes):
    revision = hashlib.sha256(content.encode()).hexdigest()
    return MetricQueryFile.model_validate(
        {
            "schema_version": "1.1",
            "name": "sales",
            "goal": "Revenue",
            "metric": {"path": ["Commerce"], "name": "revenue"},
            "datasource": "retail",
            "model_path": "subject/semantic_models/retail/shop.yml",
            "model_revision": revision,
            "model_snapshot_path": f"models/{revision}.yml",
            "metric_detail": {"path": ["Commerce"], "name": "revenue"},
            "captured_at": "2026-10-15",
            "generated_sql": "SELECT SUM(amount) FROM orders",
            **changes,
        }
    )


@pytest.mark.parametrize("source", ["disk", "bundle", "legacy"])
@pytest.mark.parametrize("length", [16, 64])
@pytest.mark.parametrize("suffix", [".yml", ".yaml", ".json"])
def test_resolves_exact_bytes_without_studio_model(tmp_path, source, length, suffix):
    content = "semantic_model: []\r\n"
    revision = hashlib.sha256(content.encode()).hexdigest()
    query = recipe(content, model_snapshot_path=f"models/{revision[:length]}{suffix}")
    options = {}
    if source == "disk":
        target = tmp_path / query.model_snapshot_path
        target.parent.mkdir()
        target.write_bytes(content.encode())
        options["artifact_dir"] = tmp_path
    elif source == "bundle":
        options["files"] = {query.model_snapshot_path: content}
    else:
        query = recipe(content, schema_version="1.0", model_snapshot_path=None, model_snapshot=content)
    resolved = resolve_metric_query_snapshot(query, **options)
    assert resolved.model_snapshot == content
    assert resolved.model_path == "subject/semantic_models/retail/shop.yml"
    assert resolved.model_revision == hashlib.sha256(content.encode()).hexdigest()


@pytest.mark.parametrize("source", ["disk", "bundle", "absent"])
@pytest.mark.parametrize("problem", ["missing", "tampered"])
@pytest.mark.parametrize("length", [16, 64])
def test_reference_never_falls_back_to_inline_model(tmp_path, source, problem, length):
    content = "semantic_model: []\r\n"
    revision = hashlib.sha256(content.encode()).hexdigest()
    query = recipe(content, model_snapshot=content, model_snapshot_path=f"models/{revision[:length]}.yml")
    files = {} if problem == "missing" else {query.model_snapshot_path: "changed model"}
    options = {}
    if source == "disk":
        for relative, value in files.items():
            target = tmp_path / relative
            target.parent.mkdir()
            target.write_text(value)
        options["artifact_dir"] = tmp_path
    elif source == "bundle":
        options["files"] = files
    with pytest.raises(ValueError, match="snapshot missing|must match model_revision"):
        resolve_metric_query_snapshot(query, **options)


def test_legacy_recipe_without_snapshot_cannot_be_published():
    with pytest.raises(ValueError, match="snapshot missing"):
        resolve_metric_query_snapshot(recipe(schema_version="1.0", model_snapshot_path=None))


@pytest.mark.parametrize(
    "path",
    [
        "../model.yml",
        "/tmp/model.yml",
        "models/../model.yml",
        "models\\model.yml",
        "models//other.yml",
        "models/./other.yml",
        "models/sub/other.yml",
        "models/other.txt",
    ],
)
def test_reference_is_confined_to_the_model_bundle_directory(path):
    with pytest.raises(ValueError, match="model file directly under models/"):
        recipe(model_snapshot_path=path)


def test_new_schema_requires_model_reference():
    with pytest.raises(ValueError, match="requires model_snapshot_path"):
        recipe(model_snapshot_path=None)


def test_model_symlink_cannot_escape_artifact(tmp_path):
    query = recipe()
    root = tmp_path / "artifact"
    (root / "models").mkdir(parents=True)
    outside = tmp_path / "private.yml"
    outside.write_text("private")
    (root / query.model_snapshot_path).symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the artifact"):
        resolve_metric_query_snapshot(query, artifact_dir=root)


@pytest.mark.parametrize("name", ["flights.yml", "jeff_shop.yml", "model.yaml", "model.json", "0" * 64 + ".yml"])
@pytest.mark.parametrize("source", ["disk", "bundle"])
def test_model_filename_is_independent_of_content_revision(tmp_path, name, source):
    content = "semantic_model: []\r\n"
    query = recipe(content, model_snapshot_path=f"models/{name}")
    files = {query.model_snapshot_path: content}
    if source == "disk":
        target = tmp_path / query.model_snapshot_path
        target.parent.mkdir()
        target.write_bytes(content.encode())
        options = {"artifact_dir": tmp_path}
    else:
        options = {"files": files}
    resolved = resolve_metric_query_snapshot(query, **options)
    assert resolved.model_snapshot == content
    assert resolved.model_revision == hashlib.sha256(content.encode()).hexdigest()
    # A human-readable name does not weaken content verification.
    if source == "disk":
        target.write_text("changed model")
    else:
        files[query.model_snapshot_path] = "changed model"
    with pytest.raises(ValueError, match="must match model_revision"):
        resolve_metric_query_snapshot(query, **options)
