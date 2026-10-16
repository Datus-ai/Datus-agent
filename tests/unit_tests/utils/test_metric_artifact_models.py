"""Frozen model references are portable, verified, and backward compatible."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from datus.schemas.metric_artifact_query import MetricQueryFile
from datus.utils.metric_artifact_models import resolve_metric_query_snapshot, save_metric_query_snapshot


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
@pytest.mark.parametrize("length", [16, 32, 64])
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
@pytest.mark.parametrize("length", [16, 32, 64])
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
        "models/other.yml",
        "models//other.yml",
    ],
)
def test_reference_is_confined_to_its_content_address(path):
    with pytest.raises(ValueError, match="content-addressed file"):
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


@pytest.mark.parametrize("length", [0, 8, 15, 17, 31, 33, 63, 65])
def test_reference_rejects_unsupported_hash_lengths(length):
    revision = recipe().model_revision
    with pytest.raises(ValueError, match="content-addressed file"):
        recipe(model_snapshot_path=f"models/{(revision + '0')[:length]}.yml")


@pytest.mark.parametrize("length", [16, 32, 64])
def test_reference_must_match_revision_prefix(length):
    with pytest.raises(ValueError, match="content-addressed file"):
        recipe(model_snapshot_path=f"models/{'0' * length}.yml")


@pytest.mark.parametrize("occupied_lengths", [[], [16], [16, 32], [16, 32, 64]])
def test_save_extends_hash_without_overwriting_existing_files(tmp_path, occupied_lengths):
    content = "semantic_model: []\r\n"
    query = recipe(content)
    (tmp_path / "models").mkdir()
    occupied = {}
    for length in occupied_lengths:
        path = tmp_path / f"models/{query.model_revision[:length]}.yml"
        occupied[path] = f"other model at {length}".encode()
        path.write_bytes(occupied[path])
    if 64 in occupied_lengths:
        with pytest.raises(ValueError, match="existing full SHA-256 filename"):
            save_metric_query_snapshot(query, content, artifact_dir=tmp_path)
    else:
        saved = save_metric_query_snapshot(query, content, artifact_dir=tmp_path)
        length = [16, 32, 64][len(occupied_lengths)]
        assert saved.model_snapshot_path == f"models/{query.model_revision[:length]}.yml"
        assert saved.model_revision == hashlib.sha256(content.encode()).hexdigest()
        assert resolve_metric_query_snapshot(saved, artifact_dir=tmp_path).model_snapshot == content
        # A repeated save reuses the same extended name, not a new file.
        assert save_metric_query_snapshot(query, content, artifact_dir=tmp_path) == saved
    for path, original in occupied.items():
        assert path.read_bytes() == original
    assert len(list((tmp_path / "models").iterdir())) == len(occupied_lengths) + (64 not in occupied_lengths)


def test_concurrent_saves_share_one_complete_snapshot(tmp_path):
    content = "semantic_model: []\r\n" * 1000
    query = recipe(content)

    def save_and_resolve(_):
        saved = save_metric_query_snapshot(query, content, artifact_dir=tmp_path)
        return saved, resolve_metric_query_snapshot(saved, artifact_dir=tmp_path).model_snapshot

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(save_and_resolve, range(24)))
    assert {saved.model_snapshot_path for saved, _ in results} == {f"models/{query.model_revision[:16]}.yml"}
    assert all(snapshot == content for _, snapshot in results)
    assert len(list((tmp_path / "models").iterdir())) == 1


@pytest.mark.parametrize("problem", ["no_reference", "tampered", "directory_escape", "file_escape", "link_failure"])
def test_save_rejects_invalid_snapshot_without_leaving_files(tmp_path, monkeypatch, problem):
    import datus.utils.metric_artifact_models as models

    content = "semantic_model: []\r\n"
    query = recipe(content)
    root = tmp_path / "artifact"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if problem == "no_reference":
        query = recipe(content, schema_version="1.0", model_snapshot_path=None, model_snapshot=content)
    elif problem == "tampered":
        content = "changed model"
    elif problem == "directory_escape":
        (root / "models").symlink_to(outside, target_is_directory=True)
    elif problem == "file_escape":
        (root / "models").mkdir()
        (root / f"models/{query.model_revision[:16]}.yml").symlink_to(outside / "model.yml")
    else:

        def fail_link(*args):
            raise OSError("link failed")

        monkeypatch.setattr(models.os, "link", fail_link)
    with pytest.raises(
        (ValueError, OSError), match="requires model_snapshot_path|match model_revision|escapes|link failed"
    ):
        save_metric_query_snapshot(query, content, artifact_dir=root)
    assert list(outside.iterdir()) == []
    assert list(root.glob("models/.model-*")) == []
