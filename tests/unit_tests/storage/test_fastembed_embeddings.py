# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace

import pytest

from datus.storage import fastembed_embeddings
from datus.storage.fastembed_embeddings import FastEmbedEmbeddings


@pytest.mark.parametrize(
    "description",
    [
        # fastembed < 0.5
        {"dim": 384, "sources": {"hf": "qdrant/all-MiniLM-L6-v2-onnx"}},
        # fastembed >= 0.5: DenseModelDescription
        SimpleNamespace(dim=384, sources=SimpleNamespace(hf="qdrant/all-MiniLM-L6-v2-onnx")),
    ],
    ids=["dict", "model-description"],
)
def test_ndims_falls_back_to_the_model_description(monkeypatch, description):
    """Without ``embedding_size`` on the model, the dim comes from its description, in either shape."""
    monkeypatch.setattr(
        fastembed_embeddings.TextEmbedding,
        "_get_model_description",
        classmethod(lambda _cls, _name: description),
    )
    embeddings = FastEmbedEmbeddings()
    embeddings._model_instance = SimpleNamespace()

    assert embeddings.ndims() == 384


@pytest.fixture
def model_factory(monkeypatch, tmp_path):
    """Replace only model construction and snapshot I/O; exercise the real wrappers."""
    from threading import Condition, Event

    cache = getattr(fastembed_embeddings, "_load_embedding_model", FastEmbedEmbeddings._create_embedding_model)
    cache.cache_clear()
    calls = []
    entered = Condition()
    release = Event()
    release.set()

    class Model:
        embedding_size = 2

        def __init__(self, **kwargs):
            with entered:
                calls.append(kwargs)
                entered.notify_all()
            assert release.wait(2), "Model constructor was not released"

        def embed(self, texts, batch_size):
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", Model)
    monkeypatch.setattr(fastembed_embeddings, "check_snapshot", lambda *_: None)
    monkeypatch.setattr(fastembed_embeddings, "_resolve_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(fastembed_embeddings, "get_embedding_device", lambda: "cpu")
    yield SimpleNamespace(calls=calls, entered=entered, release=release, model=Model)
    cache.cache_clear()


@pytest.mark.parametrize("separate_wrappers", [False, True], ids=["same-wrapper", "equivalent-wrappers"])
def test_concurrent_cold_access_shares_one_model(model_factory, separate_wrappers):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    wrappers = [FastEmbedEmbeddings() for _ in range(8 if separate_wrappers else 1)]
    start = Barrier(8)
    model_factory.release.clear()

    def access(index):
        start.wait(timeout=2)
        return wrappers[index % len(wrappers)].embedding_model

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(access, index) for index in range(8)]
        try:
            # Hold the constructor open while all workers pass the start barrier.
            # The old implementation admits eight constructors; a single-flight admits one.
            with model_factory.entered:
                model_factory.entered.wait_for(lambda: len(model_factory.calls) == 8, timeout=0.5)
        finally:
            model_factory.release.set()
        models = [future.result(timeout=2) for future in futures]

    assert len(model_factory.calls) == 1
    assert all(model is models[0] for model in models)
    assert all(wrapper.embedding_model is models[0] for wrapper in wrappers)


def test_batch_size_does_not_split_shared_session(model_factory):
    first = FastEmbedEmbeddings(batch_size=1)
    second = FastEmbedEmbeddings(batch_size=32)
    assert first.embedding_model is second.embedding_model
    assert len(model_factory.calls) == 1
    assert first.generate_embeddings(["one"])[0] == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("option", ["name", "device", "cache_dir"])
def test_session_configuration_isolation(model_factory, option):
    first = FastEmbedEmbeddings()
    second = FastEmbedEmbeddings()
    setattr(second, option, {"name": "other/model", "device": "cuda", "cache_dir": "/other/cache"}[option])
    assert first.embedding_model is not second.embedding_model
    assert len(model_factory.calls) == 2
    assert model_factory.calls[0] != model_factory.calls[1]
    assert all(call["threads"] == 1 and call["local_files_only"] is True for call in model_factory.calls)
    assert (model_factory.calls[1].get("cuda") is True) == (option == "device")


def test_failed_initialization_can_retry(model_factory, monkeypatch):
    from datus.utils.exceptions import DatusException

    first = FastEmbedEmbeddings()
    second = FastEmbedEmbeddings()
    attempts = []

    def fail_once(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("temporary construction failure")
        return model_factory.model(**kwargs)

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", fail_once)
    with pytest.raises(DatusException, match="temporary construction failure"):
        _ = first.embedding_model
    assert first._model_instance is None
    recovered = second.embedding_model
    assert first.embedding_model is recovered
    assert len(attempts) == 2


def test_cache_owns_models_without_retaining_wrappers(model_factory):
    import gc
    import weakref

    first = FastEmbedEmbeddings()
    second = FastEmbedEmbeddings()
    model = first.embedding_model
    assert second.embedding_model is model
    wrapper_ref = weakref.ref(first)
    model_ref = weakref.ref(model)
    del first, model
    gc.collect()
    assert wrapper_ref() is None
    assert second.ndims() == 2

    # Eviction releases the warm-cache reference, but never invalidates a live wrapper.
    replacement = FastEmbedEmbeddings(name="other/model")
    _ = replacement.embedding_model
    assert second.embedding_model is model_ref()
    del second
    gc.collect()
    assert model_ref() is None


def test_snapshot_failure_can_retry_without_constructing_model(model_factory, monkeypatch):
    from datus.utils.exceptions import DatusException, ErrorCode

    wrapper = FastEmbedEmbeddings()
    failure = DatusException(ErrorCode.MODEL_EMBEDDING_ERROR, message="snapshot unavailable")

    def unavailable(*_):
        raise failure

    monkeypatch.setattr(fastembed_embeddings, "check_snapshot", unavailable)
    with pytest.raises(DatusException) as caught:
        _ = wrapper.embedding_model
    assert caught.value is failure
    assert model_factory.calls == []
    assert wrapper._model_instance is None
    monkeypatch.setattr(fastembed_embeddings, "check_snapshot", lambda *_: None)
    assert wrapper.embedding_model.embedding_size == 2
    assert len(model_factory.calls) == 1


def test_model_domain_error_is_preserved(model_factory, monkeypatch):
    from datus.utils.exceptions import DatusException, ErrorCode

    failure = DatusException(ErrorCode.MODEL_EMBEDDING_ERROR, message="model unavailable")

    def unavailable(**_):
        raise failure

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", unavailable)
    wrapper = FastEmbedEmbeddings()
    with pytest.raises(DatusException) as caught:
        _ = wrapper.embedding_model
    assert caught.value is failure
    assert wrapper._model_instance is None
