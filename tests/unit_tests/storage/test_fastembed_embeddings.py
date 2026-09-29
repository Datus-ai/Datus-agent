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
