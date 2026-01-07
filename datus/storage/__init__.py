# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.storage.fastembed_embeddings import FastEmbedEmbeddings

from .base import BaseEmbeddingStore, BaseModelData, StorageBase

__all__ = [
    "BaseEmbeddingStore",
    "StorageBase",
    "BaseModelData",
    "FastEmbedEmbeddings",
]
