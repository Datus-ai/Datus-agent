# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import logging
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Union

from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from openai import AzureOpenAI, BadRequestError, OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


@register("openai")
class OpenAIEmbeddings(TextEmbeddingFunction):
    """
    An embedding function that uses the OpenAI API

    https://platform.openai.com/docs/guides/embeddings

    This can also be used for open source models that
    are compatible with the OpenAI API.

    Notes
    -----
    If you're running an Ollama server locally,
    you can just override the `base_url` parameter
    and provide the Ollama embedding model you want
    to use (https://ollama.com/library):

    ```python
    from lancedb.embeddings import get_registry
    openai = get_registry().get("openai")
    embedding_function = openai.create(
        name="<ollama-embedding-model-name>",
        base_url="http://localhost:11434",
        )
    ```

    """

    name: str = "text-embedding-ada-002"
    dim: Optional[int] = None
    base_url: Optional[str] = None
    default_headers: Optional[dict] = None
    organization: Optional[str] = None
    api_key: Optional[str] = None

    # Set true to use Azure OpenAI API
    use_azure: bool = False

    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        if hasattr(super(), name):
            super().__setattr__(name, value)
        else:
            self.__dict__[name] = value
        if name in ["name", "dim", "base_url", "use_azure", "api_key"]:
            logger.debug(f"Attribute {name} changed from {old_value} to {value}")

    def ndims(self):
        return self._ndims

    @staticmethod
    def sensitive_keys():
        return []

    @staticmethod
    def model_names():
        return [
            "text-embedding-ada-002",
            "text-embedding-3-large",
            "text-embedding-3-small",
        ]

    @cached_property
    def _ndims(self):
        if self.name == "text-embedding-ada-002":
            return 1536
        elif self.name == "text-embedding-3-large":
            return self.dim or 3072
        elif self.name == "text-embedding-3-small":
            return self.dim or 1536
        else:
            raise ValueError(f"Unknown model name {self.name}")

    def generate_embeddings(self, texts: Union[List[str], "np.ndarray"]) -> List["np.array"]:
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text:
                valid_texts.append(text)
                valid_indices.append(idx)

        try:
            kwargs = {
                "input": valid_texts,
                "model": self.name,
            }
            if self.name != "text-embedding-ada-002":
                kwargs["dimensions"] = self.dim

            logger.debug(f"Calling OpenAI API with kwargs: {kwargs}")
            rs = self._openai_client.embeddings.create(**kwargs)
            valid_embeddings = {idx: v.embedding for v, idx in zip(rs.data, valid_indices)}
            logger.debug(f"Successfully generated embeddings for {len(valid_embeddings)} texts")
        except BadRequestError:
            logger.error(f"Bad request for texts: {texts}")
            return [None] * len(texts)
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {str(e)}")
            raise
        return [valid_embeddings.get(idx, None) for idx in range(len(texts))]

    @cached_property
    def _openai_client(self) -> OpenAI:
        logger.debug(f"Creating OpenAI client with base_url={self.base_url}, use_azure={self.use_azure}")
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.organization:
            kwargs["organization"] = self.organization
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.use_azure:
            return AzureOpenAI(**kwargs)
        else:
            return OpenAI(**kwargs)
