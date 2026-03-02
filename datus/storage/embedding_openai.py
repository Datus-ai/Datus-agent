# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Union

from openai import AzureOpenAI, BadRequestError, OpenAI
from pydantic import BaseModel

from datus.storage.vector.base import EmbeddingFunction
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    import numpy as np


class OpenAIEmbeddings(BaseModel, EmbeddingFunction):
    """
    An embedding function that uses the OpenAI API.

    https://platform.openai.com/docs/guides/embeddings

    This can also be used for open source models that
    are compatible with the OpenAI API.
    """

    name: str = "text-embedding-ada-002"
    dim: Optional[int] = None
    base_url: Optional[str] = None
    default_headers: Optional[dict] = None
    organization: Optional[str] = None
    api_key: Optional[str] = None

    # Set true to use Azure OpenAI API
    use_azure: bool = False

    @classmethod
    def create(cls, **kwargs) -> "OpenAIEmbeddings":
        """Create a new instance with the given parameters."""
        return cls(**kwargs)

    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if name in {"name", "dim"}:
            self.__dict__.pop("_ndims", None)
        if name in {"base_url", "default_headers", "organization", "api_key", "use_azure"}:
            self.__dict__.pop("_openai_client", None)
        if name in ["name", "dim", "base_url", "use_azure"]:
            logger.debug(f"Attribute {name} changed from {old_value} to {value}")
        elif name == "api_key":
            logger.debug("Attribute api_key changed")

    def ndims(self):
        return self._ndims

    @staticmethod
    def sensitive_keys():
        return ["api_key"]

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
            raise DatusException(
                ErrorCode.COMMON_UNSUPPORTED,
                message=f"Unknown embedding model name '{self.name}'. "
                f"Supported models: {', '.join(self.model_names())}",
            )

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
            if self.name != "text-embedding-ada-002" and self.dim is not None:
                kwargs["dimensions"] = self.dim

            logger.debug(f"Calling OpenAI API: model={self.name}, text_count={len(valid_texts)}")
            rs = self._openai_client.embeddings.create(**kwargs)
            valid_embeddings = {idx: v.embedding for v, idx in zip(rs.data, valid_indices)}
            logger.debug(f"Successfully generated embeddings for {len(valid_embeddings)} texts")
        except BadRequestError:
            logger.error(f"Bad request when generating embeddings: model={self.name}, text_count={len(texts)}")
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
