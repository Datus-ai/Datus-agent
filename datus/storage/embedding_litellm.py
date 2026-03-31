# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Union

import litellm
from datus_storage_base.vector.base import EmbeddingFunction
from pydantic import BaseModel

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    import numpy as np


class LiteLLMEmbeddings(BaseModel, EmbeddingFunction):
    """An embedding function that uses the LiteLLM API.

    Supports any embedding model that LiteLLM supports, including
    AWS Bedrock Titan, Cohere, and other providers.

    Example model names:
        - "bedrock/amazon.titan-embed-text-v2:0"
        - "bedrock/cohere.embed-english-v3"
    """

    name: str
    dim: Optional[int] = None

    @classmethod
    def create(cls, **kwargs) -> "LiteLLMEmbeddings":
        """Create a new instance with the given parameters."""
        return cls(**kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in {"name", "dim"}:
            self.__dict__.pop("_ndims", None)
            self.__dict__.pop("_max_input_tokens", None)

    def ndims(self) -> int:
        return self._ndims

    @staticmethod
    def sensitive_keys() -> list[str]:
        return []

    @cached_property
    def _ndims(self) -> int:
        if self.dim is not None:
            return self.dim
        try:
            resp = litellm.embedding(model=self.name, input=["test"])
            return len(resp.data[0]["embedding"])
        except Exception as e:
            logger.error(f"Failed to auto-detect embedding dimensions for model '{self.name}': {e}")
            raise

    @cached_property
    def _max_input_tokens(self) -> int:
        """Auto-detect the maximum input tokens for this embedding model."""
        try:
            model_info = litellm.get_model_info(self.name)
            max_tokens = model_info.get("max_input_tokens")
            if max_tokens and isinstance(max_tokens, int) and max_tokens > 0:
                logger.debug(f"Model {self.name}: max_input_tokens={max_tokens}")
                return max_tokens
        except Exception as e:
            logger.warning(f"Failed to get model info for '{self.name}': {e}. Using default 8191 tokens.")
        return 8191  # OpenAI default fallback

    def _truncate(self, text: str) -> str:
        """Truncate text to fit within the model's token limit.

        Uses a conservative estimate of 3 characters per token.
        """
        max_chars = self._max_input_tokens * 3
        if len(text) <= max_chars:
            return text
        logger.warning(
            f"Truncating text from {len(text)} to {max_chars} chars "
            f"(~{self._max_input_tokens} tokens) for model {self.name}"
        )
        return text[:max_chars]

    def generate_embeddings(self, texts: Union[List[str], "np.ndarray"]) -> List["np.array"]:
        import numpy as np

        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text:
                valid_texts.append(self._truncate(text))
                valid_indices.append(idx)

        if not valid_texts:
            return [None] * len(texts)

        valid_embeddings: dict = {}
        for i, (text, orig_idx) in enumerate(zip(valid_texts, valid_indices)):
            try:
                resp = litellm.embedding(model=self.name, input=[text])
                valid_embeddings[orig_idx] = np.array(resp.data[0]["embedding"])
            except Exception as e:
                logger.error(f"LiteLLM embedding failed for text #{i} (len={len(text)}): {e}")

        return [valid_embeddings.get(idx, None) for idx in range(len(texts))]
