# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
import os.path
from typing import Any, List, Optional, Union

import numpy as np
from fastembed import TextEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase
from huggingface_hub.errors import LocalEntryNotFoundError
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.embeddings.utils import weak_lru
from pydantic import Field

from datus.storage.embedding_models import get_embedding_device
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@register("fastembed")
class FastEmbedEmbeddings(TextEmbeddingFunction):
    """
    LanceDB embedding function backed by ``fastembed`` text encoders.

    Parameters
    ----------
    name: str, default "sentence-transformers/all-MiniLM-L6-v2"
        The identifier of the embedding model to use. Bare model names
        (e.g. ``all-MiniLM-L6-v2``) will be resolved to the
        ``sentence-transformers`` namespace for backwards compatibility.
    device: str, default "cpu"
        Preferred device reported by Datus. ``fastembed`` currently
        supports CPU and CUDA; other values fall back to CPU.
    normalize: bool, default True
        Retained for backwards compatibility; ``fastembed`` models always
        return normalized embeddings.
    trust_remote_code: bool, default True
        Unused but preserved to avoid breaking persisted LanceDB configs.
    batch_size: int, default 256
        Batch size used when generating embeddings.
    cache_dir: str | None
        Optional cache directory for ``fastembed`` model assets. When
        omitted the library default (``FASTEMBED_CACHE_PATH`` or a temp
        directory) is used.
    local_files_only: bool, default True
        If True, only cached model artifacts will be used and downloads
        will not be attempted.
    """

    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize: bool = True
    trust_remote_code: bool = True
    batch_size: int = 256
    # Do NOT persist cache_dir into LanceDB metadata to keep artifacts portable
    cache_dir: Optional[str] = Field(default=None, exclude=True, repr=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = get_embedding_device()
        self._model_instance: Optional[TextEmbedding] = None

        normalized_name = self._normalize_model_name(self.name)
        object.__setattr__(self, "name", normalized_name)

        # Allow batch size and cache_dir overrides
        batch_size = kwargs.get("batch_size", self.batch_size)
        object.__setattr__(self, "batch_size", int(batch_size))
        self.cache_dir = str(_resolve_cache_dir())
        # If not provided via kwargs, compute a portable default cache dir.
        if not getattr(self, "cache_dir", None):
            object.__setattr__(self, "cache_dir", self.cache_dir)
        if self.device not in {"cpu", "cuda"}:
            logger.debug(f"fastembed does not support device '{self.device}', falling back to CPU.")
            self.device = "cpu"

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        return f"sentence-transformers/{model_name}"

    @property
    def embedding_model(self) -> TextEmbeddingBase:
        """
        Lazily create and cache the underlying fastembed model instance.
        """
        if self._model_instance is None:
            self._model_instance = self.get_embedding_model()
        return self._model_instance

    def ndims(self) -> int:
        if self._dim_size is None:
            _model = self.embedding_model
            if dims := getattr(_model, "embedding_size", None):
                self._dim_size = dims
            else:
                logger.info("Use model description to resolve dim size")
                self._dim_size = TextEmbedding._get_model_description(self.name).get("dim")
        return int(self._dim_size)

    def generate_embeddings(self, texts: Union[List[str], np.ndarray], *args, **kwargs) -> List[List[float]]:
        """
        Generate embeddings for the provided texts.

        Parameters
        ----------
        texts: list[str] or np.ndarray (of str)
            The texts to embed.
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        else:
            texts = list(texts)

        if not texts:
            return []

        embeddings = self.embedding_model.embed(texts, batch_size=self.batch_size)
        return [np.asarray(embedding, dtype=np.float32).tolist() for embedding in embeddings]

    @weak_lru(maxsize=1)
    def get_embedding_model(self) -> TextEmbeddingBase:
        """
        Create the fastembed ``TextEmbedding`` instance for the configured model.
        Cached so that the model is only loaded once per process.
        """
        check_snapshot(self.name, self.cache_dir)

        model_kwargs: dict[str, Any] = {
            "model_name": self.name,
            "cache_dir": self.cache_dir,
            "threads": 1,
            "local_files_only": True,
        }
        if self.device == "cuda":
            model_kwargs["cuda"] = True

        try:
            return TextEmbedding(**model_kwargs)
        except Exception as exc:  # pragma: no cover - delegated to caller
            logger.error(f"Failed to initialize fastembed model '{self.name}': {exc}")
            raise


def _resolve_cache_dir() -> str:
    """
    Define the cache directory for fastembed
    """

    from pathlib import Path

    if cache_path := os.getenv("FASTEMBED_CACHE_PATH"):
        cache_path = Path(cache_path)
    else:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_path = (home / "fastembed").resolve()
    logger.debug(f"Final fastembed cache_dir is: {cache_path}")
    return str(cache_path)


def check_snapshot(model_name: str, cache_dir: str) -> None:
    """
    Ensure the required model artifacts are present in the fastembed cache.

    When ``local_files_only`` is True we only verify cached files exist.
    Otherwise a download will be attempted if the cache is missing.
    """
    try:
        description = TextEmbedding._get_model_description(model_name)
    except ValueError as exc:
        logger.error(f"Model '{model_name}' is not supported by fastembed: {exc}")
        raise

    repo_id = None
    sources = getattr(description, "sources", None)
    if sources is not None and hasattr(sources, "hf"):
        repo_id = sources.hf  # type: ignore[assignment]
    elif isinstance(description, dict):
        repo_id = description.get("sources", {}).get("hf")  # type: ignore[assignment]
    if not repo_id:
        logger.warning(
            f"FastEmbed does not support models `{model_name}`.Support models: {TextEmbedding.list_supported_models()}"
        )
        # Some fastembed models only ship via the GCS mirror; defer to fastembed.
        return

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    try:
        snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=True)
    except LocalEntryNotFoundError:
        logger.info(f"Downloading {repo_id} to {cache_dir} via huggingface_hub")
        snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=False)
        logger.info(f"Model {repo_id} has been downloaded via huggingface_hub to {cache_dir}.")
