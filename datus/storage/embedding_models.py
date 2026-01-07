# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import multiprocessing
import os
import platform
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, Optional

from datus.utils.constants import EmbeddingProvider
from datus.utils.device_utils import get_device
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import ModelConfig

# Fix multiprocessing issues with PyTorch/sentence-transformers in Python 3.12
try:
    if platform.system() == "Windows":
        multiprocessing.set_start_method("spawn", force=True)
    else:
        multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    # set_start_method can only be called once
    pass

# Set environment variables to prevent multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

logger = get_logger(__name__)

EMBEDDING_DEVICE_TYPE = ""


@dataclass
class EmbeddingModel:
    model_name: str
    _dim_size: int
    device: str = "cpu"

    def __init__(
        self,
        model_name: str,
        dim_size: int,
        registry_name: str = EmbeddingProvider.SENTENCE_TRANSFORMERS,
        openai_config: Optional["ModelConfig"] = None,
        batch_size: int = 64,
    ):
        self.registry_name = registry_name
        self.model_name = model_name
        self._dim_size = dim_size
        self.device = EMBEDDING_DEVICE_TYPE
        self._model = None
        self.batch_size = batch_size
        self.openai_config = openai_config
        self.lock = Lock()

        # error handling
        self.is_model_failed = False
        self.model_error_message = ""
        self.model_initialization_attempted = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "model_name": self.model_name,
            "dim_size": self._dim_size,
        }

    @property
    def model(self):
        """Get the embedding model, with lazy loading and error handling."""
        # If model initialization failed before, raise exception
        if self.is_model_failed:
            # If the stored error is already a formatted DatusException message, use it directly
            if "error_code=" in self.model_error_message:
                # Extract just the core error message without the DatusException wrapper
                import re

                match = re.search(r"error_message=(.+)$", self.model_error_message)
                if match:
                    core_message = match.group(1)
                else:
                    core_message = self.model_error_message
                raise DatusException(
                    ErrorCode.MODEL_EMBEDDING_ERROR,
                    message=f"Embedding model '{self.model_name}' is not available: {core_message}",
                )
            else:
                raise DatusException(
                    ErrorCode.MODEL_EMBEDDING_ERROR,
                    message=f"Embedding model '{self.model_name}' is not available: {self.model_error_message}",
                )

        # Lazy load the model
        if self._model is None:
            with self.lock:
                if self._model is None:
                    try:
                        self.model_initialization_attempted = True
                        logger.debug(f"Loading embedding model: {self.model_name} on {self.device}")
                        self.init_model()
                    except Exception as e:
                        # Save error state, don't immediately raise exception
                        self.is_model_failed = True
                        self.model_error_message = str(e)
                        logger.error(f"Failed to load embedding model '{self.model_name}': {e}")

        return self._model

    def is_model_available(self) -> bool:
        """Check if the model is available without triggering initialization."""
        return not self.is_model_failed and (self._model is not None or not self.model_initialization_attempted)

    async def init_model_async(self):
        """Asynchronously initialize the embedding model."""
        import asyncio

        # Run the synchronous init_model in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.init_model)

    def try_init_model_silent(self) -> bool:
        """Try to initialize the model without raising exceptions.

        Returns:
            bool: True if initialization succeeded, False otherwise.
        """
        if self._model is not None:
            return True

        if self.is_model_failed:
            return False

        with self.lock:
            if self._model is not None:
                return True

            try:
                self.model_initialization_attempted = True
                logger.debug(f"Attempting silent initialization of embedding model: {self.model_name}")
                self.init_model()
                return True
            except Exception as e:
                # Save error state silently
                self.is_model_failed = True
                self.model_error_message = str(e)
                logger.warning(f"Silent initialization failed for embedding model '{self.model_name}': {e}")
                return False

    def init_model(self):
        """Pre-download the model to local cache. Now we only support sentence-transformers and openai."""
        # Additional PyTorch-specific threading controls

        if self.registry_name in (EmbeddingProvider.SENTENCE_TRANSFORMERS, EmbeddingProvider.FASTEMBED):
            logger.debug(f"Pre-downloading model {self.registry_name}/{self.model_name} by {self.device}")
            from datus.storage.fastembed_embeddings import FastEmbedEmbeddings

            try:
                # Method `get_registry` has a multi-threading problem
                self._model = FastEmbedEmbeddings.create(
                    name=self.model_name,
                    batch_size=self.batch_size,
                )
                # first download
                # self._model.generate_embeddings(["foo"])
                logger.debug(f"Model {self.registry_name}/{self.model_name} initialized successfully")
            except Exception as e:
                raise DatusException(
                    ErrorCode.MODEL_EMBEDDING_ERROR, message=f"Embedding Model initialized failed because of {str(e)}"
                ) from e

        elif self.registry_name == EmbeddingProvider.OPENAI:
            logger.debug(f"Initializing model {self.registry_name}/{self.model_name}")
            from datus.storage.embedding_openai import OpenAIEmbeddings

            if self.openai_config:
                self._model = OpenAIEmbeddings.create(
                    name=self.model_name,
                    dim=self._dim_size,
                    api_key=self.openai_config.api_key,
                    base_url=self.openai_config.base_url,
                )
            else:
                self._model = OpenAIEmbeddings.create(name=self.model_name, dim=self._dim_size)
            # check if the model is initialized
            self._model.generate_embeddings(["foo"])
            logger.debug(f"Model {self.registry_name}/{self.model_name} initialized successfully")
        else:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=f"Unsupported EmbeddingModel registration by `{self.registry_name}`",
            )

    @property
    def dim_size(self):
        if self._dim_size is None:
            self._dim_size = self.model.ndims()
        return self._dim_size


EMBEDDING_MODELS = {}
DEFAULT_MODEL_CONFIG = {"model_name": "all-MiniLM-L6-v2", "dim_size": 384}


def init_embedding_models(
    storage_config: dict[str, dict[str, Any]],
    openai_configs: Dict[str, "ModelConfig"],
    default_openai_config: "ModelConfig",
) -> dict[str, EmbeddingModel]:
    # ensure model just load once
    global EMBEDDING_DEVICE_TYPE
    EMBEDDING_DEVICE_TYPE = str(storage_config.get("embedding_device_type", ""))
    if not EMBEDDING_DEVICE_TYPE:
        EMBEDDING_DEVICE_TYPE = get_device()
    models = {}
    for name, config in storage_config.items():
        if not isinstance(config, dict):
            continue
        if config["model_name"] in models:
            target_model = models[config["model_name"]]
        else:
            target_openai_config = config.get("target_model")
            if target_openai_config:
                if target_openai_config not in openai_configs:
                    raise DatusException(
                        ErrorCode.COMMON_CONFIG_ERROR,
                        message=f"Model {target_openai_config} not found in storage openai configuration",
                    )
                target_openai_config = openai_configs[target_openai_config]
            else:
                target_openai_config = default_openai_config
            target_model = EmbeddingModel(
                model_name=config["model_name"],
                dim_size=config["dim_size"],
                registry_name=config.get("registry_name", EmbeddingProvider.SENTENCE_TRANSFORMERS),
                batch_size=config.get("batch_size", 32),
                openai_config=target_openai_config,
            )
            models[config["model_name"]] = target_model
        EMBEDDING_MODELS[name] = target_model

    return EMBEDDING_MODELS


def get_embedding_model(store_name: str) -> EmbeddingModel:
    if store_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[store_name]
    model_name = DEFAULT_MODEL_CONFIG["model_name"]
    target_model = None
    for model in EMBEDDING_MODELS.values():
        if model.model_name == model_name:
            target_model = model
            break
    if target_model is not None:
        EMBEDDING_MODELS[store_name] = target_model
        return target_model
    target_model = EmbeddingModel(model_name=str(model_name), dim_size=DEFAULT_MODEL_CONFIG["dim_size"])
    EMBEDDING_MODELS[store_name] = target_model
    return target_model


def get_db_embedding_model() -> EmbeddingModel:
    return get_embedding_model("database")


def get_document_embedding_model() -> EmbeddingModel:
    return get_embedding_model("document")


def get_metric_embedding_model() -> EmbeddingModel:
    return get_embedding_model("metric")


def get_embedding_device() -> str:
    return EMBEDDING_DEVICE_TYPE
