import multiprocessing
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

from datus.utils.constants import EmbeddingProvider
from datus.utils.device_utils import get_device
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

# Fix multiprocessing issues with PyTorch/sentence-transformers in Python 3.12
try:
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
        openai_config: Optional[dict[str, Any]] = None,
        batch_size: int = 32,
    ):
        self.registry_name = registry_name
        self.model_name = model_name
        self._dim_size = dim_size
        self.device = "cpu" if EMBEDDING_DEVICE_TYPE and "cpu" == EMBEDDING_DEVICE_TYPE else get_device()
        self._model = None
        self.batch_size = batch_size
        self.openai_config = openai_config
        self.lock = Lock()

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "model_name": self.model_name,
            "dim_size": self._dim_size,
        }

    @property
    def model(self):
        # first init
        if self._model is None:
            with self.lock:
                if self._model is None:
                    logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
                    # First try to load from local cache
                    self.init_model()

        return self._model

    def init_model(self):
        """Pre-download the model to local cache. Now we only support sentence-transformers and openai."""
        # Additional PyTorch-specific threading controls
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:
            pass

        if self.registry_name == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            logger.info(f"Pre-downloading model {self.registry_name}/{self.model_name} by {self.device}")
            from lancedb.embeddings import SentenceTransformerEmbeddings

            try:
                # Method `get_registry` has a multi-threading problem
                self._model = SentenceTransformerEmbeddings.create(name=self.model_name, device=self.device)
                # first download
                self._model.generate_embeddings(["foo"])
                logger.info(f"Model {self.registry_name}/{self.model_name} initialized successfully")
            except Exception as e:
                raise DatusException(
                    ErrorCode.MODEL_EMBEDDING_ERROR, message=f"Embedding Model initialized faield because of {str(e)}"
                ) from e

        elif self.registry_name == EmbeddingProvider.OPENAI:
            logger.info(f"Initializing model {self.registry_name}/{self.model_name}")
            from datus.storage.embedding_openai import OpenAIEmbeddings

            if self.openai_config:
                self._model = OpenAIEmbeddings.create(
                    name=self.model_name,
                    dim=self._dim_size,
                    api_key=self.openai_config["api_key"],
                    base_url=self.openai_config["base_url"],
                )
            else:
                self._model = OpenAIEmbeddings.create(name=self.model_name, dim=self._dim_size)
            # check if the model is initialized
            self._model.generate_embeddings(["foo"])
            logger.info(f"Model {self.registry_name}/{self.model_name} initialized successfully")
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
    storage_config: dict[str, dict[str, Any]], openai_config: Optional[dict[str, Any]] = None
) -> dict[str, EmbeddingModel]:
    # ensure model just load once
    global EMBEDDING_DEVICE_TYPE
    EMBEDDING_DEVICE_TYPE = str(storage_config.get("embedding_device_type", ""))
    models = {}
    for name, config in storage_config.items():
        if not isinstance(config, dict):
            continue
        if config["model_name"] in models:
            target_model = models[config["model_name"]]
        else:
            target_model = EmbeddingModel(
                model_name=config["model_name"],
                dim_size=config["dim_size"],
                registry_name=config.get("registry_name", EmbeddingProvider.SENTENCE_TRANSFORMERS),
                batch_size=config.get("batch_size", 32),
                openai_config=openai_config,
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
