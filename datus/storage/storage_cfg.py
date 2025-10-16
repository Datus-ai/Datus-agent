# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from configparser import ConfigParser
from enum import Enum
from typing import Any, Dict, List, Optional

from datus.storage.embedding_models import DEFAULT_MODEL_CONFIG, EmbeddingModel
from datus.utils.constants import EmbeddingProvider
from datus.utils.exceptions import DatusException, ErrorCode


def save_storage_config(name: str, rag_path: str, config: Optional[EmbeddingModel] = None):
    exist_config = load_storage_config(rag_path)
    if config:
        exist_config[name] = config.to_dict()
    else:
        if name in exist_config:
            exist_config.pop(name)
    save_storage_configs(exist_config, rag_path)


def save_storage_configs(configs: Dict[str, Dict[str, Any]], rag_path: str):
    """Save embedding configuration to a file."""
    os.makedirs(rag_path, exist_ok=True)
    config_path = os.path.join(rag_path, "datus_db.cfg")
    parser = ConfigParser()
    for store_type, config in configs.items():
        save_config = {}
        for k, v in config.items():
            save_config[str(k)] = str(v) if not isinstance(v, Enum) else v.value
        parser[store_type] = save_config

    with open(config_path, "w", encoding="utf-8") as f:
        parser.write(f)


def load_storage_config(rag_path: str) -> Dict[str, Dict[str, Any]]:
    """Load embedding configuration from a file."""
    config_path = os.path.join(rag_path, "datus_db.cfg")
    if not os.path.exists(config_path):
        return {}

    parser = ConfigParser()
    parser.read(config_path, encoding="utf-8")

    config = {}
    for section in parser.sections():
        config[section] = dict(parser[section])

    return config


def _find_config_differences(
    storage_type: str, existing_config: Optional[Dict[str, Any]], new_config: Optional[Dict[str, Any]]
) -> List[str]:
    """Find differences between existing and new configurations."""
    differences = []
    if not existing_config:
        existing_config = DEFAULT_MODEL_CONFIG
    if not new_config:
        new_config = DEFAULT_MODEL_CONFIG

    if "registry_name" not in existing_config:
        existing_config["registry_name"] = EmbeddingProvider.SENTENCE_TRANSFORMERS

    if "registry_name" not in new_config:
        new_config["registry_name"] = EmbeddingProvider.SENTENCE_TRANSFORMERS

    for key, value in existing_config.items():
        if key not in new_config:
            differences.append(f"Missing key '{key}' in section [{storage_type}].")
            continue
        value = str(value) if not isinstance(value, Enum) else value.value
        new_value = str(new_config[key]) if not isinstance(new_config[key], Enum) else new_config[key].value
        if value != new_value:
            differences.append(
                f"Value mismatch in section [{storage_type}] for key '{key}': "
                f"existing='{value}', new='{new_config[key]}'."
            )

    return differences


def check_storage_config(
    storage_type: str, storage_config: Optional[dict[str, Any]], rag_path: str, save_config: bool = True
):
    """Initialize embedding configuration and save it."""
    existing_config = load_storage_config(rag_path)
    if existing_config:
        differences = _find_config_differences(storage_type, existing_config.get(storage_type), storage_config)
        if differences:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message="Embedding model configuration mismatch:\n"
                + "\n".join(f"- {diff}" for diff in differences)
                + ". If you want to use the new model, initialize it first using overwrite mode.",
            )
    if save_config:
        # Convert EmbeddingModel objects to config dictionary
        existing_config[storage_type] = storage_config if storage_config else DEFAULT_MODEL_CONFIG
        save_storage_configs(existing_config, rag_path)
