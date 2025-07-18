import os
from configparser import ConfigParser
from enum import Enum
from typing import Any, Dict, List

from datus.utils.exceptions import DatusException, ErrorCode


def save_storage_config(config: Dict[str, Any], rag_base_path: str) -> None:
    """Save embedding configuration to a file."""
    os.makedirs(rag_base_path, exist_ok=True)
    config_path = os.path.join(rag_base_path, "datus_db.cfg")

    parser = ConfigParser()
    for store_type in ["database", "document", "metric"]:
        if store_type in config:
            save_config = {}
            for k, v in config[store_type].items():
                save_config[str(k)] = str(v) if not isinstance(v, Enum) else v.value
            parser[store_type] = save_config

    with open(config_path, "w", encoding="utf-8") as f:
        parser.write(f)


def load_storage_config(rag_path: str) -> Dict[str, Any]:
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


def _find_config_differences(existing: Dict[str, Any], new: Dict[str, Dict[str, Any]]) -> List[str]:
    """Find differences between existing and new configurations."""
    differences = []

    # Check for missing or extra sections
    existing_sections = set(existing.keys())
    new_sections = set(new.keys())

    if existing_sections != new_sections:
        missing = existing_sections - new_sections
        extra = new_sections - existing_sections
        if missing:
            differences.append(f"Missing sections in current config: {', '.join(missing)}.")
        if extra:
            differences.append(f"Extra sections in current config: {', '.join(extra)}.")

    # Check differences in each section
    for section in existing_sections & new_sections:
        existing_config = existing[section]
        new_config = new[section]

        for key, value in existing_config.items():
            if key not in new_config:
                differences.append(f"Missing key '{key}' in section [{section}].")
            elif str(value) != str(new_config[key]):
                differences.append(
                    f"Value mismatch in section [{section}] for key '{key}': "
                    f"existing='{value}', current='{new_config[key]}'."
                )

    return differences


def check_storage_config(storage_config: Dict[str, dict[str, Any]], rag_path: str):
    """Initialize embedding configuration and save it."""
    existing_config = load_storage_config(rag_path)

    if existing_config:
        differences = _find_config_differences(existing_config, storage_config)
        if differences:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message="Embedding model configuration mismatch:\n"
                + "\n".join(f"- {diff}" for diff in differences)
                + ". If you want to use the new model, initialize it first using overwrite mode.",
            )

    # Convert EmbeddingModel objects to config dictionary
    save_storage_config(storage_config, rag_path)
