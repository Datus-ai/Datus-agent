# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Set

import pandas as pd

from datus.storage.ext_knowledge.init_utils import exists_ext_knowledge, gen_ext_knowledge_id
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def init_ext_knowledge(
    storage: ExtKnowledgeStore,
    args: argparse.Namespace,
    build_mode: str = "overwrite",
    pool_size: int = 1,
):
    """Initialize external knowledge from CSV file.

    Args:
        storage: ExtKnowledgeStore instance
        args: Command line arguments containing ext_knowledge CSV file path
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing
    """
    if not hasattr(args, "ext_knowledge") or not args.ext_knowledge:
        logger.warning("No ext_knowledge CSV file specified in args.ext_knowledge")
        return

    if not os.path.exists(args.ext_knowledge):
        logger.error(f"External knowledge CSV file not found: {args.ext_knowledge}")
        return

    existing_knowledge = exists_ext_knowledge(storage, build_mode)
    existing_knowledge_lock = Lock()
    logger.info(f"Found {len(existing_knowledge)} existing knowledge entries (build_mode: {build_mode})")

    try:
        df = pd.read_csv(args.ext_knowledge)
        logger.info(f"Loaded CSV file with {len(df)} rows: {args.ext_knowledge}")

        # Validate required columns
        required_columns = ["subject_path", "name", "terminology", "explanation"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [
                executor.submit(process_row, storage, row.to_dict(), index, existing_knowledge, existing_knowledge_lock)
                for index, row in df.iterrows()
            ]

            processed_count = 0
            skipped_count = 0
            error_count = 0

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result == "processed":
                        processed_count += 1
                    elif result == "skipped":
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing row: {str(e)}")
                    error_count += 1

        logger.info(
            f"Processing complete - Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}"
        )

        # Create indices after bulk loading
        storage.after_init()

    except Exception as e:
        logger.error(f"Failed to initialize external knowledge: {str(e)}")
        raise


def process_row(
    storage: ExtKnowledgeStore,
    row: dict,
    index: int,
    existing_knowledge: Set[str],
    existing_knowledge_lock: Lock,
) -> str:
    """Process a single CSV row and store in database.

    Args:
        storage: ExtKnowledgeStore instance
        row: Dictionary containing row data from CSV
        index: Row index for logging
        existing_knowledge: Set of existing knowledge IDs to avoid duplicates
        existing_knowledge_lock: Lock for existing knowledge IDs

    Returns:
        Status string: "processed", "skipped", or "error"
    """
    try:
        # Extract and validate required fields
        subject_path = str(row.get("subject_path", "")).strip()
        name = str(row.get("name", "")).strip()
        terminology = str(row.get("terminology", "")).strip()
        explanation = str(row.get("explanation", "")).strip()

        # Validate required fields are not empty
        if not all([subject_path, name, terminology, explanation]):
            logger.warning(
                f"Row {index}: Missing required fields - subject_path: '{subject_path}', "
                f"name: '{name}', terminology: '{terminology}', explanation: '{explanation}'"
            )
            return "skipped"

        # Parse subject_path into path components (split by '/')
        path_components = [comp.strip() for comp in subject_path.split("/") if comp.strip()]
        if not path_components:
            logger.warning(f"Row {index}: Invalid subject_path '{subject_path}' - no valid path components found")
            return "skipped"

        # Generate unique ID using the new function that accepts path list
        knowledge_id = gen_ext_knowledge_id(path_components, terminology)

        # Check if already exists (for incremental mode)
        if knowledge_id in existing_knowledge:
            logger.debug(f"Row {index}: Knowledge '{knowledge_id}' already exists, skipping")
            return "skipped"

        storage.store_knowledge(path_components, name, terminology, explanation)

        # Add to existing set to avoid duplicates within the same batch
        with existing_knowledge_lock:
            existing_knowledge.add(knowledge_id)

        logger.debug(f"Row {index}: Successfully stored knowledge '{terminology}' at path '{subject_path}'")
        return "processed"

    except Exception as e:
        logger.error(f"Row {index}: Error processing row {row}: {str(e)}")
        return "error"
