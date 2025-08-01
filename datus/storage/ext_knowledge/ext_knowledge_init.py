import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
    logger.info(f"Found {len(existing_knowledge)} existing knowledge entries (build_mode: {build_mode})")

    try:
        df = pd.read_csv(args.ext_knowledge)
        logger.info(f"Loaded CSV file with {len(df)} rows: {args.ext_knowledge}")

        # Validate required columns
        required_columns = ["domain", "layer1", "layer2", "terminology", "explanation"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [
                executor.submit(process_row, storage, row.to_dict(), index, existing_knowledge)
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
) -> str:
    """Process a single CSV row and store in database.

    Args:
        storage: ExtKnowledgeStore instance
        row: Dictionary containing row data from CSV
        index: Row index for logging
        existing_knowledge: Set of existing knowledge IDs to avoid duplicates

    Returns:
        Status string: "processed", "skipped", or "error"
    """
    try:
        # Extract and validate required fields
        domain = str(row.get("domain", "")).strip()
        layer1 = str(row.get("layer1", "")).strip()
        layer2 = str(row.get("layer2", "")).strip()
        terminology = str(row.get("terminology", "")).strip()
        explanation = str(row.get("explanation", "")).strip()

        # Validate required fields are not empty
        if not all([domain, layer1, layer2, terminology, explanation]):
            logger.warning(
                f"Row {index}: Missing required fields - domain: '{domain}', layer1: '{layer1}', "
                f"layer2: '{layer2}', terminology: '{terminology}', explanation: '{explanation}'"
            )
            return "skipped"

        # Generate unique ID
        knowledge_id = gen_ext_knowledge_id(domain, layer1, layer2, terminology)

        # Check if already exists (for incremental mode)
        if knowledge_id in existing_knowledge:
            logger.debug(f"Row {index}: Knowledge '{knowledge_id}' already exists, skipping")
            return "skipped"

        # Store the knowledge entry
        knowledge_data = {
            "domain": domain,
            "layer1": layer1,
            "layer2": layer2,
            "terminology": terminology,
            "explanation": explanation,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        storage.store_batch([knowledge_data])

        # Add to existing set to avoid duplicates within the same batch
        existing_knowledge.add(knowledge_id)

        logger.debug(f"Row {index}: Successfully stored knowledge '{terminology}' in domain '{domain}'")
        return "processed"

    except Exception as e:
        logger.error(f"Row {index}: Error processing row {row}: {str(e)}")
        return "error"


def validate_csv_format(csv_file_path: str) -> bool:
    """Validate that CSV file has required columns and basic format.

    Args:
        csv_file_path: Path to CSV file

    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return False

        df = pd.read_csv(csv_file_path, nrows=1)  # Read just first row to check columns
        required_columns = ["domain", "layer1", "layer2", "terminology", "explanation"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"CSV file missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
            logger.info(f"Required columns: {required_columns}")
            return False

        logger.info(f"CSV file format validation passed: {csv_file_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating CSV file {csv_file_path}: {str(e)}")
        return False
