# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set

import pandas as pd

from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput
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
        required_columns = ["subject_path", "name", "search_text", "explanation"]
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
        search_text = str(row.get("search_text", "")).strip()
        explanation = str(row.get("explanation", "")).strip()

        # Validate required fields are not empty
        if not all([subject_path, name, search_text, explanation]):
            logger.warning(
                f"Row {index}: Missing required fields - subject_path: '{subject_path}', "
                f"name: '{name}', search_text: '{search_text}', explanation: '{explanation}'"
            )
            return "skipped"

        # Parse subject_path into path components (split by '/')
        path_components = [comp.strip() for comp in subject_path.split("/") if comp.strip()]
        if not path_components:
            logger.warning(f"Row {index}: Invalid subject_path '{subject_path}' - no valid path components found")
            return "skipped"

        # Generate unique ID using the new function that accepts path list
        knowledge_id = gen_ext_knowledge_id(path_components, search_text)

        # Check if already exists (for incremental mode)
        if knowledge_id in existing_knowledge:
            logger.debug(f"Row {index}: Knowledge '{knowledge_id}' already exists, skipping")
            return "skipped"

        storage.upsert_knowledge(path_components, name, search_text, explanation)

        # Add to existing set to avoid duplicates within the same batch
        with existing_knowledge_lock:
            existing_knowledge.add(knowledge_id)

        logger.debug(f"Row {index}: Successfully stored knowledge '{search_text}' at path '{subject_path}'")
        return "processed"

    except Exception as e:
        logger.error(f"Row {index}: Error processing row {row}: {str(e)}")
        return "error"


def init_success_story_knowledge(
    args: argparse.Namespace,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
) -> tuple[bool, str]:
    """
    Initialize external knowledge from success story CSV file using GenExtKnowledgeAgenticNode in workflow mode.

    Args:
        args: Command line arguments containing success_story CSV file path
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories

    Returns:
        tuple[bool, str]: (whether successful, error message)
    """
    if not os.path.exists(args.success_story):
        error_msg = f"Success story CSV file not found: {args.success_story}"
        logger.error(error_msg)
        return False, error_msg

    df = pd.read_csv(args.success_story)

    async def process_all() -> tuple[bool, List[str]]:
        errors: List[str] = []
        for idx, row in df.iterrows():
            row_idx = idx + 1
            logger.info(f"Processing row {row_idx}/{len(df)}")
            try:
                result = await process_knowledge_line(row.to_dict(), agent_config, subject_tree)
                if not result.get("successful"):
                    errors.append(f"Error processing row {row_idx}: {result.get('error')}")
            except Exception as e:
                errors.append(f"Error processing row {row_idx}: {e}")
                logger.error(f"Error processing row {row_idx}: {e}")
        return (len(df) - len(errors)) > 0, errors

    successful, errors = asyncio.run(process_all())
    error_message = "\n    ".join(errors) if errors else ""
    return successful, error_message


async def process_knowledge_line(
    row: dict,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
) -> Dict[str, any]:
    """
    Process a single line from the CSV using GenExtKnowledgeAgenticNode in workflow mode.

    Args:
        row: CSV row data containing question_id, question, sql, subject_path
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories

    Returns:
        Dict with 'successful' and 'error' keys
    """
    logger.info(f"processing line: {row}")

    question = row.get("question", "")
    sql = row.get("sql", "")
    subject_path = row.get("subject_path", "")

    if not question:
        return {"successful": False, "error": "Missing question field"}

    # Create ExtKnowledgeNodeInput with separated question and gold_sql
    # workflow mode: pass question and gold_sql directly (gold_sql accessed via tool, not in prompt)
    ext_knowledge_input = ExtKnowledgeNodeInput(
        user_message=question,  # for compatibility
        question=question,  # pass directly for workflow mode
        gold_sql=sql,  # pass directly, accessed via get_gold_sql() tool
        subject_path=subject_path if subject_path else None,
    )

    # Create GenExtKnowledgeAgenticNode (workflow mode auto-saves to database)
    ext_knowledge_node = GenExtKnowledgeAgenticNode(
        node_name="gen_ext_knowledge",
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()

    try:
        ext_knowledge_node.input = ext_knowledge_input
        async for action in ext_knowledge_node.execute_stream(action_history_manager):
            if action.status == ActionStatus.SUCCESS and action.output:
                logger.debug(f"Knowledge generation action: {action.messages}")

        logger.info(f"Generated knowledge for: {question}")
        return {"successful": True, "error": ""}

    except Exception as e:
        logger.error(f"Error generating knowledge for {question}: {e}")
        return {"successful": False, "error": f"Error generating knowledge: {str(e)}"}
