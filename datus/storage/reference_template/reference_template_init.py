# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
import re
from typing import Any, Dict, Optional

from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.storage.reference_template.init_utils import exists_reference_templates, gen_reference_template_id
from datus.storage.reference_template.store import ReferenceTemplateRAG
from datus.storage.reference_template.template_file_processor import process_template_files
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

logger = get_logger(__name__)

BIZ_NAME = "reference_template_init"

TEMPLATE_EXTRA_INSTRUCTIONS = (
    "The input is a Jinja2 SQL template (not raw SQL). "
    "It contains placeholder variables in {{ variable }} syntax that will be filled at render time. "
    "In your summary, describe what SQL this template produces when rendered, "
    "what each parameter controls, and the business scenario it addresses. "
    "In search_text, include both the business intent keywords and the parameter names.\n\n"
    "**IMPORTANT — Parameter Analysis**: In addition to the standard YAML fields, you MUST add a `parameters` "
    "list in the YAML output. For EACH {{ variable }} placeholder in the template, analyze its SQL context "
    "and output an entry with:\n"
    "- `name`: the variable name\n"
    "- `type`: one of `dimension` (filters on a table column value), `column` (a column name used in "
    "GROUP BY/SELECT/ORDER BY), `keyword` (SQL keyword like ASC/DESC), "
    "or `number` (numeric value like LIMIT or threshold)\n"
    "- `column_ref`: (dimension type ONLY) the `table.column` this parameter filters on, "
    "e.g., `frpm.\\`Educational Option Type\\``\n"
    '- `allowed_values`: (keyword type ONLY) list of valid values, e.g., ["ASC", "DESC"]\n'
    "- `description`: brief description of what this parameter controls\n\n"
    "Example parameters block in YAML:\n"
    "```yaml\n"
    "parameters:\n"
    '  - name: "school_type"\n'
    '    type: "dimension"\n'
    '    column_ref: "frpm.`Educational Option Type`"\n'
    '    description: "Type of educational option to filter by"\n'
    '  - name: "sort_order"\n'
    '    type: "keyword"\n'
    '    allowed_values: ["ASC", "DESC"]\n'
    '    description: "Sort direction for results"\n'
    '  - name: "limit"\n'
    '    type: "number"\n'
    '    description: "Maximum number of rows to return"\n'
    "```"
)


def _enrich_dimension_sample_values(params: list, agent_config: AgentConfig) -> None:
    """Enrich parameter metadata with sample values from the database.

    - dimension params: queries top 10 most common values via GROUP BY / COUNT
    - column params: queries table column names via describe_table

    Args:
        params: List of parameter dicts (modified in place)
        agent_config: Agent config to create DB connection
    """
    from datus.tools.db_tools.db_manager import db_manager_instance
    from datus.tools.func_tool.database import DBFuncTool

    try:
        db_manager = db_manager_instance(agent_config.namespaces)
        conn = db_manager.get_conn(agent_config.current_namespace, agent_config.current_database)
        db_tool = DBFuncTool(conn, agent_config=agent_config)
    except Exception as e:
        logger.debug(f"Cannot create DB connection for parameter value enrichment: {e}")
        return

    for p in params:
        ptype = p.get("type")

        if ptype == "dimension" and p.get("column_ref"):
            _enrich_dimension_param(p, db_tool)
        elif ptype == "column" and p.get("table_refs"):
            _enrich_column_param(p, db_tool)


_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_]\w*$|^`[^`]+`$")


def _is_safe_identifier(name: str) -> bool:
    """Check if a string is a safe SQL identifier (plain name or backtick-quoted)."""
    return bool(_SAFE_IDENTIFIER.match(name))


def _enrich_dimension_param(p: dict, db_tool) -> None:
    """Query top 10 most common values for a dimension parameter."""
    col_ref = p["column_ref"]
    parts = col_ref.split(".", 1)
    if len(parts) != 2:
        return
    table, column = parts
    if not _is_safe_identifier(table) or not _is_safe_identifier(column):
        logger.debug(f"Skipping sample value query for unsafe identifier: {col_ref}")
        return
    sql = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT 10"
    try:
        result = db_tool.read_query(sql)
        values = _extract_csv_values(result)
        if values:
            p["sample_values"] = values
    except Exception as e:
        logger.debug(f"Failed to query sample values for {col_ref}: {e}")


def _enrich_column_param(p: dict, db_tool) -> None:
    """Query column names from referenced tables for a column-type parameter."""
    table_refs = p["table_refs"]
    all_columns = []
    for table in table_refs:
        if not _is_safe_identifier(table):
            continue
        try:
            result = db_tool.describe_table(table)
            if result.success and result.result:
                for col_info in result.result:
                    col_name = col_info.get("column_name") or col_info.get("name", "")
                    if col_name and col_name not in all_columns:
                        all_columns.append(col_name)
        except Exception as e:
            logger.debug(f"Failed to describe table {table}: {e}")
    if all_columns:
        p["sample_values"] = all_columns


def _extract_csv_values(result) -> Optional[list]:
    """Extract values from a read_query result's compressed_data CSV."""
    if not result.success or not result.result:
        return None
    compressed = result.result.get("compressed_data", "")
    if not compressed:
        return None
    lines = compressed.strip().split("\n")
    if len(lines) <= 1:
        return None
    values = []
    for line in lines[1:]:
        _, _, val = line.partition(",")
        val = val.strip()
        if val:
            values.append(val)
    return values or None


def _action_status_value(action: Any) -> Optional[str]:
    status = getattr(action, "status", None)
    if status is None:
        return None
    return status.value if hasattr(status, "value") else str(status)


async def process_template_item(
    item: dict,
    agent_config: AgentConfig,
    build_mode: str = "incremental",
    subject_tree: Optional[list] = None,
    event_helper: Optional[BatchEventHelper] = None,
    template_id: Optional[str] = None,
    extra_instructions: Optional[str] = None,
) -> Optional[str]:
    """Process a single template item using SqlSummaryAgenticNode in workflow mode.

    Args:
        item: Dict containing template, comment, parameters, filepath fields
        agent_config: Agent configuration
        build_mode: "overwrite" or "incremental"
        subject_tree: Optional predefined subject tree categories
        event_helper: Optional BatchEventHelper to stream progress events
        template_id: Optional precomputed template identifier
        extra_instructions: Optional extra instructions for the LLM

    Returns:
        Template summary file path if successful, None otherwise
    """
    logger.debug(
        f"Processing template item: {item.get('filepath', '')}, template length: {len(item.get('template', ''))}"
    )
    template_id = template_id or gen_reference_template_id(item.get("template", ""))
    filepath = item.get("filepath")

    try:
        # Build instructions combining default template context and any extra
        instructions = TEMPLATE_EXTRA_INSTRUCTIONS
        if extra_instructions:
            instructions = f"{instructions}\n\n{extra_instructions}"

        user_message = f"Analyze and summarize this SQL query\n\n## Additional Instructions\n{instructions}"

        # Reuse SqlSummaryNodeInput - pass template content as sql_query
        # The LLM will understand it's a template via the extra_instructions
        sql_input = SqlSummaryNodeInput(
            user_message=user_message,
            sql_query=item.get("template"),
        )

        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=agent_config,
            execution_mode="workflow",
            build_mode=build_mode,
            subject_tree=subject_tree,
            storage_type="reference_template",
        )

        action_history_manager = ActionHistoryManager()
        sql_summary_file = None

        node.input = sql_input
        async for action in node.execute_stream(action_history_manager):
            if event_helper:
                event_helper.item_processing(
                    item_id=template_id,
                    action_name="gen_sql_summary",
                    status=_action_status_value(action),
                    group_id=filepath,
                    messages=action.messages,
                    output=action.output,
                )
            if action.status == ActionStatus.SUCCESS and action.output:
                output = action.output
                if isinstance(output, dict):
                    sql_summary_file = output.get("sql_summary_file")

        if not sql_summary_file:
            logger.error(
                f"Failed to generate template summary for {item.get('filepath', '')}, "
                f"template: {item.get('template', '')[:100]}..."
            )
            return None

        logger.info(f"Generated template summary: {sql_summary_file}")

        file_path = agent_config.path_manager.sql_summary_path(agent_config.current_namespace) / sql_summary_file
        import yaml

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
                if not item.get("name"):
                    item["name"] = doc.get("name", "")
                item["subject_tree"] = doc.get("subject_tree")
                # Backfill LLM-generated summary and search_text into item
                if doc.get("summary"):
                    item["summary"] = doc["summary"]
                if doc.get("search_text"):
                    item["search_text"] = doc["search_text"]
                if doc.get("tags"):
                    item["tags"] = doc["tags"]
                # Enrich parameters: sqlglot static analysis (deterministic) + LLM description
                from datus.storage.reference_template.template_file_processor import analyze_template_parameters

                static_params = analyze_template_parameters(
                    item.get("template", ""), dialect=agent_config.db_type or None
                )
                llm_parameters = doc.get("parameters")
                if static_params:
                    # Merge LLM descriptions into sqlglot-analyzed params
                    if llm_parameters and isinstance(llm_parameters, list):
                        llm_map = {p["name"]: p for p in llm_parameters if "name" in p}
                        for sp in static_params:
                            llm_p = llm_map.get(sp["name"], {})
                            if not sp.get("description"):
                                sp["description"] = llm_p.get("description", "")
                            if sp.get("type") == "keyword" and not sp.get("allowed_values"):
                                sp["allowed_values"] = llm_p.get("allowed_values", [])
                    # Query DISTINCT values for dimension params and store them
                    _enrich_dimension_sample_values(static_params, agent_config)
                    item["parameters"] = json.dumps(static_params)
                elif llm_parameters and isinstance(llm_parameters, list):
                    # Fallback to LLM-only parameters if sqlglot analysis returned nothing
                    item["parameters"] = json.dumps(llm_parameters)
        except Exception as e:
            logger.warning(f"Failed to open summary file for {file_path}: {e}")
            return None

        # Validate required metadata before treating as success
        if (
            not item.get("name")
            or not item.get("summary")
            or not item.get("search_text")
            or not item.get("subject_tree")
        ):
            logger.warning(
                f"Incomplete template summary metadata in {file_path}, "
                f"missing: {[k for k in ('name', 'summary', 'search_text', 'subject_tree') if not item.get(k)]}"
            )
            return None

        return sql_summary_file

    except Exception as e:
        logger.error(f"Error processing template item {item.get('filepath', '')}: {e}")
        return None


async def init_reference_template_async(
    storage: ReferenceTemplateRAG,
    global_config: AgentConfig,
    template_dir: str,
    validate_only: bool = False,
    build_mode: str = "overwrite",
    pool_size: int = 1,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version: Initialize reference templates from template files directory.

    Args:
        storage: ReferenceTemplateRAG instance
        template_dir: The path to the template files directory
        validate_only: If true, only validate templates.
        global_config: Global agent configuration for LLM model creation
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM

    Returns:
        Dict containing initialization results and statistics
    """
    event_helper = BatchEventHelper(BIZ_NAME, emit)

    if not template_dir:
        logger.warning("No --template_dir provided, reference template storage initialized but empty")
        return {
            "status": "success",
            "message": "reference_template storage initialized (empty - no --template_dir provided)",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": 0,
            "total_stored_entries": storage.get_reference_template_size(),
        }

    logger.info(f"Processing template files from directory: {template_dir}")

    event_helper.task_started(template_dir=template_dir)

    # Process and validate template files
    valid_items, invalid_items = process_template_files(template_dir)
    validate_errors = (
        []
        if not invalid_items
        else [
            f"Failed to validate template at {i['filepath']}:{i['line_number']} with error `{i['error']}`. "
            "Please check that the template meets expectations (e.g., valid Jinja2 syntax)."
            for i in invalid_items
        ]
    )

    event_helper.task_validated(
        total_items=len(valid_items) + len(invalid_items) if invalid_items else len(valid_items),
        valid_items=len(valid_items) if valid_items else 0,
        invalid_items=len(invalid_items) if invalid_items else 0,
    )

    if validate_only:
        logger.info(
            f"Validate-only mode: Processed {len(valid_items)} valid items and "
            f"{len(invalid_items) if invalid_items else 0} invalid items"
        )
        return {
            "status": "success",
            "message": "Template files processing completed (validate-only mode)",
            "valid_entries": len(valid_items) if valid_items else 0,
            "processed_entries": 0,
            "invalid_entries": len(invalid_items) if invalid_items else 0,
            "total_stored_entries": 0,
            "validation_errors": "\n".join(validate_errors),
        }

    if not valid_items:
        logger.info("No valid template items found to process")
        return {
            "status": "success",
            "message": f"No valid template items found in directory: {template_dir}. "
            "Please ensure the directory is correct.",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": len(invalid_items) if invalid_items else 0,
            "total_stored_entries": storage.get_reference_template_size(),
            "validation_errors": "\n".join(validate_errors),
        }

    # Filter out existing items in incremental mode
    if build_mode == "incremental":
        existing_ids = exists_reference_templates(storage, build_mode)
        new_items = []
        for item_dict in valid_items:
            item_id = gen_reference_template_id(item_dict["template"])
            if item_id not in existing_ids:
                new_items.append(item_dict)
        logger.info(f"Incremental mode: found {len(valid_items)} items, {len(new_items)} new items to process")
        items_to_process = new_items
    else:
        items_to_process = valid_items

    # Force serial processing when subject_tree is not predefined
    effective_pool_size = pool_size
    if subject_tree is None and pool_size > 1:
        logger.info(
            f"No predefined subject_tree provided, forcing serial processing "
            f"(pool_size: {pool_size} -> 1) to avoid race conditions"
        )
        effective_pool_size = 1

    processed_count = 0
    process_errors = []
    if items_to_process:
        event_helper.task_processing(total_items=len(items_to_process))

        semaphore = asyncio.Semaphore(effective_pool_size)
        logger.info(f"Processing {len(items_to_process)} template items with concurrency={effective_pool_size}")
        file_counts: Dict[str, int] = {}
        for item in items_to_process:
            file_key = str(item.get("filepath") or "unknown_file")
            file_counts[file_key] = file_counts.get(file_key, 0) + 1
        file_remaining = dict(file_counts)
        file_started: set[str] = set()
        file_lock = asyncio.Lock()

        async def emit_group_started_if_needed(file_key: str, filepath: Optional[str]) -> None:
            async with file_lock:
                if file_key in file_started:
                    return
                file_started.add(file_key)
            event_helper.group_started(
                group_id=file_key,
                total_items=file_counts.get(file_key),
                filepath=filepath,
            )

        async def emit_group_completed_if_done(file_key: str, filepath: Optional[str]) -> None:
            async with file_lock:
                remaining = file_remaining.get(file_key)
                if remaining is None:
                    return
                remaining -= 1
                file_remaining[file_key] = remaining
                if remaining != 0:
                    return
            event_helper.group_completed(
                group_id=file_key,
                total_items=file_counts.get(file_key),
                filepath=filepath,
            )

        async def process_with_semaphore(item):
            tpl_id = gen_reference_template_id(item.get("template", ""))
            filepath = item.get("filepath")
            file_key = str(filepath or "unknown_file")
            async with semaphore:
                await emit_group_started_if_needed(file_key, filepath)
                event_helper.item_started(
                    item_id=tpl_id,
                    group_id=file_key,
                    template=item.get("template", "")[:200],
                )
                error = None
                result = None
                try:
                    result = await process_template_item(
                        item,
                        global_config,
                        build_mode,
                        subject_tree,
                        event_helper=event_helper,
                        template_id=tpl_id,
                        extra_instructions=extra_instructions,
                    )
                except Exception as exc:
                    error = exc
                    event_helper.item_failed(
                        item_id=tpl_id,
                        error=str(exc),
                        group_id=file_key,
                        exception_type=type(exc).__name__,
                    )
                if result:
                    event_helper.item_completed(
                        item_id=tpl_id,
                        group_id=file_key,
                        sql_summary_file=result,
                    )
                elif error is None:
                    event_helper.item_failed(
                        item_id=tpl_id,
                        error="Failed to generate template summary",
                        group_id=file_key,
                    )
                await emit_group_completed_if_done(file_key, filepath)
                return item, tpl_id, result, error

        tasks = [asyncio.create_task(process_with_semaphore(item)) for item in items_to_process]
        _errors = []
        success_count = 0
        success_items = []
        for task in asyncio.as_completed(tasks):
            item, _tpl_id, result, error = await task
            template_preview = (item.get("template") or "")[:80]
            if error:
                logger.error(f"Template processing failed with exception `{error}`. Template: {template_preview};")
                _errors.append(
                    f"Template processing failed with exception `{str(error)}`. Template: {template_preview};"
                )
            elif result:
                success_items.append(item)
                success_count += 1
            else:
                logger.error(f"Template processing returned no result. Template: {template_preview};")
                _errors.append(f"Template processing returned no result. Template: {template_preview};")

        logger.info(f"Completed processing: {success_count}/{len(items_to_process)} successful")
        processed_count = success_count
        process_items = success_items
        if _errors:
            process_errors.extend(_errors)
        logger.info(f"Processed {processed_count} reference template entries")
    else:
        logger.info("No new items to process in incremental mode")
        process_items = []

    # Store successfully processed items into reference_template storage
    if process_items:
        store_items = []
        for item in process_items:
            tpl_id = gen_reference_template_id(item.get("template", ""))
            subject_tree_str = item.get("subject_tree", "")
            subject_path = []
            if subject_tree_str:
                parts = subject_tree_str.split("/")
                subject_path = [part.strip() for part in parts if part.strip()]

            store_items.append(
                {
                    "id": tpl_id,
                    "name": item.get("name", ""),
                    "template": item.get("template", ""),
                    "parameters": item.get("parameters", "[]"),
                    "comment": item.get("comment", ""),
                    "summary": item.get("summary", ""),
                    "search_text": item.get("search_text", ""),
                    "filepath": item.get("filepath", ""),
                    "subject_path": subject_path,
                    "tags": item.get("tags", ""),
                }
            )
        try:
            storage.upsert_batch(store_items)
            logger.info(f"Stored {len(store_items)} reference template entries to vector DB")
        except Exception as e:
            logger.error(f"Failed to store reference template entries: {e}")
            process_errors.append(f"Storage write failed: {e}")

    event_helper.task_completed(
        total_items=len(items_to_process) if items_to_process else 0,
        completed_items=processed_count,
        failed_items=len(process_errors),
    )

    storage.after_init()

    return {
        "status": "success",
        "message": f"reference_template bootstrap completed ({build_mode} mode)",
        "valid_entries": len(valid_items) if valid_items else 0,
        "processed_entries": processed_count,
        "processed_items": process_items,
        "invalid_entries": len(invalid_items) if invalid_items else 0,
        "total_stored_entries": storage.get_reference_template_size(),
        "validation_errors": "\n".join(validate_errors) if validate_errors else None,
        "process_errors": "\n".join(process_errors) if process_errors else None,
    }


def init_reference_template(
    storage: ReferenceTemplateRAG,
    global_config: AgentConfig,
    template_dir: str,
    validate_only: bool = False,
    build_mode: str = "overwrite",
    pool_size: int = 1,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync wrapper: Initialize reference templates from template files directory.

    Args:
        storage: ReferenceTemplateRAG instance
        template_dir: The path to the template files directory
        validate_only: If true, only validate templates.
        global_config: Global agent configuration for LLM model creation
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM

    Returns:
        Dict containing initialization results and statistics
    """
    with suppress_keyboard_input():
        return asyncio.run(
            init_reference_template_async(
                storage,
                global_config,
                template_dir,
                validate_only,
                build_mode,
                pool_size,
                subject_tree,
                emit,
                extra_instructions,
            )
        )
