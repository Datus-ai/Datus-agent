# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from pathlib import Path
from typing import Any, List, Literal, Optional

from rich.console import Console

from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger, print_rich_exception
from datus.utils.path_utils import safe_rmtree

logger = get_logger(__name__)
console = Console()


def detect_db_connectivity(datasource_name: str, db_config_data: dict[str, Any]) -> tuple[bool, str]:
    """Test database connectivity.

    Uses DbConfig.filter_kwargs to handle all database types uniformly.
    Adapter-specific fields are stored in the 'extra' field and expanded
    when creating the connector.
    """
    try:
        from datus.configuration.agent_config import DbConfig
        from datus.tools.db_tools.db_manager import DBManager

        db_type = db_config_data.get("type", "")
        if not db_type:
            return False, "Database type is required"

        # Handle ~ expansion for uri field
        config_data = db_config_data.copy()
        uri = config_data.get("uri", "")
        if uri:
            if uri.startswith(f"{db_type}:///"):
                db_path = uri[len(db_type) + 4 :]
                db_path = os.path.expanduser(db_path)
                config_data["uri"] = f"{db_type}:///{db_path}"

                if db_type == "sqlite" and not Path(db_path).exists():
                    return False, f"SQLite database file does not exist: {db_path}"
            else:
                config_data["uri"] = os.path.expanduser(uri)

        # Use filter_kwargs to create DbConfig
        # Unknown fields will be stored in 'extra' and expanded by DBManager
        db_config = DbConfig.filter_kwargs(DbConfig, config_data)

        # Create DB manager with minimal config
        datasource_configs = {datasource_name: {datasource_name: db_config}}
        db_manager = DBManager(datasource_configs)

        # Get connector and test connection
        connector = db_manager.get_conn(datasource_name, datasource_name)
        test_result = connector.test_connection()

        # Handle different return types from different connectors
        if isinstance(test_result, bool):
            return (test_result, "") if test_result else (False, "Connection test failed")
        elif isinstance(test_result, dict):
            success = test_result.get("success", False)
            error_msg = test_result.get("error", "Connection test failed") if not success else ""
            return success, error_msg
        else:
            return False, "Unknown connection test result format"

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Database connectivity test failed: {error_msg}")
        return False, error_msg


def init_metrics(
    success_path: Path,
    agent_config: AgentConfig,
    subject_tree: Optional[List[str]] = None,
    build_model: Literal["incremental", "overwrite"] = "overwrite",
    console: Optional[Console] = None,
    extra_instructions: Optional[str] = None,
) -> tuple[bool, Optional[dict[str, Any]]]:
    """Initialize metrics using success stories."""
    from rich.markup import escape

    from datus.schemas.batch_events import BatchEvent, BatchStage
    from datus.storage.metric.metric_init import init_success_story_metrics
    from datus.utils.stream_output import StreamOutputManager

    if not console:
        console = Console(log_path=False)
    try:
        if build_model == "overwrite":
            from datus.storage.backend_holder import create_vector_connection

            db = create_vector_connection(agent_config.project_name)
            try:
                db.drop_table("metrics", ignore_missing=True)
                logger.info("Dropped existing metrics table")
            finally:
                db.close()
            agent_config.save_storage_config("metric")

        # Create StreamOutputManager
        output_mgr = StreamOutputManager(
            console=console,
            max_message_lines=10,
            show_progress=True,
            title="Metrics Initialization",
        )

        def emit(event: BatchEvent) -> None:
            stage = event.stage

            if stage == BatchStage.TASK_STARTED:
                output_mgr.start(total_items=1, description="Initializing metrics")
                return

            if stage == BatchStage.ITEM_PROCESSING:
                payload = event.payload or {}
                messages = payload.get("messages")
                if messages:
                    output_mgr.add_llm_output(str(messages))
                # Capture summary from the final action (SemanticNodeResult with success + response)
                output = payload.get("output")
                if output and isinstance(output, dict) and output.get("success") and output.get("response"):
                    response = output["response"]
                    if isinstance(response, str) and response.strip():
                        output_mgr.summary_outputs.clear()
                        output_mgr.add_summary_content(response)
                return

            if stage == BatchStage.TASK_COMPLETED:
                output_mgr.success("Metrics processing completed.")
                return
            if stage == BatchStage.TASK_FAILED:
                output_mgr.error(f"Failed to initialize metrics: {event.error}")

        try:
            successful, error_message, metrics_result = init_success_story_metrics(
                agent_config,
                str(success_path),
                subject_tree,
                emit=emit,
                extra_instructions=extra_instructions,
            )
        finally:
            output_mgr.stop()

        # Render markdown summary after Live display stops
        output_mgr.render_markdown_summary(title="Metrics Summary")

        if successful:
            console.print("[green]Metrics initialized[/]")
            return True, metrics_result
        else:
            console.print(" [red]Error:[/] Metrics initialization failed:")
            console.print(f"    {escape(str(error_message))}")
            return False, None
    except Exception as e:
        print_rich_exception(console, e, "Metrics initialization failed", logger)
        return False, None


def init_semantic_model(
    success_path: Path,
    agent_config: AgentConfig,
    build_mode: Literal["incremental", "overwrite"] = "incremental",
    console: Optional[Console] = None,
    force: bool = False,
) -> tuple[bool, Optional[dict[str, Any]]]:
    """Initialize semantic model using success stories.

    Args:
        success_path: Path to success story CSV file
        agent_config: Agent configuration
        build_mode: Build mode (incremental or overwrite)
        console: Optional Rich console for output
        force: If True, skip confirmation prompts for deletion

    Returns:
        Tuple of (success: bool, result: Optional[dict])
    """
    from rich.markup import escape

    from datus.schemas.batch_events import BatchEvent, BatchStage
    from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model
    from datus.storage.semantic_model.store import SemanticModelRAG
    from datus.utils.stream_output import StreamOutputManager

    if not console:
        console = Console(log_path=False)

    try:
        if build_mode == "overwrite":
            from datus.storage.backend_holder import create_vector_connection

            db = create_vector_connection(agent_config.project_name)
            try:
                db.drop_table("semantic_model", ignore_missing=True)
                logger.info("Dropped existing semantic_model table")
            finally:
                db.close()
            # Clear the datasource-scoped semantic_models directory (YAML files).
            semantic_yaml_dir = agent_config.path_manager.semantic_model_path(agent_config.current_datasource)
            if semantic_yaml_dir.exists() and not safe_rmtree(
                semantic_yaml_dir,
                f"project semantic YAML directory (shared by all databases in {agent_config.project_name!r})",
                force=force,
            ):
                console.print("[yellow]Cancelled by user[/yellow]")
                return False, None
            agent_config.save_storage_config("semantic_model")

        # Create StreamOutputManager
        output_mgr = StreamOutputManager(
            console=console,
            max_message_lines=10,
            show_progress=True,
            title="Semantic Model Initialization",
        )

        def emit(event: BatchEvent) -> None:
            stage = event.stage

            if stage == BatchStage.TASK_STARTED:
                output_mgr.start(total_items=1, description="Initializing semantic model")
                return

            if stage == BatchStage.ITEM_PROCESSING:
                payload = event.payload or {}
                messages = payload.get("messages")
                if messages:
                    output_mgr.add_llm_output(str(messages))
                # Capture summary from the final action (SemanticNodeResult with success + response)
                output = payload.get("output")
                if output and isinstance(output, dict) and output.get("success") and output.get("response"):
                    response = output["response"]
                    if isinstance(response, str) and response.strip():
                        output_mgr.summary_outputs.clear()
                        output_mgr.add_summary_content(response)
                return

            if stage == BatchStage.TASK_COMPLETED:
                output_mgr.success("Semantic model processing completed.")
                return
            if stage == BatchStage.TASK_FAILED:
                output_mgr.error(f"Failed to initialize semantic model: {event.error}")

        try:
            successful, error_message = init_success_story_semantic_model(agent_config, str(success_path), emit=emit)
        finally:
            output_mgr.stop()

        # Render markdown summary after Live display stops
        output_mgr.render_markdown_summary(title="Semantic Model Summary")

        if successful:
            console.print("[green]Semantic model initialized[/]")
            # Get semantic model count
            try:
                semantic_rag = SemanticModelRAG(agent_config)
                result = {"semantic_model_count": semantic_rag.get_size()}
            except Exception:
                result = {}
            return True, result
        else:
            console.print(" [red]Error:[/] Semantic model initialization failed:")
            console.print(f"    {escape(str(error_message))}")
            return False, None
    except Exception as e:
        print_rich_exception(console, e, "Semantic model initialization failed", logger)
        return False, None
