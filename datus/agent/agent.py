# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import csv
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, AsyncGenerator, Dict, Optional, Set

from pydantic import ValidationError

from datus.agent.workflow_runner import WorkflowRunner
from datus.configuration.agent_config import AgentConfig, BenchmarkConfig
from datus.models.base import LLMBaseModel

# Import model implementations
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.agent_models import SubAgentConfig
from datus.schemas.batch_events import BatchEvent, BatchStage
from datus.schemas.node_models import SqlTask
from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.storage.metric.metric_init import init_semantic_yaml_metrics, init_success_story_metrics
from datus.storage.metric.store import MetricRAG
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema
from datus.storage.schema_metadata.benchmark_init_bird import init_dev_schema
from datus.storage.schema_metadata.local_init import init_local_schema
from datus.storage.semantic_model.semantic_model_init import (
    init_semantic_yaml_semantic_model,
    init_success_story_semantic_model,
)
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS as SUB_AGENT_COMPONENTS
from datus.storage.sub_agent_kb_bootstrap import SubAgentBootstrapper
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.utils.benchmark_utils import load_benchmark_tasks
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import to_str
from datus.utils.loggings import get_logger
from datus.utils.time_utils import format_duration_human

logger = get_logger(__name__)


class Agent:
    """
    Main entry point for the SQL Agent system.
    Handles initialization, workflow management, and execution loop.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        agent_config: AgentConfig,
        db_manager: Optional[DBManager] = None,
    ):
        """
        Initialize the Agent with configuration parameters.

        Args:
            args: Command line arguments and configuration
            agent_config: Pre-loaded agent configuration
            db_manager: Optional database manager instance
        """
        self.args = args
        self.global_config = agent_config
        if db_manager:
            self.db_manager = db_manager
        else:
            self.db_manager = db_manager_instance(self.global_config.namespaces)

        self.tools = {}
        self.storage_modules = {}
        self.metadata_store = None
        self.metrics_store = None
        self._ref_sql_file_sql_counter: Dict[str, int] = {}
        self._metrics_row_stage_seen: Dict[int, Set[str]] = {}
        self._print_lock = threading.Lock()
        self._check_storage_modules()

    def _initialize_model(self) -> LLMBaseModel:
        llm_model = LLMBaseModel.create_model(model_name="default", agent_config=self.global_config)
        logger.info(f"Using model type: {llm_model.model_config.type}, model name: {llm_model.model_config.model}")

        return llm_model

    # def _setup_database_conn(self):
    #     """
    #     Set up the environment by initializing necessary tools and connectors.
    #     """
    #     # Initialize database tools based on task type
    #     self.database_connector = self.global_config.connector()

    def _check_storage_modules(self):
        """
        Check if storage modules exist and initialize them if needed.
        """
        # Check and initialize lineage graph
        if os.path.exists(os.path.join("storage", "schema_metadata")):
            # Initialize lineage graph storage

            self.storage_modules["schema_metadata"] = True

        # Check and initialize metrics store
        if os.path.exists(os.path.join("storage", "metric_store")):
            # Initialize metrics store
            self.storage_modules["metric_store"] = True

        # Check and initialize document storage
        if os.path.exists(os.path.join("storage", "document")):
            # Initialize document storage
            self.storage_modules["document"] = True

        # Check and initialize success story storage
        if os.path.exists(os.path.join("storage", "success_story")):
            # Initialize success story storage
            self.storage_modules["success_story"] = True

        logger.info(f"Storage modules initialized: {list(self.storage_modules.keys())}")

    def create_workflow_runner(self, check_db: bool = True, run_id: Optional[str] = None) -> WorkflowRunner:
        """Create a workflow runner that can safely execute in isolation."""
        return WorkflowRunner(
            self.args, self.global_config, pre_run_callable=self.check_db if check_db else None, run_id=run_id
        )

    def run(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        check_db: bool = True,
        run_id: Optional[str] = None,
    ) -> dict:
        """
        Execute a workflow synchronously via a dedicated runner.
        """
        runner = self.create_workflow_runner(check_db=check_db, run_id=run_id)
        return runner.run(sql_task=sql_task, check_storage=check_storage)

    async def run_stream(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute a workflow with streaming progress updates."""
        runner = self.create_workflow_runner()
        async for action in runner.run_stream(
            sql_task=sql_task,
            check_storage=check_storage,
            action_history_manager=action_history_manager,
        ):
            yield action

    def check_db(self):
        """Validate database connectivity."""
        logger.info("Checking database connectivity")
        namespace = self.global_config.current_namespace
        if namespace in self.global_config.namespaces:
            connections = self.db_manager.get_connections(namespace)
            if not connections:
                logger.warning(f"No connections found for {namespace}")
                return {"status": "error", "message": f"No connections found for {namespace}"}
            if isinstance(connections, dict):
                for name, conn in connections.items():
                    conn.test_connection()
                    logger.info(f"Database connection test successful for {name}")
            else:
                connections.test_connection()
                logger.info(f"Database connection test successful {namespace}")
            return {"status": "success", "message": "Database connection test successful"}
        else:
            logger.error(f"Database connection test failed: {namespace} not found in namespaces")
            return {"status": "error", "message": f"{namespace} not found in namespaces"}

    def probe_llm(self):
        """Test LLM model connectivity."""
        logger.info("Testing LLM model connectivity")
        try:
            llm_model = LLMBaseModel.create_model(model_name="default", agent_config=self.global_config)
            logger.info(
                f"Using model type: {llm_model.model_config.type}, " f"model name: {llm_model.model_config.model}"
            )

            response = llm_model.generate("Hello, can you hear me?")
            logger.info("LLM model test successful")
            return {
                "status": "success",
                "message": "LLM model test successful",
                "response": response,
            }
        except Exception as e:
            logger.error(f"LLM model test failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _refresh_scoped_agents(self, component: str, kb_strategy: str):
        """Rebuild scoped knowledge bases for sub-agents after global bootstrap."""
        if component not in SUB_AGENT_COMPONENTS:
            return
        if kb_strategy not in {"overwrite", "incremental"}:
            return

        agent_nodes = getattr(self.global_config, "agentic_nodes", {}) or {}
        if not agent_nodes:
            return
        current_namespace = self.global_config.current_namespace
        for name, raw_config in agent_nodes.items():
            if name in SYS_SUB_AGENTS:
                continue
            try:
                sub_config = SubAgentConfig.model_validate(raw_config)
            except ValidationError as exc:
                logger.warning(f"Skipping sub-agent '{name}' due to invalid configuration: {exc}")
                continue
            if not sub_config.is_in_namespace(current_namespace):
                logger.debug(
                    f"Skipping sub-agent '{name}' for component '{component}' "
                    f" because there is no corresponding scope context configured under namespace {current_namespace}"
                )
                continue

            try:
                bootstrapper = SubAgentBootstrapper(
                    sub_agent=sub_config,
                    agent_config=self.global_config,
                )
                logger.info(
                    f"Running SubAgentBootstrapper for sub-agent '{name}' (component={component}, "
                    f"strategy=overwrite, storage={bootstrapper.storage_path})"
                )
                result = bootstrapper.run([component], "overwrite")
                if not result.should_bootstrap:
                    reason = result.reason or "No scoped context provided"
                    logger.info(f"SubAgentBootstrapper skipped for sub-agent '{name}': {reason}")
                else:
                    component_summaries = []
                    for comp_result in result.results:
                        summary = f"{comp_result.component}:{comp_result.status}"
                        if comp_result.message:
                            summary = f"{summary} ({comp_result.message})"
                        component_summaries.append(summary)
                    component_summaries_str = (
                        ", ".join(component_summaries) if component_summaries else "no component results"
                    )
                    logger.info(
                        f"Bootstrap finished for sub-agent '{name}' (storage={result.storage_path}): "
                        f"{component_summaries_str}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to refresh scoped KB for sub-agent '{name}': {exc}")

    def _reset_reference_sql_stream_state(self) -> None:
        self._ref_sql_file_sql_counter = {}

    def _reset_metrics_stream_state(self) -> None:
        self._metrics_row_stage_seen = {}

    def _print_stream_lines(self, message: Optional[object], indent: str = "  ", prefix: str = "") -> None:
        if not message:
            return
        text = str(message).strip()
        if not text:
            return
        for line in text.splitlines():
            print(f"{prefix}{indent}{line}", flush=True)

    def _next_reference_sql_number(self, filepath: str) -> int:
        with self._print_lock:
            count = self._ref_sql_file_sql_counter.get(filepath, 0) + 1
            self._ref_sql_file_sql_counter[filepath] = count
            return count

    def _format_reference_sql_line(self, sql_text: str, number: int) -> str:
        condensed = " ".join(str(sql_text).split())
        return condensed or f"sql_{number}"

    def _get_file_short_name(self, filepath: str) -> str:
        """Extract short filename for display prefix."""
        basename = os.path.basename(filepath)
        name, _ = os.path.splitext(basename)
        return name

    def _emit_reference_sql_event(self, event: BatchEvent) -> None:
        stage = event.stage
        filepath = event.group_id or "unknown_file"
        short_name = self._get_file_short_name(filepath)
        prefix = f"[{short_name}] "

        if stage == BatchStage.GROUP_STARTED:
            logger.info(f"reference_sql file start: {filepath}")
            print(f"{prefix}start processing {filepath}", flush=True)
            return

        if stage == BatchStage.GROUP_COMPLETED:
            logger.info(f"reference_sql file complete: {filepath}")
            print(f"{prefix}completed", flush=True)
            return

        if stage == BatchStage.ITEM_STARTED:
            payload = event.payload or {}
            number = self._next_reference_sql_number(filepath)
            sql_line = self._format_reference_sql_line(str(payload.get("sql") or ""), number)
            print(f"{prefix}#{number}. {sql_line}", flush=True)
            return

        if stage == BatchStage.ITEM_PROCESSING:
            payload = event.payload or {}
            self._print_stream_lines(payload.get("output", {}).get("raw_output"), prefix=prefix)
            return

        if stage == BatchStage.ITEM_FAILED:
            error = event.error
            if error:
                self._print_stream_lines(error, prefix=prefix)
            return

    def _emit_metrics_event(self, event: BatchEvent) -> None:
        stage = event.stage
        payload = event.payload or {}
        row_idx = payload.get("row_idx", "?")
        prefix = f"[row{row_idx}] "

        if stage == BatchStage.ITEM_STARTED:
            question = payload.get("question") or ""
            logger.info(f"metrics row start: {row_idx}")
            print(f"{prefix}start: {question}".strip(), flush=True)
            table_name = payload.get("table_name")
            if table_name:
                print(f"{prefix}  table: {table_name}", flush=True)
            return

        if stage == BatchStage.ITEM_PROCESSING:
            action_name = event.action_name or "action"
            with self._print_lock:
                seen = self._metrics_row_stage_seen.setdefault(row_idx, set())
                if action_name not in seen:
                    seen.add(action_name)
                    print(f"{prefix}  {action_name}:", flush=True)
            self._print_stream_lines(payload.get("output", {}).get("raw_output"), indent="    ", prefix=prefix)
            # Check for semantic model output
            output = payload.get("output")
            if isinstance(output, dict):
                semantic_model_file = output.get("semantic_model")
                if semantic_model_file:
                    print(f"{prefix}  semantic_model: {semantic_model_file}", flush=True)
            return

        if stage == BatchStage.ITEM_COMPLETED:
            logger.info(f"metrics row success: {row_idx}")
            print(f"{prefix}completed", flush=True)
            return

        if stage == BatchStage.ITEM_FAILED:
            error = event.error or "unknown error"
            logger.error(f"metrics row error: {row_idx}")
            print(f"{prefix}error: {error}", flush=True)
            return

    def bootstrap_kb(self):
        """Initialize knowledge base storage components."""
        logger.info("Initializing knowledge base components")
        results = {}
        # Get selected components from args
        selected_components = self.args.components

        kb_update_strategy = self.args.kb_update_strategy
        benchmark_platform = self.args.benchmark
        pool_size = 4 if not self.args.pool_size else self.args.pool_size
        dir_path = self.global_config.rag_storage_path()

        # Parse subject_tree from command line if provided
        subject_tree = None
        if hasattr(self.args, "subject_tree") and self.args.subject_tree:
            # Parse comma-separated string into list
            subject_tree = [s.strip() for s in self.args.subject_tree.split(",") if s.strip()]
            logger.info(f"Using predefined subject_tree categories: {subject_tree}")

        for component in selected_components:
            # db_name = component_dirs[component]``
            # Initialize corresponding stores
            if component == "metadata":
                if kb_update_strategy == "check":
                    if not os.path.exists(dir_path):
                        raise ValueError("metadata is not built, please run bootstrap_kb with overwrite strategy first")
                    else:
                        self.global_config.check_init_storage_config("database")

                        self.metadata_store = SchemaWithValueRAG(self.global_config)
                        return {
                            "status": "success",
                            "message": f"current metadata is already built, "
                            f"dir_path={dir_path},"
                            f"schema_size={self.metadata_store.get_schema_size()}, "
                            f"value_size={self.metadata_store.get_value_size()}",
                        }

                if kb_update_strategy == "overwrite":
                    self.global_config.save_storage_config("database")
                    schema_metadata_path = os.path.join(dir_path, "schema_metadata.lance")
                    if os.path.exists(schema_metadata_path):
                        shutil.rmtree(schema_metadata_path)
                        logger.info(f"Deleted existing directory {schema_metadata_path}")
                    schema_value_path = os.path.join(dir_path, "schema_value.lance")
                    if os.path.exists(schema_value_path):
                        shutil.rmtree(schema_value_path)
                        logger.info(f"Deleted existing directory {schema_value_path}")
                else:
                    self.global_config.check_init_storage_config("database")
                self.metadata_store = SchemaWithValueRAG(self.global_config)

                if not benchmark_platform:
                    self.check_db()
                    init_local_schema(
                        self.metadata_store,
                        self.global_config,
                        self.db_manager,
                        kb_update_strategy,
                        table_type=self.args.schema_linking_type,
                        init_catalog_name=self.args.catalog or "",
                        init_database_name=self.args.database_name or "",
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "spider2":
                    benchmark_path = self.global_config.benchmark_path(benchmark_platform)

                    init_snowflake_schema(
                        self.metadata_store,
                        benchmark_path,
                        kb_update_strategy,
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "bird_dev":
                    self.check_db()
                    benchmark_path = self.global_config.benchmark_path(benchmark_platform)
                    init_dev_schema(
                        self.metadata_store,
                        self.db_manager,
                        self.global_config.current_namespace,
                        benchmark_path,
                        kb_update_strategy,
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "bird_critic":
                    # TODO init bird_critic schema
                    raise DatusException(
                        ErrorCode.COMMON_VALIDATION_FAILED,
                        message=f"Unsupported benchmark platform: {benchmark_platform}",
                    )
                else:
                    raise DatusException(
                        ErrorCode.COMMON_VALIDATION_FAILED, f"Unsupported benchmark platform: {benchmark_platform}"
                    )

                result = {
                    "status": "success",
                    "message": f"metadata bootstrap completed, "
                    f"schema_size={self.metadata_store.get_schema_size()}, "
                    f"value_size={self.metadata_store.get_value_size()}",
                }
                self._refresh_scoped_agents("metadata", kb_update_strategy)
                return result

            elif component == "semantic_model":
                semantic_model_path = os.path.join(dir_path, "semantic_model.lance")
                if kb_update_strategy == "overwrite":
                    if os.path.exists(semantic_model_path):
                        shutil.rmtree(semantic_model_path)
                        logger.info(f"Deleted existing directory {semantic_model_path}")
                    self.global_config.save_storage_config("semantic_model")
                else:
                    self.global_config.check_init_storage_config("semantic_model")

                # Initialize semantic model
                if hasattr(self.args, "semantic_yaml") and self.args.semantic_yaml:
                    successful, error_message = init_semantic_yaml_semantic_model(
                        self.args.semantic_yaml, self.global_config
                    )
                else:
                    successful, error_message = init_success_story_semantic_model(self.args, self.global_config)

                if successful:
                    temp_rag = SemanticModelRAG(self.global_config)
                    result = {
                        "status": "success",
                        "message": f"semantic_model bootstrap completed, "
                        f"semantic_object_count={temp_rag.get_size()}",
                        "error": error_message,
                    }
                    self._refresh_scoped_agents("semantic_model", kb_update_strategy)
                else:
                    result = {"status": "failed", "message": error_message}
                return result

            elif component == "metrics":
                metrics_path = os.path.join(dir_path, "metrics.lance")
                if kb_update_strategy == "overwrite":
                    if os.path.exists(metrics_path):
                        shutil.rmtree(metrics_path)
                        logger.info(f"Deleted existing directory {metrics_path}")
                    self.global_config.save_storage_config("metric")  # Keep compatibility
                else:
                    self.global_config.check_init_storage_config("metric")
                self._reset_metrics_stream_state()
                # Initialize metrics using unified SemanticAgenticNode approach
                if hasattr(self.args, "semantic_yaml") and self.args.semantic_yaml:
                    successful, error_message = init_semantic_yaml_metrics(
                        self.args.semantic_yaml,
                        self.global_config,
                    )
                else:
                    successful, error_message = init_success_story_metrics(
                        self.args,
                        self.global_config,
                        subject_tree,
                        emit=self._emit_metrics_event,
                        # pool_size=pool_size,
                    )

                if successful:
                    self.metrics_store = MetricRAG(self.global_config)
                    result = {
                        "status": "success",
                        "message": f"metrics bootstrap completed, "
                        f"metrics_count={self.metrics_store.get_metrics_size()}",
                        "error": error_message,
                    }
                    self._refresh_scoped_agents("metrics", kb_update_strategy)
                else:
                    result = {"status": "failed", "message": error_message}
                return result
            elif component == "document":
                from datus.storage.document.store import document_store

                self.storage_modules["document_store"] = document_store(self.global_config.rag_storage_path())
                # self.global_config.check_init_storage_config("document")
            elif component == "ext_knowledge":
                ext_knowledge_path = os.path.join(dir_path, "ext_knowledge.lance")
                if kb_update_strategy == "overwrite":
                    if os.path.exists(ext_knowledge_path):
                        shutil.rmtree(ext_knowledge_path)
                        logger.info(f"Deleted existing directory {ext_knowledge_path}")
                    self.global_config.save_storage_config("ext_knowledge")
                else:
                    self.global_config.check_init_storage_config("ext_knowledge")
                self.ext_knowledge_store = ExtKnowledgeStore(dir_path)
                init_ext_knowledge(
                    self.ext_knowledge_store, self.args, build_mode=kb_update_strategy, pool_size=pool_size
                )
                return {
                    "status": "success",
                    "message": f"ext_knowledge bootstrap completed, "
                    f"knowledge_size={self.ext_knowledge_store.table_size()}",
                }
            elif component == "reference_sql":
                reference_sql_path = os.path.join(dir_path, "reference_sql.lance")
                if kb_update_strategy == "overwrite":
                    if os.path.exists(reference_sql_path):
                        shutil.rmtree(reference_sql_path)
                        logger.info(f"Deleted existing directory {reference_sql_path}")
                    self.global_config.save_storage_config("reference_sql")
                else:
                    self.global_config.check_init_storage_config("reference_sql")

                # Initialize reference SQL storage
                from datus.storage.reference_sql import ReferenceSqlRAG
                from datus.storage.reference_sql.reference_sql_init import init_reference_sql

                self.reference_sql_store = ReferenceSqlRAG(self.global_config)
                self._reset_reference_sql_stream_state()
                result = init_reference_sql(
                    self.reference_sql_store,
                    self.global_config,
                    self.args.sql_dir,
                    validate_only=self.args.validate_only or False,
                    build_mode=kb_update_strategy,
                    pool_size=pool_size,
                    subject_tree=subject_tree,
                    emit=self._emit_reference_sql_event,
                )
                if isinstance(result, dict) and result.get("status") != "error":
                    self._refresh_scoped_agents("reference_sql", kb_update_strategy)
                return result
            results[component] = True

        # Initialize success story storage (always created)
        success_story_path = os.path.join("storage", "success_story")
        if not os.path.exists(success_story_path):
            os.makedirs(success_story_path)
        results["success_story"] = True

        logger.info("Knowledge base components initialized successfully: " f"{', '.join(selected_components)}")
        return {
            "status": "success",
            "message": "Knowledge base initialized",
            "components": results,
        }

    def benchmark(self):
        logger.info("Benchmarking begins")
        benchmark_platform = self.args.benchmark
        benchmark_path = self.global_config.benchmark_path(benchmark_platform)

        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark_path not found: {benchmark_path}")

        target_task_ids = getattr(self.args, "benchmark_task_ids", [])
        target_task_ids = set(target_task_ids) if target_task_ids else None
        import time
        from datetime import datetime

        # Generate a shared run_id for this benchmark run
        benchmark_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Benchmark run_id: {benchmark_run_id}")

        start = time.perf_counter()
        if benchmark_platform == "semantic_layer":
            self.global_config.check_init_storage_config("metric")
            result = self.benchmark_semantic_layer(benchmark_path, target_task_ids, run_id=benchmark_run_id)
        else:
            self.global_config.check_init_storage_config("database")
            self.global_config.check_init_storage_config("metric")
            result = self.do_benchmark(benchmark_platform, target_task_ids, run_id=benchmark_run_id)
        end = time.perf_counter()

        time_spends = end - start
        result["time_spends"] = format_duration_human(time_spends)
        result["time_spends_seconds"] = str(time_spends)
        result["run_id"] = benchmark_run_id
        return result

    def do_benchmark(
        self, benchmark_platform: str, target_task_ids: Optional[Set[str]] = None, run_id: Optional[str] = None
    ):
        _, conn = db_manager_instance(self.global_config.namespaces).first_conn_with_name(
            self.global_config.current_namespace
        )
        self.check_db()

        def run_single_task(task_id: str, benchmark_config: BenchmarkConfig, task_item: Dict[str, Any]):
            """Execute a single benchmark task"""
            task = task_item.get(benchmark_config.question_key)
            if not task:
                logger.warning(
                    f"The question content was not obtained through {benchmark_config.question_key}, "
                    "please check your benchmark configuration."
                )
                return task_id, ""
            database_name = task_item.get(benchmark_config.db_key) or conn.database_name or ""
            logger.info(f"start benchmark with {task_id}: {task}")
            use_tables = None if not benchmark_config.use_tables_key else task_item.get(benchmark_config.use_tables_key)

            # Use hierarchical save directory structure
            output_dir = self.global_config.get_save_run_dir(run_id) if run_id else self.global_config.output_dir

            result = self.run(
                SqlTask(
                    id=task_id,
                    database_type=conn.dialect,
                    task=task,
                    database_name=database_name,
                    output_dir=output_dir,
                    current_date=self.args.current_date,
                    tables=use_tables,
                    external_knowledge=""
                    if not benchmark_config.ext_knowledge_key
                    else task_item.get(benchmark_config.ext_knowledge_key, ""),
                    schema_linking_type="full",
                ),
                check_storage=False,
                check_db=False,
                run_id=run_id,
            )
            logger.info(f"Finish benchmark with {task_id}, " f"file saved in {output_dir}/{task_id}.csv.")
            return task_id, result

        max_workers = getattr(self.args, "max_workers", 1) or 1
        logger.info(f"Loaded tasks from {benchmark_platform} benchmark")
        benchmark_config = self.global_config.benchmark_config(benchmark_platform)
        task_id_key = benchmark_config.question_id_key or "_task_id"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task_item in load_benchmark_tasks(self.global_config, benchmark_platform):
                raw_task_id = task_item.get(task_id_key)
                if raw_task_id in (None, ""):
                    logger.warning(f"Task id {raw_task_id} was not found, please check your benchmark configuration.")
                    continue
                else:
                    task_id = str(raw_task_id)
                task_item[task_id_key] = task_id
                if not target_task_ids or task_id in target_task_ids:
                    f = executor.submit(run_single_task, task_id, benchmark_config, task_item)
                    future_to_task[f] = task_item

            # Wait for completion
            for future in as_completed(future_to_task):
                task_item = future_to_task[future]
                try:
                    task_id, _ = future.result()
                    logger.debug(f"Task {task_id} completed successfully")
                except Exception as exc:
                    task_id = task_item.get(task_id_key) or task_item.get("_task_id")
                    logger.error(f"Task {task_id} generated an exception: {exc}")
        logger.info("Benchmark execution completed.")
        return {"status": "success", "message": "Benchmark tasks executed successfully"}

    def benchmark_semantic_layer(
        self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None, run_id: Optional[str] = None
    ):
        task_file = self.args.testing_set
        self._check_benchmark_file(task_file)

        # Clean up previous execution results to avoid interference
        self._cleanup_benchmark_output_paths(benchmark_path)

        tasks = []
        with open(task_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, 1):
                logger.debug(f"line {line_no}: {row}")
                if "question" in row and "sql" in row and row["question"].strip() and row["sql"].strip():
                    task_data = {"question_id": line_no, "question": row["question"].strip(), "sql": row["sql"].strip()}
                    # Check if ext_knowledge column exists and add it to task data
                    if "external_knowledge" in row and row["external_knowledge"].strip():
                        task_data["external_knowledge"] = row["external_knowledge"].strip()
                    tasks.append(task_data)

        logger.info(f"Loaded {len(tasks)} tasks from semantic_layer benchmark")

        metric_meta = self.global_config.current_metric_meta(self.args.metric_meta)

        for task in tasks:
            task_id = str(task["question_id"])
            if target_task_ids and task_id not in target_task_ids:
                continue

            question = task["question"]
            logger.info(f"start benchmark with {task_id}: {question}")
            current_db_config = self.global_config.current_db_config()

            # Merge external knowledge from file with metric_meta
            combined_ext_knowledge = metric_meta.ext_knowledge
            if "external_knowledge" in task and task["external_knowledge"]:
                if combined_ext_knowledge:
                    # Combine both knowledge sources
                    combined_ext_knowledge = f"{combined_ext_knowledge}\n\n{task['external_knowledge']}"
                else:
                    # Use only file knowledge if metric_meta doesn't have any
                    combined_ext_knowledge = task["external_knowledge"]

            # Use hierarchical save directory structure
            output_dir = self.global_config.get_save_run_dir(run_id) if run_id else self.global_config.output_dir

            if metric_meta.subject_path and metric_meta.subject_path.strip():
                subject_path = [c.strip() for c in metric_meta.subject_path.split("/") if c.strip()]
            else:
                subject_path = None
            self.run(
                SqlTask(
                    id=task_id,
                    database_type=current_db_config.type,
                    task=question,
                    database_name=current_db_config.database,
                    schema_name=current_db_config.schema,
                    subject_path=subject_path,
                    output_dir=output_dir,
                    external_knowledge=combined_ext_knowledge,
                    current_date=self.args.current_date,
                ),
                run_id=run_id,
            )

            logger.info(f"Finish benchmark with {task_id}, " f"file saved in {output_dir}/{task_id}.csv.")

        return {"status": "success", "message": "Benchmark tasks executed successfully"}

    def _check_benchmark_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmarking task file not found, file_path={file_path}")

    def _cleanup_benchmark_output_paths(self, benchmark_path: str):
        """Clean up previous benchmark execution results to avoid interference."""
        current_namespace = self.global_config.current_namespace

        # Clean up namespace directory in output directory
        output_dir = self.global_config.output_dir
        namespace_dir = os.path.join(output_dir, current_namespace)
        if os.path.exists(namespace_dir):
            logger.info(f"Cleaning up namespace directory: {namespace_dir}")
            try:
                shutil.rmtree(namespace_dir)
                logger.info(f"Successfully removed namespace directory: {namespace_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean namespace directory {namespace_dir}: {e}")

        # Clean up gold directory (which contains exec_result)
        gold_path = os.path.join(benchmark_path, "gold")
        if os.path.exists(gold_path):
            logger.info(f"Cleaning up gold directory: {gold_path}")
            try:
                shutil.rmtree(gold_path)
                logger.info(f"Successfully removed gold directory: {gold_path}")
            except Exception as e:
                logger.warning(f"Failed to clean gold directory {gold_path}: {e}")

    def benchmark_bird_critic(self):
        pass

    def evaluation(self, log_summary: bool = True) -> Dict[str, Any]:
        """Evaluate the benchmarking"""
        benchmark_platform = self.args.benchmark
        if benchmark_platform in ("semantic_layer", "bird_critic"):
            return {
                "status": "failed",
                "message": "Benchmark bird_critic and semantic_layer evaluation is not supported at the moment",
            }

        from datus.utils.benchmark_utils import evaluate_benchmark_and_report

        run_id = getattr(self.args, "run_id", None)
        summary_report_file = getattr(self.args, "summary_report_file", None)
        evaluation_result = evaluate_benchmark_and_report(
            agent_config=self.global_config,
            benchmark_platform=benchmark_platform,
            target_task_ids=self.args.task_ids,
            output_file=self.args.output_file,
            log_summary=log_summary,
            run_id=run_id,
            summary_report_file=summary_report_file,
        )
        return {
            "status": evaluation_result.get("status"),
            "generated_time": evaluation_result.get("generated_time"),
            "message": evaluation_result.get("error"),
        }

    def generate_dataset(self):
        """Generate dataset from trajectory files."""
        logger.info("Generating dataset from trajectory files")

        import glob
        import json

        import yaml

        trajectory_dir = self.args.trajectory_dir
        dataset_name = self.args.dataset_name
        output_format = getattr(self.args, "format", "json")
        benchmark_task_ids = getattr(self.args, "benchmark_task_ids", None)

        if not os.path.exists(trajectory_dir):
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

        # Parse benchmark_task_ids if provided
        allowed_task_ids = None
        if benchmark_task_ids:
            allowed_task_ids = [task_id.strip() for task_id in benchmark_task_ids.split(",")]
            logger.info(f"Filtering by task IDs: {allowed_task_ids}")

        # Find all trajectory YAML files
        trajectory_files = glob.glob(os.path.join(trajectory_dir, "*_*.yaml"))
        logger.info(f"Found {len(trajectory_files)} trajectory files")

        dataset_data = []

        for trajectory_file in trajectory_files:
            try:
                # Extract task_id from filename (e.g., "0_1750662901.yaml" -> "0")
                filename = os.path.basename(trajectory_file)
                task_id = filename.split("_")[0]

                # Filter by task_id if benchmark_task_ids is provided
                if allowed_task_ids and task_id not in allowed_task_ids:
                    logger.debug(f"Skipping trajectory file {filename} (task_id {task_id} not in allowed list)")
                    continue

                logger.info(f"Processing trajectory file: {filename}")

                # Load trajectory YAML file
                with open(trajectory_file, "r", encoding="utf-8") as f:
                    trajectory_data = yaml.safe_load(f)

                # Extract sql_contexts from the workflow
                sql_contexts = None
                first_sql_node_id = None

                if "workflow" in trajectory_data and "nodes" in trajectory_data["workflow"]:
                    for node in trajectory_data["workflow"]["nodes"]:
                        if node.get("type") in ["reasoning", "generate_sql"]:
                            if "result" in node and "sql_contexts" in node["result"]:
                                sql_contexts = node["result"]["sql_contexts"]
                                first_sql_node_id = node["id"]
                                break

                if not sql_contexts or not first_sql_node_id:
                    logger.warning(f"No sql_contexts found in {filename}")
                    continue

                # Load node details from the corresponding node file
                node_file = os.path.join(trajectory_dir, task_id, f"{first_sql_node_id}.yml")
                if not os.path.exists(node_file):
                    logger.warning(f"Node file not found: {node_file}")
                    continue

                with open(node_file, "r", encoding="utf-8") as f:
                    node_data = yaml.safe_load(f)

                # Extract required fields
                user_prompt = node_data.get("user_prompt", "")
                system_prompt = node_data.get("system_prompt", "")
                reason_content = node_data.get("reason_content", [])
                output_content = node_data.get("output_content", "")

                # Create dataset entry
                dataset_entry = {
                    "task_id": task_id,
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "reason_content": reason_content,
                    "sql_contexts": sql_contexts,
                    "output_content": output_content,
                }

                dataset_data.append(dataset_entry)
                logger.info(f"Successfully processed {filename}")

            except Exception as e:
                logger.error(f"Error processing {trajectory_file}: {str(e)}")
                continue

        # Save dataset to file based on format
        if output_format == "json":
            output_file = f"{dataset_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        elif output_format == "parquet":
            try:
                import pandas as pd

                output_file = f"{dataset_name}.parquet"

                # Convert the dataset to a pandas DataFrame
                # For nested structures, we'll convert them to strings
                df_data = []
                for entry in dataset_data:
                    df_entry = {
                        "user_prompt": entry["user_prompt"],
                        "system_prompt": entry["system_prompt"],
                        "reason_content": to_str(entry["reason_content"]),
                        "sql_contexts": to_str(entry["sql_contexts"]),
                        "output_content": entry["output_content"],
                    }
                    df_data.append(df_entry)

                df = pd.DataFrame(df_data)
                df.to_parquet(output_file, index=False)

            except ImportError:
                logger.error(
                    "pandas is required for parquet format. Please install it with: pip install pandas pyarrow"
                )
                return {
                    "status": "error",
                    "message": "pandas is required for parquet format. "
                    "Please install it with: pip install pandas pyarrow",
                }

        filter_info = f" (filtered by task IDs: {allowed_task_ids})" if allowed_task_ids else ""
        logger.info(f"Dataset generated successfully: {output_file}")
        logger.info(f"Total entries: {len(dataset_data)}{filter_info}")

        return {
            "status": "success",
            "message": f"Dataset generated successfully: {output_file}",
            "total_entries": len(dataset_data),
            "output_file": output_file,
            "format": output_format,
            "filtered_task_ids": allowed_task_ids,
        }
