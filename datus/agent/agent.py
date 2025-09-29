import argparse
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncGenerator, Optional, Set

from datus.agent.evaluate import evaluate_result, setup_node_input
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.models.base import LLMBaseModel

# Import model implementations
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import BaseResult, SqlTask
from datus.storage.document import DocumentStore
from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.storage.metric.metrics_init import init_semantic_yaml_metrics, init_success_story_metrics
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema
from datus.storage.schema_metadata.benchmark_init_bird import init_dev_schema
from datus.storage.schema_metadata.local_init import init_local_schema
from datus.storage.schema_metadata.store import rag_by_configuration
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.utils.benchmark_utils import (
    evaluate_and_report_accuracy,
    generate_gold_standard_results,
    load_bird_dev_tasks,
)
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

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
        self.workflow = None
        self.workflow_ready = False
        # self._setup_database_conn()
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

    def initialize_workflow(self, sql_task: SqlTask):
        """
        Generate an initial workflow using the planning module.
        """
        # Use plan from args if provided, otherwise use config default
        plan_type = getattr(self.args, "plan", None) or self.global_config.workflow_plan

        self.workflow = generate_workflow(
            task=sql_task,
            plan_type=plan_type,
            agent_config=self.global_config,
        )

        # Display the initial workflow
        self.workflow.display()
        logger.info("Initial workflow generated")

    def resume_workflow(self, config: argparse.Namespace):
        """
        Resume workflow from YAML config file and continue execution
        Args:
            workflow_config: Path to YAML configuration file
        """
        logger.info(f"Resuming workflow from config: {config}")

        try:
            # Load workflow from YAML
            self.workflow = Workflow.load(config.load_cp)
            self.workflow.global_config = self.global_config
            # self.workflow.metadata = {"args": config}

            self.workflow.resume()

            # Display the initial workflow
            self.workflow.display()
            logger.info("resume workflow from {workflow_config} successfully")
        except Exception as e:
            logger.error(f"Failed to resume workflow: {str(e)}")
            return {"status": "error", "message": str(e)}

    def is_complete(self):
        """
        Check if the workflow is complete.
        Returns:
            True if the workflow is none or complete, False otherwise
        """
        if self.workflow is None:
            return True
        return self.workflow.is_complete()

    def init_or_load_workflow(self, sql_task: SqlTask):
        """
        Initialize the workflow
        """
        # load_cp will be used to resume a workflow from a checkpoint and generate sqltask
        if self.args.load_cp:
            self.workflow_ready = False
            self.workflow = None
            self.resume_workflow(self.args)
        elif sql_task:
            self.workflow_ready = False
            self.workflow = None
            self.initialize_workflow(sql_task)
        elif self.workflow_ready:
            # if workflow is ready, just skip the initialization and run the workflow
            pass
        else:
            logger.error("Failed to initialize workflow. need a sql_task or to load from checkpoint.")
            return None

        if not self.workflow:
            logger.error("Failed to initialize workflow. Exiting.")
            return None
        self.workflow_ready = True
        return True

    @optional_traceable(name="agent")
    def run(self, sql_task: Optional[SqlTask] = None, check_storage: bool = False) -> dict:
        """
        Main execution loop for the agent.

        Returnsfinish benchmark_ids with:
            The final result of the workflow execution
        """
        if check_storage:
            self.global_config.check_init_storage_config("database")
        logger.info("Starting agent execution")

        if not self.init_or_load_workflow(sql_task):
            return {}
        self.check_db()
        # Main execution loop
        step_count = 0

        # Skip the first node (Start Node), and setup input for the next one
        if self.workflow.current_node_index == 0:
            self.workflow.get_current_node().complete(BaseResult(success=True))
            next_node = self.workflow.advance_to_next_node()
            setup_node_input(next_node, self.workflow)

        while not self.workflow.is_complete() and step_count < self.args.max_steps:
            # Get the next task
            current_node = self.workflow.get_current_node()
            if not current_node:
                logger.warning("No more tasks to execute. Exiting.")
                break

            # Execute the task
            logger.info(f"Executing task: {current_node.description}")
            current_node.run()
            if current_node.status == "failed":
                if current_node.type == NodeType.TYPE_PARALLEL:
                    try:
                        has_any_success = False
                        if current_node.result and hasattr(current_node.result, "child_results"):
                            for v in current_node.result.child_results.values():
                                ok = v.get("success", False) if isinstance(v, dict) else getattr(v, "success", False)
                                if ok:
                                    has_any_success = True
                                    break
                        if has_any_success:
                            logger.warning("Parallel node partial failure, continue to selection")
                        else:
                            logger.warning(f"Parallel node all failed: {current_node.description}")
                            break
                    except Exception:
                        logger.warning(f"Node failed: {current_node.description}")
                        break
                else:
                    logger.warning(f"Node failed: {current_node.description}")
                    break
            # evaluate the task result, update the context and setup the next node input if needed
            evaluation = evaluate_result(current_node, self.workflow)
            logger.debug(f"Evaluation result for {current_node.type}: {evaluation}")
            if not evaluation["success"]:
                logger.error(f"Setting {current_node.type} status to failed due to evaluation failure")
                current_node.status = "failed"
                break

            # Check for human intervention, use a new now to handle human feedback
            # if self.args.human_in_loop:
            #    human_feedback = handle_human_intervention(self.workflow)
            #    if human_feedback["modified"]:
            #        logger.info("Workflow modified by human intervention")

            self.workflow.advance_to_next_node()
            step_count += 1

        # Log if max steps reached
        if step_count >= self.args.max_steps:
            logger.warning(f"Workflow execution stopped after reaching max steps: {self.args.max_steps}")

        # Save trajectories and collect final results
        self.workflow.display()
        file_name = self.workflow.task.id
        timestamp = int(time.time())
        trajectory_dir = self.global_config.trajectory_dir

        # Ensure trajectory directory exists
        os.makedirs(trajectory_dir, exist_ok=True)

        save_path = f"{trajectory_dir}/{file_name}_{timestamp}.yaml"
        self.workflow.save(save_path)
        logger.info(f"Workflow saved to {save_path}")
        final_result = self.workflow.get_final_result()
        logger.info(f"Agent execution completed. Steps:{step_count}")

        return final_result

    def _create_action_history(
        self, action_id: str, messages: str, action_type: str, input_data: dict = None
    ) -> ActionHistory:
        """Helper method to create ActionHistory objects with consistent structure."""
        return ActionHistory(
            action_id=action_id,
            role=ActionRole.WORKFLOW,
            messages=messages,
            action_type=action_type,
            input=input_data or {},
            status=ActionStatus.PROCESSING,
        )

    def _update_action_status(self, action: ActionHistory, success: bool, output_data: dict = None, error: str = None):
        """Helper method to update ActionHistory status consistently."""
        if success:
            action.status = ActionStatus.SUCCESS
            action.output = output_data or {}
        else:
            action.status = ActionStatus.FAILED
            action.output = {"error": error or "Unknown error"}
            if output_data:
                action.output.update(output_data)

    async def run_stream(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Main execution loop for the agent with streaming support.

        Yields ActionHistory objects for each workflow step and node execution.

        Args:
            sql_task: SQL task to execute
            check_storage: Whether to check storage configuration
            action_history_manager: Manager for tracking action history

        Yields:
            ActionHistory: Progress updates throughout execution
        """
        if check_storage:
            self.global_config.check_init_storage_config("database")

        logger.info("Starting agent execution with streaming")

        # Workflow initialization action
        init_action = self._create_action_history(
            action_id="workflow_initialization",
            messages="Initializing workflow and checking prerequisites",
            action_type="workflow_init",
            input_data={
                "has_sql_task": bool(sql_task),
                "check_storage": check_storage,
                "load_from_checkpoint": bool(self.args.load_cp),
            },
        )
        yield init_action

        try:
            # Initialize workflow
            if not self.init_or_load_workflow(sql_task):
                self._update_action_status(init_action, success=False, error="Failed to initialize workflow")
                return

            self.check_db()

            self._update_action_status(
                init_action,
                success=True,
                output_data={
                    "workflow_ready": True,
                    "total_nodes": len(self.workflow.nodes) if self.workflow else 0,
                    "current_node_index": self.workflow.current_node_index if self.workflow else 0,
                },
            )

        except Exception as e:
            self._update_action_status(init_action, success=False, error=str(e))
            logger.error(f"Workflow initialization failed: {e}")
            return

        # Main execution loop with streaming
        step_count = 0

        # Skip the first node (Start Node), and setup input for the next one
        if self.workflow.current_node_index == 0:
            self.workflow.get_current_node().complete(BaseResult(success=True))
            next_node = self.workflow.advance_to_next_node()
            setup_node_input(next_node, self.workflow)

        while not self.workflow.is_complete() and step_count < self.args.max_steps:
            # Get the next task
            current_node = self.workflow.get_current_node()
            if not current_node:
                logger.warning("No more tasks to execute. Exiting.")
                break

            # Node execution start action
            node_start_action = self._create_action_history(
                action_id=f"node_execution_{current_node.id}",
                messages=f"Executing node: {current_node.description}",
                action_type="node_execution",
                input_data={
                    "node_id": current_node.id,
                    "node_type": current_node.type,
                    "description": current_node.description,
                    "step_count": step_count,
                },
            )
            yield node_start_action

            try:
                logger.info(f"Executing task: {current_node.description}")

                # Execute node with streaming support
                async for node_action in current_node.run_stream(action_history_manager):
                    yield node_action

                if current_node.status == "failed":
                    self._update_action_status(
                        node_start_action, success=False, error=f"Node execution failed: {current_node.description}"
                    )
                    logger.warning(f"Node failed: {current_node.description}")
                    break

                self._update_action_status(
                    node_start_action,
                    success=True,
                    output_data={
                        "node_completed": True,
                        "execution_successful": True,
                    },
                )

            except Exception as e:
                self._update_action_status(node_start_action, success=False, error=str(e))
                logger.error(f"Node execution error: {e}")
                break

            try:
                evaluation = evaluate_result(current_node, self.workflow)
                logger.debug(f"Evaluation result: {evaluation}")

                if not evaluation["success"]:
                    current_node.status = "failed"
                    break

            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                break

            self.workflow.advance_to_next_node()
            step_count += 1

        # Workflow completion action
        completion_action = self._create_action_history(
            action_id="workflow_completion",
            messages="Finalizing workflow execution and saving results",
            action_type="workflow_completion",
            input_data={
                "steps_completed": step_count,
                "max_steps_reached": step_count >= self.args.max_steps,
                "workflow_complete": self.workflow.is_complete(),
            },
        )
        yield completion_action

        try:
            # Save trajectories and collect final results
            self.workflow.display()
            file_name = self.workflow.task.id
            timestamp = int(time.time())
            trajectory_dir = self.global_config.trajectory_dir

            # Ensure trajectory directory exists
            os.makedirs(trajectory_dir, exist_ok=True)

            save_path = f"{trajectory_dir}/{file_name}_{timestamp}.yaml"
            self.workflow.save(save_path)
            logger.info(f"Workflow saved to {save_path}")

            final_result = self.workflow.get_final_result()
            logger.info(f"Agent execution completed. Steps:{step_count}")

            self._update_action_status(
                completion_action,
                success=True,
                output_data={
                    "workflow_saved": True,
                    "save_path": save_path,
                    "steps_completed": step_count,
                    "final_result_available": bool(final_result),
                },
            )

        except Exception as e:
            self._update_action_status(completion_action, success=False, error=str(e))
            logger.error(f"Workflow completion error: {e}")

        # Yield the updated completion action with final status
        yield completion_action

        # Log if max steps reached
        if step_count >= self.args.max_steps:
            logger.warning(f"Workflow execution stopped after reaching max steps: {self.args.max_steps}")

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
                    try:
                        conn.test_connection()
                        logger.info(f"Database connection test successful for {name}")
                    except Exception as e:
                        logger.error(f"Database connection test failed for {name}: {str(e)}", exc_info=False)
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
        for component in selected_components:
            # db_name = component_dirs[component]``
            # Initialize corresponding stores
            if component == "metadata":
                if kb_update_strategy == "check":
                    if not os.path.exists(dir_path):
                        raise ValueError("metadata is not built, please run bootstrap_kb with overwrite strategy first")
                    else:
                        self.global_config.check_init_storage_config("database")

                        self.metadata_store = rag_by_configuration(self.global_config)
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
                self.metadata_store = rag_by_configuration(self.global_config)

                if not benchmark_platform:
                    self.check_db()
                    init_local_schema(
                        self.metadata_store,
                        self.global_config,
                        self.db_manager,
                        kb_update_strategy,
                        table_type=self.args.schema_linking_type,
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "spider2":
                    benchmark_path = os.path.expanduser(
                        self.args.benchmark_path or self.global_config.benchmark_path(benchmark_platform)
                    )

                    init_snowflake_schema(
                        self.metadata_store,
                        benchmark_path,
                        kb_update_strategy,
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "bird_dev":
                    self.check_db()
                    benchmark_path = os.path.expanduser(
                        self.args.benchmark_path or self.global_config.benchmark_path(benchmark_platform)
                    )
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
                    raise ValueError(f"Unsupported benchmark platform: {benchmark_platform}")
                else:
                    raise ValueError(f"Unsupported benchmark platform: {benchmark_platform}")

                return {
                    "status": "success",
                    "message": f"metadata bootstrap completed, "
                    f"schema_size={self.metadata_store.get_schema_size()}, "
                    f"value_size={self.metadata_store.get_value_size()}",
                }

            elif component == "metrics":
                semantic_model_path = os.path.join(dir_path, "semantic_model.lance")
                metrics_path = os.path.join(dir_path, "metrics.lance")
                if kb_update_strategy == "overwrite":
                    if os.path.exists(semantic_model_path):
                        shutil.rmtree(semantic_model_path)
                        logger.info(f"Deleted existing directory {semantic_model_path}")
                    if os.path.exists(metrics_path):
                        shutil.rmtree(metrics_path)
                        logger.info(f"Deleted existing directory {metrics_path}")
                    self.global_config.save_storage_config("metric")
                else:
                    self.global_config.check_init_storage_config("metric")
                self.metrics_store = SemanticMetricsRAG(dir_path)
                if hasattr(self.args, "semantic_yaml") and self.args.semantic_yaml:
                    init_semantic_yaml_metrics(
                        self.metrics_store, self.args, self.global_config, build_mode=kb_update_strategy
                    )
                else:
                    init_success_story_metrics(
                        self.metrics_store, self.args, self.global_config, build_mode=kb_update_strategy
                    )
                return {
                    "status": "success",
                    "message": f"metrics bootstrap completed, "
                    f"semantic_model_size={self.metrics_store.get_semantic_model_size()}, "
                    f"metrics_size={self.metrics_store.get_metrics_size()}",
                }
            elif component == "document":
                self.storage_modules["document_store"] = DocumentStore(dir_path)
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
            elif component == "sql_history":
                sql_history_path = os.path.join(dir_path, "sql_history.lance")
                if kb_update_strategy == "overwrite":
                    if os.path.exists(sql_history_path):
                        shutil.rmtree(sql_history_path)
                        logger.info(f"Deleted existing directory {sql_history_path}")
                    self.global_config.save_storage_config("sql_history")
                else:
                    self.global_config.check_init_storage_config("sql_history")

                # Initialize SQL history storage
                from datus.storage.sql_history import SqlHistoryRAG
                from datus.storage.sql_history.sql_history_init import init_sql_history

                self.sql_history_store = SqlHistoryRAG(dir_path)
                result = init_sql_history(
                    self.sql_history_store,
                    self.args,
                    self.global_config,
                    build_mode=kb_update_strategy,
                    pool_size=pool_size,
                )
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

        benchmark_path = self.args.benchmark_path or self.global_config.benchmark_path(benchmark_platform)
        if not benchmark_path:
            raise ValueError("benchmark_path is not set, please setup in config file or set --benchmark_path")

        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark_path not found: {benchmark_path}")

        target_task_ids = getattr(self.args, "benchmark_task_ids", [])
        target_task_ids = set(target_task_ids) if target_task_ids else None
        if benchmark_platform == "spider2":
            self.global_config.check_init_storage_config("database")
            return self.benchmark_spider2(benchmark_path, target_task_ids)
        elif benchmark_platform == "bird_dev":
            self.global_config.check_init_storage_config("database")
            return self.benchmark_bird_dev(benchmark_path, target_task_ids)
        elif benchmark_platform == "semantic_layer":
            self.global_config.check_init_storage_config("metric")
            return self.benchmark_semantic_layer(benchmark_path, target_task_ids)
        elif benchmark_platform == "bird_critic":
            self.global_config.check_init_storage_config("database")
            logger.info(f"Benchmark {benchmark_platform} not support now, please wait for update.")
        return {"status": "success", "message": "Benchmarking completed"}

    def benchmark_spider2(self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None):
        task_file = os.path.join(benchmark_path, "spider2-snow.jsonl")
        self._check_benchmark_file(task_file)

        with open(task_file, "r") as f:
            task_configs = [json.loads(line) for line in f]

        # Filter tasks by target_task_ids if specified
        filtered_tasks = []
        for task_config in task_configs:
            task_id = task_config["instance_id"]
            if target_task_ids and task_id not in target_task_ids:
                continue
            filtered_tasks.append(task_config)

        logger.info(f"Loaded {len(filtered_tasks)} tasks from Spider2 benchmark")
        logger.info("Phase 1: Running agent benchmark tests...")

        def run_single_spider2_task(task_config):
            """Execute a single Spider2 benchmark task"""
            task_id = task_config["instance_id"]
            task = task_config["instruction"]
            database_name = task_config["db_id"]
            logger.info(f"start benchmark with {task_id}: {task}")

            result = self.run(
                SqlTask(
                    id=task_id,
                    database_type="snowflake",
                    task=task,
                    database_name=database_name,
                    output_dir=self.global_config.output_dir,
                    current_date=self.args.current_date,
                )
            )
            logger.info(
                f"Finish benchmark with {task_id}, " f"file saved in {self.global_config.output_dir}/{task_id}.csv."
            )
            return task_id, result

        # Get concurrency level from args or default to 1
        max_workers = getattr(self.args, "max_workers", 1)
        logger.info(f"Using {max_workers} worker threads for parallel execution")

        # Execute tasks with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_single_spider2_task, task_config): task_config for task_config in filtered_tasks
            }

            # Wait for completion
            for future in as_completed(future_to_task):
                task_config = future_to_task[future]
                try:
                    task_id, _ = future.result()
                    logger.debug(f"Task {task_id} completed successfully")
                except Exception as exc:
                    task_id = task_config["instance_id"]
                    logger.error(f"Task {task_id} generated an exception: {exc}")

        logger.info("Phase 2: Evaluating benchmark accuracy...")
        return evaluate_and_report_accuracy(
            benchmark_path,
            self.global_config.trajectory_dir,
            self.global_config.current_namespace,
            self.global_config.output_dir,
            target_task_ids,
        )

    def benchmark_bird_dev(self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None):
        tasks = load_bird_dev_tasks(benchmark_path)
        current_namespace = self.global_config.current_namespace

        # Convert Bird tasks to format expected by generate_gold_standard_results
        task_size = 0
        group_task_ids = defaultdict(list)
        # Prepare filtered tasks for parallel execution
        filtered_bird_tasks = []
        for task in tasks:
            task_id = str(task["question_id"])
            if target_task_ids and task_id not in target_task_ids:
                continue
            db_id = task["db_id"]
            filtered_bird_tasks.append(task)
            group_task_ids[db_id].append(
                {"question_id": task["question_id"], "sql": task["SQL"], "question": task["question"]}
            )
            task_size += 1
        if task_size == 0:
            logger.warning("There are no benchmarks that need to be run.")
            return {}
        logger.info(f"Loaded {task_size} tasks from Bird benchmark")
        logger.info("Phase 1: Generating gold standard results...")

        for db_id, converted_tasks in group_task_ids.items():
            generate_gold_standard_results(
                converted_tasks,
                benchmark_path,
                self.db_manager.get_conn(current_namespace, db_name=db_id),
                target_task_ids,
            )

        logger.info("Phase 2: Running agent benchmark tests...")

        def run_single_bird_task(task):
            """Execute a single Bird benchmark task"""
            task_id = str(task["question_id"])
            question = task["question"]
            database_name = task["db_id"]
            logger.info(f"start benchmark with {task_id}: {question}")

            result = self.run(
                SqlTask(
                    id=task_id,
                    database_type=DBType.SQLITE,
                    task=question,
                    database_name=database_name,
                    external_knowledge="" if "evidence" not in task else task["evidence"],
                    output_dir=self.global_config.output_dir,
                    current_date=self.args.current_date,
                )
            )
            logger.info(
                f"Finish benchmark with {task_id}, " f"file saved in {self.global_config.output_dir}/{task_id}.csv."
            )
            return task_id, result

        # Get concurrency level from args or default to 1
        max_workers = getattr(self.args, "max_workers", 1)
        logger.info(f"Using {max_workers} worker threads for parallel execution")

        # Execute tasks with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_single_bird_task, task): task for task in filtered_bird_tasks}

            # Wait for completion
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task_id, _ = future.result()
                    logger.debug(f"Task {task_id} completed successfully")
                except Exception as exc:
                    task_id = str(task["question_id"])
                    logger.error(f"Task {task_id} generated an exception: {exc}")

        logger.info("Phase 3: Evaluating benchmark accuracy...")
        return evaluate_and_report_accuracy(
            benchmark_path,
            self.global_config.trajectory_dir,
            self.global_config.current_namespace,
            self.global_config.output_dir,
            target_task_ids,
        )

    def benchmark_semantic_layer(self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None):
        task_file = self.args.testing_set
        self._check_benchmark_file(task_file)

        # Clean up previous execution results to avoid interference
        self._cleanup_benchmark_paths(benchmark_path)

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
        logger.info("Phase 1: Generating gold standard results...")

        generate_gold_standard_results(
            tasks, benchmark_path, self.db_manager.get_conn(self.global_config.current_namespace), target_task_ids
        )
        metric_meta = self.global_config.current_metric_meta(self.args.metric_meta)

        logger.info("Phase 2: Running agent benchmark tests...")
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

            self.run(
                SqlTask(
                    id=task_id,
                    database_type=current_db_config.type,
                    task=question,
                    database_name=current_db_config.database,
                    schema_name=current_db_config.schema,
                    domain=metric_meta.domain,
                    layer1=metric_meta.layer1,
                    layer2=metric_meta.layer2,
                    output_dir=self.global_config.output_dir,
                    external_knowledge=combined_ext_knowledge,
                    current_date=self.args.current_date,
                )
            )

            logger.info(
                f"Finish benchmark with {task_id}, " f"file saved in {self.global_config.output_dir}/{task_id}.csv."
            )

        logger.info("Phase 3: Evaluating benchmark accuracy...")
        return evaluate_and_report_accuracy(
            benchmark_path,
            self.global_config.trajectory_dir,
            self.global_config.current_namespace,
            self.global_config.output_dir,
            target_task_ids,
        )

    def _check_benchmark_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmarking task file not found, file_path={file_path}")

    def _cleanup_benchmark_paths(self, benchmark_path: str):
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
                        "reason_content": json.dumps(entry["reason_content"], ensure_ascii=False),
                        "sql_contexts": json.dumps(entry["sql_contexts"], ensure_ascii=False),
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
