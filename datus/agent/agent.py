import argparse
import json
import os
import shutil
import time
from typing import Optional, Set

from langsmith import traceable

from datus.agent.evaluate import evaluate_result, setup_node_input
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.models.base import LLMBaseModel
from datus.models.claude_model import ClaudeModel
from datus.models.deepseek_model import DeepSeekModel
from datus.models.openai_model import OpenAIModel
from datus.models.qwen_model import QwenModel

# Import model implementations
from datus.schemas.node_models import BaseResult, SqlTask
from datus.storage.document import DocumentStore
from datus.storage.metric.metrics_init import init_success_story_metrics
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema
from datus.storage.schema_metadata.benchmark_init_bird import init_dev_schema
from datus.storage.schema_metadata.local_init import init_local_schema
from datus.storage.schema_metadata.store import rag_by_configuration
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.utils.loggings import get_logger

logger = get_logger("sql_agent")

MODEL_TYPE_MAP = {
    "deepseek": DeepSeekModel,
    "qwen": QwenModel,
    "openai": OpenAIModel,
    "claude": ClaudeModel,
    # "llama": LlamaModel,
}


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

    @traceable(name="agent", run_type="chain")
    def run(self, sql_task: Optional[SqlTask] = None, check_storage: bool = False):
        """
        Main execution loop for the agent.

        Returnsfinish benchmark_ids with:
            The final result of the workflow execution
        """
        if check_storage:
            self.global_config.check_init_storage_config()
        logger.info("Starting agent execution")

        if not self.init_or_load_workflow(sql_task):
            return None

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
                logger.warning(f"Node failed: {current_node.description}")
                break
            # evaluate the task result, update the context and setup the next node input if needed
            evaluation = evaluate_result(current_node, self.workflow)
            logger.debug(f"Evaluation result: {evaluation}")
            if not evaluation["success"]:
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

    def check_db(self):
        """Validate database connectivity."""
        logger.info("Checking database connectivity")

        try:
            namespace = self.global_config.current_namespace
            if namespace in self.global_config.namespaces:
                db_config = self.global_config.namespaces[namespace]
                if isinstance(db_config, DbConfig):
                    self.db_manager.get_conn(namespace, db_config.type).test_connection()
                else:
                    for db_name, db_conf in db_config.items():
                        self.db_manager.get_conn(namespace, db_conf.type, db_name).test_connection()

                logger.info(f"Database connection test successful {namespace}")
                return {"status": "success", "message": "Database connection test successful"}
            else:
                logger.error(f"Database connection test failed: {namespace} not found in namespaces")
                return {"status": "error", "message": f"{namespace} not found in namespaces"}
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def check_mcp(self):
        """Check MCP server connectivity for the current namespace."""
        logger.info("Checking MCP server connectivity")

        try:
            from datus.tools.mcp_server import MCPServer

            db_configs = self.global_config.current_dbconfigs()
            db_type = self.global_config.db_type

            logger.info(f"Checking MCP server for database type: {db_type}")

            # Use the encapsulated method to check connectivity
            return MCPServer.check_connectivity(db_type, db_configs)

        except Exception as e:
            logger.error(f"MCP server check failed: {str(e)}")
            return {"status": "error", "message": f"MCP server check failed: {str(e)}"}

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
        self.check_db()
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
                        self.global_config.check_init_storage_config()
                        self.metadata_store = rag_by_configuration(self.global_config)
                        return {
                            "status": "success",
                            "message": f"current metadata is already built, "
                            f"dir_path={dir_path},"
                            f"schema_size={self.metadata_store.get_schema_size()}, "
                            f"value_size={self.metadata_store.get_value_size()}",
                        }

                if kb_update_strategy == "overwrite" and os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted existing directory {dir_path}")
                self.global_config.check_init_storage_config()
                self.metadata_store = rag_by_configuration(self.global_config)

                if not benchmark_platform:
                    init_local_schema(
                        self.metadata_store,
                        self.global_config,
                        self.db_manager,
                        kb_update_strategy,
                        table_type=self.args.schema_linking_type,
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "spider2":
                    benchmark_path = self.args.benchmark_path or self.global_config.benchamrk_path(benchmark_platform)

                    init_snowflake_schema(
                        self.metadata_store,
                        benchmark_path,
                        kb_update_strategy,
                        pool_size=pool_size,
                    )
                elif benchmark_platform == "bird_dev":
                    benchmark_path = self.args.benchmark_path or self.global_config.benchamrk_path(benchmark_platform)
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
                self.metrics_store = SemanticMetricsRAG(dir_path)
                init_success_story_metrics(self.metrics_store, self.args, self.global_config)
                return {
                    "status": "success",
                    "message": f"metrics bootstrap completed, "
                    f"semantic_model_size={self.metrics_store.get_semantic_model_size()}, "
                    f"metrics_size={self.metrics_store.get_metrics_size()}",
                }
            elif component == "document":
                self.storage_modules["document_store"] = DocumentStore(dir_path)
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

        benchmark_path = self.args.benchmark_path or self.global_config.benchamrk_path(benchmark_platform)
        if not benchmark_path:
            raise ValueError("benchmark_path is not set, please setup in config file or set --benchmark_path")

        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark_path not found: {benchmark_path}")

        self.global_config.check_init_storage_config()
        target_task_ids = set(getattr(self.args, "benchmark_task_ids", []))
        if benchmark_platform == "spider2":
            return self.benchmark_spider2(benchmark_path, target_task_ids)
        elif benchmark_platform == "bird_dev":
            return self.benchmark_bird_dev(benchmark_path, target_task_ids)
        elif benchmark_platform == "bird_critic":
            logger.info(f"Benchmark {benchmark_platform} not support now, please wait for update.")
        return {"status": "success", "message": "Benchmarking completed"}

    def benchmark_spider2(self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None):
        task_file = os.path.join(benchmark_path, "spider2-snow.jsonl")
        self._check_benchmark_file(task_file)

        with open(task_file, "r") as f:
            task_configs = [json.loads(line) for line in f]
        for task_config in task_configs:
            task_id = task_config["instance_id"]
            if target_task_ids and task_id not in target_task_ids:
                continue

            task = task_config["instruction"]
            database_name = task_config["db_id"]
            logger.info(f"start benchmark with {task_config['instance_id']}, database: {database_name}")
            self.run(
                SqlTask(
                    id=task_id,
                    database_type="snowflake",
                    task=task,
                    database_name=database_name,
                    output_dir=self.global_config.output_dir,
                )
            )
            logger.info(
                f"Finish benchmark_ids with {task_id}, "
                f"database: {database_name}, "
                f"file saved in {self.global_config.output_dir}/{task_id}.csv."
            )

    def benchmark_bird_dev(self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None):
        task_file = f"{benchmark_path}/dev.json"
        self._check_benchmark_file(task_file)
        with open(task_file, "r") as f:
            tasks = json.load(f)

        logger.info(f"Benchmarking tasks: {target_task_ids}")
        task_group = {}
        # group tasks by database_name
        for task in tasks:
            task_id = str(task["question_id"])
            if target_task_ids and task_id not in target_task_ids:
                continue
            database_name = task["db_id"]
            if database_name not in task_group:
                task_group[database_name] = []
            task_group[database_name].append(task)
        for database_name, tasks in task_group.items():
            for task in tasks:
                self.run(
                    SqlTask(
                        id=str(task["question_id"]),
                        database_type="sqlite",
                        task=task["question"],
                        database_name=database_name,
                        external_knowledge="" if "evidence" not in task else task["evidence"],
                        output_dir=self.global_config.output_dir,
                    )
                )
                task_id = str(task["question_id"])
                logger.info(
                    f"Finish benchmark_ids with {task_id}, "
                    f"database: {database_name}, "
                    f"file saved in {self.global_config.output_dir}/{task_id}.csv."
                )

    def _check_benchmark_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmarking task file not found, file_path={file_path}")

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

        if not os.path.exists(trajectory_dir):
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

        # Find all trajectory YAML files
        trajectory_files = glob.glob(os.path.join(trajectory_dir, "*_*.yaml"))
        logger.info(f"Found {len(trajectory_files)} trajectory files")

        dataset_data = []

        for trajectory_file in trajectory_files:
            try:
                # Extract task_id from filename (e.g., "0_1750662901.yaml" -> "0")
                filename = os.path.basename(trajectory_file)
                task_id = filename.split("_")[0]

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

        logger.info(f"Dataset generated successfully: {output_file}")
        logger.info(f"Total entries: {len(dataset_data)}")

        return {
            "status": "success",
            "message": f"Dataset generated successfully: {output_file}",
            "total_entries": len(dataset_data),
            "output_file": output_file,
            "format": output_format,
        }
