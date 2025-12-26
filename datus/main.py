#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import os
import sys
from datetime import datetime

from datus.cli.namespace_manager import NamespaceManager
from datus.cli.tutorial import BenchmarkTutorial
from datus.utils.async_utils import setup_windows_policy

# Add path fixing to ensure proper imports
if __package__ is None:
    # Add parent directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datus import __version__
from datus.agent.agent import Agent
from datus.cli.interactive_init import InteractiveInit
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.node_models import SqlTask
from datus.utils.exceptions import setup_exception_handler
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    # Create a parent parser for global options that will be shared across all subcommands
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--debug", action="store_true", help="Enable debug level logging")
    global_parser.add_argument("--config", type=str, help="Path to configuration file (default: conf/agent.yml)")
    global_parser.add_argument(
        "--save_llm_trace",
        action="store_true",
        help="Enable saving LLM input/output traces to YAML files",
    )

    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Datus: AI-powered SQL Agent for data engineering",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add version argument
    parser.add_argument("-v", "--version", action="version", version=f"Datus Agent {__version__}")

    # Create subparsers for different commands, inheriting global options
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # init command
    subparsers.add_parser(
        "init",
        help="Interactive setup wizard for basic configuration",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # namespace command
    namespace_parser = subparsers.add_parser(
        "namespace",
        help="Manage namespaces",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    namespace_parser.add_argument("command", help="namespace manage command", choices=["list", "add", "delete"])

    # probe-llm command
    probe_parser = subparsers.add_parser(
        "probe-llm",
        help="Test LLM connectivity",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    probe_parser.add_argument("--model", type=str, help="Model to test", required=False)

    # check-db command
    check_db_parser = subparsers.add_parser(
        "check-db",
        help="Check database connectivity",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check_db_parser.add_argument("--namespace", type=str, required=True, help="Database namespace to check")

    # bootstrap-kb command
    bootstrap_parser = subparsers.add_parser(
        "bootstrap-kb",
        help="Initialize knowledge base",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bootstrap_parser.add_argument(
        "--kb_update_strategy",
        type=str,
        choices=["check", "overwrite", "incremental"],
        default="check",
        help="Knowledge base update strategy: check (verify paths and data), overwrite (careful!), or incremental",
    )
    bootstrap_parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        choices=[
            "metrics",
            "metadata",
            "semantic_model",
            "table_lineage",
            "document",
            "ext_knowledge",
            "reference_sql",
        ],
        default=["metadata"],
        help="Knowledge base components to initialize",
    )
    bootstrap_parser.add_argument("--storage_path", type=str, help="Parent directory for all storage components")
    bootstrap_parser.add_argument(
        "--benchmark", type=str, choices=["spider2", "bird_dev", "bird_critic"], help="Benchmark dataset to use"
    )
    bootstrap_parser.add_argument("--namespace", type=str, required=True, help="Database namespace")
    bootstrap_parser.add_argument(
        "--schema_linking_type",
        type=str,
        choices=["table", "view", "mv", "full"],
        default="full",
        help="Schema linking type for the task, (mv for materialized view, full for all types)",
    )
    bootstrap_parser.add_argument(
        "--database_name",
        type=str,
        default="",
        help="Database name to be initialized: It represents duckdb, schema_name in Snowflake; "
        "database names in MySQL, StarRocks, PostgreSQL, etc.; SQLite is not supported.",
    )
    bootstrap_parser.add_argument(
        "--pool_size",
        type=int,
        default=4,
        help="Number of threads to initialize bootstrap-kb, default is 4",
    )
    bootstrap_parser.add_argument(
        "--success_story",
        type=str,
        default="benchmark/semantic_layer/success_story.csv",
        help="Path to success story file",
    )
    bootstrap_parser.add_argument(
        "--semantic_yaml",
        type=str,
        help="Path to semantic model YAML file",
    )
    bootstrap_parser.add_argument(
        "--metric_meta", type=str, default="default", help="Metric meta for the success story"
    )
    bootstrap_parser.add_argument("--catalog", type=str, help="Catalog of the success story")

    bootstrap_parser.add_argument("--subject_path", type=str, help="Subject path of the success story")
    bootstrap_parser.add_argument("--ext_knowledge", type=str, help="Path to external knowledge CSV file")
    bootstrap_parser.add_argument(
        "--sql_dir", type=str, help="Directory containing SQL files for reference_sql component"
    )
    bootstrap_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only process and validate SQL files, then exit (for reference_sql component)",
    )
    bootstrap_parser.add_argument(
        "--subject_tree",
        type=str,
        help='Comma-separated subject tree categories (e.g., "Sales/Reporting/Daily,Sales/Analytics/Trends"). '
        "If provided, only these predefined categories can be used. "
        "If not provided, existing categories from LanceDB will be reused or new ones created.",
    )

    # benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        # choices=["spider2", "bird_dev", "semantic_layer"],
        help="Benchmark type to run",
    )
    benchmark_parser.add_argument(
        "--benchmark_task_ids", type=str, nargs="+", help="Specific benchmark task IDs to run"
    )
    benchmark_parser.add_argument("--namespace", type=str, required=True, help="Database namespace")
    benchmark_parser.add_argument("--task_db_name", type=str, help="Database name for the task")
    benchmark_parser.add_argument("--task_schema", type=str, help="Schema name for the task")
    benchmark_parser.add_argument("--metric_meta", type=str, default="default", help="Metric meta for the task")
    benchmark_parser.add_argument("--subject_path", type=str, help="Subject path for the task")
    benchmark_parser.add_argument("--task_ext_knowledge", type=str, default="", help="External knowledge for the task")
    benchmark_parser.add_argument(
        "--current_date",
        type=str,
        default=None,
        help="Current date reference for relative time expressions (e.g., '2025-07-01')",
    )
    benchmark_parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of worker threads for parallel execution (default: 1)",
    )
    benchmark_parser.add_argument(
        "--testing_set",
        type=str,
        default="benchmark/semantic_layer/testing_set.csv",
        help="Full path to testing set file for semantic_layer benchmark",
    )
    benchmark_parser.add_argument(
        "--plan-mode",
        action="store_true",
        help="Enable plan mode for benchmark execution (generates plan then auto-executes without confirmation)",
    )

    # generate-dataset command
    generate_dataset_parser = subparsers.add_parser(
        "generate-dataset",
        help="Generate dataset from trajectory files",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # trajectory_dir parameter has been deprecated - trajectory path is now fixed at {agent.home}/trajectory
    generate_dataset_parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name for the output dataset file"
    )
    generate_dataset_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "parquet"],
        default="json",
        help="Output format for the dataset (default: json)",
    )
    generate_dataset_parser.add_argument(
        "--benchmark_task_ids",
        type=str,
        help="list of task IDs to include (e.g., '1,2,3,4,10'). If not specified, all tasks will be processed.",
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run SQL agent",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument("--namespace", type=str, required=True, help="Database namespace")
    run_parser.add_argument("--task", type=str, required=True, help="Natural language task description")
    run_parser.add_argument(
        "--task_id",
        type=str,
        help="Task ID for the task, it's used for output file name. If not set, it will be generated by datetime.",
    )
    run_parser.add_argument("--task_catalog", type=str, default="", help="Catalog of the task")
    run_parser.add_argument(
        "--task_db_name",
        type=str,
        required=True,
        help="Database name for the task (format: schema.database)",
    )
    run_parser.add_argument("--task_schema", type=str, default="", help="Schema of the task")
    run_parser.add_argument(
        "--schema_linking_type",
        type=str,
        choices=["table", "view", "mv", "full"],
        default="table",
        help="Schema linking type for the task, (mv for materialized view, full for all types)",
    )
    run_parser.add_argument("--task_ext_knowledge", type=str, default="", help="External knowledge for the task")
    run_parser.add_argument(
        "--current_date",
        type=str,
        default=None,
        help="Current date reference for relative time expressions (e.g., '2025-07-01')",
    )
    run_parser.add_argument("--subject_path", type=str, default="", help="Subject path of the success story")

    # evaluation for benchmark
    evaluation_parser = subparsers.add_parser(
        "eval",
        aliases=["evaluation", "evaluate"],
        help="Run evaluation for benchmark",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluation_parser.add_argument("--namespace", type=str, required=True, help="Database namespace")
    evaluation_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark type to run, choice for spider2, bird_dev, semantic_layer and subagents",
    )
    evaluation_parser.add_argument("--task_ids", type=str, nargs="+", help="Specific benchmark task IDs to run")
    evaluation_parser.add_argument("--output_file", help="Output file name, if not set, the report file is not output")
    evaluation_parser.add_argument(
        "--run_id",
        type=str,
        help="Specific run ID to evaluate. If not provided, evaluates the latest run for the namespace",
    )
    evaluation_parser.add_argument(
        "--summary_report_file",
        type=str,
        help="Path to summary report file. Reports will be appended to this file.",
    )

    bi_subparser = subparsers.add_parser(
        "bootstrap-bi",
        help="Build subagent by bi dashboard url",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bi_subparser.add_argument("--namespace", type=str, required=True, help="Database namespace")

    # tutorial command
    subparsers.add_parser(
        "tutorial",
        help="Benchmarking tutorial using a dataset from California schools",
        parents=[global_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Node configuration group (available for run and benchmark)
    for p in [run_parser, benchmark_parser]:
        node_group = p.add_argument_group("Node Configuration")
        # output_dir parameter deprecated - save path is now fixed at {agent.home}/save
        node_group.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help=argparse.SUPPRESS,  # Hide from help to discourage use
        )
        # trajectory_dir parameter deprecated - trajectory path is now fixed at {agent.home}/trajectory
        node_group.add_argument(
            "--schema_linking_rate",
            type=str,
            choices=["fast", "medium", "slow", "from_llm"],
            default="fast",
            help="Schema linking node strategy",
        )

        node_group.add_argument(
            "--search_metrics_rate",
            type=str,
            choices=["fast", "medium", "slow"],
            default="fast",
            help="Search metrics node query strategy",
        )

    # Workflow configuration group (available for run and benchmark)
    for p in [run_parser, benchmark_parser]:
        workflow_group = p.add_argument_group("Workflow Configuration")
        workflow_group.add_argument(
            "--workflow",
            type=str,
            help="Workflow planning strategy (can be builtin: fixed, reflection, dynamic, empty or custom plan name)",
        )
        workflow_group.add_argument("--max_steps", type=int, default=20, help="Maximum workflow steps")
        workflow_group.add_argument("--load_cp", type=str, help="Load workflow from checkpoint file")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        return 1

    # Handle init command separately as it doesn't require existing configuration
    if args.action == "init":
        configure_logging(args.debug, console_output=False)
        init = InteractiveInit()
        return init.run()

    if args.action == "tutorial":
        configure_logging(args.debug, console_output=False)
        tutorial = BenchmarkTutorial(args.config)
        return tutorial.run()

    if args.action == "namespace":
        configure_logging(args.debug, console_output=False)
        namespace_manager = NamespaceManager(args.config or "")
        return namespace_manager.run(args.command)

    configure_logging(args.debug)
    setup_exception_handler()

    # Load agent configuration
    agent_config = load_agent_config(**vars(args))
    if args.action == "bootstrap-bi":
        configure_logging(args.debug, console_output=False)
        from datus.cli.bi_dashboard import BiDashboardCommands

        return BiDashboardCommands(agent_config).cmd()

    # Initialize agent with both args and config
    agent = Agent(args, agent_config)
    result = None
    # Execute different functions based on action
    if args.action == "check-db":
        result = agent.check_db()
    elif args.action == "probe-llm":
        result = agent.probe_llm()
    elif args.action == "bootstrap-kb":
        result = agent.bootstrap_kb()
    elif args.action == "run":
        if args.load_cp:
            result = agent.run(check_storage=True)  # load task from checkpoint
        else:
            db_name, db_type = agent_config.current_db_name_type(args.task_db_name)
            task_id = args.task_id or datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
            subject_path = [c.strip() for c in args.subject_path.split("/") if c.strip()] if args.subject_path else None
            result = agent.run(
                SqlTask(
                    id=task_id,
                    database_type=db_type,
                    catalog_name=args.task_catalog,
                    database_name=db_name,
                    schema_name=args.task_schema,
                    task=args.task,
                    external_knowledge=args.task_ext_knowledge,
                    output_dir=agent_config.output_dir,
                    schema_linking_type=args.schema_linking_type,
                    subject_path=subject_path,
                    current_date=args.current_date,
                ),
                True,
            )
    elif args.action == "benchmark":
        result = agent.benchmark()
    elif args.action == "generate-dataset":
        result = agent.generate_dataset()
    elif args.action in ("eval", "evaluation", "evaluate"):
        result = agent.evaluation()
    if result:
        logger.info(f"\nFinal Result: {result}")

    return 0


if __name__ == "__main__":
    setup_windows_policy()
    sys.exit(main())
