# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set

from rich.markup import escape

from datus.cli.interactive_init import console
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.schemas.agent_models import SubAgentConfig
from datus.utils.loggings import get_logger, print_rich_exception
from datus.utils.path_manager import get_path_manager
from datus.utils.sub_agent_manager import SubAgentManager

logger = get_logger(__name__)


def _parse_subject_tree(subject_tree: Optional[str]) -> Optional[list]:
    if not subject_tree:
        return None
    return [item.strip() for item in subject_tree.split(",") if item.strip()]


def _print_stream_lines(message: Optional[object], indent: str = "  ") -> None:
    if not message:
        return
    text = str(message).strip()
    if not text:
        return
    for line in text.splitlines():
        console.print(f"{indent}{escape(line)}")


class BenchmarkTutorial:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.namespace_name = "california_schools"

    def _ensure_files(self):
        if not self.benchmark_path.exists():
            self.benchmark_path.mkdir(parents=True)
        from datus.cli.interactive_init import copy_data_file

        sub_benchmark_path = self.benchmark_path / self.namespace_name
        if not sub_benchmark_path.exists():
            sub_benchmark_path.mkdir(parents=True)
        copy_data_file(
            resource_path="sample_data/california_schools",
            target_dir=sub_benchmark_path,
        )

    def _ensure_config(self) -> bool:
        if self.config_path and not Path(self.config_path).expanduser().resolve().exists():
            console.print(
                f" ❌Configuration file `{self.config_path}` not found, "
                "please check it or run `datus-agent init` first."
            )
            return False
        agent_config = load_agent_config(config=self.config_path)
        path_manager = get_path_manager(datus_home=agent_config.home)
        self.benchmark_path = path_manager.benchmark_dir
        if (
            self.namespace_name not in agent_config.benchmark_configs
            or self.namespace_name not in agent_config.namespaces
        ):
            from datus.configuration.agent_config_loader import configuration_manager

            namespace_config = {
                "california_schools": {
                    "type": "sqlite",
                    "name": "california_schools",
                    "uri": "~/.datus/benchmark/california_schools/california_schools.sqlite",
                },
            }
            config_manager = configuration_manager()
            config_manager.update_item(
                "namespace",
                namespace_config,
                delete_old_key=False,
                save=False,
            )
            console.print("Namespace configuration added:")

            from rich.syntax import Syntax

            console.print(Syntax(dict_to_yaml_str(namespace_config), lexer="yaml"))

            benchmark_config = {
                self.namespace_name: {
                    "question_file": "california_schools.csv",
                    "question_id_key": "task_id",
                    "question_key": "question",
                    "ext_knowledge_key": "evidence",
                    "gold_sql_path": "california_schools.csv",
                    "gold_sql_key": "gold_sql",
                    "gold_result_path": "california_schools.csv",
                },
            }

            config_manager.update_item(
                "benchmark",
                benchmark_config,
                delete_old_key=False,
                save=True,
            )
            console.print("Benchmark configuration added:")

            console.print(Syntax(dict_to_yaml_str(benchmark_config), lexer="yaml"))
        return True

    def run(self):
        try:
            console.print("[bold cyan]Welcome to Datus benchmark data preparation tutorial 🎉[/bold cyan]")
            console.print(
                "Let's start learning how to prepare for benchmarking step by step using a dataset "
                "from California schools."
            )
            console.print("[bold yellow][1/5] Ensure data files and configuration[/bold yellow]")
            with console.status("Ensuring...") as status:
                if not self._ensure_config():
                    return 1
                self._ensure_files()
                console.print("Data files are ready.")
                status.update("Ensuring configuration...")
            console.print("Configuration is ready.")
            california_schools_path = self.benchmark_path / self.namespace_name
            from datus.cli.interactive_init import init_metadata_and_log_result, init_sql_and_log_result

            console.print("[bold yellow][2/5] Initialize Metadata using command: [/bold yellow]")
            console.print(
                f"    [bold green]datus-agent[/] [bold]bootstrap-kb --config {self.config_path} "
                "--namespace california_schools "
                "--components metadata --kb_update_strategy overwrite[/]"
            )
            init_metadata_and_log_result(namespace_name=self.namespace_name, config_path=self.config_path)

            console.print("[bold yellow][3/5] Initialize Metrics using command: [/bold yellow]")
            success_path = self.benchmark_path / self.namespace_name / "success_story.csv"
            console.print(
                f"    [bold green]datus-agent[/] [bold]bootstrap-kb --config {self.config_path} "
                f"--namespace california_schools "
                f"--components metrics --kb_update_strategy overwrite --success_story {success_path} "
                '--subject_tree "california_schools/Continuation_School/Free_Rate,'
                'california_schools/Charter/Education_Location"'
                "[/]"
            )
            console.print("Metrics initializing...")
            self._init_metrics(success_path)

            console.print("[bold yellow][4/5] Initialize Reference SQL using command: [/bold yellow]")
            console.print(
                f"    [bold green]datus-agent[/] [bold]bootstrap-kb --config {self.config_path} "
                "--namespace california_schools --components reference_sql --kb_update_strategy overwrite "
                f"--sql_dir {str(california_schools_path / 'reference_sql')} "
                f'--subject_tree "'
                "california_schools/Continuation/Free_Rate,"
                "california_schools/Charter/Education_Location,"
                "california_schools/Charter-Fund/Phone,"
                "california_schools/SAT_Score/Average,"
                "california_schools/SAT_Score/Excellence_Rate,"
                "california_schools/FRPM_Enrollment/Rate,"
                "california_schools/Enrollment/Total"
                '" [/]'
            )
            init_sql_and_log_result(
                namespace_name=self.namespace_name,
                sql_dir=str(california_schools_path / "reference_sql"),
                subject_tree="california_schools/Continuation/Free_Rate,"
                "california_schools/Charter/Education_Location,"
                "california_schools/Charter-Fund/Phone,"
                "california_schools/SAT_Score/Average,"
                "california_schools/SAT_Score/Excellence_Rate,"
                "california_schools/FRPM_Enrollment/Rate,"
                "california_schools/Enrollment/Total",
                config_path=self.config_path,
            )
            console.print("[bold yellow][5/5] Building sub-agents and workflows: [/bold yellow]")

            with console.status("Sub-Agents Building...") as status:
                self.add_sub_agents()
                status.update("Workflows Building...")
                self.add_workflows()

                console.print(
                    "[bold cyan]The sub-agents and workflow are now configured. "
                    "You can use them in the following ways:\n"
                    "  1. Conduct multi-turn conversations in the CLI via `/datus_schools <your question>` or  "
                    "`/datus_schools_context <your question>`\n"
                    "  2. Use them in benchmark by running the command "
                    "`datus-agent benchmark --workflow datus_schools`.[/]"
                )

            console.print(" 🎉 [bold green]Now you can start with the benchmarking section of the guidance document[/]")
            return 0
        except Exception as e:
            print_rich_exception(console, e, "Tutorial failed", logger)
            return 1

    def _init_metrics(self, success_path: Path):
        """Initialize metrics using success stories."""
        from datus.storage.metric.metrics_init import init_success_story_metrics
        from datus.storage.metric.store import SemanticMetricsRAG

        logger.info(f"Metrics initialization with {self.benchmark_path}/{self.namespace_name}/success_story.csv")
        try:
            agent_config = load_agent_config(reload=True, config=self.config_path)
            agent_config.current_namespace = self.namespace_name
            kb_update_strategy = "overwrite"
            storage_path = agent_config.rag_storage_path()
            semantic_model_path = os.path.join(storage_path, "semantic_model.lance")
            metrics_path = os.path.join(storage_path, "metrics.lance")
            if kb_update_strategy == "overwrite":
                if os.path.exists(semantic_model_path):
                    shutil.rmtree(semantic_model_path)
                    logger.info(f"Deleted existing directory {semantic_model_path}")
                if os.path.exists(metrics_path):
                    shutil.rmtree(metrics_path)
                    logger.info(f"Deleted existing directory {metrics_path}")
                agent_config.save_storage_config("metric")
            else:
                agent_config.check_init_storage_config("metric")

            subject_tree = _parse_subject_tree(
                "california_schools/Continuation_School/Free_Rate," "california_schools/Charter/Education_Location"
            )
            stage_seen: Dict[int, Set[str]] = {}

            def emit(event: dict) -> None:
                event_type = event.get("type")
                if not event_type:
                    return

                if event_type == "metrics_init.row_start":
                    row_idx = event.get("row_idx")
                    question = str(event.get("question") or "")
                    console.print(f"row {row_idx} start: {escape(question)}")
                    table_name = event.get("table_name")
                    if table_name:
                        console.print(f"  table: {escape(str(table_name))}")
                    return

                if event_type == "metrics_init.action":
                    row_idx = event.get("row_idx")
                    stage = str(event.get("stage") or "action")
                    seen = stage_seen.setdefault(row_idx, set())
                    if stage not in seen:
                        seen.add(stage)
                        console.print(f"  {escape(stage)}:")
                    _print_stream_lines(event.get("messages"), indent="    ")
                    return

                if event_type == "metrics_init.semantic_model_ready":
                    semantic_model_file = event.get("semantic_model_file")
                    if semantic_model_file:
                        console.print(f"  semantic_model: {escape(str(semantic_model_file))}")
                    return

                if event_type == "metrics_init.row_success":
                    row_idx = event.get("row_idx")
                    console.print(f"row {row_idx} end")
                    return

                if event_type == "metrics_init.row_error":
                    row_idx = event.get("row_idx")
                    error = event.get("error") or "unknown error"
                    console.print(f"row {row_idx} error: {escape(str(error))}")
                    return

            args = argparse.Namespace(success_story=str(success_path))
            successful, error_message = init_success_story_metrics(
                args,
                agent_config,
                subject_tree,
                emit=emit,
            )

            if successful:
                metrics_store = SemanticMetricsRAG(agent_config)
                metrics_size = metrics_store.get_metrics_size()
                if metrics_size > 0:
                    console.print(f"  → Processed {metrics_size} metrics")
                    if error_message:
                        console.print(" ⚠️ [bold]The metrics has not been fully initialised successfully: [/]")
                        console.print(f"    {escape(error_message)}")
                    else:
                        console.print(" ✅ Metrics initialized")
                else:
                    if error_message:
                        console.print(" ❌[bold]There are some errors in the processing:[/]")
                        console.print(f"    {escape(error_message)}")
                    else:
                        console.print(" ⚠️No metrics initialized")
            else:
                console.print(" ❌[bold]Metrics initialization failed:[/]")
                console.print(f"    {escape(str(error_message))}")
            return True
        except Exception as e:
            print_rich_exception(console, e, "Metrics initialization failed", logger)
            return False

    def add_sub_agents(self):
        agent_config = load_agent_config(reload=True)
        manager = SubAgentManager(
            configuration_manager=configuration_manager(config_path=self.config_path, reload=True),
            namespace=self.namespace_name,
            agent_config=agent_config,
        )
        manager.save_agent(
            SubAgentConfig(
                system_prompt="datus_schools",
                prompt_version="1.0",
                prompt_language="en",
                agent_description="",
                rules=[],
                tools="db_tools, date_parsing_tools",
            ),
            previous_name="datus_schools",
        )
        console.print("  ✅ Sub-agent `datus_schools` have been added. It can work using database tools.")

        manager.save_agent(
            SubAgentConfig(
                system_prompt="datus_schools_context",
                prompt_version="1.0",
                prompt_language="en",
                agent_description="",
                rules=[],
                tools="context_search_tools, db_tools, date_parsing_tools",
            ),
            previous_name="datus_schools_context",
        )
        console.print(
            "  ✅ Sub-agent `datus_schools_context` have been added. "
            "It can work using metrics, relevant SQL and database tools."
        )

    def add_workflows(self):
        config_manager = configuration_manager(self.config_path, reload=True)
        config_manager.update_item(
            "workflow",
            value={
                "datus_schools": ["datus_schools", "execute_sql", "output"],
                "datus_schools_context": ["datus_schools_context", "execute_sql", "output"],
            },
            delete_old_key=False,
            save=True,
        )
        console.print("  ✅ Workflow `datus_schools` and `datus_schools_context` have been added.")


def dict_to_yaml_str(data: Dict[str, Any]) -> str:
    import io

    import yaml

    result = ""
    with io.StringIO() as stream:
        try:
            yaml.safe_dump(data, stream)
            result = stream.getvalue()
        except Exception as e:
            logger.warning(f"Failed to convert data to yaml: {e}")

    return result
