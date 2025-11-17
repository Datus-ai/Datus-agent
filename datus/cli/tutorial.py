from pathlib import Path
from typing import Any, Dict

from datus.cli.interactive_init import console, create_agent
from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)


class BenchmarkTutorial:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.namespace_name = "california_schools"
        path_manager = get_path_manager()
        self.benchmark_path = path_manager.benchmark_dir
        path_manager.ensure_dirs("sample")

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
        if not self.config_path or not Path(self.config_path).expanduser().resolve().exists():
            console.print(
                f" ❌Configuration file `{self.config_path}` not found, "
                "please check it or run `datus-agent init` first."
            )
            return False
        agent_config = load_agent_config(config=self.config_path)
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
        console.print("[bold cyan] Welcome to Datus benchmark data preparation tutorial 🎉[/bold cyan]")
        console.print(
            "Let's start learning how to prepare for benchmarking step by step using a dataset from California schools."
        )
        console.print("[bold yellow][1/4] Ensure data files and configuration[/bold yellow]")
        with console.status("Ensuring...") as status:
            self._ensure_files()
            console.print("Data files are ready.")
            status.update("Ensuring configuration...")
            if not self._ensure_config():
                return 1
        console.print("Configuration is ready.")
        california_schools_path = self.benchmark_path / self.namespace_name
        from datus.cli.interactive_init import init_metadata_and_log_result, init_sql_and_log_result

        console.print("[bold yellow][2/4] Initialize Metadata using command: [/bold yellow]")
        console.print(
            f"    [bold green]datus-agent[/] [bold]bootstrap-kb --config {self.config_path} "
            "--namespace california_schools "
            "--components metadata --kb_update_strategy overwrite[/]"
        )
        init_metadata_and_log_result(namespace_name=self.namespace_name, config_path=self.config_path)

        console.print("[bold yellow][3/4] Initialize Metrics using command: [/bold yellow]")
        success_path = self.benchmark_path / self.namespace_name / "success_story.csv"
        console.print(
            f"    [bold green]datus-agent[/] [bold]bootstrap-kb --config {self.config_path} "
            f"--namespace california_schools "
            f"--components metrics --kb_update_strategy overwrite --success_story {success_path}"
            f" [/]"
        )
        console.print(
            "[bold cyan]This step needs to be done using DeepSeek or Claude, otherwise you will get an error[/]"
        )
        with console.status("Metrics initializing..."):
            self._init_metrics(success_path)

        console.print("[bold yellow][4/4] Initialize Reference SQL using command: [/bold yellow]")
        console.print(
            f"    [bold green]datus-agent[/] [bold]bootstrap-kb --config {self.config_path} "
            "--namespace california_schools --components reference_sql --kb_update_strategy overwrite "
            f"--sql_dir {str(california_schools_path / 'reference_sql')} "
            f'--subject_tree "'
            "bird/california_schools/FRPM_Meal_Analysis,"
            "bird/california_schools/Enrollment_Demographics,"
            "bird/california_schools/SAT_Academic_Performance,"
            'bird/debit_card_specializing,bird/student_club"'
            " [/]"
        )
        init_sql_and_log_result(
            namespace_name=self.namespace_name,
            sql_dir=str(california_schools_path / "reference_sql"),
            subject_tree=(
                "bird/california_schools/FRPM_Meal_Analysis,"
                "bird/california_schools/Enrollment_Demographics,"
                "bird/california_schools/SAT_Academic_Performance,"
                "bird/debit_card_specializing,bird/student_club"
            ),
            config_path=self.config_path,
        )
        return 0

    def _init_metrics(self, success_path: Path):
        """Initialize metrics using success stories."""
        logger.info(f"Metrics initialization with {self.benchmark_path}/{self.namespace_name}/success_story.csv")
        try:
            agent = create_agent(
                namespace_name=self.namespace_name,
                components=["metrics"],
                success_story=success_path,
                validate_only=False,
                config_path=self.config_path,
            )
            result = agent.bootstrap_kb()
            logger.info(f"Metrics bootstrap result: {result}")
            metrics_size = 0 if not agent.metrics_store else agent.metrics_store.get_metrics_size()
            if metrics_size > 0:
                console.print(f"  → Processed {metrics_size} metrics")
                console.print(" ✅ Metrics initialized")
            else:
                console.print(" ⚠️No metrics initialized")
            return True
        except Exception as e:
            logger.error(f"Metrics initialization failed: {e}")
            return False


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
