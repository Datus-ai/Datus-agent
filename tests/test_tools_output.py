from pathlib import Path

import pytest
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.schemas.node_models import ExecuteSQLInput, OutputInput
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.config import SQLiteConfig
from datus.tools.db_tools.sqlite_connector import SQLiteConnector
from datus.tools.output_tools.output import OutputTool
from datus.utils.constants import DBType
from datus.utils.sql_utils import extract_table_names


class TestBirdDevOutput:
    @pytest.fixture
    def global_config(self) -> AgentConfig:
        return load_agent_config()

    @pytest.fixture
    def test_data(self) -> dict:
        """Load test data from YAML file"""
        yaml_path = Path(__file__).parent / "data/OutputInput.yaml"
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def llm_model(self, global_config: AgentConfig) -> LLMBaseModel:
        return LLMBaseModel.create_model(agent_config=global_config)

    @pytest.mark.acceptance
    def test_output(self, test_data: dict, llm_model: LLMBaseModel, global_config: AgentConfig):
        for benchmark, data in test_data.items():
            print(f"switch benchmark to {benchmark}")
            global_config.current_namespace = data["namespace"]
            self._do_execute(benchmark, data, global_config, llm_model)

    def _do_execute(
        self,
        benchmark_platform: str,
        test_data: dict,
        global_config: AgentConfig,
        llm_model: LLMBaseModel,
    ):
        task_group = {}
        # Use db_path from test data if provided, otherwise use benchmark_path
        if "db_path" in test_data:
            # db_path in test data is relative, resolve it based on home directory
            benchmark_path = str(Path(test_data["db_path"]).expanduser().resolve())
        else:
            benchmark_path = global_config.benchmark_path(benchmark_platform)

        rag_storage = SchemaWithValueRAG(global_config)
        output_dir = test_data["output_dir"]
        for task in test_data["tasks"]:
            db_name = task["database_name"]
            if db_name not in task_group:
                task_group[db_name] = []
            task_group[db_name].append(task)

        tool = OutputTool()
        for db_name, tasks in task_group.items():
            db_path = f"{benchmark_path}/dev_databases/{db_name}/{db_name}.sqlite"
            config = SQLiteConfig(db_path=db_path)
            sql_connector = SQLiteConnector(config)
            for task in tasks:
                task_gen_sql = task["gen_sql"]
                table_names = extract_table_names(task_gen_sql, dialect=DBType.SQLITE)
                table_schemas, _ = rag_storage.search_tables(tables=table_names, database_name=db_name)
                sql_result = sql_connector.execute(ExecuteSQLInput(sql_query=task_gen_sql))
                output_result = tool.execute(
                    input_data=OutputInput(
                        finished=True,
                        task_id=f'bird_dev_{task["question_id"]}',
                        task=task["question"],
                        database_name=db_name,
                        table_schemas=table_schemas,
                        gen_sql=task_gen_sql,
                        external_knowledge=("" if "dev_evidence" not in task else task["dev_evidence"]),
                        sql_result=sql_result.sql_return,
                        row_count=sql_result.row_count,
                        metrics=[],
                        output_dir=output_dir,
                    ),
                    sql_connector=sql_connector,
                    model=llm_model,
                )
                with open(f"{output_dir}/bird_dev_{task['question_id']}_gold.csv", "w") as f:
                    f.write(sql_connector.execute(ExecuteSQLInput(sql_query=task["gold_sql"])).sql_return)
                print(output_result)
