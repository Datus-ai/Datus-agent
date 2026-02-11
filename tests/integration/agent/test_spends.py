from datetime import datetime

from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.utils.constants import DBType


def test_storage_spends():
    for _ in range(10):
        first_start = datetime.now()
        start = datetime.now()
        agent_config = load_agent_config(**{"namespace": "bird_sqlite"})
        end = datetime.now()
        print("init agent config spends", (end - start).total_seconds() * 1000)
        start = datetime.now()
        rag = SchemaWithValueRAG(agent_config)
        end = datetime.now()
        print("init storage spends", (end - start).total_seconds() * 1000)
        start = datetime.now()
        tool = SchemaLineageTool(storage=rag)
        end = datetime.now()
        print("init tool spends", (end - start).total_seconds() * 1000)
        start = datetime.now()
        tool.execute(
            input_param=SchemaLinkingInput(
                input_text="What is the publisher name of the superhero ID 38?",
                database_name="superhero",
                database_type=DBType.SQLITE,
                matching_rate="medium",
            )
        )
        end = datetime.now()
        print("execute tool spends", (end - start).total_seconds() * 1000)
        print("total spends", (end - first_start).total_seconds() * 1000, "-" * 100)
