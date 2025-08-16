from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow


class HitlNode(Node):
    def update_context(self, workflow: Workflow) -> Dict:
        pass

    def setup_input(self, workflow: Workflow) -> Dict:
        pass

    def execute(self):
        pass

    async def execute_stream(self):
        pass
