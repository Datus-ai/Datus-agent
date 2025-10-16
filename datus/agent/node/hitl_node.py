# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

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
