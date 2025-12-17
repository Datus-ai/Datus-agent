# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from .ext_knowledge_init import init_ext_knowledge
from .store import ExtKnowledgeStore

__all__ = ["ExtKnowledgeStore", "init_ext_knowledge"]
