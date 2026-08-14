# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Compatibility tests for the retired gen_semantic_model node."""

from unittest.mock import MagicMock

import pytest

from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
from datus.utils.exceptions import DatusException

pytestmark = pytest.mark.acceptance


def test_direct_legacy_semantic_model_node_recommends_semantic_modeling_on_dosi():
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "dosi"

    with pytest.raises(DatusException, match="gen_semantic_model is retired. Use semantic_modeling instead"):
        GenSemanticModelAgenticNode(config)


def test_direct_legacy_semantic_model_node_requires_migration_on_legacy_project():
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "osi"

    with pytest.raises(DatusException, match="This project is query-only"):
        GenSemanticModelAgenticNode(config)
