# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Compatibility tests for the retired gen_metrics node."""

from unittest.mock import MagicMock

import pytest

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.utils.exceptions import DatusException

pytestmark = pytest.mark.acceptance


def test_direct_legacy_metrics_node_recommends_semantic_modeling_on_dosi():
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "dosi"

    with pytest.raises(DatusException, match="gen_metrics is retired. Use semantic_modeling instead"):
        GenMetricsAgenticNode(config)


def test_direct_legacy_metrics_node_requires_migration_on_legacy_project():
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "metricflow"

    with pytest.raises(DatusException, match="This project is query-only"):
        GenMetricsAgenticNode(config)
