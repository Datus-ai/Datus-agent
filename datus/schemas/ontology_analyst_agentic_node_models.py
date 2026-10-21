# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OntologyAnalyst Agentic Node models."""

from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput, AskMetricsNodeResult


class OntologyAnalystNodeInput(AskMetricsNodeInput):
    """Input model for OntologyAnalystAgenticNode."""


class OntologyAnalystNodeResult(AskMetricsNodeResult):
    """Result model for OntologyAnalystAgenticNode."""
