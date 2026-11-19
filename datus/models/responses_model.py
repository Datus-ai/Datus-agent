# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Datus transport for native OpenAI Responses."""

from datus.models.observed_model import ObservedResponsesModel


class DatusResponsesModel(ObservedResponsesModel):
    """Preserve the SDK's completion and recovery semantics."""
