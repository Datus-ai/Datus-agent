# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Common envelopes for configured agents and plugin-defined business results."""

from typing import Any

from pydantic import ConfigDict, Field

from datus.schemas.at_context import AtContextInput
from datus.schemas.base import BaseResult


class ConfigurableNodeInput(AtContextInput):
    """A conversational request with the same datasource and @-context carrier as builtins."""

    user_message: str = Field(min_length=1)
    catalog: str | None = None
    database: str | None = None
    db_schema: str | None = None
    database_is_physical: bool = False
    prompt_language: str = "en"
    prompt_version: str | None = None
    plan_mode: bool = False


class ConfigurableNodeResult(BaseResult):
    """Keep framework-owned fields and plugin business payloads through serialization."""

    model_config = ConfigDict(extra="allow")

    response: str = ""
    data: Any = None
    tokens_used: int = 0
