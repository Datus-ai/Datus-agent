# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Datus compatibility checks for the native OpenAI Responses transport."""

from typing import Any

from datus.models.observed_model import ObservedResponsesModel
from datus.models.output_truncation import check_responses_output, checked_responses_stream


class DatusResponsesModel(ObservedResponsesModel):
    """Reject output exhaustion before the SDK treats partial output as executable."""

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        result = await super()._fetch_response(*args, **kwargs)
        if hasattr(result, "__aiter__"):
            return checked_responses_stream(result)
        check_responses_output(result)
        return result
