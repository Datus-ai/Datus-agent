# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Formatting helpers for the compact metric queryability contract."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def summarize_queryability_contracts(contracts: Iterable[Dict[str, Any]]) -> str:
    """Summarize complete metric and dimension combinations for error messages."""
    parts = []
    for contract in contracts:
        dimensions = ", ".join(contract.get("dimensions") or [])
        output_ids = ", ".join(contract.get("metric_output_ids") or [])
        metric_names = ", ".join(contract.get("metric_hints") or [])
        if not dimensions:
            continue
        metrics = metric_names or output_ids
        part = (
            f"{contract.get('contract_id') or contract.get('source_id') or 'source SQL'} "
            f"group-by [{dimensions}] metrics [{metrics}]"
        )
        if contract.get("time_grain"):
            part += f" (time_granularity='{contract['time_grain']}')"
        parts.append(part)
    return "; ".join(parts)
