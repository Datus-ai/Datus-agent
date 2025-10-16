# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-friendly text generation for metrics.

This module provides utilities to convert metric definitions into
LLM-readable text format for better semantic search and retrieval.
"""


def generate_metric_llm_text(metric_doc: dict, data_source: dict = None) -> str:
    """
    Generate LLM-friendly text representation of a metric definition.

    Args:
        metric_doc: Metric definition from YAML
        data_source: Optional data_source definition for measure lookup

    Returns:
        str: Formatted LLM-friendly text
    """
    lines = []

    # Header: Metric name
    metric_name = metric_doc.get("name", "")
    lines.append(f"Metric: {metric_name}")

    # Description
    description = metric_doc.get("description", "")
    if description:
        lines.append(description)

    lines.append("")

    # Metric type
    metric_type = metric_doc.get("type", "").lower()
    lines.append(f"Type: {metric_type}")

    # Type-specific parameters
    type_params = metric_doc.get("type_params", {})

    if metric_type == "measure_proxy":
        # Handle measure_proxy type
        measure_name = type_params.get("measure", "")
        measures = type_params.get("measures", [])

        if measure_name:
            # Single measure
            lines.append(f"Measure: {measure_name}")
            if data_source:
                measure_def = _find_measure_in_data_source(measure_name, data_source)
                if measure_def:
                    lines.append(f"  - Aggregation: {measure_def.get('agg', 'N/A')}")
                    lines.append(f"  - Expression: {measure_def.get('expr', 'N/A')}")
                    if measure_def.get("description"):
                        lines.append(f"  - Description: {measure_def.get('description')}")
        elif measures:
            # Multiple measures
            lines.append("Measures:")
            for m in measures:
                if isinstance(m, str):
                    lines.append(f"  - {m}")
                elif isinstance(m, dict):
                    m_name = m.get("name", "")
                    lines.append(f"  - {m_name}")
                    if m.get("constraint"):
                        lines.append(f"    Constraint: {m['constraint']}")

    elif metric_type == "ratio":
        # Handle ratio type
        numerator = type_params.get("numerator", {})
        denominator = type_params.get("denominator", {})

        # Format numerator
        num_str = _format_measure_reference(numerator, data_source)
        den_str = _format_measure_reference(denominator, data_source)

        lines.append(f"Formula: [{num_str}] / [{den_str}]")

    elif metric_type == "expr":
        # Handle expression type
        expr = type_params.get("expr", "")
        measures = type_params.get("measures", [])

        lines.append(f"Expression: {expr}")

        if measures:
            lines.append("Using measures:")
            for m in measures:
                if isinstance(m, str):
                    lines.append(f"  - {m}")
                    if data_source:
                        measure_def = _find_measure_in_data_source(m, data_source)
                        if measure_def:
                            lines.append(f"    {measure_def.get('agg', 'N/A')}({measure_def.get('expr', 'N/A')})")
                elif isinstance(m, dict):
                    m_name = m.get("name", "")
                    lines.append(f"  - {m_name}")
                    if m.get("constraint"):
                        lines.append(f"    Constraint: {m['constraint']}")

    elif metric_type == "cumulative":
        # Handle cumulative type
        measures = type_params.get("measures", [])
        window = type_params.get("window")
        grain_to_date = type_params.get("grain_to_date")

        if measures:
            lines.append(f"Measure: {measures[0] if isinstance(measures[0], str) else measures[0].get('name', '')}")
            if data_source and measures:
                measure_name = measures[0] if isinstance(measures[0], str) else measures[0].get("name", "")
                measure_def = _find_measure_in_data_source(measure_name, data_source)
                if measure_def:
                    lines.append(f"  - Aggregation: {measure_def.get('agg', 'N/A')}")
                    lines.append(f"  - Expression: {measure_def.get('expr', 'N/A')}")

        if window:
            lines.append(f"Window: {window} (cumulative)")
        elif grain_to_date:
            lines.append(f"Window: {grain_to_date}-to-date (cumulative)")

    elif metric_type == "derived":
        # Handle derived type
        expr = type_params.get("expr", "")
        metrics = type_params.get("metrics", [])

        lines.append(f"Expression: {expr}")

        if metrics:
            lines.append("Based on metrics:")
            for m in metrics:
                if isinstance(m, str):
                    lines.append(f"  - {m}")
                elif isinstance(m, dict):
                    lines.append(f"  - {m.get('name', '')}")

    # Source table
    if data_source:
        sql_table = data_source.get("sql_table", "")
        if sql_table:
            lines.append(f"Source: {sql_table}")

    lines.append("")

    # Constraint/Filter
    constraint = metric_doc.get("constraint", "") or metric_doc.get("where_constraint", "")
    if constraint:
        lines.append(f"Filter: {constraint}")

    # Locked metadata
    locked_metadata = metric_doc.get("locked_metadata", {})
    if locked_metadata:
        display_name = locked_metadata.get("display_name", "")
        if display_name:
            lines.append(f"Display Name: {display_name}")

        unit = locked_metadata.get("unit", "")
        if unit:
            lines.append(f"Unit: {unit}")

        value_format = locked_metadata.get("value_format", "")
        if value_format:
            lines.append(f"Value Format: {value_format}")

        tags = locked_metadata.get("tags", [])
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")

        increase_is_good = locked_metadata.get("increase_is_good")
        if increase_is_good is not None:
            lines.append(f"Increase is Good: {increase_is_good}")

    return "\n".join(lines)


def _find_measure_in_data_source(measure_name: str, data_source: dict) -> dict:
    """Find measure definition in data_source."""
    measures = data_source.get("measures", [])
    for measure in measures:
        if measure.get("name") == measure_name:
            return measure
    return {}


def _format_measure_reference(measure_ref, data_source: dict = None) -> str:
    """Format a measure reference (for ratio numerator/denominator)."""
    if isinstance(measure_ref, str):
        measure_name = measure_ref
        constraint = None
    elif isinstance(measure_ref, dict):
        measure_name = measure_ref.get("name", "")
        constraint = measure_ref.get("constraint", "")
    else:
        return "?"

    # Look up measure definition
    if data_source:
        measure_def = _find_measure_in_data_source(measure_name, data_source)
        if measure_def:
            agg = measure_def.get("agg", "UNKNOWN")
            expr = measure_def.get("expr", "?")
            result = f"{agg}({expr})"
        else:
            result = measure_name
    else:
        result = measure_name

    # Add constraint if present
    if constraint:
        result += f" WHERE {constraint}"

    return result
