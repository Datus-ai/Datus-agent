# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def create_selection_prompt(
    candidates: Dict[str, Any], prompt_version: Optional[str] = None, max_text_length: int = 500
) -> str:
    """
    Create prompt for LLM-based candidate selection.

    Args:
        candidates: Dictionary of candidate results to analyze
        prompt_version: Version of the prompt template to use
        max_text_length: Maximum length for text fields to avoid overly long prompts

    Returns:
        Formatted prompt string for candidate selection
    """
    # Truncate long text fields to avoid overwhelming the prompt
    processed_candidates = {}

    for candidate_id, candidate in candidates.items():
        processed_candidate = candidate.copy() if isinstance(candidate, dict) else candidate

        if isinstance(processed_candidate, dict):
            # Handle result truncation if needed
            result = processed_candidate.get("result")
            if result and hasattr(result, "sql_result_final"):
                if len(str(result.sql_result_final)) > max_text_length:
                    # Create a copy to avoid modifying original
                    result_copy = type(result)(**result.__dict__)
                    result_copy.sql_result_final = str(result.sql_result_final)[:max_text_length] + "\n... (truncated)"
                    processed_candidate = processed_candidate.copy()
                    processed_candidate["result"] = result_copy

            # Handle error truncation
            error = processed_candidate.get("error")
            if error and len(str(error)) > max_text_length:
                processed_candidate = processed_candidate.copy()
                processed_candidate["error"] = str(error)[:max_text_length] + "\n... (truncated)"

        elif len(str(processed_candidate)) > max_text_length:
            processed_candidate = str(processed_candidate)[:max_text_length] + "\n... (truncated)"

        processed_candidates[candidate_id] = processed_candidate

    # Render the template
    prompt = prompt_manager.render_template(
        "selection_analysis", candidates=processed_candidates, version=prompt_version
    )

    logger.info(f"Generated selection prompt for {len(candidates)} candidates")
    return prompt
