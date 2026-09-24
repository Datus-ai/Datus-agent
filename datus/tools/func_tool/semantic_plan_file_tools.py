# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Narrow file storage for an optional, skill-authored semantic plan."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from datus.storage.semantic_model.artifact_file import atomic_write_text, path_mutation_lock
from datus.tools.func_tool.base import FuncToolResult


def _reject_non_finite_number(value: str):
    raise ValueError(f"{value} is not valid JSON")


class SemanticPlanFileTools:
    """Store a reviewable plan without validating or approving its design."""

    permission_category = "semantic_tools"
    _MODEL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self._plan_ids: dict[str, str] = {}

    def available_tools(self):
        from datus.tools.func_tool import trans_to_function_tool

        return [trans_to_function_tool(self.write_semantic_model_plan_file)]

    def write_semantic_model_plan_file(self, model_name: str, plan_json: str) -> FuncToolResult:
        """Write an optional planning JSON file, without changing the semantic model.

        Use with a planning workflow that prepares the JSON for human review.

        Args:
            model_name: Selected semantic-model name; revisions reuse its plan directory.
            plan_json: JSON object following the loaded planning skill's framework.
        """
        if not isinstance(model_name, str) or not self._MODEL_NAME.fullmatch(model_name):
            return FuncToolResult(success=0, error="model_name must be a simple semantic-model name")
        try:
            plan = json.loads(plan_json, parse_constant=_reject_non_finite_number)
        except (TypeError, ValueError) as exc:
            return FuncToolResult(success=0, error=f"plan_json must be valid JSON: {exc}")
        if not isinstance(plan, dict):
            return FuncToolResult(success=0, error="plan_json must be a JSON object")
        try:
            content = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            return FuncToolResult(success=0, error=f"plan_json must be valid JSON: {exc}")

        plan_id = self._plan_ids.get(model_name)
        if plan_id is None:
            plan_id = uuid4().hex
            self._plan_ids[model_name] = plan_id
        relative_path = Path(".datus/semantic-model-plans") / plan_id / "semantic-model-plan.json"
        target = self.project_root / relative_path
        if not target.resolve(strict=False).is_relative_to(self.project_root):
            return FuncToolResult(success=0, error="Semantic plan path must stay inside the project workspace")
        try:
            with path_mutation_lock(target):
                atomic_write_text(target, content)
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Could not write semantic plan: {exc}")
        return FuncToolResult(result={"path": relative_path.as_posix()})
