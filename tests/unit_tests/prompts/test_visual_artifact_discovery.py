# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Execute the discovery calls advertised by visual-artifact prompts."""

import json
import re

import pytest

from datus.prompts.prompt_manager import PromptManager
from datus.tools.func_tool import DashboardFilesystemFuncTool, ReportFilesystemFuncTool
from datus.utils.path_manager import DatusPathManager


@pytest.mark.parametrize(
    ("kind", "filesystem_cls"),
    [("dashboard", DashboardFilesystemFuncTool), ("report", ReportFilesystemFuncTool)],
)
@pytest.mark.parametrize("gitignored", [False, True])
def test_prompt_discovery_calls_find_existing_artifacts(tmp_path, kind, filesystem_cls, gitignored):
    """Every concrete discovery example must work with the file-only glob tool."""
    project_root = tmp_path / "project"
    artifact_root = f"{kind}s"
    manifests = set()
    for slug in ("asset_usage", "asset_usage_v2"):
        artifact_dir = project_root / artifact_root / slug
        (artifact_dir / "render").mkdir(parents=True)
        (artifact_dir / "manifest.json").write_text(
            json.dumps({"slug": slug, "name": slug, "kind": kind}), encoding="utf-8"
        )
        (artifact_dir / "render" / "app.jsx").write_text("export default () => null;", encoding="utf-8")
        manifests.add(f"{artifact_root}/{slug}/manifest.json")
    if gitignored:
        (project_root / ".gitignore").write_text(f"{artifact_root}/\n", encoding="utf-8")

    node_name = f"gen_visual_{kind}"
    manager = PromptManager(path_manager=DatusPathManager(tmp_path / "home"))
    prompt = manager.render_template(f"{node_name}_system")
    # Consume the actual tool-call examples rather than duplicating their patterns.
    # Render-tree examples contain <slug> and are not artifact discovery calls.
    patterns = re.findall(rf"glob\('({artifact_root}/[^'<>]+)'\)", prompt)
    assert patterns, "The prompt must advertise an artifact discovery call"

    filesystem = filesystem_cls(root_path=str(project_root), current_node=node_name, strict=True)
    for pattern in patterns:
        result = filesystem.glob(pattern)
        assert result.success == 1, (pattern, result.error)
        assert set(result.result["files"]) == manifests, pattern
