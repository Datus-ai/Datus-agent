"""Test setup for plan calls made after exact OSI target selection."""

from hashlib import sha256

from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState
from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools


def select_plan_model(state, path, model_name):
    exists = path.exists()
    state.select(
        {
            "absolute_path": str(path.resolve()),
            "semantic_model_file": path.name,
            "semantic_model_name": model_name,
            "artifact_sha256": sha256(path.read_bytes()).hexdigest() if exists else "",
            "exists": exists,
        },
        mode="bound" if exists else "planned",
    )


def plan_writer(root, model_name="sales", *, chat_id="chat-1", model_path=None):
    state = OsiSemanticModelTargetState()
    select_plan_model(state, model_path or root / "subject" / "semantic_models" / f"{model_name}.yaml", model_name)
    return SemanticPlanFileTools(root, target_state=state, chat_identity=lambda: ("owner", chat_id))
