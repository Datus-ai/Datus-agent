"""Select fixed product journeys and affected deterministic tests for a PR."""

from pathlib import Path

from ci.harness.test_plan import ROOT, collapse_targets, load_map, normalize_path, unit_owner, unit_test_plan

SHARED_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "requirements-test.txt",
    "pytest.ini",
    "datus/__init__.py",
    "datus/configuration/",
    "datus/schemas/",
    "datus/models/base.py",
    "datus/models/__init__.py",
    "datus/models/session_manager.py",
    "datus/models/sqlite_session.py",
    "datus/models/litellm_model.py",
    "datus/models/responses_model.py",
    "datus/models/tool_result_stream.py",
    "datus/agent/node/agentic_node.py",
    "datus/agent/workflow.py",
    "datus/tools/__init__.py",
    "datus/utils/__init__.py",
    "datus/tools/func_tool/base.py",
    "datus/tools/func_tool/__init__.py",
    "datus/tools/policy_runtime.py",
    "datus/tools/sql_read_context.py",
    "conf/",
    "tests/conftest.py",
    "tests/helpers/",
    "tests/data/",
    "tests/conf/",
)
PR_MARK_EXPR = "not quarantine"


def validate_integration_rules(data: dict, root: Path = ROOT) -> list[str]:
    """Reject stale configured contracts instead of treating them as deleted diff tests."""
    rules = data.get("execution", {}).get("integration_rules")
    if not isinstance(rules, list) or not rules:
        return ["execution.integration_rules: expected a non-empty list of ownership rules"]
    errors, ids = [], set()
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("id"), str) or not rule["id"]:
            errors.append("integration_rules: expected a rule with a non-empty id")
            continue
        name = rule["id"]
        if name in ids:
            errors.append(f"integration_rules.{name}: duplicate rule id")
        ids.add(name)
        for field, prefix in (("paths", "datus/"), ("targets", "tests/integration/")):
            values = rule.get(field)
            if not isinstance(values, list) or not values:
                errors.append(f"integration_rules.{name}.{field}: expected a non-empty list of paths")
                continue
            for path in values:
                if not isinstance(path, str) or not path.startswith(prefix) or ".." in Path(path).parts:
                    errors.append(f"integration_rules.{name}.{field}: invalid path: {path!r}")
                elif field == "targets" and not (root / path).is_file():
                    errors.append(f"integration_rules.{name}: integration target does not exist: {path}")
                elif field == "paths" and not (root / path).exists():
                    errors.append(f"integration_rules.{name}: source path does not exist: {path}")
                elif field == "paths" and (root / path).is_dir() and not path.endswith("/"):
                    errors.append(f"integration_rules.{name}: directory source needs a trailing slash: {path}")
    return errors


def integration_groups(changed: list[str], rules: list[dict], unit_targets: list[str]) -> set[str]:
    all_groups = {rule["id"] for rule in rules}
    groups = set(all_groups) if "tests/unit_tests/" in unit_targets else set()
    for path in changed:
        if any(path == prefix or (prefix.endswith("/") and path.startswith(prefix)) for prefix in SHARED_PATHS):
            groups.update(all_groups)
        for rule in rules:
            if any(path == prefix or (prefix.endswith("/") and path.startswith(prefix)) for prefix in rule["paths"]):
                groups.add(rule["id"])
    return groups


def validate_integration_consumers(data: dict, root: Path = ROOT) -> list[str]:
    """Keep every registered integration contract selected for its literal imports."""
    from ci.harness.unit_consumers import find_test_consumers

    rules = data["execution"]["integration_rules"]
    targets = sorted({target for rule in rules for target in rule["targets"]})
    consumers = find_test_consumers(root, [root / target for target in targets])
    errors = []
    for source, tests in consumers.items():
        if not tests:
            continue
        # The caller already validated these owners against the requested root.
        owner = unit_owner(source, data["execution"]["unit_rules"])
        unit_targets = owner[1]["targets"] if owner else []
        groups = integration_groups([source], rules, unit_targets)
        selected = {target for rule in rules if rule["id"] in groups for target in rule["targets"]}
        for test in sorted(tests - selected):
            errors.append(f"integration_rules: uncovered source/test reference: {source} -> {test}")
    return errors


def nightly_pr_contracts() -> list[str]:
    """Legacy dual-marked local Dosi contracts also belong to scheduled P0 suites.

    Keep their Nightly registration; these use the installed binding and local
    DuckDB fixtures, without credentials, remote models or service provisioning.
    """
    data = load_map()
    semantic = next(rule["targets"] for rule in data["execution"]["integration_rules"] if rule["id"] == "semantic")
    return [*data["execution"]["core_nodeids"], *semantic]


def pr_test_plan(changed: list[str], data: dict | None = None, root: Path = ROOT) -> dict:
    """Keep marked unit tests in their owner suite instead of excluding them."""
    data = data or load_map()
    changed = sorted({normalize_path(path) for path in changed})
    unit = unit_test_plan(changed, data)
    errors = validate_integration_rules(data, root)
    if errors:
        raise ValueError("Invalid integration ownership rules:\n" + "\n".join(errors))
    rules = data["execution"]["integration_rules"]
    groups = integration_groups(changed, rules, unit["targets"])
    integration = {target for rule in rules if rule["id"] in groups for target in rule["targets"]}
    for path in changed:
        if path.startswith("tests/integration/"):
            if Path(path).name.startswith("test_") and path.endswith(".py"):
                integration.add(path)
            else:
                integration.update(target for rule in rules for target in rule["targets"])
    core = list(data["execution"]["core_nodeids"])
    missing_core = [target for target in core if not (root / target.split("::", 1)[0]).is_file()]
    if missing_core:
        raise ValueError(f"Missing mandatory core tests: {missing_core}")
    targets = collapse_targets(sorted(set(core + unit["targets"] + list(integration))))
    selected = [t for t in targets if (root / t.split("::", 1)[0]).exists()]
    excel = "tests/integration/tools/db_tools/test_data_file_loader_excel.py"
    return {
        **unit,
        "core_nodeids": core,
        "integration_targets": sorted(integration),
        "integration_groups": sorted(groups),
        "pytest_targets": selected,
        "needs_excel": any(t == excel or t.startswith(excel + "::") for t in selected),
        "deleted_targets": [t for t in targets if not (root / t.split("::", 1)[0]).exists()],
    }
