"""Audit explicit unit ownership against statically visible test consumers.

This does not import application modules or infer runtime execution coverage.
Factories, dependency injection and computed imports still need conservative
package owners and explicit consumer contracts in the coverage map.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from importlib.util import resolve_name
from pathlib import Path

from ci.harness.test_plan import ROOT, unit_owner


def _module_name(path: Path, repo_root: Path) -> str:
    parts = path.relative_to(repo_root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _from_module(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if not node.level:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    return resolve_name("." * node.level + (node.module or ""), package)


def audit_unit_consumers(data: dict, repo_root: Path = ROOT) -> dict:
    """Report every test file whose visible source reference its owner omits.

    Includes function-local imports, package re-exports, and literal module or
    patch paths. Parents are included because importing a child executes its
    package initializers. Non-literal dynamic imports are outside this audit.
    """
    modules = {_module_name(p, repo_root): p for p in (repo_root / "datus").rglob("*.py")}
    exports = defaultdict(lambda: defaultdict(set))
    for module, path in modules.items():
        if path.name != "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                origin = _from_module(node, module, True)
                for name in node.names:
                    exports[module][name.asname or name.name].add((origin, name.name))
            elif isinstance(node, ast.Import):
                for name in node.names:
                    exports[module][name.asname or name.name.split(".")[0]].add((name.name, ""))

    def references(module: str, name: str = "", seen: frozenset = frozenset()) -> set[str]:
        key = (module, name)
        if key in seen:
            return set()
        result = {module}
        if name:
            result.add(module + "." + name)
            if name == "*":
                for exported in exports[module]:
                    result.update(references(module, exported, seen | {key}))
            elif name in exports[module]:
                for origin in exports[module][name]:
                    result.update(references(*origin, seen | {key}))
        return result

    consumers = {p.relative_to(repo_root).as_posix(): set() for p in modules.values()}
    test_files = sorted((repo_root / "tests/unit_tests").rglob("test_*.py"))
    for path in test_files:
        module = _module_name(path, repo_root)
        refs = set()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                refs.update(name.name for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                origin = _from_module(node, module, False)
                for name in node.names:
                    refs.update(references(origin, name.name))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("datus."):
                refs.add(node.value.split(":", 1)[0])
        test = path.relative_to(repo_root).as_posix()
        for ref in refs:
            while ref:
                if ref in modules:
                    consumers[modules[ref].relative_to(repo_root).as_posix()].add(test)
                ref = ref.rpartition(".")[0]

    missing = []
    for source, tests in sorted(consumers.items()):
        match = unit_owner(source, data["execution"]["unit_rules"])
        if match is None:
            missing.append({"rule": "<unowned>", "source": source, "test": ""})
            continue
        _, rule = match
        for test in sorted(tests):
            if not any(test == t or (t.endswith("/") and test.startswith(t)) for t in rule["targets"]):
                missing.append({"rule": rule["id"], "source": source, "test": test})
    return {
        "source_modules": len(modules),
        "test_files": len(test_files),
        "consumers": {s: sorted(t) for s, t in sorted(consumers.items())},
        "missing": missing,
    }


def validate_unit_consumers(data: dict, repo_root: Path = ROOT) -> list[str]:
    gaps = defaultdict(list)
    for gap in audit_unit_consumers(data, repo_root)["missing"]:
        gaps[gap["rule"]].append(f"{gap['source']} -> {gap['test'] or 'no unit owner'}")
    return [
        f"unit_rules.{rule}: {len(refs)} uncovered source/test references: " + "; ".join(refs[:3])
        for rule, refs in sorted(gaps.items())
    ]
