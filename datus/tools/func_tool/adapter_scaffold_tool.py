# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Adapter scaffolding tool for generating Datus adapter project skeletons."""

import inspect
import os
import textwrap
from typing import List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Adapter type configuration table
# ---------------------------------------------------------------------------

ADAPTER_TYPE_CONFIG = {
    "semantic": {
        "base_class": "BaseSemanticAdapter",
        "base_import": "from datus_semantic_core import BaseSemanticAdapter",
        "config_base_class": "SemanticAdapterConfig",
        "config_import": "from datus_semantic_core import SemanticAdapterConfig",
        "registry_import": "from datus_semantic_core import semantic_adapter_registry",
        "entry_point_group": "datus.semantic_adapters",
        "abstract_methods": [
            ("list_metrics", "self, path=None, limit=100, offset=0", "List[MetricDefinition]"),
            ("get_dimensions", "self, metric_name, path=None", "List[DimensionInfo]"),
            (
                "query_metrics",
                "self, metrics, dimensions=None, path=None, time_start=None, time_end=None, "
                "time_granularity=None, where=None, limit=None, order_by=None, dry_run=False",
                "QueryResult",
            ),
            ("validate_semantic", "self", "ValidationResult"),
        ],
    },
    "bi": {
        "base_class": "BIAdapterBase",
        "base_import": "# from datus_bi_core import BIAdapterBase  # Install datus-bi-core",
        "config_base_class": "BaseModel",
        "config_import": "from pydantic import BaseModel, Field",
        "registry_import": "# from datus_bi_core import adapter_registry  # Install datus-bi-core",
        "entry_point_group": "datus.bi_adapters",
        "abstract_methods": [
            ("platform_name", "self", "str"),
            ("list_dashboards", "self, search='', page_size=20", "list"),
            ("get_dashboard_info", "self, dashboard_id", "object"),
            ("list_charts", "self, dashboard_id", "list"),
            ("list_datasets", "self, dashboard_id", "list"),
        ],
    },
    "db": {
        "base_class": "BaseSqlConnector",
        "base_import": "# from datus_db_core import BaseSqlConnector  # Install datus-db-core",
        "config_base_class": "BaseModel",
        "config_import": "from pydantic import BaseModel, Field",
        "registry_import": "# from datus_db_core import connector_registry  # Install datus-db-core",
        "entry_point_group": "datus.adapters",
        "abstract_methods": [
            ("execute", "self, input_params, result_format=None", "object"),
            ("test_connection", "self", "bool"),
            ("get_databases", "self, catalog_name='', include_sys=False", "list"),
        ],
    },
    "scheduler": {
        "base_class": "BaseSchedulerAdapter",
        "base_import": ("# from datus_scheduler_core import BaseSchedulerAdapter  # Install datus-scheduler-core"),
        "config_base_class": "BaseModel",
        "config_import": "from pydantic import BaseModel, Field",
        "registry_import": ("# from datus_scheduler_core import scheduler_registry  # Install datus-scheduler-core"),
        "entry_point_group": "datus.schedulers",
        "abstract_methods": [
            ("platform_name", "self", "str"),
            ("test_connection", "self", "bool"),
            ("submit_job", "self, payload", "object"),
            ("get_job", "self, job_id", "object"),
            ("list_jobs", "self, project=None, status=None, limit=50, offset=0", "list"),
        ],
    },
}

# Package name prefix by adapter type
_PACKAGE_PREFIX = {
    "semantic": "datus_semantic_{platform}",
    "bi": "datus_bi_{platform}",
    "db": "datus_{platform}",
    "scheduler": "datus_scheduler_{platform}",
}

# Project name prefix by adapter type (hyphenated, for pyproject.toml)
_PROJECT_NAME = {
    "semantic": "datus-semantic-{platform}",
    "bi": "datus-bi-{platform}",
    "db": "datus-{platform}",
    "scheduler": "datus-scheduler-{platform}",
}

_COPYRIGHT_HEADER = textwrap.dedent(
    """\
    # Copyright 2025-present DatusAI, Inc.
    # Licensed under the Apache License, Version 2.0.
    # See http://www.apache.org/licenses/LICENSE-2.0 for details.
    """
)


def _to_pascal_case(name: str) -> str:
    """Convert snake_case or hyphenated name to PascalCase."""
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


def _write(path: str, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------


def _gen_init_py(
    *,
    adapter_type: str,
    platform: str,
    class_name: str,
    display_name: str,
    cfg: dict,
) -> str:
    registry_import = cfg["registry_import"]
    entry_group = cfg["entry_point_group"]

    if adapter_type == "semantic":
        register_body = textwrap.dedent(
            f"""\
            def register():
                \"\"\"Register this adapter with the Datus semantic adapter registry.\"\"\"
                {registry_import}
                semantic_adapter_registry.register(
                    service_type="{platform}",
                    adapter_class={class_name}Adapter,
                    config_class={class_name}Config,
                    display_name="{display_name}",
                )
            """
        )
    elif adapter_type == "bi":
        register_body = textwrap.dedent(
            f"""\
            def register():
                \"\"\"Register this adapter with the Datus BI adapter registry.\"\"\"
                {registry_import}
                # adapter_registry.register(
                #     platform="{platform}",
                #     adapter_class={class_name}Adapter,
                #     config_class={class_name}Config,
                #     display_name="{display_name}",
                # )
            """
        )
    elif adapter_type == "db":
        register_body = textwrap.dedent(
            f"""\
            def register():
                \"\"\"Register this connector with the Datus DB connector registry.\"\"\"
                {registry_import}
                # connector_registry.register(
                #     db_type="{platform}",
                #     connector_class={class_name}Adapter,
                #     config_class={class_name}Config,
                # )
            """
        )
    else:  # scheduler
        register_body = textwrap.dedent(
            f"""\
            def register():
                \"\"\"Register this adapter with the Datus scheduler registry.\"\"\"
                {registry_import}
                # scheduler_registry.register(
                #     platform="{platform}",
                #     adapter_class={class_name}Adapter,
                #     config_class={class_name}Config,
                #     display_name="{display_name}",
                # )
            """
        )

    lines = [
        _COPYRIGHT_HEADER,
        f'"""{display_name} adapter for Datus — entry point group: {entry_group}."""\n',
        f"from .adapter import {class_name}Adapter",
        f"from .config import {class_name}Config",
        "",
        f'__all__ = ["{class_name}Adapter", "{class_name}Config", "register"]',
        "",
        "",
        register_body,
    ]
    return "\n".join(lines)


def _gen_adapter_py(
    *,
    adapter_type: str,
    platform: str,
    class_name: str,
    display_name: str,
    cfg: dict,
) -> str:
    base_import = cfg["base_import"]
    base_class = cfg["base_class"]
    methods = cfg["abstract_methods"]
    is_async = adapter_type == "semantic"

    method_blocks = []
    for method_name, sig, return_type in methods:
        if is_async:
            method_blocks.append(
                textwrap.dedent(
                    f"""\
                    async def {method_name}({sig}) -> {return_type}:
                        raise NotImplementedError("TODO: Implement {method_name}")
                    """
                )
            )
        else:
            method_blocks.append(
                textwrap.dedent(
                    f"""\
                    def {method_name}({sig}) -> {return_type}:
                        raise NotImplementedError("TODO: Implement {method_name}")
                    """
                )
            )

    indented_methods = "\n".join(
        "    " + line if line else "" for block in method_blocks for line in block.splitlines()
    )

    if adapter_type == "semantic":
        init_block = textwrap.dedent(
            f"""\
                def __init__(self, config: "{class_name}Config"):
                    super().__init__(config)
            """
        )
    else:
        init_block = textwrap.dedent(
            f"""\
                def __init__(self, config: "{class_name}Config"):
                    super().__init__()
                    self.config = config
            """
        )

    lines = [
        _COPYRIGHT_HEADER,
        f'"""{display_name} adapter implementation."""\n',
        base_import,
        "",
        f"from .config import {class_name}Config",
        "",
        "",
        f"class {class_name}Adapter({base_class}):",
        f'    """{display_name} adapter for Datus."""',
        "",
        init_block,
        "",
        indented_methods,
    ]
    return "\n".join(lines)


def _gen_config_py(
    *,
    adapter_type: str,
    platform: str,
    class_name: str,
    display_name: str,
    cfg: dict,
) -> str:
    config_import = cfg["config_import"]
    config_base = cfg["config_base_class"]

    if adapter_type == "semantic":
        extra_field = f'    service_type: str = "{platform}"'
    else:
        extra_field = f'    platform: str = "{platform}"'

    lines = [
        _COPYRIGHT_HEADER,
        f'"""{display_name} adapter configuration."""\n',
        config_import,
        "",
        "",
        f"class {class_name}Config({config_base}):",
        f'    """{display_name} adapter configuration."""',
        "",
        extra_field,
        "    # TODO: Add connection/authentication fields here",
        "    # host: str = Field(..., description='Host address')",
        "    # port: int = Field(443, description='Port')",
        "    # api_key: str = Field(..., description='API key')",
        "",
    ]
    return "\n".join(lines)


def _gen_py_typed() -> str:
    return ""


def _gen_test_skeleton(*, class_name: str, package_name: str, display_name: str) -> str:
    lines = [
        _COPYRIGHT_HEADER,
        f'"""Unit tests for {display_name} adapter."""\n',
        "import pytest",
        "",
        f"from {package_name}.adapter import {class_name}Adapter",
        f"from {package_name}.config import {class_name}Config",
        "",
        "",
        f"class Test{class_name}Adapter:",
        f'    """Tests for {class_name}Adapter."""',
        "",
        "    def setup_method(self):",
        "        self.config = {class_name}Config()".replace("{class_name}", class_name),
        "",
        "    def test_instantiation(self):",
        "        # TODO: adapter may raise NotImplementedError in __init__ — adjust as needed",
        f"        adapter = {class_name}Adapter(config=self.config)",
        f"        assert isinstance(adapter, {class_name}Adapter)",
        "",
        "    def test_register(self):",
        f"        from {package_name} import register",
        "        # register() should be callable without raising an import error",
        "        # (real registration requires the core package to be installed)",
        "        assert callable(register)",
        "",
    ]
    return "\n".join(lines)


def _gen_conftest() -> str:
    lines = [
        _COPYRIGHT_HEADER,
        '"""pytest configuration for adapter tests."""\n',
        "import pytest",
        "",
        "",
        "# Add shared fixtures here",
        "",
    ]
    return "\n".join(lines)


def _gen_pyproject_toml(
    *,
    adapter_type: str,
    platform: str,
    package_name: str,
    display_name: str,
    cfg: dict,
) -> str:
    project_name = _PROJECT_NAME[adapter_type].format(platform=platform.replace("_", "-"))
    entry_group = cfg["entry_point_group"]

    return textwrap.dedent(
        f"""\
        [project]
        name = "{project_name}"
        version = "0.1.0"
        description = "{display_name} adapter for Datus"
        requires-python = ">=3.12"

        [project.entry-points."{entry_group}"]
        {platform} = "{package_name}:register"

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["{package_name}"]
        """
    )


def _gen_readme(*, display_name: str, adapter_type: str, platform: str, package_name: str) -> str:
    return textwrap.dedent(
        f"""\
        # {display_name} Adapter for Datus

        This package provides a **{adapter_type}** adapter for the **{platform}** platform.

        ## Installation

        ```bash
        pip install -e .
        ```

        ## Usage

        ```python
        from {package_name} import register

        register()  # Register with the Datus adapter registry
        ```

        ## Development

        Implement the `TODO` stubs in `{package_name}/adapter.py` and add configuration
        fields to `{package_name}/config.py`.

        Run tests:

        ```bash
        pytest tests/unit/
        ```
        """
    )


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class AdapterScaffoldTool:
    """FuncTool for generating Datus adapter project scaffolding."""

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    def available_tools(self) -> List[Tool]:
        return [
            trans_to_function_tool(self.scaffold_adapter),
            trans_to_function_tool(self.validate_adapter),
            trans_to_function_tool(self.list_adapter_types),
        ]

    # ── Public tool methods ────────────────────────────────────────────────

    def list_adapter_types(self) -> FuncToolResult:
        """List all supported Datus adapter types with metadata.

        Returns:
            FuncToolResult with result as a list of adapter type descriptor dicts,
            each containing: type, base_class, entry_point_group, package_prefix.
        """
        rows = []
        for adapter_type, cfg in ADAPTER_TYPE_CONFIG.items():
            rows.append(
                {
                    "type": adapter_type,
                    "base_class": cfg["base_class"],
                    "entry_point_group": cfg["entry_point_group"],
                    "package_prefix": _PACKAGE_PREFIX[adapter_type].replace("{platform}", "<platform>"),
                }
            )
        return FuncToolResult(result=rows)

    def scaffold_adapter(
        self,
        adapter_type: str,
        platform: str,
        output_dir: str,
        display_name: str = "",
    ) -> FuncToolResult:
        """Generate a complete Datus adapter project skeleton.

        Args:
            adapter_type: One of "semantic", "bi", "db", "scheduler".
            platform: Platform identifier, e.g. "cube", "metabase", "clickhouse", "dagster".
            output_dir: Filesystem path where the project directory will be created.
            display_name: Human-readable name (defaults to platform.title() if omitted).

        Returns:
            FuncToolResult with result containing output_dir, files_created list, and next_steps.
        """
        if adapter_type not in ADAPTER_TYPE_CONFIG:
            valid = ", ".join(sorted(ADAPTER_TYPE_CONFIG.keys()))
            return FuncToolResult(
                success=0,
                error=f"Invalid adapter_type '{adapter_type}'. Must be one of: {valid}",
            )

        try:
            cfg = ADAPTER_TYPE_CONFIG[adapter_type]
            if not display_name:
                display_name = platform.replace("_", " ").title()

            class_name = _to_pascal_case(platform)
            package_name = _PACKAGE_PREFIX[adapter_type].format(platform=platform)

            project_root = os.path.abspath(output_dir)

            files_created: List[str] = []

            def write_file(rel_path: str, content: str) -> None:
                abs_path = os.path.join(project_root, rel_path)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                files_created.append(rel_path)

            # Package files
            write_file(
                os.path.join(package_name, "__init__.py"),
                _gen_init_py(
                    adapter_type=adapter_type,
                    platform=platform,
                    class_name=class_name,
                    display_name=display_name,
                    cfg=cfg,
                ),
            )
            write_file(
                os.path.join(package_name, "adapter.py"),
                _gen_adapter_py(
                    adapter_type=adapter_type,
                    platform=platform,
                    class_name=class_name,
                    display_name=display_name,
                    cfg=cfg,
                ),
            )
            write_file(
                os.path.join(package_name, "config.py"),
                _gen_config_py(
                    adapter_type=adapter_type,
                    platform=platform,
                    class_name=class_name,
                    display_name=display_name,
                    cfg=cfg,
                ),
            )
            write_file(os.path.join(package_name, "py.typed"), _gen_py_typed())

            # Test files
            write_file(os.path.join("tests", "__init__.py"), "")
            write_file(os.path.join("tests", "unit", "__init__.py"), "")
            write_file(
                os.path.join("tests", "unit", "test_adapter.py"),
                _gen_test_skeleton(
                    class_name=class_name,
                    package_name=package_name,
                    display_name=display_name,
                ),
            )
            write_file(os.path.join("tests", "conftest.py"), _gen_conftest())

            # Project files
            write_file(
                "pyproject.toml",
                _gen_pyproject_toml(
                    adapter_type=adapter_type,
                    platform=platform,
                    package_name=package_name,
                    display_name=display_name,
                    cfg=cfg,
                ),
            )
            write_file(
                "README.md",
                _gen_readme(
                    display_name=display_name,
                    adapter_type=adapter_type,
                    platform=platform,
                    package_name=package_name,
                ),
            )

            next_steps = (
                f"1. cd {project_root}\n"
                f"2. Implement stubs in {package_name}/adapter.py\n"
                f"3. Add config fields in {package_name}/config.py\n"
                f"4. pip install -e .\n"
                f"5. pytest tests/unit/"
            )

            logger.info(
                "Scaffolded %s adapter '%s' at %s (%d files)",
                adapter_type,
                platform,
                project_root,
                len(files_created),
            )

            return FuncToolResult(
                result={
                    "output_dir": project_root,
                    "files_created": files_created,
                    "next_steps": next_steps,
                }
            )

        except Exception as exc:
            logger.exception("scaffold_adapter failed for %s/%s", adapter_type, platform)
            return FuncToolResult(success=0, error=str(exc))

    def validate_adapter(self, adapter_module_path: str) -> FuncToolResult:
        """Validate a generated adapter module for correctness.

        Checks that the module:
        - Can be imported successfully
        - Exposes a ``register()`` callable
        - Contains an adapter class
        - Has abstract methods implemented (not just NotImplementedError stubs)

        Args:
            adapter_module_path: Dotted module path, e.g. "datus_bi_metabase" or
                                 "datus_semantic_cube".

        Returns:
            FuncToolResult with result containing "valid" (bool) and "issues" (list of strings).
        """
        issues: List[str] = []

        try:
            import importlib

            mod = importlib.import_module(adapter_module_path)
        except ImportError as exc:
            return FuncToolResult(
                success=0,
                error=f"Cannot import module '{adapter_module_path}': {exc}",
            )
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"Error importing module '{adapter_module_path}': {exc}",
            )

        # 1. Check register() callable
        register_fn = getattr(mod, "register", None)
        if register_fn is None:
            issues.append("Missing 'register' function")
        elif not callable(register_fn):
            issues.append("'register' attribute is not callable")

        # 2. Find an adapter class
        adapter_class = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if (
                obj is not None
                and isinstance(obj, type)
                and attr_name.endswith("Adapter")
                and not attr_name.startswith("Base")
            ):
                adapter_class = obj
                break

        if adapter_class is None:
            issues.append("No adapter class found (expected a class named *Adapter)")
        else:
            # 3. Check for unimplemented abstract methods (NotImplementedError stubs)
            for name, method in inspect.getmembers(adapter_class, predicate=inspect.isfunction):
                if name.startswith("_"):
                    continue
                try:
                    src = inspect.getsource(method)
                    if "NotImplementedError" in src:
                        issues.append(f"Method '{name}' is not implemented (raises NotImplementedError)")
                except (OSError, TypeError):
                    # Source not available — skip
                    pass

        valid = len(issues) == 0
        return FuncToolResult(result={"valid": valid, "issues": issues})
