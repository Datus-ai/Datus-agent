#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Workspace initialization command for Datus Agent.

Generates AGENTS.md in the current project directory. Requires a configured
LLM (use `/model` inside the CLI). Reads configured services from agent.yml,
scans the directory structure, and uses the LLM to produce a project-level
AGENTS.md with Architecture, Directory Map, Services, and Artifacts sections.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.prompt import Prompt

from datus.utils.loggings import get_logger, print_rich_exception

logger = get_logger(__name__)


def _scan_directory(root: str, max_depth: int = 3) -> str:
    """Scan directory structure and return a tree-like string."""
    lines = []
    root_path = Path(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and common noise
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d
            not in ("__pycache__", "node_modules", ".git", ".venv", "venv", ".tox", ".mypy_cache", "dist", "build")
        ]

        depth = len(Path(dirpath).relative_to(root_path).parts)
        if depth > max_depth:
            dirnames.clear()
            continue

        indent = "  " * depth
        dir_name = os.path.basename(dirpath) if depth > 0 else "."
        lines.append(f"{indent}{dir_name}/")

        # Show key files at each level
        key_files = [f for f in filenames if not f.startswith(".") and not f.endswith(".pyc")]
        if len(key_files) <= 8:
            for f in sorted(key_files):
                lines.append(f"{indent}  {f}")
        else:
            for f in sorted(key_files)[:5]:
                lines.append(f"{indent}  {f}")
            lines.append(f"{indent}  ... ({len(key_files) - 5} more files)")

    return "\n".join(lines)


def _detect_project_type(root: str) -> str:
    """Detect project type from common files."""
    indicators = {
        "pyproject.toml": "Python (pyproject.toml)",
        "setup.py": "Python (setup.py)",
        "package.json": "Node.js",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java (Gradle)",
        "docker-compose.yml": "Docker Compose",
        "Dockerfile": "Docker",
        "dbt_project.yml": "dbt",
        "Makefile": "Make",
    }
    detected = []
    for filename, label in indicators.items():
        if os.path.exists(os.path.join(root, filename)):
            detected.append(label)
    return ", ".join(detected) if detected else "Unknown"


def _build_services_section(datasources: Dict[str, Any]) -> str:
    """Build Services section from configured datasources."""
    if not datasources:
        return "No services configured. Use `/datasource` inside the CLI to add datasources.\n"

    lines = ["| Name | Type | Connection |", "|------|------|------------|"]
    for name, cfg in datasources.items():
        conn = ""
        if hasattr(cfg, "uri") and cfg.uri:
            conn = cfg.uri
        elif hasattr(cfg, "host") and cfg.host:
            conn = f"{cfg.host}:{cfg.port}"
        elif hasattr(cfg, "account") and cfg.account:
            conn = f"account={cfg.account}"
        lines.append(f"| {name} | {cfg.type} | {conn} |")
    return "\n".join(lines) + "\n"


class InitWorkspace:
    """Initialize project workspace by generating AGENTS.md."""

    def __init__(self, args):
        self.args = args
        self.console = Console(log_path=False)
        self.project_dir = os.getcwd()
        self.project_name = os.path.basename(self.project_dir)
        self.agents_md_path = os.path.join(self.project_dir, "AGENTS.md")

    def run(self) -> int:
        """Main entry point."""
        try:
            # Check agent.yml exists
            from datus.configuration.agent_config_loader import load_agent_config

            try:
                agent_config = load_agent_config(
                    config=getattr(self.args, "config", "") or "", action="service", reload=True
                )
            except Exception as e:
                self.console.print(f"[red]Failed to load configuration: {e}[/red]")
                self.console.print("Run 'datus init' first to set up the configuration.")
                return 1

            # Check for existing AGENTS.md
            if os.path.exists(self.agents_md_path):
                self.console.print(f"[yellow]AGENTS.md already exists at {self.agents_md_path}[/yellow]")
                action = Prompt.ask(
                    "What would you like to do?",
                    choices=["overwrite", "cancel"],
                    default="cancel",
                )
                if action == "cancel":
                    self.console.print("Init cancelled.")
                    return 0

            self.console.print(f"\n[bold cyan]Initializing project: {self.project_name}[/bold cyan]\n")

            # Scan directory
            self.console.print("[dim]Scanning directory structure...[/dim]")
            dir_tree = _scan_directory(self.project_dir)
            project_type = _detect_project_type(self.project_dir)

            # Build services section from config
            services_section = _build_services_section(agent_config.services.datasources)

            # Probe database schema if --datasource specified
            db_schema_info = ""
            db_name = getattr(self.args, "datasource", "")
            if db_name:
                db_schema_info = self._probe_database(agent_config, db_name)

            # Try LLM-assisted generation
            content = self._generate_with_llm(agent_config, dir_tree, project_type, services_section, db_schema_info)
            if not content:
                # Fallback to template
                self.console.print("[yellow]LLM generation failed, using template.[/yellow]")
                content = self._generate_template(dir_tree, project_type, services_section)

            # Write AGENTS.md
            with open(self.agents_md_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.console.print(f"\n[bold green]AGENTS.md created at {self.agents_md_path}[/bold green]")
            self.console.print("You can edit this file to refine the project description.")
            return 0

        except KeyboardInterrupt:
            self.console.print("\nInit cancelled by user")
            return 1
        except Exception as e:
            print_rich_exception(self.console, e, "Init failed", logger)
            return 1

    def _probe_database(self, agent_config, db_name: str) -> str:
        """Probe database schema and return summary for LLM context."""
        try:
            from datus.tools.db_tools.db_manager import DBManager

            self.console.print(f"[dim]Probing database '{db_name}'...[/dim]")
            db_config = agent_config.services.datasources.get(db_name)
            if not db_config:
                self.console.print(f"[yellow]Database '{db_name}' not found in config, skipping probe.[/yellow]")
                return ""

            datasource_configs = {db_name: {db_name: db_config}}
            db_manager = DBManager(datasource_configs)
            connector = db_manager.get_conn(db_name, db_name)

            tables = connector.get_tables()
            if not tables:
                return f"Database '{db_name}' ({db_config.type}): no tables found.\n"

            lines = [f"Database '{db_name}' ({db_config.type}) — {len(tables)} tables:"]
            for t in tables[:30]:  # Limit to 30 tables
                table_name = t.get("table_name", t.get("name", str(t)))
                lines.append(f"  - {table_name}")
            if len(tables) > 30:
                lines.append(f"  ... and {len(tables) - 30} more tables")

            self.console.print(f"[dim]Found {len(tables)} tables in '{db_name}'[/dim]")
            return "\n".join(lines) + "\n"

        except Exception as e:
            logger.warning(f"Database probe failed for '{db_name}': {e}")
            self.console.print(f"[yellow]Could not probe database '{db_name}': {e}[/yellow]")
            return ""

    def _generate_with_llm(
        self, agent_config, dir_tree: str, project_type: str, services_section: str, db_schema_info: str = ""
    ) -> Optional[str]:
        """Use configured LLM to generate AGENTS.md content."""
        try:
            from datus.models.base import LLMBaseModel

            model_config = agent_config.active_model()
            model_type = model_config.type
            model_class_name = LLMBaseModel.MODEL_TYPE_MAP.get(model_type)
            if not model_class_name:
                logger.warning(f"Unsupported model type: {model_type}")
                return None
            module = __import__(f"datus.models.{model_type}_model", fromlist=[model_class_name])
            model_class = getattr(module, model_class_name)
            llm = model_class(model_config)

            # Read README if exists
            readme_content = ""
            for readme_name in ("README.md", "readme.md", "README.rst", "README"):
                readme_path = os.path.join(self.project_dir, readme_name)
                if os.path.exists(readme_path):
                    try:
                        with open(readme_path, encoding="utf-8") as f:
                            readme_content = f.read()[:3000]
                    except Exception:
                        pass
                    break

            db_section = ""
            if db_schema_info:
                db_section = f"\nDatabase schema:\n{db_schema_info}\n"

            prompt = f"""Generate an AGENTS.md file for the project "{self.project_name}".

Project type: {project_type}
Project directory: {self.project_dir}

Directory structure:
```
{dir_tree}
```

{f"README excerpt:\n{readme_content}\n" if readme_content else ""}\
Configured services (databases):
{services_section}
{db_section}\
Generate AGENTS.md with these exact sections (use ## headers):

1. **# {self.project_name}** — One-line project description

2. **## Architecture** — Brief architecture description based on the directory structure and README.
   If this is a data project, describe the data flow. If a web app, describe the stack.
   Include an ASCII diagram if appropriate.

3. **## Directory Map** — Table with columns: Directory | Purpose | Key Entry Point | Consumer
   Cover the main directories found in the scan.

4. **## Services** — Use the configured services table provided above.
   Add any additional services detected from docker-compose.yml, Dockerfile, etc.
{
                '''
5. **## Data Tables** — List tables from the database schema above.
   Table with columns: Table | Type | Description
'''
                if db_schema_info
                else ""
            }\
{
                f"{'6' if db_schema_info else '5'}"
            }. **## Artifacts** — Describe data artifacts, configs, or outputs this project produces.
   Examples: data catalogs, semantic models, SQL files, reports, API schemas.

Output ONLY the markdown content, no code fences around the entire document."""

            self.console.print("[dim]Generating AGENTS.md with LLM...[/dim]")
            response = llm.generate(prompt)

            if response and len(response) > 100:
                return response
            return None

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return None

    def _generate_template(self, dir_tree: str, project_type: str, services_section: str) -> str:
        """Generate a template AGENTS.md without LLM."""
        return f"""# {self.project_name}

> Project type: {project_type}

## Architecture

<!-- Describe your project architecture here -->

```
<!-- Add architecture diagram -->
```

## Directory Map

| Directory | Purpose | Key Entry Point | Consumer |
|-----------|---------|-----------------|----------|
| `.` | Project root | `README.md` | — |

<!-- Add more directories -->

## Services

{services_section}

## Artifacts

<!-- Describe data artifacts, configs, or outputs this project produces -->

| Artifact | Type | Location | Description |
|----------|------|----------|-------------|
"""
