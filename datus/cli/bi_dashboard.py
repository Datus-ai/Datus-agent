# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import re
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from datus.cli._cli_utils import prompt_input
from datus.cli.init_util import init_metrics, init_semantic_model
from datus.cli.interactive_init import ReferenceSqlStreamHandler
from datus.configuration.agent_config import AgentConfig, DashboardConfig
from datus.configuration.agent_config_loader import configuration_manager
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.storage.reference_sql.reference_sql_init import init_reference_sql
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.schema_metadata.local_init import init_local_schema
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.bi_tools.base_adaptor import AuthParam, AuthType, BIAdaptorBase, ChartInfo, DashboardInfo, DatasetInfo
from datus.tools.bi_tools.dashboard_assembler import (
    ChartSelection,
    DashboardAssembler,
    DashboardAssemblyResult,
    SelectedSqlCandidate,
)
from datus.tools.bi_tools.registry import adaptor_registry
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager
from datus.utils.reference_paths import quote_path_segment
from datus.utils.sql_utils import metadata_identifier, parse_table_name_parts
from datus.utils.stream_output import StreamOutputManager
from datus.utils.sub_agent_manager import SubAgentManager
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI


@dataclass(slots=True)
class DashboardCliOptions:
    platform: str
    dashboard_url: str
    api_base_url: str
    auth_params: AuthParam | None = None
    dialect: Optional[str] = None


def _parse_subject_path_for_metrics(tags: List[str]) -> Optional[str]:
    if not tags:
        return None
    for tag in tags:
        if tag.startswith("subject_tree:"):
            parts = [p.strip() for p in tag[13:].strip().split("/") if p.strip()]
            if parts:
                return ".".join(parts)
    return None


class BiDashboardCommands:
    def __init__(
        self, agent_config: AgentConfig | "DatusCLI", console: Optional[Console] = None, force: bool = False
    ) -> None:
        self.cli: Optional["DatusCLI"] = None
        if hasattr(agent_config, "agent_config"):
            self.cli = agent_config
            self.agent_config = agent_config.agent_config
            self.console = console or agent_config.console
            self._configuration_manager = getattr(agent_config, "configuration_manager", None)
        else:
            self.agent_config = agent_config
            self.console = console or Console(log_path=False)
            self._configuration_manager = None
        self._adaptor_registry = self._discover_adaptors()
        self._force = force
        self.db_manager = db_manager_instance(self.agent_config.namespaces)

    def current_database_context(self) -> Tuple[str, str, str]:
        current_con = self.db_manager.get_conn(self.agent_config.current_namespace)
        return (
            getattr(current_con, "catalog_name", ""),
            getattr(current_con, "database_name", ""),
            getattr(current_con, "schema_name", ""),
        )

    @optional_traceable(name="bootstrap_bi")
    def cmd(self, args: str = "") -> None:
        try:
            options = self._prompt_options()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Cancelled.[/]")
            return
        except Exception as exc:
            logger.error("Failed to initialize BI dashboard options", exc_info=True)
            self.console.print(f"[bold red]Error:[/] {exc}")
            return

        adaptor = self._create_adaptor(options)
        default_catalog, default_database, default_schema = self._resolve_default_table_context()
        assembler = DashboardAssembler(
            adaptor,
            default_dialect=options.dialect,
            default_catalog=default_catalog,
            default_database=default_database,
            default_schema=default_schema,
        )

        try:
            dashboard, dashboard_id = self._confirm_dashboard(adaptor, options.dashboard_url)
            if dashboard is None:
                return

            with self.console.status("Loading charts..."):
                chart_metas = adaptor.list_charts(dashboard_id)
            if not chart_metas:
                self.console.print("[yellow]No charts found in this dashboard.[/]")
                return

            chart_details = self._hydrate_charts(adaptor, dashboard_id, chart_metas)

            if not chart_details:
                self.console.print("[yellow]No charts found in this dashboard.[/]")
                return

            self._render_chart_table(chart_details, title="Charts")

            # Select charts for reference SQL initialization
            ref_selection_input = self._prompt_input(
                "Select chart indexes to init reference SQL (e.g. 1,3,... or all)", default="all"
            )
            chart_indices_ref = self._parse_selection(ref_selection_input, len(chart_details))
            chart_selections_ref = self._load_chart_selections(
                chart_details, chart_indices_ref, purpose="reference SQL"
            )

            # Select charts for metrics initialization
            self.console.print(
                "[dim]Tip: For metrics, select charts with aggregation SQL " "(e.g. SUM, COUNT, AVG, MAX, MIN)[/]"
            )

            metrics_selection_input = self._prompt_input(
                "Select chart indexes to init metrics (e.g. 1,3,... or all)", default="all"
            )
            chart_indices_metrics = self._parse_selection(metrics_selection_input, len(chart_details))
            chart_selections_metrics = self._load_chart_selections(
                chart_details, chart_indices_metrics, purpose="metrics"
            )

            if not chart_selections_ref and not chart_selections_metrics:
                self.console.print("[yellow]No charts selected for reference SQL or metrics. Aborting.[/]")
                return

            with self.console.status("Loading datasets..."):
                datasets = adaptor.list_datasets(dashboard_id)

            result = assembler.assemble(dashboard, chart_selections_ref, chart_selections_metrics, datasets)

            result.tables = self._review_tables(result.tables)

            self._save_sub_agent(options.platform, dashboard, result)
            self.console.print("[green]Sub-Agent build successful.[/]")
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Cancelled.[/]")
        finally:
            try:
                adaptor.close()
            except Exception:
                pass

    def _prompt_options(self) -> DashboardCliOptions:
        platforms = sorted(self._adaptor_registry)
        if not platforms:
            raise ValueError("No BI adaptor implementations found.")
        platform = self._prompt_input("Select BI platform", default=platforms[0], choices=platforms)
        if platform not in self._adaptor_registry:
            raise ValueError(f"Unsupported platform '{platform}'")

        dashboard_url = self._prompt_input("Dashboard URL")
        if not dashboard_url:
            raise ValueError("Dashboard URL is required.")

        api_base_url = self._derive_api_base(dashboard_url)
        api_base_url = self._prompt_input("API base URL (e.g. https://host)", default=api_base_url)
        if not api_base_url:
            raise ValueError("API base URL is required.")
        metadata = adaptor_registry.get_metadata(platform)
        if metadata is None:
            raise ValueError(f"Missing BI adaptor metadata for '{platform}'")
        auth_param = self._resolve_auth_params(platform, metadata.auth_type)
        if auth_param is None:
            auth_param = self._prompt_auth_params(platform, metadata.auth_type)

        default_dialect = self.agent_config.db_type

        return DashboardCliOptions(
            platform=platform,
            dashboard_url=dashboard_url,
            api_base_url=api_base_url,
            auth_params=auth_param,
            dialect=default_dialect,
        )

    def _resolve_auth_params(self, platform: str, auth_type: AuthType) -> Optional[AuthParam]:
        configs = getattr(self.agent_config, "dashboard_config", None) or {}
        config = self._lookup_dashboard_config(configs, platform)
        if config is None:
            return None

        username = (config.username or "").strip()
        password = (config.password or "").strip()
        api_key = (config.api_key or "").strip()
        extra = config.extra or {}

        auth_param = AuthParam()
        if auth_type == AuthType.LOGIN:
            if not username or not password:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"Dashboard auth config for '{platform}' requires username and password.",
                )
            auth_param.username = username
            auth_param.password = password
            auth_param.extra = extra
        elif auth_type == AuthType.API_KEY:
            if not api_key:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR, message=f"Dashboard auth config for '{platform}' requires api_key."
                )
            auth_param.api_key = api_key
            auth_param.extra = extra
        else:
            raise ValueError(f"Unsupported auth type '{auth_type}'.")
        return auth_param

    def _lookup_dashboard_config(self, configs: dict, platform: str) -> Optional[DashboardConfig]:
        if platform in configs:
            return configs[platform]
        key = (platform or "").strip().lower()
        if key in configs:
            return configs[key]
        for name, config in configs.items():
            if (name or "").strip().lower() == key:
                return config
        return None

    def _prompt_auth_params(self, platform: str, auth_type: AuthType) -> AuthParam:
        auth_param = AuthParam()
        if auth_type == AuthType.LOGIN:
            auth_param.username = self._prompt_input(f"{platform.capitalize()} username")
            if not auth_param.username:
                raise ValueError("Username is required.")

            auth_param.password = self._prompt_password(f"{platform.capitalize()} password")
            if not auth_param.password:
                raise ValueError("Password is required.")
        elif auth_type == AuthType.API_KEY:
            auth_param.api_key = self._prompt_password(f"{platform.capitalize()} API key")
            if not auth_param.api_key:
                raise ValueError("API key is required.")
        else:
            raise ValueError(f"Unsupported auth type '{auth_type}'.")
        return auth_param

    def _confirm_dashboard(
        self, adaptor: BIAdaptorBase, dashboard_url: str
    ) -> tuple[Optional[DashboardInfo], Optional[Union[int, str]]]:
        while True:
            dashboard_id = adaptor.parse_dashboard_id(dashboard_url)
            try:
                with self.console.status("Loading dashboard..."):
                    dashboard = adaptor.get_dashboard_info(dashboard_id)
            except Exception as exc:
                # Sanitize URL to avoid leaking sensitive query parameters in logs
                parsed = urlparse(dashboard_url)
                safe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                logger.error(f"Failed to load dashboard from {safe_url}", exc_info=True)
                self.console.print(f"[bold red]Failed to load dashboard:[/] {exc}")
                dashboard = None

            if dashboard:
                self.console.print("\n[bold]Dashboard[/]")
                self.console.print(f"ID: {dashboard.id}")
                self.console.print(f"Name: {dashboard.name}")
                if dashboard.description:
                    self.console.print(f"Description: {dashboard.description}")
                confirm = self._prompt_input("Use this dashboard?", default="y", choices=["y", "n"])
                if confirm == "y":
                    return dashboard, dashboard_id
            else:
                retry = self._prompt_input("Enter another dashboard URL?", default="y", choices=["y", "n"])
                if retry == "y":
                    dashboard_url = self._prompt_input("Dashboard URL")
                    if not dashboard_url:
                        return None, None
                    continue
            return None, None

    def _create_adaptor(self, options: DashboardCliOptions) -> BIAdaptorBase:
        adaptor_cls = self._adaptor_registry.get(options.platform)
        if adaptor_cls is None:
            raise ValueError(f"Unsupported platform '{options.platform}'")
        return adaptor_cls(
            api_base_url=options.api_base_url, auth_params=options.auth_params, dialect=self.agent_config.db_type
        )

    def _derive_api_base(self, dashboard_url: str) -> str:
        parsed = urlparse(dashboard_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return ""

    def _resolve_default_table_context(self) -> tuple[str, str, str]:
        catalog = ""
        database = ""
        schema = ""

        cli_context = getattr(self.cli, "cli_context", None) if self.cli else None
        if cli_context:
            catalog = (cli_context.current_catalog or "").strip()
            database = (cli_context.current_db_name or "").strip()
            schema = (cli_context.current_schema or "").strip()

        if not (catalog and database and schema):
            try:
                db_config = self.agent_config.current_db_config(self.agent_config.current_database)
            except Exception:
                db_config = None
            if db_config:
                if not catalog:
                    catalog = db_config.catalog or ""
                if not database:
                    database = db_config.database or ""
                if not schema:
                    schema = db_config.schema or ""

        return catalog, database, schema

    def _prompt_password(self, label: str) -> str:
        try:
            from prompt_toolkit import prompt
            from prompt_toolkit.formatted_text import HTML
            from prompt_toolkit.history import InMemoryHistory

            prompt_text = f"{label}: "
            return prompt(
                HTML(f"<ansigreen><b>{prompt_text}</b></ansigreen>"),
                is_password=True,
                history=InMemoryHistory(),
            ).strip()
        except Exception:
            return getpass(f"{label}: ").strip()

    def _prompt_input(
        self,
        message: str,
        default: str = "",
        choices: list | None = None,
        multiline: bool = False,
    ) -> str:
        return prompt_input(
            self.console,
            message,
            default=default,
            choices=choices,
            multiline=multiline,
            allow_interrupt=True,
        )

    def _hydrate_charts(
        self,
        adaptor: BIAdaptorBase,
        dashboard_id: Union[int, str],
        chart_metas: Sequence[ChartInfo],
    ) -> List[ChartInfo]:
        charts: List[ChartInfo] = []
        total = len(chart_metas)
        with self.console.status("Loading chart details...") as status:
            for idx, chart_meta in enumerate(chart_metas, start=1):
                status.update(f"Loading chart {idx}/{total}...")
                try:
                    chart_detail = adaptor.get_chart(chart_meta.id, dashboard_id)
                except Exception as exc:
                    logger.warning(f"Failed to load chart {chart_meta.id}: {exc}")
                    self.console.print(f"[yellow]Failed to load chart {chart_meta.id}:[/] {exc}")
                    chart_detail = None
                charts.append(chart_detail or chart_meta)
        return charts

    def _load_chart_selections(
        self,
        charts: Sequence[ChartInfo],
        indices: Sequence[int],
        purpose: str = "reference SQL",
    ) -> List[ChartSelection]:
        selections: List[ChartSelection] = []
        if not indices:
            return selections

        for idx in indices:
            chart = charts[idx]
            sqls = chart.query.sql or [] if chart.query else []
            sql_indices = list(range(len(sqls)))
            selections.append(ChartSelection(chart=chart, sql_indices=sql_indices))
        return selections

    def _select_datasets(self, datasets: Sequence[DatasetInfo]) -> List[DatasetInfo]:
        if not datasets:
            self.console.print("[yellow]No datasets reported for this dashboard.[/]")
            return []

        table = Table(title="Datasets")
        table.add_column("Index", style="cyan", width=4)
        table.add_column("Dataset ID", style="green")
        table.add_column("Name", style="white")
        table.add_column("Dialect", style="magenta")
        table.add_column("Tables", style="blue", justify="right")
        table.add_column("Metrics", style="blue", justify="right")
        table.add_column("Dimensions", style="blue", justify="right")

        for idx, dataset in enumerate(datasets, start=1):
            table.add_row(
                str(idx),
                str(dataset.id),
                dataset.name or "",
                dataset.dialect or "",
                str(len(dataset.tables or [])),
                str(len(dataset.metrics or [])),
                str(len(dataset.dimensions or [])),
            )

        self.console.print(table)
        selection_input = self._prompt_input("Select datasets (e.g. 1,2,... or all)", default="all")
        indices = self._parse_selection(selection_input, len(datasets))
        return [datasets[idx] for idx in indices]

    def _review_tables(self, tables: Sequence) -> List:
        if not tables:
            return []

        table_view = Table(title="Tables")
        table_view.add_column("#", style="cyan", width=4)
        table_view.add_column("Identifier", style="white")
        for idx, table in enumerate(tables, start=1):
            table_view.add_row(str(idx), str(table or ""))
        self.console.print(table_view)
        selection_input = self._prompt_input(
            "Select table index for scoped context(e.g. 1,2,... or all)", default="all"
        )
        indices = self._parse_selection(selection_input, len(tables))
        return [tables[idx] for idx in indices]

    def _save_sub_agent(
        self,
        platform: str,
        dashboard: DashboardInfo,
        result: DashboardAssemblyResult,
    ) -> None:
        if not getattr(self.agent_config, "current_namespace", ""):
            self.console.print("[yellow]No namespace set. Skipping sub-agent save.[/]")
            return

        sub_agent_name = self._build_sub_agent_name(platform, dashboard.name or "")
        if sub_agent_name in SYS_SUB_AGENTS:
            self.console.print(f"[bold red]Error:[/] '{sub_agent_name}' is reserved for built-in sub-agents.")
            return
        table_names = self._dedupe_values([table for table in result.tables if table])
        table_names = self._qualify_table_names(table_names)
        # Generate metadata first (before semantic model)
        self.console.log("[bold cyan]Start building metadata[/]")
        self._gen_metadata(table_names)
        self.console.log("[bold cyan]Building metadata completed[/]")

        self.console.log("[bold cyan]Start building reference SQL[/]")

        ref_sqls = self._gen_reference_sqls(result.reference_sqls, platform, dashboard)

        # Generate semantic model (before metrics)
        self.console.log("[bold cyan]Start building semantic model[/]")
        semantic_model_success = self._gen_semantic_model(result.metric_sqls, platform, dashboard)

        # Generate metrics (after semantic model, skip if semantic model failed)
        metrics = []
        if semantic_model_success:
            self.console.log("[bold cyan]Building semantic model completed[/]")
            self.console.log("[bold cyan]Start building metrics[/]")
            metrics = self._gen_metrics(result.metric_sqls, platform, dashboard)
            if metrics:
                self.console.log("[bold cyan]Building metrics completed[/]")
            else:
                self.console.log("[yellow]Metrics generation failed or no metrics generated[/]")
        else:
            self.console.log("[yellow]Skipping metrics generation due to semantic model failure[/]")

        scoped_context: Optional[ScopedContext] = None
        if table_names or ref_sqls or metrics:
            scoped_context = ScopedContext(
                tables=",".join(table_names) if table_names else None,
                sqls=",".join(ref_sqls) if ref_sqls else None,
                metrics=",".join(metrics) if metrics else None,
            )

        if scoped_context is None:
            self.console.log("[yellow]No scoped context derived. Skipping sub-agent save.[/]")
            return

        description = dashboard.description or dashboard.name or ""

        manager = SubAgentManager(
            configuration_manager=self._configuration_manager or configuration_manager(),
            namespace=self.agent_config.current_namespace,
            agent_config=self.agent_config,
        )
        self._do_save_sub_agent(
            sub_agent_manager=manager,
            sub_agent=SubAgentConfig(
                system_prompt=sub_agent_name,
                agent_description=description,
                tools="context_search_tools,db_tools.search_table,db_tools.describe_table,db_tools.read_query",
                scoped_context=scoped_context,
            ),
        )

        # Create attribution subagent (gen_report type) for metric attribution analysis

        self._do_save_sub_agent(
            sub_agent_manager=manager,
            sub_agent=SubAgentConfig(
                system_prompt=f"{sub_agent_name}_attribution",
                agent_description=f"Attribution analysis for {description}",
                node_class="gen_report",
                tools="semantic_tools,context_search_tools.list_subject_tree",
                scoped_context=scoped_context,
            ),
        )

        self._refresh_agent_config(manager)

    def _do_save_sub_agent(self, sub_agent_manager: SubAgentManager, sub_agent: SubAgentConfig, prefix: str = ""):
        sub_agent_name = sub_agent.system_prompt
        log_prefix = "" if not prefix else f"{prefix} "

        log_prefix = f"{log_prefix}Sub-Agent `{sub_agent_name}`"

        try:
            sub_agent_manager.save_agent(sub_agent, previous_name=sub_agent_name)
            self.console.log(f"[bold green]{log_prefix} saved.")
        except Exception as exc:
            self.console.log(f"[bold yellow] {log_prefix} persist failed:[/] {exc}")
        else:
            result = sub_agent_manager.bootstrap_agent(
                sub_agent, components=["metadata", "semantic_model", "metrics", "reference_sql"]
            )
            if not result.should_bootstrap:
                self.console.log(f"[bold yellow]{log_prefix} bootstrap skipped: {result.reason}[/]")
                return

            errors = [r for r in result.results if r.status == "error"]
            if errors:
                for err in errors:
                    self.console.log(f"[bold red]{log_prefix} bootstrap [{err.component}] error: {err.message}[/]")
            else:
                self.console.log(f"[bold green]{log_prefix} bootstrapped.")

    def _refresh_agent_config(self, manager: SubAgentManager) -> None:
        try:
            agents = manager.list_agents()
        except Exception:
            return

        try:
            self.agent_config.agentic_nodes = agents
        except Exception:
            pass

        if self.cli and self.cli.available_subagents:
            try:
                self.cli.available_subagents.update(name for name in agents.keys() if name != "chat")
            except Exception:
                pass

    def _build_sub_agent_name(self, platform: str, dashboard_name: str) -> str:
        platform_token = self._normalize_identifier(platform, fallback="bi")
        dashboard_token = self._normalize_identifier(dashboard_name, max_words=3, fallback="dashboard")
        name = f"{platform_token}_{dashboard_token}".strip("_")
        if not name or not name[0].isalpha():
            name = f"dashboard_{name}" if name else "dashboard_agent"
        return name

    def _normalize_identifier(self, text: str, max_words: Optional[int] = None, fallback: str = "item") -> str:
        """Normalize a free-form label into a filesystem/identifier-friendly token.

        Notes:
        - Keeps ASCII alphanumerics and CJK (Chinese) characters.
        - Collapses runs of non-token characters into separators.
        - Lower-cases ASCII only; CJK characters are preserved.
        """

        raw = (text or "").strip()
        if not raw:
            return fallback

        # Match either ASCII alphanumerics or a run of CJK Unified Ideographs.
        # This makes identifiers derived from Chinese names stable and readable.
        pattern = r"[A-Za-z0-9]+|[\u4E00-\u9FFF]+"
        tokens = re.findall(pattern, raw)

        if max_words is not None and len(tokens) > max_words:
            tokens = tokens[:max_words]

        if not tokens:
            return fallback

        normalized: List[str] = []
        for tok in tokens:
            # Lower-case ASCII tokens; keep CJK as-is.
            normalized.append(tok.lower() if tok.isascii() else tok)

        # Join with underscore and remove accidental leading/trailing underscores.
        out = "_".join(part for part in normalized if part)
        out = re.sub(r"_+", "_", out).strip("_")
        return out or fallback

    def _dedupe_values(self, values: Sequence[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            cleaned = (value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    def _qualify_table_names(self, table_names: List[str]) -> List[str]:
        """Qualify under-qualified table names using the current database context.

        Uses ``parse_table_name_parts`` to split each name according to the
        dialect's field hierarchy, fills in missing qualifier fields (catalog,
        database, schema) from the live connection, then rebuilds via
        ``metadata_identifier`` so the bootstrap's right-alignment matches.
        """
        catalog, database, schema = self.current_database_context()
        dialect = self.agent_config.db_type or ""
        qualified: List[str] = []
        for name in table_names:
            if not (name or "").strip():
                continue
            parts = parse_table_name_parts(name, dialect)
            if not parts.get("catalog_name") and catalog:
                parts["catalog_name"] = catalog
            if not parts.get("database_name") and database:
                parts["database_name"] = database
            if not parts.get("schema_name") and schema:
                parts["schema_name"] = schema
            qualified.append(
                metadata_identifier(
                    catalog_name=parts.get("catalog_name", ""),
                    database_name=parts.get("database_name", ""),
                    schema_name=parts.get("schema_name", ""),
                    table_name=parts.get("table_name", ""),
                    dialect=dialect,
                )
            )
        return qualified

    def _gen_reference_sqls(
        self, reference_sqls: List[SelectedSqlCandidate], platform: str, dashboard: DashboardInfo
    ) -> List[str]:
        sql_dir = self._write_chart_sql_files(reference_sqls, platform, dashboard)
        output_mgr = StreamOutputManager(
            console=self.console,
            max_message_lines=10,
            show_progress=True,
            title="Reference SQL Initialization",
        )

        # Create stream handler
        stream_handler = ReferenceSqlStreamHandler(output_mgr)
        # Derive subject_tree constraint from dashboard info
        subject_tree_hint = (
            f"{platform}/{self._normalize_identifier(dashboard.name or '', max_words=3, fallback='dashboard')}"
        )
        extra_instructions = (
            f"IMPORTANT: All SQL summaries from this batch MUST use the SAME subject_tree classification. "
            f'Suggested subject_tree: "{subject_tree_hint}". '
            f"You may adjust the classification based on SQL content, but ensure consistency across all items."
        )

        result = init_reference_sql(
            storage=ReferenceSqlRAG(self.agent_config),
            global_config=self.agent_config,
            build_mode="incremental",
            sql_dir=str(sql_dir),
            subject_tree=None,
            emit=stream_handler.handle_event,
            extra_instructions=extra_instructions,
        )
        output_mgr.stop()

        # Render markdown summary for the last 1 processed items
        output_mgr.render_markdown_summary(title="Reference SQL Summary", last_n=1)

        # Print statistics
        valid_entries = result.get("valid_entries", 0)
        invalid_entries = result.get("invalid_entries", 0)
        processed_entries = result.get("processed_entries", 0)
        if invalid_entries > 0:
            self.console.print(f"  [yellow]Warning: {invalid_entries} invalid SQL items skipped[/]")
        if valid_entries > processed_entries:
            skipped = valid_entries - processed_entries
            self.console.print(f"  [dim]({skipped} items already existed, skipped in incremental mode)[/]")

        ref_sqls = []
        if result.get("status") != "success":
            self.console.log(f"[bold red]Processed reference SQL failed: {result.get('error')}[/]")
        else:
            self.console.log("[bold cyan]Processed reference SQL succeeded.[/]")
            subject_trees = set()
            for item in result.get("processed_items", []):
                subject_tree = item.get("subject_tree")
                if subject_tree:
                    parts = [p.strip() for p in subject_tree.split("/") if p.strip()]
                    name = (item.get("name") or "").strip()
                    if name:
                        parts.append(quote_path_segment(name))
                    if parts:
                        subject_trees.add(".".join(parts))
            ref_sqls.extend(subject_trees)
        return ref_sqls

    def _ensure_file_name(self, platform: str, dashboard: DashboardInfo, suffix: str = ".sql") -> Path:
        sql_root = get_path_manager(self.agent_config.home).dashboard_path() / platform
        file_name = self._build_sql_file_name(platform, dashboard)
        sql_root.mkdir(parents=True, exist_ok=True)
        return sql_root / f"{file_name}{suffix}"

    def _write_chart_sql_files(
        self,
        reference_sqls: Sequence[SelectedSqlCandidate],
        platform: str,
        dashboard: DashboardInfo,
    ) -> Optional[Path]:
        if not reference_sqls:
            return None

        target_file = self._ensure_file_name(platform, dashboard)

        grouped: dict[str, List[SelectedSqlCandidate]] = {}
        for item in reference_sqls:
            key = str(item.chart_id)
            grouped.setdefault(key, []).append(item)

        with open(target_file, "w", encoding="utf-8") as target_f:
            for _, items in grouped.items():
                lines: List[str] = []
                for sql_item in items:
                    comment_lines = self._build_sql_comment_lines(sql_item, dashboard)
                    lines.extend(comment_lines)
                    sql_text = (sql_item.sql or "").strip()
                    if sql_text:
                        if not sql_text.endswith(";"):
                            sql_text = f"{sql_text};"
                        lines.append(sql_text)
                        # Split SQL
                        lines.append("")

                if lines:
                    target_f.write("\n".join(lines))
        return target_file

    def _build_sql_file_name(self, platform: str, dashboard: DashboardInfo) -> str:
        from datetime import datetime

        platform_token = self._normalize_identifier(platform, fallback="bi")
        dashboard_token = self._normalize_identifier(dashboard.name or "", max_words=3, fallback="dashboard")
        parts = [part for part in (platform_token, dashboard_token) if part] + [
            str(datetime.now().strftime("%Y%m%d%H%M"))
        ]
        return "_".join(parts)

    def _build_sql_comment_lines(
        self,
        sql_item: SelectedSqlCandidate,
        dashboard: DashboardInfo,
    ) -> List[str]:
        lines = [
            f"-- Dashboard={self._clean_comment_text(dashboard.name or '')};",
            f"-- Chart={self._clean_comment_text(sql_item.chart_name or str(sql_item.chart_id))};",
        ]
        if sql_item.description:
            lines.append(f"-- Description={self._clean_comment_text(sql_item.description)};")
        return lines

    def _clean_comment_text(self, text: str) -> str:
        return " ".join(str(text).split())

    def _split_subject_tree(self, subject_tree: str) -> tuple[str, str, str]:
        parts = [part.strip() for part in (subject_tree or "").split("/") if part.strip()]
        domain = parts[0] if len(parts) > 0 else ""
        layer1 = parts[1] if len(parts) > 1 else ""
        layer2 = parts[2] if len(parts) > 2 else ""
        return domain, layer1, layer2

    def _gen_metrics(self, sqls: List[SelectedSqlCandidate], platform: str, dashboard: DashboardInfo) -> List[str]:
        if not sqls:
            return []
        target_file = self._ensure_file_name(platform, dashboard, suffix=".csv")
        file_data = []
        for sql_item in sqls:
            question = (
                f"Dashboard={self._clean_comment_text(dashboard.name or '')};"
                f"Chart={self._clean_comment_text(sql_item.chart_name or str(sql_item.chart_id))};"
            )
            if sql_item.description:
                question += f"Description={self._clean_comment_text(sql_item.description)};"
            file_data.append({"question": question, "sql": sql_item.sql})

        with open(target_file, "w", encoding="utf-8") as target_f:
            pd.DataFrame(file_data, columns=["question", "sql"]).to_csv(target_f, index=False)

        # Derive subject_tree constraint from dashboard info
        subject_tree_hint = (
            f"{platform}/{self._normalize_identifier(dashboard.name or '', max_words=3, fallback='dashboard')}"
        )
        extra_instructions = (
            f"IMPORTANT: All metrics from this batch MUST use the SAME subject_tree classification. "
            f'Suggested subject_tree: "{subject_tree_hint}". '
            f"You may adjust the classification based on SQL content, but ensure consistency across all metrics."
        )

        successful, metrics_result = init_metrics(
            target_file,
            agent_config=self.agent_config,
            console=self.console,
            build_model="incremental",
            extra_instructions=extra_instructions,
        )

        if not successful:
            return None

        # Validate semantic model after metrics generation
        if not self._validate_semantic_model():
            self.console.log("[yellow]Metrics validation failed[/]")
            return None

        metrics = set()
        if files := metrics_result.get("semantic_models", []):
            # Get base directory for semantic models
            base_dir = get_path_manager(self.agent_config.home).semantic_model_path(self.agent_config.current_namespace)
            for file in files:
                # Convert relative path to absolute path if needed
                file_path = file if Path(file).is_absolute() else base_dir / file
                with open(file_path, "r", encoding="utf-8") as f:
                    # multi documents
                    for metrics_meta in yaml.safe_load_all(f):
                        meta = metrics_meta.get("metric")
                        if not meta:
                            continue
                        name = meta.get("name")
                        subject_tree = _parse_subject_path_for_metrics(meta.get("locked_metadata", {}).get("tags", []))
                        if name and subject_tree:
                            metrics.add(f"{subject_tree}.{quote_path_segment(name)}")

        return list(metrics)

    def _gen_semantic_model(self, sqls: List[SelectedSqlCandidate], platform: str, dashboard: DashboardInfo) -> bool:
        """Generate semantic model from SQL queries.

        Args:
            sqls: List of SQL candidates from charts
            platform: BI platform name
            dashboard: Dashboard info

        Returns:
            True if successful, False otherwise
        """
        if not sqls:
            self.console.log("[yellow]No SQL queries for semantic model generation[/]")
            return False

        # Reuse the same CSV file as metrics (or create one if not exists)
        target_file = self._ensure_file_name(platform, dashboard, suffix=".csv")

        # Check if file exists, if not create it
        if not target_file.exists():
            file_data = []
            for sql_item in sqls:
                question = (
                    f"Dashboard={self._clean_comment_text(dashboard.name or '')};"
                    f"Chart={self._clean_comment_text(sql_item.chart_name or str(sql_item.chart_id))};"
                )
                if sql_item.description:
                    question += f"Description={self._clean_comment_text(sql_item.description)};"
                file_data.append({"question": question, "sql": sql_item.sql})

            with open(target_file, "w", encoding="utf-8") as target_f:
                pd.DataFrame(file_data, columns=["question", "sql"]).to_csv(target_f, index=False)

        successful, result = init_semantic_model(
            target_file,
            agent_config=self.agent_config,
            console=self.console,
            build_mode="incremental",
            force=self._force,
        )

        if successful:
            # Validate semantic model after generation
            validation_success = self._validate_semantic_model()
            if validation_success:
                count = result.get("semantic_model_count", 0) if result else 0
                self.console.log(f"[green]Semantic model generated successfully (count={count})[/]")
            else:
                self.console.log("[yellow]Semantic model validation failed[/]")
                return False
        else:
            self.console.log("[yellow]Semantic model generation failed or skipped[/]")

        return successful

    def _gen_metadata(self, table_names: List[str]) -> bool:
        """Generate metadata for tables used in the dashboard.

        Args:
            table_names: List of table names to build metadata for

        Returns:
            True if successful, False otherwise
        """
        if not table_names:
            self.console.log("[yellow]No tables for metadata generation[/]")
            return False

        try:
            # Get db_manager from cli or create new instance
            if self.cli and hasattr(self.cli, "db_manager"):
                db_manager = self.cli.db_manager
            else:
                db_manager = db_manager_instance(self.agent_config.namespaces)

            # Create metadata store
            metadata_store = SchemaWithValueRAG(self.agent_config)

            # Build metadata using incremental mode
            init_local_schema(
                metadata_store,
                self.agent_config,
                db_manager,
                build_mode="incremental",
                table_type="full",
            )

            self.console.log(f"[green]Metadata generated for {len(table_names)} tables[/]")
            return True

        except Exception as exc:
            logger.error(f"Metadata generation failed for tables: {table_names}", exc_info=True)
            self.console.log(f"[yellow]Metadata generation failed: {exc}[/]")
            return False

    def _validate_semantic_model(self) -> bool:
        """Validate semantic model configuration using SemanticTools.

        Returns:
            True if validation passed, False otherwise
        """
        try:
            # Get adapter_type from agentic_nodes config, default to metricflow
            adapter_type = "metricflow"
            agentic_nodes = getattr(self.agent_config, "agentic_nodes", None) or {}
            node_config = agentic_nodes.get("gen_semantic_model", {})
            if isinstance(node_config, dict) and node_config.get("semantic_adapter"):
                adapter_type = node_config.get("semantic_adapter")

            semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                adapter_type=adapter_type,
            )

            # Check if adapter is available
            if not semantic_tools.adapter:
                self.console.log(
                    "[red]Semantic adapter not available. " "Install with: pip install datus-semantic-metricflow[/]"
                )
                return False

            result = semantic_tools.validate_semantic()
            if not result.success:
                error_msg = result.error or "Semantic validation failed"
                self.console.log(f"[red]Validation error: {error_msg}[/]")
                return False

            return True

        except Exception as exc:
            logger.warning(f"Semantic model validation check failed: {exc}")
            self.console.log(f"[yellow]Validation check failed: {exc}[/]")
            return False

    def _render_summary(self, result: DashboardAssemblyResult) -> None:
        summary = Table(title="Dashboard Assembly Summary")
        summary.add_column("Item", style="cyan")
        summary.add_column("Count", style="green", justify="right")
        summary.add_row("Charts", str(len(result.charts)))
        summary.add_row("Datasets", str(len(result.datasets)))
        summary.add_row("Reference SQL", str(len(result.reference_sqls)))
        summary.add_row("Metrics", str(len(result.metric_sqls)))
        summary.add_row("Tables", str(len(result.tables)))
        self.console.print(summary)

    def _parse_selection(self, raw: str, max_index: int) -> List[int]:
        if max_index <= 0:
            return []
        text = (raw or "").strip().lower()
        if not text or text in ("all", "*"):
            return list(range(max_index))
        if text in ("none", "n", "no"):
            return []

        selections: List[int] = []
        for token in re.split(r"[,\s]+", text):
            if not token:
                continue
            if "-" in token:
                start, end = token.split("-", 1)
                if start.isdigit() and end.isdigit():
                    for idx in range(int(start), int(end) + 1):
                        if 1 <= idx <= max_index:
                            selections.append(idx - 1)
                continue
            if token.isdigit():
                idx = int(token)
                if 1 <= idx <= max_index:
                    selections.append(idx - 1)

        seen = set()
        deduped: List[int] = []
        for idx in selections:
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(idx)
        return deduped

    def _render_chart_table(self, charts: Sequence[ChartInfo], title: str) -> None:
        table = Table(title=title, show_lines=True)
        table.add_column("Index", style="cyan", width=4)
        table.add_column("Chart ID", style="green")
        table.add_column("Name", style="white")
        table.add_column("Type", style="magenta")
        table.add_column("SQL", style="white")

        for idx, chart in enumerate(charts, start=1):
            table.add_row(
                str(idx),
                str(chart.id),
                chart.name or "",
                chart.chart_type or "",
                _sql_format(chart.query.sql if chart.query else []),
            )

        self.console.print(table)

    def _discover_adaptors(self) -> dict[str, type[BIAdaptorBase]]:
        return adaptor_registry.list_adaptors()

    def _hydrate_datasets(
        self,
        assembler: DashboardAssembler,
        datasets: Sequence[DatasetInfo],
        dashboard_id: Union[int, str],
    ) -> List[DatasetInfo]:
        with self.console.status("Loading dataset details..."):
            return assembler.hydrate_datasets(datasets, dashboard_id)


def _sql_format(sqls: Optional[List[str]]) -> Syntax | str:
    sqls = list(sqls or [])
    final_sqls = []
    for sql in sqls:
        if not sql.endswith(";"):
            final_sqls.append(sql + ";")
        else:
            final_sqls.append(sql)
    if final_sqls:
        return Syntax("\n".join(final_sqls), lexer="sql", word_wrap=True)
    return "-"
