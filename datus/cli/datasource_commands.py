# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/datasource`` slash command — interactive datasource management.

Entry point
-----------

  - ``/datasource``               — open the interactive picker.
  - ``/datasource <name>``        — switch directly (backward compatible).

The interactive path delegates to :class:`~datus.cli.datasource_app.DatasourceApp`,
a single prompt_toolkit Application that hosts datasource listing, action
sub-menus, database type selection, and an inline config form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from datus.cli._cli_utils import _run_sub_application
from datus.cli.cli_styles import print_error, print_info, print_success, print_warning
from datus.cli.datasource_app import DatasourceApp, DatasourceSelection
from datus.cli.datasource_manager import serialize_services_section
from datus.cli.init_util import detect_db_connectivity
from datus.configuration.agent_config import DbConfig
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class DatasourceCommands:
    """Handlers for the ``/datasource`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console

    # ── Entry point ──────────────────────────────────────────────

    def cmd(self, args: str) -> None:
        token = (args or "").strip()
        if not token:
            self._run_menu()
            return
        if token in (self.cli.agent_config.datasource_configs or {}):
            self._switch(token)
            return
        print_error(self.console, f"Unknown datasource '{token}'. Run /datasource to see available options.")

    # ── Menu loop ────────────────────────────────────────────────

    def _run_menu(self) -> None:
        while True:
            app = DatasourceApp(self.cli.agent_config, self.console)
            selection = self._run_app(app)
            if selection is None:
                return

            if selection.kind == "switch":
                self._switch(selection.name)
                return

            if selection.kind == "needs_install":
                if self._install_plugin(f"datus-{selection.db_type}"):
                    continue
                continue

            if selection.kind == "add_submit":
                self._handle_add_submit(selection)
                continue

            if selection.kind == "edit_submit":
                self._handle_edit_submit(selection)
                continue

            if selection.kind == "delete":
                self._run_delete(selection.name)
                continue

            if selection.kind == "set_default":
                self._set_default(selection.name)
                continue

            if selection.kind == "install":
                self._install_plugin(f"datus-{selection.db_type}")
                continue

            logger.debug("DatasourceApp returned unknown kind=%s", selection.kind)
            return

    def _run_app(self, app: DatasourceApp) -> Optional[DatasourceSelection]:
        return _run_sub_application(app)

    # ── Switch ───────────────────────────────────────────────────

    def _switch(self, name: str) -> None:
        if self.cli.agent_config.current_datasource == name:
            print_warning(self.console, f"Already on datasource '{name}'.")
            return
        try:
            db_name, connector = self.cli.db_manager.first_conn_with_name(name)
            self.cli.agent_config.current_datasource = name
            self.cli.db_connector = connector
            db_logic_name = db_name or name
            self.cli.cli_context.update_database_context(
                catalog=connector.catalog_name,
                db_name=connector.database_name,
                schema=connector.schema_name,
                db_logic_name=db_logic_name,
            )
            self.cli.reset_session()
            self.cli.chat_commands.update_chat_node_tools()
            self._persist_default_datasource(name)
            print_success(self.console, f"Datasource changed to: {name}")
        except Exception as e:
            print_error(self.console, f"Failed to switch datasource: {e}")

    def _persist_default_datasource(self, name: str) -> None:
        """Write ``default_datasource`` to ``./.datus/config.yml`` so the
        choice survives process restarts."""
        try:
            from datus.configuration.project_config import ProjectOverride, load_project_override, save_project_override

            project_root = str(getattr(self.cli.agent_config, "_project_root", None) or "")
            current = load_project_override(cwd=project_root) or ProjectOverride()
            current.default_datasource = name
            save_project_override(current, cwd=project_root)
        except Exception as e:
            logger.debug("Failed to persist default_datasource: %s", e)

    # ── Add submit ───────────────────────────────────────────────

    def _handle_add_submit(self, selection: DatasourceSelection) -> None:
        payload = selection.payload or {}
        ds_name = payload.pop("_name", "")
        if not ds_name:
            print_error(self.console, "Datasource name is missing.")
            return

        if not self._test_connectivity(ds_name, payload):
            return

        if not self.cli.agent_config.services.datasources:
            payload["default"] = True

        db_config = DbConfig.filter_kwargs(DbConfig, payload)
        db_config.logic_name = ds_name
        self.cli.agent_config.services.datasources[ds_name] = db_config
        if self._save():
            print_success(self.console, f"Datasource '{ds_name}' added successfully.", symbol=True)
            self._reload_runtime()

    # ── Edit submit ──────────────────────────────────────────────

    @staticmethod
    def _merge_password_fields(old_config: DbConfig, payload: Dict[str, Any]) -> None:
        """Carry over password-type fields from *old_config* when they are
        absent from *payload* (i.e. the user left them blank in the form)."""
        from datus.tools.db_tools import connector_registry

        old_dict = old_config.to_dict()
        if isinstance(getattr(old_config, "extra", None), dict):
            old_dict.update(old_config.extra)

        adapter_meta = connector_registry.list_available_adapters().get(payload.get("type", ""))
        if adapter_meta:
            for fn, fi in adapter_meta.get_config_fields().items():
                is_pw = fi.get("input_type") == "password" or fn == "password"
                if is_pw and fn not in payload and old_dict.get(fn):
                    payload[fn] = old_dict[fn]
        elif "password" not in payload and old_dict.get("password"):
            payload["password"] = old_dict["password"]

    def _handle_edit_submit(self, selection: DatasourceSelection) -> None:
        name = selection.name
        payload = selection.payload or {}
        old_config = self.cli.agent_config.services.datasources.get(name)
        if not old_config:
            print_error(self.console, f"Datasource '{name}' not found.")
            return

        self._merge_password_fields(old_config, payload)

        if not self._test_connectivity(name, payload):
            return

        payload["default"] = old_config.default
        new_config = DbConfig.filter_kwargs(DbConfig, payload)
        new_config.logic_name = old_config.logic_name or name
        new_config.default = old_config.default
        self.cli.agent_config.services.datasources[name] = new_config
        if self._save():
            print_success(self.console, f"Datasource '{name}' updated successfully.", symbol=True)
            self._reload_runtime()

    # ── Delete ───────────────────────────────────────────────────

    def _run_delete(self, name: str) -> None:
        if name not in self.cli.agent_config.services.datasources:
            print_error(self.console, f"Datasource '{name}' not found.")
            return

        del self.cli.agent_config.services.datasources[name]

        if self._save():
            print_success(self.console, f"Datasource '{name}' deleted.", symbol=True)

            if self.cli.agent_config.current_datasource == name:
                remaining = list(self.cli.agent_config.services.datasources.keys())
                if remaining:
                    default_ds = self.cli.agent_config.services.default_datasource or remaining[0]
                    self._switch(default_ds)
                else:
                    self.cli.agent_config.current_datasource = ""
                    print_warning(self.console, "No datasources remaining.")

            self._reload_runtime()

    # ── Set default ──────────────────────────────────────────────

    def _set_default(self, name: str) -> None:
        if name not in self.cli.agent_config.services.datasources:
            print_error(self.console, f"Datasource '{name}' not found.")
            return
        for cfg in self.cli.agent_config.services.datasources.values():
            cfg.default = False
        self.cli.agent_config.services.datasources[name].default = True
        if self._save():
            print_success(self.console, f"'{name}' set as default datasource.", symbol=True)

    # ── Install adapter ──────────────────────────────────────────

    def _install_plugin(self, package: str) -> bool:
        import shutil
        import subprocess
        import sys

        print_info(self.console, f"Installing {package}...")
        python = sys.executable
        uv_path = shutil.which("uv")
        if uv_path:
            cmd = [uv_path, "pip", "install", "--python", python, package]
        else:
            cmd = [python, "-m", "pip", "install", package]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print_error(self.console, f"Install failed: {result.stderr.strip()}")
                return False

            import importlib

            importlib.invalidate_caches()
            db_type = package.removeprefix("datus-").replace("-", "_")
            try:
                module = importlib.import_module(f"datus_{db_type}")
                if hasattr(module, "register"):
                    module.register()
                    logger.info("Loaded adapter after install: %s", db_type)
            except Exception as e:
                logger.warning("Adapter installed but failed to load in current session: %s", e)
            print_success(self.console, f"{package} installed successfully.", symbol=True)
            return True
        except subprocess.TimeoutExpired:
            print_error(self.console, "Install timed out.")
            return False
        except Exception as e:
            print_error(self.console, f"Install failed: {e}")
            return False

    # ── Connectivity test ────────────────────────────────────────

    def _test_connectivity(self, name: str, config_data: Dict[str, Any]) -> bool:
        test_data = {k: v for k, v in config_data.items() if k not in ("default", "_name")}
        print_info(self.console, "Testing database connectivity...")
        success, error_msg = detect_db_connectivity(name, test_data)
        if success:
            print_success(self.console, "Database connection test successful.", symbol=True)
            self.console.print()
            return True
        print_error(self.console, f"Connectivity test failed: {error_msg}")
        return False

    # ── Persistence ──────────────────────────────────────────────

    def _save(self) -> bool:
        try:
            from datus.configuration.agent_config_loader import configuration_manager

            mgr = configuration_manager()
            services_section = serialize_services_section(self.cli.agent_config.services)
            mgr.update(updates={"services": services_section}, delete_old_key=True)
            return True
        except Exception as e:
            print_error(self.console, f"Failed to save configuration: {e}")
            logger.error("Failed to save datasource config: %s", e)
            return False

    def _reload_runtime(self) -> None:
        try:
            from datus.tools.db_tools.db_manager import _cli_cache, db_manager_instance

            _cli_cache.clear()
            self.cli.db_manager = db_manager_instance(self.cli.agent_config.datasource_configs)
            self.cli._init_connection()
            self.cli.reset_session()
        except Exception as e:
            logger.debug("Runtime reload after datasource change: %s", e)
