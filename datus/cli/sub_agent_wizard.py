# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
This module provides an interactive wizard for adding a new sub-agent.
It uses prompt_toolkit to create a terminal-based user interface (TUI)
with side-by-side input and preview panes.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import yaml
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import WindowAlign
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Button, CheckboxList, Dialog, Frame, Label, TextArea
from pygments.lexers.data import YamlLexer
from pygments.lexers.html import HtmlLexer

from datus.agent.node.gen_sql_agentic_node import prepare_template_context
from datus.cli.autocomplete import TableCompleter
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.tools.mcp_tools import MCPTool
from datus.utils.constants import SYS_SUB_AGENTS, DBType
from datus.utils.loggings import get_logger
from datus.utils.reference_paths import normalize_reference_path

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.cli import DatusCLI


class LineCompleter(Completer):
    """
    A completer that wraps another completer and applies it to the current line only.
    """

    def __init__(self, completer: Completer):
        self.completer = completer

    def get_completions(self, document: Document, complete_event):
        text = document.current_line
        cursor_pos_in_line = len(document.current_line_before_cursor)
        line_document = Document(text, cursor_pos_in_line)
        yield from self.completer.get_completions(line_document, complete_event)


class SubAgentWizard:
    """
    An interactive wizard for creating a new agent configuration.
    """

    def __init__(self, cli_instance: "DatusCLI", data: Optional[Union[SubAgentConfig, Dict[str, Any]]] = None):
        self.cli_instance = cli_instance
        if not data:
            self.prompt_template_name = "sql_system"
            self.data = SubAgentConfig(system_prompt="", agent_description="", scoped_context=ScopedContext())
        else:
            if isinstance(data, SubAgentConfig):
                self.data = data
            else:
                self.data: SubAgentConfig = SubAgentConfig.model_validate(data)
            self.prompt_template_name = f"{self.data.system_prompt}_system"
        # Keep track of original name for edit-mode validation
        self._original_name: Optional[str] = self.data.system_prompt
        self._reserved_template_names = self._load_reserved_template_names()
        self.step = 0
        self.done = False
        self.error_dialog = None

        # Preview updates are suspended during initialization until edit-state is applied
        self._suspend_preview_updates = True
        self.table_completer: TableCompleter | None = None

        # For step 4: Rules list
        self.selected_rule_index = 0
        self.edit_mode = False
        self.editing_rule_buffer: Optional[Buffer] = None
        self.original_text_on_edit: Optional[str] = None
        self.is_new_rule = False
        self._current_rule_editor: Optional[TextArea] = None
        # Scoped context selections
        self.selected_tables: List[str] = []
        self.selected_metrics: List[str] = []
        self.selected_sqls: List[str] = []
        self.mcp_tools = MCPTool()

        # UI components
        self._init_components()
        self._init_key_bindings()
        self._init_layout()

        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True,
            mouse_support=True,
        )
        # Start lightweight watchers to reflect checkbox changes triggered by mouse
        # interactions too (not just keyboard). This keeps preview and headers in sync.
        self._native_selection_snapshot: Dict[int, frozenset] = {}
        self._mcp_selection_snapshot: Dict[int, frozenset] = {}
        self._start_selection_watchers()
        # Kick off async loading of MCP server tools if available
        try:
            self._load_mcp_tools_async()
        except Exception:
            pass

        # Apply edit-mode data to form inputs and selections (if provided)
        try:
            self._apply_initial_form_state()
        except Exception:
            # Non-fatal: continue even if prefill hits an issue
            self._suspend_preview_updates = False
            self._update_previews()

    # ------------------ Edit Mode Prefill Helpers ------------------
    def _load_reserved_template_names(self) -> Set[str]:
        """Collect template base names that would conflict with agent names."""
        try:
            templates = prompt_manager.list_templates()
        except Exception:
            return set()

        system_pattern = re.compile(r".*_system(?:_.+)?$")
        reserved = {name for name in templates if system_pattern.match(name)}

        if self._original_name:
            original_pattern = re.compile(rf"^{re.escape(self._original_name)}_system(?:_.+)?$")
            reserved = {name for name in reserved if not original_pattern.match(name)}

        return reserved

    def _split_comma_tokens(self, value: Any) -> List[str]:
        """Split comma-delimited values (recursively) into normalized tokens."""
        if value is None:
            return []
        if isinstance(value, list):
            tokens: List[str] = []
            for item in value:
                tokens.extend(self._split_comma_tokens(item))
            return [t for t in tokens if t]

        text = str(value).replace("\n", ",")
        return [part.strip() for part in text.split(",") if part.strip()]

    def _normalize_scoped_text(self, value: Any) -> str:
        """Convert scoped-context values into newline-delimited text."""
        if value is None:
            return ""
        if isinstance(value, list):
            tokens: List[str] = []
            for item in value:
                tokens.extend(self._split_comma_tokens(item))
        else:
            tokens = self._split_comma_tokens(value)
        return "\n".join(tokens)

    def _apply_initial_form_state(self):
        """Populate inputs and checkbox selections from self.data when editing."""
        # 1) Basic fields
        try:
            self.name_buffer.text = self.data.system_prompt or ""
        except Exception:
            self.name_buffer.text = ""
        try:
            self.description_area.text = self.data.agent_description or ""
        except Exception:
            self.description_area.text = ""

        # 2) Rules (preserve order)
        try:
            self.selected_rule_index = 0
            if isinstance(self.data.rules, list):
                self._update_rules_display()
        except Exception:
            pass

        # 3) Scoped context
        scoped_ctx = getattr(self.data, "scoped_context", None)
        try:
            tables_value = getattr(scoped_ctx, "tables", None)
            metrics_value = getattr(scoped_ctx, "metrics", None)
            sqls_value = getattr(scoped_ctx, "sqls", None)
            self.catalogs_area.text = self._normalize_scoped_text(tables_value)
            self.metrics_area.text = self._normalize_scoped_text(metrics_value)
            self.sqls_area.text = self._normalize_scoped_text(sqls_value)
        except Exception:
            self.catalogs_area.text = ""
            self.metrics_area.text = ""
            self.sqls_area.text = ""

        # 4) Native tool selections
        self._apply_initial_native_selection(self.data.tools)

        # 5) MCP selections
        self._apply_initial_mcp_selection(self.data.mcp)

        # 6) Refresh previews (now safe to update underlying model)
        self._suspend_preview_updates = False
        self._update_previews()

    def _apply_initial_native_selection(self, tools_value: Any):
        """Prefill native tool checkboxes based on 'tools' value.
        Accepts formats like:
          - "db_tools, context_search_tools" (select all in categories)
          - "db_tools.list_tables, context_search_tools.search_metrics"
          - legacy: ["list_tables", "search_metrics"]
        """
        try:
            tokens = self._split_comma_tokens(tools_value)
            if not tokens:
                return

            categories: Dict[str, Dict[str, Any]] = {}
            for entry in getattr(self, "native_category_entries", []):
                name = entry.get("name")
                cbl = entry.get("tools_cbl")
                if not name or not cbl:
                    continue
                all_values = [v for v, _ in getattr(cbl, "values", [])]
                categories[name] = {"cbl": cbl, "all_values": all_values}

            per_category: Dict[str, set] = {cat: set() for cat in categories}
            full_categories: set = set()

            for token in tokens:
                item = token.strip()
                if not item:
                    continue
                normalized = item.replace(":", ".")
                if "." in normalized:
                    cat, tool = normalized.split(".", 1)
                    cat, tool = cat.strip(), tool.strip()
                    if not cat:
                        continue
                    data = categories.get(cat)
                    if not data or not tool:
                        continue
                    available = data["all_values"]
                    if tool in available:
                        per_category.setdefault(cat, set()).add(tool)
                    else:
                        # Fallback: match suffix if config stored fully qualified names
                        for candidate in available:
                            if candidate.endswith(tool):
                                per_category.setdefault(cat, set()).add(candidate)
                else:
                    if item in categories:
                        full_categories.add(item)
                    else:
                        for cat, data in categories.items():
                            if item in data["all_values"]:
                                per_category.setdefault(cat, set()).add(item)

            for cat, data in categories.items():
                cbl: CheckboxList = data["cbl"]
                all_values = data["all_values"]
                if cat in full_categories:
                    cbl.current_values = list(all_values)
                else:
                    selections = [v for v in per_category.get(cat, set()) if v in all_values]
                    cbl.current_values = selections
        except Exception:
            pass

    def _apply_initial_mcp_selection(self, mcp_value: Any):
        """Prefill MCP selections based on 'mcp' value.
        Accepts formats like:
          - "server_a, server_b" (select all tools)
          - "server_a.tool1, server_b.tool2" (partial)
          - legacy list items "server:tool"
        """
        try:
            tokens = self._split_comma_tokens(mcp_value)
            if not tokens:
                return

            selections: Dict[str, Dict[str, Any]] = {}
            for token in tokens:
                item = token.strip()
                if not item:
                    continue
                server = item
                tool = ""
                normalized = item.replace(":", ".")
                if "." in normalized:
                    server, tool = normalized.split(".", 1)
                    server, tool = server.strip(), tool.strip()
                else:
                    server = server.strip()

                if not server:
                    continue

                state = selections.setdefault(server, {"all": False, "tools": set()})
                if tool:
                    state["tools"].add(tool)
                else:
                    state["all"] = True

            for entry in getattr(self, "mcp_server_entries", []):
                name = entry.get("name")
                if not name or name not in selections:
                    continue
                state = selections[name]
                if state.get("all"):
                    entry["preselect_all"] = True
                if state.get("tools"):
                    entry["preselect_specific"] = set(state["tools"])  # type: ignore

                tools_cbl = entry.get("tools_cbl")
                if tools_cbl and getattr(tools_cbl, "values", None):
                    if entry.get("preselect_all"):
                        tools_cbl.current_values = [v for v, _ in tools_cbl.values]
                    else:
                        desired = set(entry.get("preselect_specific", set()))
                        picks = []
                        for value, _ in tools_cbl.values:
                            if ":" not in value:
                                continue
                            srv, tool_name = value.split(":", 1)
                            if srv == name and tool_name in desired:
                                picks.append(value)
                        tools_cbl.current_values = picks
                try:
                    self._update_single_mcp_header(entry)
                except Exception:
                    pass
        except Exception:
            pass

    def _start_editing(self, is_new: bool = False):
        if self.edit_mode:
            return

        self.is_new_rule = is_new
        if is_new:
            new_rule = ""
            self.data.rules.append(new_rule)
            self.selected_rule_index = len(self.data.rules) - 1
            self.original_text_on_edit = new_rule
        else:
            if not (0 <= self.selected_rule_index < len(self.data.rules)):
                return
            self.original_text_on_edit = self.data.rules[self.selected_rule_index]

        self.edit_mode = True
        self.editing_rule_buffer = Buffer(
            document=Document(self.original_text_on_edit, len(self.original_text_on_edit))
        )
        self._update_rules_display()
        self._focus_rules_list()

    def _finish_editing(self):
        if not self.edit_mode:
            return

        new_text = self.editing_rule_buffer.text.strip()
        if new_text:
            self.data.rules[self.selected_rule_index] = new_text
        else:
            # If the rule is empty (new or edited), remove it
            self.data.rules.pop(self.selected_rule_index)
            if self.selected_rule_index >= len(self.data.rules):
                self.selected_rule_index = max(0, len(self.data.rules) - 1)

        self.edit_mode = False
        self.editing_rule_buffer = None
        self.original_text_on_edit = None
        self.is_new_rule = False
        self._update_rules_display()
        self._update_previews()
        self._focus_rules_list()

    def _cancel_editing(self):
        if not self.edit_mode:
            return

        if self.is_new_rule:
            self.data.rules.pop(self.selected_rule_index)
            if self.selected_rule_index >= len(self.data.rules):
                self.selected_rule_index = max(0, len(self.data.rules) - 1)
        # No need to restore text if it wasn't a new rule, as it was never changed in the main list

        self.edit_mode = False
        self.editing_rule_buffer = None
        self.original_text_on_edit = None
        self.is_new_rule = False
        self._update_rules_display()
        self._focus_rules_list()

    def _update_rules_display(self):
        """Dynamically update the widgets in the rules container."""
        children = []
        # Reset current rule editor reference; it will be set when building the editing widget
        self._current_rule_editor = None

        for i, rule_text in enumerate(self.data.rules):
            is_selected = (i == self.selected_rule_index) and not self.edit_mode
            is_editing = i == self.selected_rule_index and self.edit_mode

            prefix_text = f"[{i + 1}] "

            if is_editing:
                rule_control = BufferControl(buffer=self.editing_rule_buffer, focusable=True)
                rule_widget = Window(
                    content=rule_control,
                    height=1,
                    dont_extend_height=True,
                    style="class:rule.editing",
                )
                # Keep a reference so we can focus it reliably
                self._current_rule_editor = rule_widget
                prefix_window = Window(
                    content=FormattedTextControl(prefix_text),
                    style="class:label",
                    dont_extend_width=True,
                )
                container = VSplit([prefix_window, rule_widget])
                children.append(container)
            else:
                style = "class:rule.selected" if is_selected else "class:rule"
                display_text = [(style, f"{prefix_text}{rule_text}")]
                rule_widget = Window(
                    FormattedTextControl(display_text, focusable=True),
                    height=1,
                    dont_extend_height=True,
                )
                children.append(rule_widget)

        self.rules_container.children = children
        if not self.data.rules and not self.edit_mode:
            # Show an informational line when there are no rules.
            # Use a Window with FormattedTextControl (Label is a Container and cannot be used as Window content).
            self.rules_container.children = [
                Window(
                    content=FormattedTextControl("(No rules defined yet. Press 'a' to add)"),
                    style="class:label",
                    height=1,
                    dont_extend_height=True,
                )
            ]

        if hasattr(self, "app") and self.app:
            self.app.invalidate()

    def _focus_rules_list(self):
        """Focus the appropriate widget for the rules area.
        - If editing, focus the TextArea editor.
        - Else, focus the selected rule line (make sure it's focusable).
        - If list is empty, do not attempt to focus the container.
        """
        if not hasattr(self, "app") or not self.app:
            return
        if self.edit_mode and self._current_rule_editor is not None:
            self.app.layout.focus(self._current_rule_editor)
            return
        # Non-edit mode: focus the selected line if present
        children = getattr(self.rules_container, "children", [])
        if children:
            idx = min(self.selected_rule_index, len(children) - 1)
            try:
                self.app.layout.focus(children[idx])
            except Exception:
                # If child is somehow not focusable, ignore
                pass

    def _init_components(self):
        self._init_step1_basic()
        self._init_step2_tools_mcp()
        self._init_step3_scoped_context()
        self._init_step4_rules()
        self._init_previews_and_picker_state()

    def _init_step1_basic(self):
        self.name_buffer = Buffer()
        self.name_buffer.on_text_changed += self._update_previews
        self.description_area = TextArea(text="", multiline=True, wrap_lines=True, style="class:textarea")
        self.description_area.buffer.on_text_changed += self._update_previews

    def _init_step2_tools_mcp(self):
        # Native Tools
        self.tool_checkbox_lists = []
        self.category_buttons = []
        self.native_category_entries: List[Dict[str, Any]] = []
        tool_widgets = []
        for category, tools in sorted(self.native_tools_choices().items()):
            if not tools:
                continue
            cbl = CheckboxList(values=[(tool, tool) for tool in tools])
            self.tool_checkbox_lists.append(cbl)

            def make_handler(current_cbl):
                def handler():
                    all_values = [v[0] for v in current_cbl.values]
                    if not all(v in current_cbl.current_values for v in all_values):
                        current_cbl.current_values = all_values
                    else:
                        current_cbl.current_values = []
                    self._update_previews()

                return handler

            button = Button(text=f"[ ] {category}", handler=make_handler(cbl), left_symbol="", right_symbol="")
            button.window = Window(
                button.control,
                align=WindowAlign.LEFT,
                height=1,
                width=Dimension(weight=1),
                dont_extend_width=False,
                dont_extend_height=True,
            )
            self.category_buttons.append((button, cbl, category))
            indented_cbl = VSplit(
                [Window(width=4, char=" "), Box(body=cbl, padding=0, width=Dimension(weight=1))],
                width=Dimension(weight=1),
            )
            cat_container = HSplit([button.window, indented_cbl], width=Dimension(weight=1))
            self.native_category_entries.append(
                {
                    "name": category,
                    "header_btn": button,
                    "tools_cbl": cbl,
                    "container": cat_container,
                }
            )
            tool_widgets.append(cat_container)
        self.tools_container = (
            HSplit(tool_widgets, width=Dimension(weight=1)) if tool_widgets else Window(width=Dimension(weight=1))
        )

        # MCP Servers
        self.mcp_tool_checklists = []
        self.mcp_server_entries = []
        server_result = self.mcp_tools.list_servers()
        if not server_result.success:
            error_msg = server_result.message or "Failed to load MCP servers"
            self.mcp_container = Frame(
                HSplit(
                    [
                        Window(content=FormattedTextControl("MCP Servers (unavailable)"), height=1),
                        Window(height=1, char=" "),
                        Window(content=FormattedTextControl(f"Error: {error_msg}"), height=1),
                    ]
                ),
                title="MCP Servers",
                width=Dimension(weight=1),
                height=Dimension(),
            )
        else:
            servers = server_result.result.get("servers", [])
            mcp_children = []
            for server in servers:
                name = server.get("name", "")

                def make_header_handler(server_name: str):
                    def handler():
                        entry = next((e for e in self.mcp_server_entries if e.get("name") == server_name), None)
                        if not entry:
                            return
                        tools_cbl = entry.get("tools_cbl")
                        if tools_cbl:
                            all_values = [v for v, _ in tools_cbl.values]
                            if len(tools_cbl.current_values) < len(all_values):
                                tools_cbl.current_values = all_values
                            else:
                                tools_cbl.current_values = []
                        else:
                            entry["preselect_all"] = not entry.get("preselect_all", False)
                        self._update_single_mcp_header(entry)
                        self._update_previews()

                    return handler

                header_btn = Button(
                    text=f"[ ] {name}", handler=make_header_handler(name), left_symbol="", right_symbol=""
                )
                header_btn.window = Window(
                    header_btn.control,
                    align=WindowAlign.LEFT,
                    height=1,
                    width=Dimension(weight=1),
                    dont_extend_width=False,
                    dont_extend_height=True,
                )

                placeholder = Window(content=FormattedTextControl("Loading tools..."), height=1)
                tools_line = VSplit([Window(width=4, char=" "), placeholder], width=Dimension(weight=1))
                server_container = HSplit([header_btn.window, tools_line], width=Dimension(weight=1))

                entry = {
                    "name": name,
                    "header_btn": header_btn,
                    "tools_cbl": None,
                    "placeholder": placeholder,
                    "container": server_container,
                    "preselect_all": False,
                }
                self.mcp_server_entries.append(entry)
                mcp_children.append(server_container)

            self.mcp_container = Frame(
                HSplit(mcp_children, width=Dimension(weight=1)),
                title="MCP Servers (Space/Enter select; 'a' toggle all tools)",
                width=Dimension(weight=1),
                height=Dimension(),
            )

            def _schedule_tools_loading():
                import threading

                def worker(server_name: str, entry_ref: Dict[str, Any]):
                    res = self.mcp_tools.list_tools(server_name)

                    def apply_update():
                        if res.success:
                            tools = res.result.get("tools", []) if isinstance(res.result, dict) else []
                            values = []
                            for tool in tools:
                                tname = tool.get("name", "") if isinstance(tool, dict) else str(tool)
                                values.append((f"{server_name}:{tname}", tname))
                            tools_cbl = CheckboxList(values=values)
                            self.mcp_tool_checklists.append(tools_cbl)
                            indented = VSplit(
                                [Window(width=4, char=" "), Box(tools_cbl, padding=0, width=Dimension(weight=1))],
                                width=Dimension(weight=1),
                            )
                            entry_ref["tools_cbl"] = tools_cbl
                            entry_ref["container"].children[1] = indented
                            if entry_ref.get("preselect_all"):
                                tools_cbl.current_values = [v for v, _ in tools_cbl.values]
                            elif entry_ref.get("preselect_specific"):
                                wanted = set(entry_ref.get("preselect_specific") or [])
                                picks = []
                                for v, _ in tools_cbl.values:
                                    if ":" in v:
                                        srv, tool = v.split(":", 1)
                                        if srv == server_name and tool in wanted:
                                            picks.append(v)
                                tools_cbl.current_values = picks
                            try:
                                self._update_single_mcp_header(entry_ref)
                            except Exception:
                                pass
                        else:
                            err_text = f"Error: {res.message}"
                            entry_ref["container"].children[0] = Window(
                                content=FormattedTextControl(f"{server_name} (unavailable)"), height=1
                            )
                            entry_ref["container"].children[1] = VSplit(
                                [
                                    Window(width=4, char=" "),
                                    Window(content=FormattedTextControl(err_text), height=1),
                                ],
                                width=Dimension(weight=1),
                            )
                            entry_ref["header_btn"] = None
                        if hasattr(self, "app") and self.app:
                            self.app.invalidate()

                    try:
                        self.app.loop.call_soon_threadsafe(apply_update)
                    except Exception:
                        pass

                for entry in self.mcp_server_entries:
                    threading.Thread(target=worker, args=(entry["name"], entry), daemon=True).start()

            self._load_mcp_tools_async = _schedule_tools_loading

    def _init_step3_scoped_context(self):
        if self.cli_instance.agent_config.db_type == DBType.SQLITE:
            self.table_completer = TableCompleter(self.cli_instance.agent_config, True)
        else:
            self.table_completer = self.cli_instance.at_completer.table_completer
        self.catalogs_area = TextArea(
            text="",
            multiline=True,
            wrap_lines=True,
            style="class:textarea",
            completer=LineCompleter(self.table_completer),
            complete_while_typing=False,
        )
        self.metrics_area = TextArea(
            text="",
            multiline=True,
            wrap_lines=True,
            style="class:textarea",
            completer=LineCompleter(self.cli_instance.at_completer.metric_completer),
            complete_while_typing=False,
        )
        self.sqls_area = TextArea(
            text="",
            multiline=True,
            wrap_lines=True,
            style="class:textarea",
            completer=LineCompleter(self.cli_instance.at_completer.sql_completer),
            complete_while_typing=False,
        )

    def _init_step4_rules(self):
        self.rules_container = HSplit([], height=Dimension(min=3, preferred=8))
        self._update_rules_display()

    def _init_previews_and_picker_state(self):
        self.yaml_preview_buffer = Buffer(read_only=False)
        self.prompt_preview_buffer = Buffer(read_only=False)
        self._update_previews()
        self._picker_dialog = None
        self._active_picker_kind = None

    # --- Selection watchers (for mouse-driven toggles) ---
    def _snapshot_checkbox_state(self):
        try:
            self._native_selection_snapshot = {
                id(cbl): frozenset(getattr(cbl, "current_values", []) or []) for cbl in self.tool_checkbox_lists
            }
        except Exception:
            self._native_selection_snapshot = {}
        try:
            self._mcp_selection_snapshot = {
                id(cbl): frozenset(getattr(cbl, "current_values", []) or []) for cbl in self.mcp_tool_checklists
            }
        except Exception:
            self._mcp_selection_snapshot = {}

    def _detect_selection_changes(self) -> bool:
        changed = False
        try:
            for cbl in self.tool_checkbox_lists:
                key = id(cbl)
                prev = self._native_selection_snapshot.get(key, frozenset())
                cur = frozenset(getattr(cbl, "current_values", []) or [])
                if cur != prev:
                    changed = True
                    break
            if not changed:
                for cbl in self.mcp_tool_checklists:
                    key = id(cbl)
                    prev = self._mcp_selection_snapshot.get(key, frozenset())
                    cur = frozenset(getattr(cbl, "current_values", []) or [])
                    if cur != prev:
                        changed = True
                        break
        except Exception:
            # Fail-safe: if anything goes wrong, assume no change
            changed = False
        return changed

    def _selection_watch_tick(self):
        try:
            if self.done:
                return
            if self.step == 1 and self._detect_selection_changes():
                # Update parent headers and previews to reflect latest checkbox states
                try:
                    self._sync_mcp_after_toggle()
                except Exception:
                    pass
                self._update_previews()
                # Refresh snapshots after applying updates
                self._snapshot_checkbox_state()
        finally:
            # Re-arm the timer with a small delay
            try:
                if hasattr(self.app, "loop") and hasattr(self.app.loop, "call_later"):
                    self.app.loop.call_later(0.2, self._selection_watch_tick)
                elif hasattr(self.app, "loop") and hasattr(self.app.loop, "call_soon"):
                    self.app.loop.call_soon(self._selection_watch_tick)
            except Exception:
                pass

    def _start_selection_watchers(self):
        # Initialize baseline snapshots and arm periodic check
        self._snapshot_checkbox_state()
        try:
            if hasattr(self, "app") and hasattr(self.app, "loop") and hasattr(self.app.loop, "call_later"):
                self.app.loop.call_later(0.2, self._selection_watch_tick)
            elif hasattr(self, "app") and hasattr(self.app, "loop") and hasattr(self.app.loop, "call_soon"):
                self.app.loop.call_soon(self._selection_watch_tick)
        except Exception:
            pass

    def _flatten_picker_items(self, kind: str):
        """Return a list of (value, display) for the given kind."""
        items = []
        try:
            if kind == "tables":
                if not self.table_completer.flatten_data:
                    self.table_completer.reload_data()
                if flatten_data := self.table_completer.flatten_data:
                    for key, meta in flatten_data.items():
                        disp = key
                        if isinstance(meta, dict) and meta.get("table_type"):
                            disp = f"{key} [{str(meta.get('table_type'))}]"
                        items.append((key, disp))
                else:
                    data = self.table_completer.get_data() or []
                    if isinstance(data, list):
                        items = [(v, v) for v in data]
            elif kind == "metrics":
                if not self.cli_instance.at_completer.metric_completer.flatten_data:
                    self.cli_instance.at_completer.metric_completer.reload_data()
                if self.cli_instance.at_completer.metric_completer.flatten_data:
                    for key, meta in self.cli_instance.at_completer.metric_completer.flatten_data.items():
                        desc = meta.get("agent_description") if isinstance(meta, dict) else None
                        disp = f"{key} - {desc}" if desc else key
                        if len(disp) > 80:
                            disp = disp[:77] + "..."
                        items.append((key, disp))
            elif kind == "sqls":
                if not self.cli_instance.at_completer.sql_completer.flatten_data:
                    self.cli_instance.at_completer.sql_completer.reload_data()
                if self.cli_instance.at_completer.sql_completer.flatten_data:
                    for key, meta in self.cli_instance.at_completer.sql_completer.flatten_data.items():
                        summary = meta.get("summary") if isinstance(meta, dict) else None
                        disp = f"{key} - {summary}" if summary else key
                        if len(disp) > 80:
                            disp = disp[:77] + "..."
                        items.append((key, disp))
        except Exception:
            pass
        # De-dup and sort for stable UI
        seen = set()
        uniq = []
        for val, lab in items:
            if val not in seen:
                uniq.append((val, lab))
                seen.add(val)
        return sorted(uniq, key=lambda x: x[0])

    def _fill_scoped_text(self, area_component: TextArea, new_text: str):
        """Insert picker selections at the cursor rather than replacing all text."""
        if not new_text:
            self.app.layout.focus(area_component)
            return

        buffer = area_component.buffer
        document = buffer.document

        prefix = ""
        if document.cursor_position > 0 and document.char_before_cursor != "\n":
            prefix = "\n"

        text_to_insert = f"{prefix}{new_text}"
        if not text_to_insert.endswith("\n"):
            text_to_insert += "\n"

        buffer.insert_text(text_to_insert, overwrite=False)
        self.app.layout.focus(area_component)

    def _open_scoped_picker(self, kind: str):
        """Open a searchable multi-select picker for scoped context fields."""
        values = self._flatten_picker_items(kind)
        if not values:
            self._show_error_dialog(
                "No candidates loaded. Please ensure configuration and KB initialization are complete."
            )
            return

        # Build filter + checkbox list
        filter_area = TextArea(text="", multiline=False, wrap_lines=False, style="class:input-window")
        cbl = CheckboxList(values=values)

        def apply_filter():
            term = filter_area.text.strip().lower()
            if not term:
                cbl.values = values
            else:
                cbl.values = [(v, d) for (v, d) in values if term in d.lower() or term in v.lower()]
            self.app.invalidate()

        filter_area.buffer.on_text_changed += lambda *_: apply_filter()

        def on_ok():
            selected = list(cbl.current_values)
            text = "\n".join(selected)
            if kind == "tables":
                self._fill_scoped_text(self.catalogs_area, text)
            elif kind == "metrics":
                self._fill_scoped_text(self.metrics_area, text)
            else:
                self._fill_scoped_text(self.sqls_area, text)
            self._close_picker()
            self._update_previews()

        def on_cancel():
            self._close_picker()

        body = HSplit(
            [
                Label(text="Type to filter; Space to toggle; Enter to confirm", style="class:label"),
                filter_area,
                cbl,
            ]
        )

        self._picker_dialog = Dialog(
            title={"tables": "Choose Tables", "metrics": "Choose Metrics", "sqls": "Choose Reference SQL"}.get(
                kind, "Pick"
            ),
            body=body,
            buttons=[Button(text="OK", handler=on_ok), Button(text="Cancel", handler=on_cancel)],
            with_background=True,
        )

        float_container = self.layout.container
        if isinstance(float_container, FloatContainer):
            float_container.floats.append(Float(self._picker_dialog))
        self.app.layout.focus(self._picker_dialog)
        self._active_picker_kind = kind
        self.app.invalidate()

    def _close_picker(self):
        float_container = self.layout.container
        if isinstance(float_container, FloatContainer) and float_container.floats:
            # Remove last float (our picker)
            float_container.floats.pop()
        self._picker_dialog = None
        kind = self._active_picker_kind
        self._active_picker_kind = None
        # Refocus original field
        if kind == "tables":
            self.app.layout.focus(self.catalogs_area)
        elif kind == "metrics":
            self.app.layout.focus(self.metrics_area)
        elif kind == "sqls":
            self.app.layout.focus(self.sqls_area)
        self.app.invalidate()

    def _update_single_mcp_header(self, entry: Dict[str, Any]):
        """Update one MCP server header label based on its tools selection state."""
        name = entry.get("name", "")
        header_btn: Button = entry.get("header_btn")
        tools_cbl: CheckboxList = entry.get("tools_cbl")
        if not header_btn:
            return
        symbol = "[ ]"
        if tools_cbl and tools_cbl.values:
            total = len(tools_cbl.values)
            selected = len(tools_cbl.current_values)
            if selected == 0:
                symbol = "[ ]"
            elif selected == total:
                symbol = "[x]"
            else:
                symbol = "[-]"
        else:
            # If tools not loaded yet, reflect preselect state
            if entry.get("preselect_all"):
                symbol = "[x]"
        header_btn.text = f"{symbol} {name}"

    def _sync_mcp_after_toggle(self):
        """Synchronize header/tool selections and header labels after a toggle."""
        for entry in getattr(self, "mcp_server_entries", []):
            header_btn = entry.get("header_btn")
            tools_cbl = entry.get("tools_cbl")
            # name = entry.get("name")
            # If focus is on this header, propagate selection to all tools
            try:
                if header_btn and self.app.layout.has_focus(header_btn.window) and tools_cbl:
                    # No automatic propagation here; header toggle handled by button handler
                    pass
            except Exception:
                pass
            # Always refresh header label to reflect current tool selection state
            self._update_single_mcp_header(entry)
        if hasattr(self, "app") and self.app:
            self.app.invalidate()

    def _init_key_bindings(self):
        self.kb = KeyBindings()

        is_editing = Condition(lambda: self.edit_mode)
        is_not_editing = ~is_editing

        @Condition
        def is_name_input_focused():
            try:
                return self.step == 0 and self.app.layout.has_focus(self.name_buffer)
            except Exception:
                return False

        @Condition
        def is_scoped_context_input_focused():
            try:
                if self.step != 2:
                    return False
                return any(
                    self.app.layout.has_focus(w) for w in (self.catalogs_area, self.metrics_area, self.sqls_area)
                )
            except Exception:
                return False

        @Condition
        def is_completion_open():
            try:
                buf = self.app.current_buffer
                return getattr(buf, "complete_state", None) is not None
            except Exception:
                return False

        @self.kb.add("c-c")
        @self.kb.add("c-q")
        def _(event):
            self.done = True
            event.app.exit(result=None)

        @self.kb.add("escape", filter=is_not_editing)
        def _(event):
            # If a dialog is open, close it instead of exiting
            if getattr(self, "error_dialog", None) is not None:
                try:
                    self._close_error_dialog()
                    return
                except Exception:
                    pass
            if self._picker_dialog is not None:
                try:
                    self._close_picker()
                    return
                except Exception:
                    pass
            # Otherwise, cancel the wizard
            self.done = True
            event.app.exit(result=None)

        @self.kb.add("c-n")
        def _(event):
            if self.edit_mode:
                self._finish_editing()

            def _process_next_step_or_field():
                self._update_previews()
                # In step 2 (Scoped Context), Ctrl+N cycles fields; at last field, advance step
                if self.step == 2:
                    if self.app.layout.has_focus(self.catalogs_area):
                        self.app.layout.focus(self.metrics_area)
                        return
                    if self.app.layout.has_focus(self.metrics_area):
                        self.app.layout.focus(self.sqls_area)
                        return
                    # If at last or not focused on scoped fields, move to next step
                if self.step == 3:
                    self._collect_data()
                    if self._validate_step():
                        self.done = True
                        result_config = self.data.model_copy(deep=True)
                        event.app.exit(result=result_config)
                else:
                    if self._validate_step():
                        self.step += 1
                        self._update_layout()
                        event.app.invalidate()

            event.app.loop.call_soon(_process_next_step_or_field)

        @Condition
        def is_checkbox_focused():
            try:
                if self.step != 1:
                    return False
                lists = list(self.tool_checkbox_lists)
                # Only include MCP tool checklists; headers are buttons
                lists.extend(getattr(self, "mcp_tool_checklists", []))
                return any(self.app.layout.has_focus(c) for c in lists)
            except Exception:
                return False

        @self.kb.add("space", filter=is_checkbox_focused)
        @self.kb.add("enter", filter=is_checkbox_focused)
        def _(event):
            # After the widget processes the toggle, sync MCP headers/tools and refresh previews
            def _after_toggle():
                try:
                    self._sync_mcp_after_toggle()
                finally:
                    self._update_previews()

            try:
                event.app.loop.call_soon(_after_toggle)
            except Exception:
                _after_toggle()

        # 'a' to select/deselect all tools under the focused MCP server
        @Condition
        def is_mcp_focus():
            try:
                if self.step != 1:
                    return False
                for entry in getattr(self, "mcp_server_entries", []):
                    if entry.get("header_btn") and self.app.layout.has_focus(entry["header_btn"].window):
                        return True
                    if entry.get("tools_cbl") and self.app.layout.has_focus(entry["tools_cbl"]):
                        return True
                return False
            except Exception:
                return False

        @self.kb.add("a", filter=is_mcp_focus)
        def _(event):
            # Find focused entry
            target_entry = None
            for entry in getattr(self, "mcp_server_entries", []):
                if entry.get("header_btn") and self.app.layout.has_focus(entry["header_btn"].window):
                    target_entry = entry
                    break
                if entry.get("tools_cbl") and self.app.layout.has_focus(entry["tools_cbl"]):
                    target_entry = entry
                    break
            if not target_entry:
                return

            tools_cbl = target_entry.get("tools_cbl")
            if not tools_cbl:
                return
            all_values = [v for v, _ in tools_cbl.values]
            if len(tools_cbl.current_values) < len(all_values):
                tools_cbl.current_values = all_values
            else:
                tools_cbl.current_values = []
            # Update header label to reflect state
            self._update_single_mcp_header(target_entry)
            self._update_previews()

        # Shift+A (invert selection) for MCP server under focus
        @self.kb.add("A", filter=is_mcp_focus)
        def _(event):
            # Determine focused entry
            target_entry = None
            for entry in getattr(self, "mcp_server_entries", []):
                if entry.get("header_btn") and self.app.layout.has_focus(entry["header_btn"].window):
                    target_entry = entry
                    break
                if entry.get("tools_cbl") and self.app.layout.has_focus(entry["tools_cbl"]):
                    target_entry = entry
                    break
            if not target_entry:
                return
            tools_cbl = target_entry.get("tools_cbl")
            if not tools_cbl:
                return
            all_values = [v for v, _ in tools_cbl.values]
            cur = set(tools_cbl.current_values)
            inverted = [v for v in all_values if v not in cur]
            tools_cbl.current_values = inverted
            self._update_single_mcp_header(target_entry)
            self._update_previews()

        # Move focus with Tab only when no completion menu is open and not in scoped-context inputs
        @self.kb.add("tab", filter=is_not_editing & ~is_scoped_context_input_focused & ~is_completion_open)
        def _(event):
            event.app.layout.focus_next()

        @self.kb.add("s-tab", filter=is_not_editing & ~is_scoped_context_input_focused & ~is_completion_open)
        def _(event):
            event.app.layout.focus_previous()

        # In Step 0, pressing Enter on Agent Name moves focus to Description
        @self.kb.add("enter", filter=is_name_input_focused)
        def _(event):
            event.app.layout.focus(self.description_area)

        # Toggle completion menu for scoped inputs (Ctrl-P)
        @self.kb.add("c-p", filter=is_scoped_context_input_focused, eager=True)
        def _(event):
            try:
                buf = event.app.current_buffer
                # Toggle: close if open, else start completion
                if getattr(buf, "complete_state", None) is not None:
                    buf.cancel_completion()
                else:
                    buf.start_completion(select_first=False)
            except Exception:
                pass

        # Open searchable picker (F2) for the focused scoped-context field
        @self.kb.add("f2", filter=is_scoped_context_input_focused)
        def _(event):
            try:
                if self.app.layout.has_focus(self.catalogs_area):
                    self._open_scoped_picker("tables")
                elif self.app.layout.has_focus(self.metrics_area):
                    self._open_scoped_picker("metrics")
                elif self.app.layout.has_focus(self.sqls_area):
                    self._open_scoped_picker("sqls")
            except Exception:
                pass

        # Space/Enter on MCP header buttons to toggle all tools
        @Condition
        def is_mcp_header_focused():
            try:
                if self.step != 1:
                    return False
                for entry in getattr(self, "mcp_server_entries", []):
                    btn = entry.get("header_btn")
                    if btn and self.app.layout.has_focus(btn.window):
                        return True
                return False
            except Exception:
                return False

        @self.kb.add("space", filter=is_mcp_header_focused)
        @self.kb.add("enter", filter=is_mcp_header_focused)
        def _(event):
            # Find focused header and invoke its handler
            for entry in getattr(self, "mcp_server_entries", []):
                btn = entry.get("header_btn")
                if btn and self.app.layout.has_focus(btn.window):
                    # Call the handler to toggle selection
                    try:
                        btn.handler()
                    except Exception:
                        pass
                    break

        # Space/Enter on Native Tools header buttons to toggle all items in the category
        @Condition
        def is_native_header_focused():
            try:
                if self.step != 1:
                    return False
                for entry in getattr(self, "native_category_entries", []):
                    btn = entry.get("header_btn")
                    if btn and self.app.layout.has_focus(btn.window):
                        return True
                return False
            except Exception:
                return False

        @self.kb.add("space", filter=is_native_header_focused)
        @self.kb.add("enter", filter=is_native_header_focused)
        def _(event):
            for entry in getattr(self, "native_category_entries", []):
                btn = entry.get("header_btn")
                if btn and self.app.layout.has_focus(btn.window):
                    try:
                        btn.handler()
                    except Exception:
                        pass
                    break

        # Native Tools: a to select/deselect all; Shift+a to invert selection
        @Condition
        def is_native_focus():
            try:
                if self.step != 1:
                    return False
                for entry in getattr(self, "native_category_entries", []):
                    btn = entry.get("header_btn")
                    cbl = entry.get("tools_cbl")
                    if btn and self.app.layout.has_focus(btn.window):
                        return True
                    if cbl and self.app.layout.has_focus(cbl):
                        return True
                return False
            except Exception:
                return False

        @self.kb.add("a", filter=is_native_focus)
        def _(event):
            target_entry = None
            for entry in getattr(self, "native_category_entries", []):
                btn = entry.get("header_btn")
                cbl = entry.get("tools_cbl")
                if (btn and self.app.layout.has_focus(btn.window)) or (cbl and self.app.layout.has_focus(cbl)):
                    target_entry = entry
                    break
            if not target_entry:
                return
            tools_cbl = target_entry.get("tools_cbl")
            if not tools_cbl:
                return
            all_values = [v for v, _ in tools_cbl.values]
            if len(tools_cbl.current_values) < len(all_values):
                tools_cbl.current_values = all_values
            else:
                tools_cbl.current_values = []
            # Native headers are updated via _update_previews tri-state sync
            self._update_previews()

        @self.kb.add("A", filter=is_native_focus)
        def _(event):
            target_entry = None
            for entry in getattr(self, "native_category_entries", []):
                btn = entry.get("header_btn")
                cbl = entry.get("tools_cbl")
                if (btn and self.app.layout.has_focus(btn.window)) or (cbl and self.app.layout.has_focus(cbl)):
                    target_entry = entry
                    break
            if not target_entry:
                return
            tools_cbl = target_entry.get("tools_cbl")
            if not tools_cbl:
                return
            all_values = [v for v, _ in tools_cbl.values]
            cur = set(tools_cbl.current_values)
            inverted = [v for v in all_values if v not in cur]
            tools_cbl.current_values = inverted
            self._update_previews()

        @Condition
        def is_rules_step():
            return self.step == 3

        # Fallback: whenever we're in Step 1 (tools & MCP), any Space/Enter should
        # trigger a preview + header refresh. This covers cases where focus detection
        # for CheckboxList containers is inconsistent across prompt_toolkit versions.
        @self.kb.add("space", filter=Condition(lambda: self.step == 1))
        @self.kb.add("enter", filter=Condition(lambda: self.step == 1))
        def _(event):
            def _after_toggle():
                try:
                    self._sync_mcp_after_toggle()
                finally:
                    self._update_previews()

            try:
                event.app.loop.call_soon(_after_toggle)
            except Exception:
                _after_toggle()

        # --- Keybindings for NORMAL mode ---
        @self.kb.add("up", filter=is_rules_step & is_not_editing)
        def _(event):
            if self.selected_rule_index > 0:
                self.selected_rule_index -= 1
                self._update_rules_display()
                self._focus_rules_list()

        @self.kb.add("down", filter=is_rules_step & is_not_editing)
        def _(event):
            if self.selected_rule_index < len(self.data.rules) - 1:
                self.selected_rule_index += 1
                self._update_rules_display()
                self._focus_rules_list()

        @self.kb.add("u", filter=is_rules_step & is_not_editing)
        def _(event):
            if self.data.rules and self.selected_rule_index > 0:
                self.data.rules.insert(self.selected_rule_index - 1, self.data.rules.pop(self.selected_rule_index))
                self.selected_rule_index -= 1
                self._update_rules_display()
                self._update_previews()
                self._focus_rules_list()

        @self.kb.add("j", filter=is_rules_step & is_not_editing)
        def _(event):
            if self.data.rules and self.selected_rule_index < len(self.data.rules) - 1:
                self.data.rules.insert(self.selected_rule_index + 1, self.data.rules.pop(self.selected_rule_index))
                self.selected_rule_index += 1
                self._update_rules_display()
                self._update_previews()
                self._focus_rules_list()

        @self.kb.add("x", filter=is_rules_step & is_not_editing)
        def _(event):
            if self.data.rules and 0 <= self.selected_rule_index < len(self.data.rules):
                self.data.rules.pop(self.selected_rule_index)
                if self.selected_rule_index >= len(self.data.rules):
                    self.selected_rule_index = len(self.data.rules) - 1
                self._update_rules_display()
                self._update_previews()
                self._focus_rules_list()

        @self.kb.add("a", filter=is_rules_step & is_not_editing)
        def _(event):
            self._start_editing(is_new=True)

        @self.kb.add("e", filter=is_rules_step & is_not_editing)
        def _(event):
            if self.data.rules:
                self._start_editing(is_new=False)

        # --- Keybindings for EDITING mode ---
        @self.kb.add("enter", filter=is_rules_step & is_editing)
        def _(event):
            self._finish_editing()

        @self.kb.add("escape", filter=is_rules_step & is_editing)
        def _(event):
            self._cancel_editing()

    def _validate_step(self) -> bool:
        """Validate current step's data."""
        if self.step == 0:
            name = self.name_buffer.text.strip()
            description = self.description_area.text.strip()
            if not name or not description:
                self._show_error_dialog("Agent Name and Description are required.")
                return False
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
                self._show_error_dialog(f"Invalid Agent Name: {name}. Must match: ^[a-zA-Z][a-zA-Z0-9_]*$")
                return False

            # Prevent using reserved built-in sub-agent names
            if name in SYS_SUB_AGENTS and (self._original_name is None or name != self._original_name):
                self._show_error_dialog(
                    (f"Agent name '{name}' is reserved for built-in functionality. " "Please choose a different name.")
                )
                return False

            # Check for existing agent name
            existing_agents = self.cli_instance.configuration_manager.get("agentic_nodes", {})
            # Allow same-name when editing the existing agent
            if name in existing_agents and (self._original_name is None or name != self._original_name):
                self._show_error_dialog(f"Agent '{name}' already exists. Please choose a different name.")
                return False

            is_original_name = self._original_name is not None and name == self._original_name
            if not is_original_name and self._reserved_template_names:
                conflict_pattern = re.compile(rf"^{re.escape(name)}_system(?:_.+)?$")
                conflicts: List[str] = [
                    template for template in sorted(self._reserved_template_names) if conflict_pattern.match(template)
                ]
                if conflicts:
                    conflict_list = ", ".join(f"{template}_<version>.j2" for template in conflicts)
                    self._show_error_dialog(
                        (
                            f"Agent name '{name}' conflicts with existing prompt template(s): {conflict_list}. "
                            "Please choose a different name."
                        )
                    )
                    return False

        if self.step == 1:
            # Must pick at least one native tool or one MCP tool
            if not (self.data.tools and self.data.tools.strip()) and not (self.data.mcp and self.data.mcp.strip()):
                self._show_error_dialog("At least one tool (native or MCP) must be selected.")
                return False
        return True

    def _refocus_step_element(self):
        if self.step == 0:
            self.app.layout.focus(self.name_buffer)
        elif self.step == 1:
            if self.tool_checkbox_lists:
                self.app.layout.focus(self.tool_checkbox_lists[0])
            else:
                # Focus first MCP server header if available
                entries = getattr(self, "mcp_server_entries", [])
                if entries and entries[0].get("header_btn"):
                    self.app.layout.focus(entries[0]["header_btn"].window)
        elif self.step == 2:
            self.app.layout.focus(self.catalogs_area)
        elif self.step == 3:
            self._focus_rules_list()

    def _close_error_dialog(self):
        float_container = self.layout.container
        if isinstance(float_container, FloatContainer) and float_container.floats:
            float_container.floats.pop()
        self.error_dialog = None
        self._refocus_step_element()
        self.app.invalidate()

    def _show_error_dialog(self, text: str):
        self.error_dialog = Dialog(
            title="Validation Error",
            body=Label(text=text, dont_extend_height=True),
            buttons=[Button(text="OK", handler=self._close_error_dialog)],
            with_background=True,
        )

        # Add dialog to the float container
        float_container = self.layout.container
        if isinstance(float_container, FloatContainer):
            float_container.floats.append(Float(self.error_dialog))

        self.app.layout.focus(self.error_dialog)
        self.app.invalidate()

    def _collect_data(self):
        """Collect data from all steps into self.data (SubAgentConfig)."""
        self.data.system_prompt = self.name_buffer.text.strip()
        self.data.agent_description = self.description_area.text.strip()
        # Build native tools string with category awareness
        native_parts: List[str] = []
        for entry in getattr(self, "native_category_entries", []):
            try:
                category = entry.get("name")
                cbl: CheckboxList = entry.get("tools_cbl")
                if not category or not cbl:
                    continue
                all_values = [v for v, _ in getattr(cbl, "values", [])]
                selected = list(getattr(cbl, "current_values", []))
                if not selected:
                    continue
                if len(selected) == len(all_values):
                    # All tools selected for this category -> just the category name
                    native_parts.append(category)
                else:
                    # Partial selection -> prefix each with category.
                    native_parts.extend([f"{category}.{t}" for t in selected])
            except Exception:
                continue
        self.data.tools = ", ".join(native_parts)
        # MCP selections -> comma-separated string with grouping rules
        mcp_parts: List[str] = []
        for entry in getattr(self, "mcp_server_entries", []):
            name = entry.get("name")
            tools_cbl: CheckboxList = entry.get("tools_cbl")
            try:
                if tools_cbl and tools_cbl.values:
                    total = len(tools_cbl.values)
                    selected_vals = list(getattr(tools_cbl, "current_values", []) or [])
                    if not selected_vals:
                        continue
                    if len(selected_vals) == total:
                        # All tools from this server -> just the server name
                        mcp_parts.append(name)
                    else:
                        # Partial -> server.tool for each selected
                        for sv in selected_vals:
                            sval = str(sv)
                            # stored as "server:tool"; convert to "server.tool"
                            if ":" in sval:
                                _, tool = sval.split(":", 1)
                                mcp_parts.append(f"{name}.{tool}")
                            else:
                                # Fallback: if format unexpected, still prefix
                                mcp_parts.append(f"{name}.{sval}")
                else:
                    # Tools not loaded yet: respect preselect_all if set
                    if entry.get("preselect_all") and name:
                        mcp_parts.append(name)
            except Exception:
                continue
        self.data.mcp = ", ".join(mcp_parts)
        scoped_context: Dict[str, str] = {}

        def _normalize_entries(entries: List[str]) -> List[str]:
            normalized: List[str] = []
            for entry in entries:
                normalized_entry = normalize_reference_path(entry)
                if normalized_entry:
                    normalized.append(normalized_entry)
            return normalized

        table_entries = self.selected_tables or [
            line.strip() for line in self.catalogs_area.text.split("\n") if line.strip()
        ]
        tables = _normalize_entries(table_entries)
        if tables:
            scoped_context["tables"] = ", ".join(tables)

        metric_entries = self.selected_metrics or [
            line.strip() for line in self.metrics_area.text.split("\n") if line.strip()
        ]
        metrics = _normalize_entries(metric_entries)
        if metrics:
            scoped_context["metrics"] = ", ".join(metrics)

        sql_entries = self.selected_sqls or [line.strip() for line in self.sqls_area.text.split("\n") if line.strip()]
        sqls = _normalize_entries(sql_entries)
        if sqls:
            scoped_context["sqls"] = ", ".join(sqls)

        if scoped_context:
            self.data.scoped_context = ScopedContext(**scoped_context)
        else:
            self.data.scoped_context = None
        # self.data.rules is now updated in real-time

    def _export_result(self) -> Dict[str, Any]:
        """Export SubAgentConfig to plain dict expected by callers and templates."""
        sc = self.data.scoped_context
        res: Dict[str, Any] = {
            "system_prompt": self.data.system_prompt or "your_agent_name",
            "prompt_version": self.data.prompt_version,
            "agent_description": self.data.agent_description,
            "prompt_language": self.data.prompt_language,
            "tools": self.data.tools,
            "mcp": self.data.mcp,
            "rules": list(self.data.rules or []),
        }
        scoped = {}
        if sc:
            if sc.tables:
                scoped["tables"] = sc.tables
            if sc.metrics:
                scoped["metrics"] = sc.metrics
            if sc.sqls:
                scoped["sqls"] = sc.sqls
        if scoped:
            res["scoped_context"] = scoped
        return res

    def _update_previews(self, *args, **kwargs):
        """Update YAML and prompt preview panes based on current inputs."""
        if getattr(self, "_suspend_preview_updates", False):
            return
        # Sync category button states
        if hasattr(self, "category_buttons"):
            for button, cbl, category in self.category_buttons:
                all_values = [v[0] for v in cbl.values]
                selected_values = cbl.current_values

                if not selected_values:
                    button.text = f"[ ] {category}"
                elif len(selected_values) == len(all_values):
                    button.text = f"[x] {category}"
                else:
                    button.text = f"[-] {category}"

        self._collect_data()

        # Generate YAML preview from model export
        exported = self._export_result()
        agent_name_key = exported.get("system_prompt") or "your_agent_name"
        yaml_data = {"agentic_nodes": {agent_name_key: exported}}
        try:
            self.yaml_preview_buffer.text = yaml.dump(yaml_data, sort_keys=False, allow_unicode=True, indent=2)
        except Exception:
            pass  # Ignore errors during preview update

        # Generate prompt preview using sub-agent prompt template
        prompt_context = prepare_template_context(
            node_config=self.data,
            agent_config=self.cli_instance.agent_config,
            workspace_root=self.cli_instance.agent_config.workspace_root,
        )
        try:
            prompt_text = prompt_manager.render_template(self.prompt_template_name, **prompt_context)
        except FileNotFoundError:
            prompt_text = prompt_manager.render_template("sql_system", **prompt_context)
        try:
            self.prompt_preview_buffer.text = prompt_text
        except Exception:
            pass  # Ignore errors during preview update

        if hasattr(self, "app") and self.app:
            self.app.invalidate()

    def _get_step_layout(self):
        """Return the layout for the current step."""
        if self.step == 0:
            return HSplit(
                [
                    Label(
                        "Tips: Enter → Description | Ctrl-N Next | Esc/Ctrl-C Cancel",
                        style="class:tip",
                    ),
                    Label("Agent Name (required, a-zA-Z0-9_, letter start):", style="class:label"),
                    Window(
                        BufferControl(buffer=self.name_buffer),
                        height=1,
                        style="class:input-window",
                    ),
                    Window(height=1, char=" "),
                    Label("Description / System Prompt (required, multi-line):", style="class:label"),
                    self.description_area,
                ]
            )
        elif self.step == 1:
            return HSplit(
                [
                    Label(
                        (
                            "Tips: Space/Enter toggle | a Select/Deselect All | Shift+a Invert | Headers toggle group |"
                            " Mouse supported"
                        ),
                        style="class:tip",
                    ),
                    Frame(
                        self.tools_container,
                        title="Native Tools (Space/Enter to select, can be empty):",
                        width=Dimension(weight=1),
                        height=Dimension(),
                    ),
                    self.mcp_container,
                ],
                width=Dimension(weight=1),
            )
        elif self.step == 2:
            return HSplit(
                [
                    Label(
                        "Tips: Focus a field then F2 opens picker | Ctrl-P inline suggestions",
                        style="class:tip",
                    ),
                    Label("Tables (optional):", style="class:label"),
                    self.catalogs_area,
                    Window(height=1, char=" "),
                    Label("Metrics (optional):", style="class:label"),
                    self.metrics_area,
                    Window(height=1, char=" "),
                    Label("Reference SQL (optional):", style="class:label"),
                    self.sqls_area,
                ]
            )
        elif self.step == 3:
            return HSplit(
                [
                    Label(
                        "Tips: ↑↓ Select | a Add | e Edit | x Delete | u Up | j Down | Enter Confirm",
                        style="class:tip",
                        align=WindowAlign.CENTER,
                    ),
                    Frame(self.rules_container),
                ]
            )
        return Window()  # Should not happen

    def _update_layout(self):
        """Update the layout for the new step."""
        self.left_pane.children = [self._get_step_layout()]
        self.title_control.text = f" Agent Add Wizard - Step {self.step + 1}/4 "

        if self.step == 0:
            self.app.layout.focus(self.name_buffer)
        elif self.step == 1:
            if self.tool_checkbox_lists:
                self.app.layout.focus(self.tool_checkbox_lists[0])
            else:
                entries = getattr(self, "mcp_server_entries", [])
                if entries and entries[0].get("header_btn"):
                    self.app.layout.focus(entries[0]["header_btn"].window)
        elif self.step == 2:
            self.app.layout.focus(self.catalogs_area)
        elif self.step == 3:
            self._update_rules_display()
            self._focus_rules_list()

    def _init_layout(self):
        self.title_control = FormattedTextControl(f" Agent Add Wizard - Step {self.step + 1}/4 ")

        self.left_pane = HSplit([self._get_step_layout()])

        body = VSplit(
            [
                Box(self.left_pane, padding=0, width=Dimension(weight=1)),
                Window(width=1, char="│", style="class:separator"),
                Box(
                    HSplit(
                        [
                            Label("Preview (YAML - agent.yml)"),
                            Window(
                                content=BufferControl(
                                    buffer=self.yaml_preview_buffer,
                                    lexer=PygmentsLexer(YamlLexer),
                                    focusable=False,
                                ),
                                wrap_lines=False,
                            ),
                            Window(height=1, char="─", style="class:separator"),
                            Label("Preview (Rendered Prompt)"),
                            Window(
                                content=BufferControl(
                                    buffer=self.prompt_preview_buffer,
                                    lexer=PygmentsLexer(HtmlLexer),
                                    focusable=False,
                                ),
                                wrap_lines=True,
                            ),
                        ]
                    ),
                    padding=0,
                    width=Dimension(weight=2),
                ),
            ]
        )

        root_container = HSplit(
            [
                Window(
                    content=self.title_control,
                    height=1,
                    style="class:status-bar",
                    align=WindowAlign.LEFT,
                ),
                body,
                Window(
                    height=1,
                    content=FormattedTextControl(
                        (
                            "[Ctrl-N] Next Field/Step | [Tab] Next Input (outside step 3) "
                            "| [Ctrl+P] Inline Suggestions | [F2] Scoped Picker | [a] Select All (Tools/MCP) "
                            "| [Shift+A] Invert (Tools/MCP) | [Ctrl-C/Esc] Cancel"
                        )
                    ),
                    style="class:status-bar",
                ),
            ]
        )

        self.layout = Layout(FloatContainer(root_container, floats=[]), focused_element=self.name_buffer)

        self.style = Style.from_dict(
            {
                "status-bar": "bg:#000044 #ffffff",
                "input-window": "fg:ansigreen",
                "textarea": "fg:ansigreen",
                "label": "fg:ansicyan",
                "tip": "fg:ansiyellow bold",
                "separator": "fg:ansigray",
                "dialog": "bg:#444444",
                "dialog frame.label": "fg:#ffffff bg:#000000",
                "dialog.body": "bg:#444444 fg:#ffffff",
                "dialog shadow": "bg:#000000",
                "rule": "",
                "rule.selected": "bg:ansiblue fg:ansiwhite",
                "rule.editing": "bg:ansigreen fg:ansiwhite",
            }
        )

    def run(self) -> Optional[SubAgentConfig]:
        """Run the wizard application."""
        # Ensure selection watchers start exactly when the event loop is running.
        try:
            return self.app.run(pre_run=self._start_selection_watchers)
        except TypeError:
            # Fallback for older prompt_toolkit where pre_run is not supported
            return self.app.run()

    def native_tools_choices(self) -> Dict[str, List[str]]:
        from datus.tools.func_tool import ContextSearchTools, DBFuncTool

        return {
            "db_tools": DBFuncTool.all_tools_name(),
            "context_search_tools": ContextSearchTools.all_tools_name(),
            "date_parsing_tools": ["parse_temporal_expressions", "get_current_date"],
        }


def run_wizard(
    cli_instance: "DatusCLI", data: Optional[Union[SubAgentConfig, Dict[str, Any]]] = None
) -> Optional[SubAgentConfig]:
    """
    Initializes and runs the agent creation wizard.
    """
    wizard = SubAgentWizard(cli_instance, data)
    return wizard.run()
