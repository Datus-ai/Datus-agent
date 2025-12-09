# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Streamlit Chatbot for Datus Agent

This module provides a web-based chatbot interface using Streamlit,
maximizing reuse of existing Datus CLI components including:
- DatusCLI for real CLI functionality
- ChatCommands for chat processing
- ActionHistoryDisplay for execution visualization
- CollapsibleActionContentGenerator for detail views
"""

import csv
import math
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import structlog

# Import Datus components to reuse
from datus.cli.repl import DatusCLI
from datus.cli.web.chat_executor import ChatExecutor
from datus.cli.web.config_manager import ConfigManager
from datus.cli.web.session_loader import SessionLoader
from datus.cli.web.ui_components import UIComponents
from datus.models.session_manager import SessionManager
from datus.schemas.action_history import ActionHistory
from datus.schemas.node_models import ExecuteSQLResult
from datus.utils.loggings import configure_logging, setup_web_chatbot_logging

# Logging setup shared with CLI entry point
logger = structlog.get_logger("web_chatbot")
_LOGGING_INITIALIZED = False


def initialize_logging(debug: bool = False, log_dir: str = None) -> None:
    """Configure logging for the Streamlit subprocess to match CLI behavior."""

    global _LOGGING_INITIALIZED, logger

    if _LOGGING_INITIALIZED:
        return

    # Use path manager default if not specified
    if log_dir is None:
        from datus.utils.path_manager import get_path_manager

        log_dir = str(get_path_manager().logs_dir)

    configure_logging(debug=debug, log_dir=log_dir, console_output=False)
    logger = setup_web_chatbot_logging(debug=debug, log_dir=log_dir)
    _LOGGING_INITIALIZED = True


class StreamlitChatbot:
    """Main Streamlit Chatbot class that wraps Datus CLI components"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.session_loader = SessionLoader()
        self.chat_executor = ChatExecutor()
        self.config_manager = ConfigManager()

        # Get server host and port from Streamlit config with fallback
        self.server_host = st.get_option("server.address") or "localhost"
        self.server_port = st.get_option("server.port") or 8501

        # Initialize UI components
        self.ui = UIComponents(self.server_host, self.server_port)

        # Initialize session state with defaults
        defaults = {
            "messages": [],
            "current_actions": [],
            "chat_session_initialized": False,
            "cli_instance": None,
            "current_chat_id": None,
            "subagent_name": None,
            "view_session_id": None,
            "session_readonly_mode": False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def sanitize_csv_field(value: Optional[str]) -> Optional[str]:
        """
        Sanitize a CSV field to prevent formula injection.

        If the field starts with =, +, -, or @, prefix it with a single quote
        to neutralize Excel formula injection attacks.

        Args:
            value: The field value to sanitize

        Returns:
            Sanitized value safe for CSV export
        """
        if value is None:
            return None

        if not isinstance(value, str):
            value = str(value)

        # Check if first character is a formula trigger
        if value and value[0] in "=+-@":
            return "'" + value

        return value

    @property
    def cli(self) -> DatusCLI:
        """Get CLI instance from session state"""
        return st.session_state.cli_instance

    @cli.setter
    def cli(self, value):
        """Set CLI instance in session state"""
        st.session_state.cli_instance = value

    @property
    def current_subagent(self) -> Optional[str]:
        """Get current subagent from URL query parameters"""
        return st.query_params.get("subagent")

    @property
    def should_hide_sidebar(self) -> bool:
        """Check if sidebar should be hidden (embed mode)"""
        # Return from session_state (persists across reruns)
        # Query params are read in run() method after set_page_config
        return st.session_state.get("embed_mode", False)

    def setup_config(
        self, config_path: str = "conf/agent.yml", namespace: str = None, catalog: str = "", database: str = ""
    ) -> bool:
        """Delegate to ConfigManager for agent configuration setup."""
        # Check if already initialized to prevent repeated initialization
        if self.cli is not None:
            logger.info("CLI already initialized, skipping...")
            return True

        try:
            self.cli = self.config_manager.setup_config(config_path, namespace, catalog, database)
            st.session_state.chat_session_initialized = True
            self.ui.agent_config = self.cli.agent_config
            return True
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            logger.error(f"Configuration loading error: {e}")
            return False

    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with configuration information"""
        # Skip sidebar rendering in embed mode, but keep config loading
        if self.should_hide_sidebar:
            # Still need to initialize config if not done
            if not self.cli and not st.session_state.get("initialization_attempted", False):
                startup_config = st.session_state.get("startup_config_path", "conf/agent.yml")
                startup_namespace = st.session_state.get("startup_namespace", None)
                startup_catalog = st.session_state.get("startup_catalog", "")
                startup_database = st.session_state.get("startup_database", "")

                st.session_state.initialization_attempted = True

                if self.setup_config(
                    startup_config, startup_namespace, catalog=startup_catalog, database=startup_database
                ):
                    st.rerun()
                else:
                    st.session_state.initialization_attempted = False

            # Update subagent name from URL
            if self.cli:
                st.session_state.subagent_name = self.current_subagent

            return {"config_loaded": self.cli is not None}

        with st.sidebar:
            st.header("📊 Datus Chat")

            # Auto-load config with startup parameters (only once)
            if not self.cli and not st.session_state.get("initialization_attempted", False):
                startup_config = st.session_state.get("startup_config_path", "conf/agent.yml")
                startup_namespace = st.session_state.get("startup_namespace", None)
                startup_catalog = st.session_state.get("startup_catalog", "")
                startup_database = st.session_state.get("startup_database", "")

                # Mark that we've attempted initialization
                st.session_state.initialization_attempted = True

                with st.spinner("Loading configuration..."):
                    if self.setup_config(
                        startup_config, startup_namespace, catalog=startup_catalog, database=startup_database
                    ):
                        st.success("✅ Configuration loaded!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load configuration")
                        st.session_state.initialization_attempted = False

            # Show current configuration info
            if self.cli:
                # Set subagent name directly from URL (always refresh from URL)
                st.session_state.subagent_name = self.current_subagent

                # Current subagent info
                if self.current_subagent:
                    st.subheader("🤖 Current Subagent")
                    st.info(f"**{st.session_state.subagent_name}** (GenSQL Mode)")

                # Current namespace info
                st.subheader("🏷️ Current Namespace")
                if hasattr(self.cli.agent_config, "current_namespace"):
                    st.info(f"**{self.cli.agent_config.current_namespace}**")

                # Model selection
                st.subheader("🤖 Chat Model")
                available_models = self.get_available_models()
                current_model = self.get_current_chat_model()

                if available_models:
                    selected_model = st.selectbox(
                        "Select Model:",
                        options=available_models,
                        index=available_models.index(current_model) if current_model in available_models else 0,
                        help="Choose the model for chat conversations",
                    )

                    if selected_model != current_model:
                        st.info(f"Model changed to: {selected_model}")
                        # Note: Model switching would require config reload
                        # For now, just show the selection

                # Session controls
                st.markdown("---")
                st.subheader("💬 Session")

                # Session info
                if self.cli.chat_commands and self.cli.chat_commands.chat_node:
                    session_info = self.cli.chat_commands.chat_node.get_session_info()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Messages", session_info.get("action_count", 0))
                    with col2:
                        st.metric("Tokens", session_info.get("token_count", 0))

                # Clear chat button
                if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                    self.clear_chat()
                    st.rerun()

                # Session History section
                st.markdown("---")
                st.subheader("📚 Session History")

                # List only recent 10 sessions (already sorted by modification time)
                recent_sessions = self.session_manager.list_sessions(limit=10, sort_by_modified=True)
                if recent_sessions:
                    # Get session info for the recent sessions
                    session_infos = []
                    for sid in recent_sessions:
                        info = self.session_manager.get_session_info(sid)
                        if info.get("exists"):
                            session_infos.append(info)

                    # Display recent sessions
                    if session_infos:
                        st.caption(f"Showing {len(session_infos)} recent session(s)")
                        for info in session_infos:
                            self.ui.render_session_item(info)
                else:
                    st.caption("No saved sessions yet")

                # Report Issue section
                st.markdown("---")

                # Get session ID
                session_id = self.get_current_session_id()

                # Render Report Issue button
                import streamlit.components.v1 as components

                components.html(self.ui.generate_report_issue_html(session_id), height=150)

                # Debug section
                st.markdown("---")
                st.subheader("🔍 Debug Info")
                with st.expander("Debug Details", expanded=False):
                    st.write("Query Params:", dict(st.query_params))
                    st.write("Startup Subagent:", st.session_state.get("startup_subagent_name"))
                    st.write("Current Subagent:", st.session_state.get("subagent_name"))
                    st.write("Session ID:", self.get_current_session_id())
                    if self.cli and self.cli.chat_commands:
                        st.write("Has current_node:", self.cli.chat_commands.current_node is not None)
                        st.write("Has chat_node:", self.cli.chat_commands.chat_node is not None)
                        if self.cli.chat_commands.current_node:
                            st.write(
                                "current_node.session_id:",
                                getattr(self.cli.chat_commands.current_node, "session_id", None),
                            )

            else:
                st.warning("⚠️ Loading configuration...")

            return {"config_loaded": self.cli is not None}

    def get_available_models(self) -> List[str]:
        """Delegate to ConfigManager for getting available models."""
        self.config_manager.cli = self.cli
        return self.config_manager.get_available_models()

    def get_current_chat_model(self) -> str:
        """Delegate to ConfigManager for getting current chat model."""
        config_path = st.session_state.get("startup_config_path", "conf/agent.yml")
        self.config_manager.cli = self.cli
        return self.config_manager.get_current_chat_model(config_path)

    def clear_chat(self):
        """Clear chat history and session"""
        st.session_state.messages = []
        st.session_state.current_actions = []
        st.session_state.current_chat_id = None

        if self.cli and self.cli.chat_commands:
            self.cli.chat_commands.cmd_clear_chat("")

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Delegate to SessionLoader for loading messages from database."""
        return self.session_loader.get_session_messages(session_id)

    def get_current_session_id(self) -> Optional[str]:
        """Delegate to SessionLoader for getting current session ID."""
        return self.session_loader.get_current_session_id(self.cli)

    def _store_session_id(self) -> None:
        """Store current session_id in session_state for sidebar access"""
        session_id = self.get_current_session_id()
        if session_id:
            st.session_state.current_session_id = session_id

    def save_success_story(self, sql: str, user_message: str):
        """
        Save a success story to CSV file with session link.

        Args:
            sql: The generated SQL query
            user_message: The user's original question
        """
        # Get current session ID
        session_id = self.get_current_session_id()
        if not session_id:
            st.warning("No active session found. Cannot save success story.")
            logger.warning("Attempted to save success story without active session")
            return

        # Get subagent name (for metadata and directory organization)
        subagent_name = st.session_state.get("subagent_name") or "default"

        # Generate session link with current server host and port
        session_link = f"http://{self.server_host}:{self.server_port}?session={session_id}"

        # Create benchmark directory safely (sanitize and contain)
        from datus.utils.path_manager import get_path_manager

        base_dir = get_path_manager().benchmark_dir.resolve()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", subagent_name)
        target_dir = (base_dir / safe_name).resolve()
        try:
            target_dir.relative_to(base_dir)
        except ValueError:
            logger.warning(f"Rejected unsafe subagent_name: {subagent_name!r}")
            st.error("Unsafe subagent name.")
            return
        target_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        csv_path = str(target_dir / "success_story.csv")

        # Prepare row data with CSV injection protection
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "session_link": self.sanitize_csv_field(session_link),
            "session_id": session_id,
            "subagent_name": self.sanitize_csv_field(subagent_name),
            "user_message": self.sanitize_csv_field(user_message),
            "sql": self.sanitize_csv_field(sql),
            "timestamp": timestamp,
        }

        try:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(csv_path)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                fieldnames = ["session_link", "session_id", "subagent_name", "user_message", "sql", "timestamp"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Write the success story
                writer.writerow(row)

            st.success(f"✅ Success story saved! Session link: {session_link}")
            logger.info(f"Saved success story for session {session_id}")

        except Exception as e:
            st.error(f"Failed to save success story: {e}")
            logger.error(f"Failed to save success story: {e}")

    def load_session_from_url(self):
        """
        Load a session from URL query parameter if present.
        Sets the app to read-only mode when viewing a shared session.
        """
        # Check URL query params for session parameter
        session_id = st.query_params.get("session")

        # Check if we've already loaded this specific session
        if st.session_state.get("view_session_id") == session_id:
            return
        if not session_id:
            return

        # Verify session exists
        if not self.session_manager.session_exists(session_id):
            st.error(f"Session {session_id} not found or has no data.")
            logger.warning(f"Attempted to load non-existent session: {session_id}")
            return

        # Load messages from session
        try:
            messages = self.get_session_messages(session_id)
            if not messages:
                st.warning(f"Session {session_id} has no messages to display.")
                return

            # Populate session state with loaded messages
            st.session_state.messages = messages
            st.session_state.view_session_id = session_id
            st.session_state.session_readonly_mode = True

            logger.info(f"Loaded session {session_id} with {len(messages)} messages in read-only mode")

        except Exception as e:
            st.error(f"Failed to load session: {e}")
            logger.error(f"Failed to load session {session_id}: {e}")

    def extract_sql_and_response(self, actions: List[ActionHistory]) -> Tuple[Optional[str], Optional[str]]:
        """Delegate to ChatExecutor for SQL and response extraction."""
        return self.chat_executor.extract_sql_and_response(actions, self.cli)

    def _format_action_for_stream(self, action: ActionHistory) -> str:
        """Delegate to ChatExecutor for action formatting."""
        return self.chat_executor.format_action_for_stream(action)

    def execute_chat_stream(self, user_message: str):
        """Delegate to ChatExecutor for streaming execution."""
        for msg in self.chat_executor.execute_chat_stream(user_message, self.cli, self.current_subagent):
            yield msg
        # Store session_id and actions
        self._store_session_id()
        # Note: actions are stored in session_state by caller

    def do_download(self, sql: str, markdown: str, data: ExecuteSQLResult, sql_id: Optional[str], display_column):
        """Execute SQL, build a multi-sheet Excel workbook, and expose it for download."""

        if not sql_id:
            sql_id = str(uuid.uuid4())

        if not self.cli or not getattr(self.cli, "db_connector", None):
            st.error("Database connector is not initialized. Please configure the agent first.")
            logger.error("Download requested without an active database connector")
            return
        markdown = markdown or ""

        def _normalize_dataframe(sql_return: Any) -> pd.DataFrame:
            """Convert different result payloads into a DataFrame."""
            if isinstance(sql_return, pd.DataFrame):
                return sql_return
            if sql_return is None:
                return pd.DataFrame()
            if isinstance(sql_return, list):
                return pd.DataFrame(sql_return)
            if isinstance(sql_return, dict):
                return pd.DataFrame([sql_return])
            return pd.DataFrame({"result": [sql_return]})

        def _write_text_block(worksheet, text: str):
            """
            Write text into a single merged cell spanning multiple rows/columns.

            The goal is to end up with a single logical cell (one merged region)
            that contains the full text (with line breaks), instead of one row
            per line.
            """
            text = text or ""
            lines = text.splitlines() or [""]

            # Estimate how many columns we want based on the longest line length
            max_len = max((len(line) for line in lines), default=1)
            approx_chars_per_col = 45
            col_span = max(1, min(8, math.ceil(max_len / approx_chars_per_col)))
            row_span = max(1, len(lines))

            # Configure column widths
            for col in range(col_span):
                worksheet.set_column(col, col, approx_chars_per_col)

            text_format = writer.book.add_format(
                {
                    "text_wrap": True,
                    "valign": "top",
                    "align": "left",
                }
            )

            # If the merged region would cover more than a single cell, use merge_range.
            # Otherwise, just write into the single top-left cell to avoid xlsxwriter's
            # "Can't merge single cell" warning.
            if row_span > 1 or col_span > 1:
                worksheet.merge_range(0, 0, row_span - 1, col_span - 1, text, text_format)
            else:
                worksheet.write(0, 0, text or " ", text_format)

        with st.spinner("Preparing Excel..."):
            buffer = BytesIO()
            try:
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    if data and getattr(data, "success", False):
                        df = _normalize_dataframe(data.sql_return)
                        df.to_excel(writer, sheet_name="result", index=False)
                    else:
                        df = pd.DataFrame()
                        worksheet = writer.book.add_worksheet("result")
                        writer.sheets["result"] = worksheet
                        error_text = (getattr(data, "error", None) or "Unknown error").strip()
                        error_format = writer.book.add_format(
                            {
                                "text_wrap": True,
                                "font_color": "#9C0006",
                                "bold": True,
                                "align": "left",
                                "valign": "top",
                            }
                        )
                        worksheet.set_column(0, 3, 50)
                        worksheet.merge_range(0, 0, 0, 3, error_text, error_format)

                    sql_sheet = writer.book.add_worksheet("SQL")
                    writer.sheets["SQL"] = sql_sheet
                    from datus.utils.sql_utils import format_sql_to_pretty

                    _write_text_block(
                        sql_sheet,
                        format_sql_to_pretty(
                            sql, getattr(self.cli.db_connector, "dialect", None) or self.cli.agent_config.db_type
                        ),
                    )

                    markdown_sheet = writer.book.add_worksheet("report")
                    writer.sheets["report"] = markdown_sheet
                    _write_text_block(markdown_sheet, markdown)
            except Exception as exc:
                st.error(f"Failed to generate Excel: {exc}")
                logger.error(f"Failed to build download workbook: {exc}", exc_info=True)
                return

            buffer.seek(0)
            file_bytes = buffer.getvalue()

        file_name = f"{sql_id}.xlsx"

        with display_column:
            st.download_button(
                label="⬇ Save results as Excel",
                data=file_bytes,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_download_{sql_id}",
                use_container_width=False,
                disabled=not df.empty and len(df) > self.cli.agent_config.max_export_lines,
            )

    def run(self):
        """Main Streamlit app runner"""
        # Read query params and update session_state
        hide_param = st.query_params.get("hide_sidebar")
        if hide_param is not None:
            st.session_state.embed_mode = hide_param == "true"

        # Initialize logging for web interface
        if "log_manager_initialized" not in st.session_state:
            st.session_state.log_manager_initialized = True
            logger.info("Web chatbot logging initialized")

        # Hide deploy button and toolbar
        st.set_option("client.toolbarMode", "viewer")

        # Update session_id in session_state at the beginning of each render
        # This ensures sidebar always has the latest session_id
        current_session_id = self.get_current_session_id()
        if current_session_id:
            st.session_state.current_session_id = current_session_id

        # Custom CSS for chat styling
        if self.should_hide_sidebar:
            # Hide sidebar completely in embed mode
            st.markdown(
                """
                <style>
                .stChatMessage {
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                }
                .user-message {
                    background-color: #e3f2fd;
                }
                .assistant-message {
                    background-color: #f5f5f5;
                }
                .stExpander {
                    border: 1px solid #ddd;
                    border-radius: 0.5rem;
                    margin-bottom: 0.5rem;
                }
                [data-testid="stSidebar"] {
                    display: none;
                }
                [data-testid="stSidebarCollapsedControl"] {
                    display: none;
                }
                [aria-label="Download as CSV"] {
                    display: none !important;
                    visibility: hidden !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <style>
                .stChatMessage {
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                }
                .user-message {
                    background-color: #e3f2fd;
                }
                .assistant-message {
                    background-color: #f5f5f5;
                }
                .stExpander {
                    border: 1px solid #ddd;
                    border-radius: 0.5rem;
                    margin-bottom: 0.5rem;
                }
                [aria-label="Download as CSV"] {
                    display: none !important;
                    visibility: hidden !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        # Load session from URL if present
        self.load_session_from_url()

        # Show read-only banner if viewing shared session
        if st.session_state.session_readonly_mode:
            session_id_short = st.session_state.view_session_id[:8] if st.session_state.view_session_id else "unknown"
            st.info(f"📖 Viewing Shared Session (Read-Only) - ID: {session_id_short}...")

        # Title and description with subagent support - detect directly from URL
        if self.current_subagent:
            st.title(f"🤖 Datus AI Chat Assistant - {self.current_subagent.title()}")
            st.caption(f"Specialized {self.current_subagent} subagent for SQL generation - Natural Language to SQL")
        else:
            st.title("🤖 Datus AI Chat Assistant")
            st.caption("Intelligent database query assistant based on Datus Agent - Natural Language to SQL")
            # Only show available subagents when NOT in subagent mode and NOT in readonly mode
            if not st.session_state.session_readonly_mode:
                self.ui.show_available_subagents(self.cli.agent_config if self.cli else None)

        # Render sidebar and get config status
        sidebar_state = self.render_sidebar()

        # Main chat interface
        if not sidebar_state["config_loaded"]:
            st.warning("⚠️ Please wait for configuration to load or check the sidebar")
            st.info("Configuration file contains database connections, model settings, etc.")
            return
        self.ui.reset_sql_display_state()
        self._display_chat_actions()

        if not self.cli or not self.cli.chat_commands:
            st.warning("⚠️ Something went wrong, please try restarting.")
            return
        # Chat input - disabled in read-only mode
        if not st.session_state.session_readonly_mode:
            if prompt := st.chat_input("Enter your data query question..."):
                # Create unique chat ID for this conversation
                import uuid

                chat_id = str(uuid.uuid4())
                st.session_state.current_chat_id = chat_id

                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt, "chat_id": chat_id})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate assistant response
                with st.chat_message("assistant"):
                    # Use status container for progress display
                    status_placeholder = st.empty()
                    progress_messages = []

                    with status_placeholder.container():
                        with st.status("Processing your query... ", expanded=True) as status:
                            step_index = 0
                            from datus.cli.action_history_display import ActionContentGenerator

                            content_generator = ActionContentGenerator(enable_truncation=False)

                            for action in self.execute_chat_stream(prompt):
                                step_index += 1
                                self.ui.render_action_item(chat_id, step_index, action, content_generator)
                            status.update(label=f"✓ Completed {step_index} steps", state="complete", expanded=True)
                    # Get complete actions from chat executor
                    actions = self.chat_executor.last_actions
                    logger.info(f"Chat execution completed: {len(actions) if actions else 0} actions collected")

                    # Extract SQL and response
                    sql, response = self.extract_sql_and_response(actions)

                    # Display final response
                    if response:
                        self.ui.display_markdown_response(response)
                    else:
                        st.markdown(
                            "Sorry, unable to generate a valid response. "
                            "Please check execution details for more information."
                        )

                    # Display SQL if available
                    if sql:
                        self.ui.display_sql(sql, self.cli.db_connector.execute_pandas(sql))

                    # # Display collapsed action history at bottom
                    if actions:
                        self.ui.render_action_history(actions, chat_id, expanded=False)

                    # Save to chat history with complete data
                    assistant_message = {
                        "role": "assistant",
                        "content": response or "Unable to generate valid response",
                        "sql": sql,
                        "actions": actions,
                        "chat_id": chat_id,
                        "progress_messages": progress_messages,
                    }
                    logger.info(
                        f"Saving message: chat_id={chat_id}, has_sql={sql is not None}, "
                        f"actions_count={len(actions) if actions else 0}"
                    )
                    st.session_state.messages.append(assistant_message)

                    # Trigger rerun to update sidebar with new session_id and display via history loop
                    st.rerun()
        else:
            # Show disabled input in read-only mode
            st.chat_input("Read-only mode - cannot send messages", disabled=True)

        # Display conversation statistics
        if st.session_state.messages:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                conversation_count = len([m for m in st.session_state.messages if m["role"] == "user"])
                st.metric("Conversation Turns", conversation_count)

            with col2:
                total_chars = sum(len(m["content"]) for m in st.session_state.messages)
                st.metric("Total Characters", f"{total_chars:,}")

            with col3:
                current_model = self.get_current_chat_model()
                st.metric("Current Model", current_model)

    def _display_chat_actions(self):
        # Display chat history
        readonly_mode = st.session_state.session_readonly_mode
        last_chat_id = None if not self.chat_executor.last_actions else self.chat_executor.last_actions[-1]["chat_id"]
        for _, message in enumerate(st.session_state.messages):
            if last_chat_id and last_chat_id == message["chat_id"]:
                continue
            with st.chat_message(message["role"]):
                # Display user messages
                if message["role"] == "user":
                    st.markdown(message["content"])

                # Display assistant messages
                elif message["role"] == "assistant":
                    # Different display based on readonly mode
                    if readonly_mode:
                        # Session page: Only show expanded action history
                        actions_data = message.get("actions", [])
                        chat_id = message.get("chat_id", "default")
                        if actions_data:
                            self.ui.render_action_history(actions_data, chat_id, expanded=True)
                    else:
                        # Normal page: Show progress summary, AI response, SQL, and collapsed details at bottom
                        # Show progress summary if available
                        # Show collapsed action history at bottom
                        actions_data = message.get("actions", [])
                        chat_id = message.get("chat_id", "default")
                        if actions_data:
                            self.ui.render_action_history(actions_data, chat_id, expanded=False)

                        # Show AI response summary
                        st.markdown(message["content"])

                        # Show SQL if available
                        if "sql" in message and message["sql"]:
                            sql = message["sql"]
                            user_msg = ""
                            if st.session_state.messages:
                                user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
                                if user_msgs:
                                    user_msg = user_msgs[-1]
                            data = self.cli.db_connector.execute_pandas(sql)
                            sql_id = self.ui.display_sql(sql, data)
                            col1, col2, col3 = st.columns([1, 1, 12])
                            with col1:
                                self.ui.display_success_button(sql, user_msg, sql_id, self.save_success_story)
                            with col2:
                                self.ui.display_download(sql, message["content"], data, col3, sql_id, self.do_download)


def run_web_interface(args):
    """Launch Streamlit web interface"""
    import os
    import subprocess
    import sys

    try:
        # Get the path to the web chatbot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        web_chatbot_path = os.path.join(current_dir, "chatbot.py")

        if not os.path.exists(web_chatbot_path):
            print(f"❌ Error: Web chatbot not found at {web_chatbot_path}")
            sys.exit(1)

        print("🚀 Starting Datus Web Interface...")
        if args.namespace:
            print(f"🔗 Using namespace: {args.namespace}")
        if args.config:
            print(f"⚙️ Using config: {args.config}")
        if args.database:
            print(f"📚 Using database: {args.database}")
        print(f"🌐 Starting server at http://{args.host}:{args.port}")
        print("⏹️ Press Ctrl+C to stop server")
        print("-" * 50)

        # Prepare streamlit command
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            web_chatbot_path,
            "--server.port",
            str(args.port),
            "--server.address",
            args.host,
            "--browser.serverAddress",
            args.host,
        ]

        # Add arguments to pass to the web app
        web_args = []
        if args.namespace:
            web_args.extend(["--namespace", args.namespace])
        if args.config:
            web_args.extend(["--config", args.config])
        if args.database:
            web_args.extend(["--database", args.database])
        if getattr(args, "debug", False):
            web_args.append("--debug")

        if web_args:
            cmd.extend(["--"] + web_args)

        # Launch streamlit
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 Web server stopped")
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    import sys

    # Page configuration - MUST be the first Streamlit command
    st.set_page_config(
        page_title="Datus AI Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )

    # Parse command line arguments
    namespace = None
    config_path = "conf/agent.yml"
    database = ""
    subagent_name = None
    debug = False

    # Simple argument parsing for Streamlit
    for i, arg in enumerate(sys.argv):
        if arg == "--namespace" and i + 1 < len(sys.argv):
            namespace = sys.argv[i + 1]
        elif arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        elif arg == "--database" and i + 1 < len(sys.argv):
            database = sys.argv[i + 1]
        elif arg == "--debug":
            debug = True

    # Initialize logging once per process
    initialize_logging(debug=debug)

    # Note: Subagent detection is now handled directly in the interface, not here

    # Store in session state for use by the app
    if "startup_namespace" not in st.session_state:
        st.session_state.startup_namespace = namespace
    if "startup_config_path" not in st.session_state:
        st.session_state.startup_config_path = config_path
    if "startup_subagent_name" not in st.session_state:
        st.session_state.startup_subagent_name = subagent_name
    if "startup_database" not in st.session_state:
        st.session_state.startup_database = database
    if "startup_debug" not in st.session_state:
        st.session_state.startup_debug = debug

    chatbot = StreamlitChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
