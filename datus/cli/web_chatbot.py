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
import hashlib
import os
from argparse import Namespace
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import structlog
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Import Datus components to reuse
from datus.cli.action_history_display import ActionContentGenerator
from datus.cli.repl import DatusCLI
from datus.cli.screen.action_display_app import CollapsibleActionContentGenerator
from datus.configuration.agent_config_loader import parse_config_path
from datus.models.session_manager import SessionManager
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import configure_logging, setup_web_chatbot_logging

# Logging setup shared with CLI entry point
logger = structlog.get_logger("web_chatbot")
_LOGGING_INITIALIZED = False


def initialize_logging(debug: bool = False, log_dir: str = "~/.datus/logs") -> None:
    """Configure logging for the Streamlit subprocess to match CLI behavior."""

    global _LOGGING_INITIALIZED, logger

    if _LOGGING_INITIALIZED:
        return

    configure_logging(debug=debug, log_dir=log_dir, console_output=False)
    logger = setup_web_chatbot_logging(debug=debug, log_dir=log_dir)
    _LOGGING_INITIALIZED = True


def get_available_namespaces(config_path: str = "") -> List[str]:
    """Extract available namespaces from config file"""
    try:
        config = _load_config_cached(config_path)
        if "agent" in config and "namespace" in config["agent"]:
            return list(config["agent"]["namespace"].keys())
        elif "namespace" in config:
            return list(config["namespace"].keys())
        return []
    except Exception as e:
        logger.error(f"Failed to read namespaces from config: {e}")
        return []


def create_cli_args(config_path: str = "", namespace: str = None, catalog: str = "", database: str = "") -> Namespace:
    """Create CLI arguments for DatusCLI initialization"""
    args = Namespace()
    args.config = parse_config_path(config_path)
    args.namespace = namespace  # Add namespace parameter
    args.history_file = ".datus_history"
    args.db_type = "sqlite"
    args.db_path = None
    args.database = database
    args.catalog = catalog
    args.schema = ""
    # Add missing attributes that DatusCLI expects
    args.debug = bool(st.session_state.get("startup_debug", False)) if hasattr(st, "session_state") else False
    args.no_color = False

    # Read storage path from config file
    args.storage_path = get_storage_path_from_config(config_path)

    args.save_llm_trace = False
    # Add non-interactive mode flags
    args.non_interactive = True
    args.disable_detail_views = True
    return args


@lru_cache(maxsize=1)
def _load_config_cached(config_path: str) -> Dict[str, Any]:
    """Load and cache YAML configuration"""
    import yaml

    config_path = parse_config_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_storage_path_from_config(config_path: str) -> str:
    """Extract storage base_path from configuration file"""
    try:
        config = _load_config_cached(config_path)
        storage_config = config.get("agent", {}).get("storage", {})
        base_path = storage_config.get("base_path", "data")
        return os.path.expanduser(base_path)
    except Exception as e:
        logger.warning(f"Failed to read storage path from config: {e}")
        return "data"


class StreamlitActionRenderer:
    """Converts Rich components to Streamlit components"""

    def __init__(self):
        self.content_generator = ActionContentGenerator(enable_truncation=False)

    def render_rich_content(self, content) -> None:
        """Render Rich content in Streamlit"""
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, Panel):
            title = getattr(content, "title", None)
            if title:
                st.markdown(f"**{title}**")
            # Extract content from panel
            panel_content = str(content.renderable)
            st.markdown(panel_content)
        elif isinstance(content, Table):
            self._render_rich_table(content)
        elif isinstance(content, Syntax):
            language = getattr(content, "lexer_name", "text")
            code = str(content.code)
            st.code(code, language=language)
        elif isinstance(content, Markdown):
            st.markdown(str(content.markup))
        else:
            st.text(str(content))

    def _render_rich_table(self, table: Table) -> None:
        """Convert Rich Table to Streamlit dataframe"""
        try:
            # Extract table data
            if not table.columns:
                return

            # Get column headers
            headers = [col.header for col in table.columns]

            # Get row data
            rows = []
            for row in table.rows:
                row_data = []
                for cell in row:
                    # Convert cell content to string
                    if hasattr(cell, "plain"):
                        row_data.append(cell.plain)
                    else:
                        row_data.append(str(cell))
                rows.append(row_data)

            if rows:
                import pandas as pd

                df = pd.DataFrame(rows, columns=headers)
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            logger.debug(f"Failed to render Rich table: {e}")
            st.text(str(table))


class StreamlitChatbot:
    """Main Streamlit Chatbot class that wraps Datus CLI components"""

    def __init__(self):
        self.action_renderer = StreamlitActionRenderer()
        self.collapsible_generator = CollapsibleActionContentGenerator()
        self.session_manager = SessionManager()

        # Get server host and port from Streamlit config with fallback
        self.server_host = st.get_option("server.address") or "localhost"
        self.server_port = st.get_option("server.port") or 8501

        # Initialize session state with defaults
        defaults = {
            "messages": [],
            "current_actions": [],
            "chat_session_initialized": False,
            "cli_instance": None,
            "rendered_action_ids": set(),
            "current_chat_id": None,
            "subagent_name": None,
            "view_session_id": None,
            "session_readonly_mode": False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

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

    def _get_session_id(self) -> Optional[str]:
        """Unified session ID retrieval: try session_state first, then node"""
        if sid := st.session_state.get("current_session_id"):
            return sid
        return self.get_current_session_id()

    def setup_config(
        self, config_path: str = "conf/agent.yml", namespace: str = None, catalog: str = "", database: str = ""
    ) -> bool:
        """Setup agent configuration by initializing real DatusCLI"""
        # Check if already initialized to prevent repeated initialization
        if self.cli is not None:
            logger.info("CLI already initialized, skipping...")
            return True

        try:
            # Create CLI arguments
            args = create_cli_args(config_path, namespace, catalog, database=database)

            # Initialize real DatusCLI
            self.cli = DatusCLI(args)

            # Set Streamlit mode flag to skip interactive prompts
            self.cli.streamlit_mode = True

            st.session_state.chat_session_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            logger.error(f"Configuration loading error: {e}")
            return False

    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with configuration information"""
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

                # List all sessions
                all_sessions = self.session_manager.list_sessions()
                if all_sessions:
                    # Get session info and sort by modified time
                    session_infos = []
                    for sid in all_sessions:
                        info = self.session_manager.get_session_info(sid)
                        if info.get("exists"):
                            session_infos.append(info)

                    session_infos.sort(key=lambda x: x.get("file_modified", 0), reverse=True)

                    # Display recent 10 sessions
                    st.caption(f"Showing {min(len(session_infos), 10)} of {len(session_infos)} sessions")

                    for info in session_infos[:10]:
                        self._render_session_item(info)
                else:
                    st.caption("No saved sessions yet")

                # Report Issue section
                st.markdown("---")

                # Get session ID using unified method
                session_id = self._get_session_id()

                # Render Report Issue button
                import streamlit.components.v1 as components

                components.html(self._generate_report_issue_html(session_id), height=150)

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
        """Get list of available model names"""
        if not self.cli or not hasattr(self.cli.agent_config, "models"):
            return []

        try:
            return list(self.cli.agent_config.models.keys())
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def get_current_chat_model(self) -> str:
        """Get current chat model from configuration"""
        try:
            config_path = st.session_state.get("startup_config_path", "conf/agent.yml")
            config = _load_config_cached(config_path)
            chat_model = config.get("agent", {}).get("nodes", {}).get("chat", {}).get("model", "")
            if chat_model:
                return chat_model
            available_models = self.get_available_models()
            return available_models[0] if available_models else "unknown"
        except Exception as e:
            logger.error(f"Failed to get current chat model: {e}")
            return "unknown"

    def clear_chat(self):
        """Clear chat history and session"""
        st.session_state.messages = []
        st.session_state.current_actions = []
        st.session_state.rendered_action_ids = set()
        st.session_state.current_chat_id = None

        if self.cli and self.cli.chat_commands:
            self.cli.chat_commands.cmd_clear_chat("")

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages from a session stored in SQLite.

        Args:
            session_id: Session ID to load messages from

        Returns:
            List of message dictionaries with role, content, and timestamp
        """
        import json
        import sqlite3

        messages = []
        db_path = os.path.join(os.path.expanduser("~/.datus/sessions"), f"{session_id}.db")

        if not os.path.exists(db_path):
            logger.warning(f"Session database not found: {db_path}")
            return messages

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT message_data, created_at
                    FROM agent_messages
                    WHERE session_id = ?
                    ORDER BY created_at
                    """,
                    (session_id,),
                )

                for message_data, created_at in cursor.fetchall():
                    try:
                        message_json = json.loads(message_data)
                        role = message_json.get("role", "")
                        content = message_json.get("content", "")

                        # Only include user and assistant messages
                        if role in ["user", "assistant"]:
                            messages.append({"role": role, "content": content, "timestamp": created_at})
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Skipping malformed message: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to load session messages for {session_id}: {e}")

        return messages

    def get_current_session_id(self) -> Optional[str]:
        """
        Get the current session ID from the active chat node.

        Returns:
            Session ID if available, None otherwise
        """
        if self.cli and self.cli.chat_commands:
            # Prefer current_node over chat_node (for subagent support)
            node = self.cli.chat_commands.current_node or self.cli.chat_commands.chat_node
            if node:
                return node.session_id
        return None

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

        # Create benchmark directory
        benchmark_dir = os.path.expanduser(f"~/.datus/benchmark/{subagent_name}")
        os.makedirs(benchmark_dir, exist_ok=True)

        # CSV file path
        csv_path = os.path.join(benchmark_dir, "success_story.csv")

        # Prepare row data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "session_link": session_link,
            "session_id": session_id,
            "subagent_name": subagent_name,
            "user_message": user_message,
            "sql": sql,
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
        """Extract SQL and clean response from actions using existing logic"""
        if not actions:
            return None, None

        final_action = actions[-1]
        if not (
            final_action.output
            and isinstance(final_action.output, dict)
            and final_action.status == ActionStatus.SUCCESS
        ):
            return None, None

        sql = final_action.output.get("sql")
        response = final_action.output.get("response")

        # Extract SQL and output using ChatCommands
        extracted_sql, extracted_output = None, response
        if self.cli and self.cli.chat_commands:
            extracted_sql, extracted_output = self.cli.chat_commands._extract_sql_and_output_from_content(response)
            sql = sql or extracted_sql

        # Determine clean output
        if sql:
            return sql, extracted_output or response

        if isinstance(extracted_output, dict):
            return None, extracted_output.get("raw_output", str(extracted_output))

        try:
            import ast

            response_dict = ast.literal_eval(response)
            if isinstance(response_dict, dict):
                return None, response_dict.get("raw_output", response)
        except (ValueError, SyntaxError):
            pass

        return None, response

    def _render_session_item(self, info: dict) -> None:
        """Render a single session item in sidebar"""
        sid_short = info["session_id"][:8]
        with st.expander(f"📝 {sid_short}...", expanded=False):
            st.caption(f"**Created:** {info.get('created_at', 'N/A')}")
            st.caption(f"**Messages:** {info.get('message_count', 0)}")
            latest_msg = info.get("latest_user_message", "")
            if latest_msg:
                st.caption(f"**Latest:** {latest_msg[:50]}...")
            if st.button("🔗 Load Session", key=f"load_{info['session_id']}", use_container_width=True):
                st.query_params.update({"session": info["session_id"]})
                st.rerun()

    def _generate_report_issue_html(self, session_id: Optional[str] = None) -> str:
        """Generate Report Issue button HTML with JavaScript"""
        if session_id:
            session_link = f"http://{self.server_host}:{self.server_port}?session={session_id}"
            return f"""
            <div style="width: 100%;">
                <button id="reportIssueBtn" style="width: 100%; padding: 0.5rem 1rem;
                    background-color: #ff4b4b; color: white; border: none; border-radius: 0.5rem;
                    cursor: pointer; font-size: 1rem; font-weight: 500; transition: all 0.3s ease;">
                    🐛 Report Issue</button>
                <div id="feedbackMsg" style="width: 100%; margin-top: 0.5rem; padding: 0.75rem;
                    border-radius: 0.5rem; font-size: 0.875rem; display: none;
                    transition: all 0.3s ease; text-align: center; box-sizing: border-box;
                    min-height: 3rem; line-height: 1.5;"></div>
            </div>
            <script>
            document.getElementById('reportIssueBtn').addEventListener('click', function() {{
                const sessionLink = '{session_link}';
                const btn = this;
                const feedbackMsg = document.getElementById('feedbackMsg');
                const originalHTML = btn.innerHTML;
                const originalBgColor = btn.style.backgroundColor;

                const textArea = document.createElement('textarea');
                textArea.value = sessionLink;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();

                try {{
                    const successful = document.execCommand('copy');
                    if (successful) {{
                        btn.innerHTML = '✓ Copied!';
                        btn.style.backgroundColor = '#00c853';
                        feedbackMsg.innerHTML = '✓ Session link copied to clipboard';
                        feedbackMsg.style.backgroundColor = '#d4edda';
                        feedbackMsg.style.color = '#155724';
                        feedbackMsg.style.border = '1px solid #c3e6cb';
                        feedbackMsg.style.display = 'block';
                        setTimeout(() => {{
                            btn.innerHTML = originalHTML;
                            btn.style.backgroundColor = originalBgColor;
                            feedbackMsg.style.display = 'none';
                        }}, 3000);
                    }} else if (navigator.clipboard && navigator.clipboard.writeText) {{
                        navigator.clipboard.writeText(sessionLink)
                            .then(() => {{
                                btn.innerHTML = '✓ Copied!';
                                btn.style.backgroundColor = '#00c853';
                                feedbackMsg.innerHTML = '✓ Session link copied to clipboard';
                                feedbackMsg.style.backgroundColor = '#d4edda';
                                feedbackMsg.style.color = '#155724';
                                feedbackMsg.style.border = '1px solid #c3e6cb';
                                feedbackMsg.style.display = 'block';
                                setTimeout(() => {{
                                    btn.innerHTML = originalHTML;
                                    btn.style.backgroundColor = originalBgColor;
                                    feedbackMsg.style.display = 'none';
                                }}, 3000);
                            }})
                            .catch(() => {{
                                feedbackMsg.innerHTML = '⚠ Failed to copy. Link: ' + sessionLink;
                                feedbackMsg.style.backgroundColor = '#fff3cd';
                                feedbackMsg.style.color = '#856404';
                                feedbackMsg.style.border = '1px solid #ffeaa7';
                                feedbackMsg.style.display = 'block';
                                feedbackMsg.style.wordBreak = 'break-all';
                                setTimeout(() => {{
                                    feedbackMsg.style.display = 'none';
                                }}, 5000);
                            }});
                    }} else {{
                        feedbackMsg.innerHTML = '⚠ Copy not supported. Link: ' + sessionLink;
                        feedbackMsg.style.backgroundColor = '#fff3cd';
                        feedbackMsg.style.color = '#856404';
                        feedbackMsg.style.border = '1px solid #ffeaa7';
                        feedbackMsg.style.display = 'block';
                        feedbackMsg.style.wordBreak = 'break-all';
                        setTimeout(() => {{
                            feedbackMsg.style.display = 'none';
                        }}, 5000);
                    }}
                }} catch (err) {{
                    console.error('Copy failed:', err);
                    feedbackMsg.innerHTML = '⚠ Copy failed. Link: ' + sessionLink;
                    feedbackMsg.style.backgroundColor = '#fff3cd';
                    feedbackMsg.style.color = '#856404';
                    feedbackMsg.style.border = '1px solid #ffeaa7';
                    feedbackMsg.style.display = 'block';
                    feedbackMsg.style.wordBreak = 'break-all';
                    setTimeout(() => {{
                        feedbackMsg.style.display = 'none';
                    }}, 5000);
                }}

                document.body.removeChild(textArea);
            }});
            </script>
            """
        else:
            return """
            <div style="width: 100%;">
                <button id="reportIssueBtn" style="width: 100%; padding: 0.5rem 1rem;
                    background-color: #ff4b4b; color: white; border: none; border-radius: 0.5rem;
                    cursor: pointer; font-size: 1rem; font-weight: 500; transition: all 0.3s ease;">
                    🐛 Report Issue</button>
                <div id="feedbackMsg" style="width: 100%; margin-top: 0.5rem; padding: 0.75rem;
                    border-radius: 0.5rem; font-size: 0.875rem; display: none;
                    transition: all 0.3s ease; text-align: center; box-sizing: border-box;
                    min-height: 3rem; line-height: 1.5;"></div>
            </div>
            <script>
            document.getElementById('reportIssueBtn').addEventListener('click', function() {
                const feedbackMsg = document.getElementById('feedbackMsg');
                feedbackMsg.innerHTML = 'ℹ No active session. Please run a query first.';
                feedbackMsg.style.backgroundColor = '#d4edda';
                feedbackMsg.style.color = '#155724';
                feedbackMsg.style.border = '1px solid #c3e6cb';
                feedbackMsg.style.display = 'block';
                setTimeout(() => {
                    feedbackMsg.style.display = 'none';
                }, 3000);
            });
            </script>
            """

    def _get_available_subagents(self):
        """Get list of available subagents from agent config, excluding 'chat' (default)."""
        if not self.cli or not hasattr(self.cli.agent_config, "agentic_nodes"):
            return {}

        # Get all agentic_nodes except 'chat' since it's the default
        agentic_nodes = self.cli.agent_config.agentic_nodes
        return {name: config for name, config in agentic_nodes.items() if name != "chat"}

    def _show_available_subagents(self):
        """Show available subagents with dynamic routing."""
        available_subagents = self._get_available_subagents()

        if not available_subagents:
            return

        with st.expander("🔧 Access Specialized Subagents", expanded=False):
            st.markdown("**Available specialized subagents:**")

            # Current URL
            current_host = "http://localhost:8501"

            # Display each available subagent
            for subagent_name, subagent_config in available_subagents.items():
                model_name = subagent_config.get("model", "unknown")
                system_prompt = subagent_config.get("system_prompt", "general")
                tools = subagent_config.get("tools", "")
                workspace_root = subagent_config.get("workspace_root")

                # Create columns for better layout
                col1, col2 = st.columns([3, 1])

                with col1:
                    subagent_url = f"{current_host}/?subagent={subagent_name}"
                    st.markdown(f"**{subagent_name.title()} Subagent**: `{subagent_url}`")

                    # Show subagent details
                    details = [f"Model: {model_name}", f"Prompt: {system_prompt}"]
                    if workspace_root:
                        details.append(f"Workspace: {workspace_root}")
                    if "context_search_tools." in tools:
                        # Show specific tools if they're specified
                        specific_tools = [t.strip() for t in tools.split(",") if "context_search_tools." in t]
                        if specific_tools:
                            tool_names = []
                            for tool in specific_tools:
                                if tool.endswith(".*"):
                                    tool_names.append("all context tools")
                                else:
                                    tool_names.append(tool.split(".")[-1])
                            details.append(f"Context Tools: {', '.join(tool_names)}")

                    st.caption(" | ".join(details))

                with col2:
                    if st.button(f"🚀 Use {subagent_name}", key=f"switch_{subagent_name}"):
                        st.query_params.update({"subagent": subagent_name})
                        st.rerun()

            st.markdown("---")
            st.info("💡 **Tip**: Bookmark subagent URLs for direct access!")

    def display_sql_with_copy(self, sql: str):
        """Display SQL with syntax highlighting and save button"""
        if not sql:
            return

        st.markdown("### 🔧 Generated SQL")

        # Display SQL with syntax highlighting
        st.code(sql, language="sql")

        # Get last user message for save functionality
        user_msg = ""
        if st.session_state.messages:
            user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            if user_msgs:
                user_msg = user_msgs[-1]

        # Save button (only show if not in readonly mode)
        if not st.session_state.session_readonly_mode:
            # Create unique ID for this SQL block
            sql_id = hashlib.md5(sql.encode()).hexdigest()[:8]

            if st.button("👍 Success", key=f"save_{sql_id}", help="Save this query as a success story"):
                self.save_success_story(sql, user_msg)

    def display_markdown_response(self, response: str):
        """Display clean response as formatted markdown"""
        if not response:
            return

        st.markdown("### 💬 AI Response")
        st.markdown(response)

    def render_action_history(self, actions: List[ActionHistory], chat_id: str = None):
        """Render action history using existing ActionContentGenerator"""
        if not actions:
            return

        chat_id = chat_id or "default"

        # Use expander without key parameter (not supported in this Streamlit version)
        with st.expander("🔍 Execution Step Details", expanded=False):
            content_generator = ActionContentGenerator(enable_truncation=False)

            for action in actions:
                # Use container without key parameter (not supported in this Streamlit version)
                with st.container():
                    # Action header
                    dot = content_generator._get_action_dot(action)

                    # Format title similar to CLI version
                    if action.role == ActionRole.TOOL:
                        function_name = "unknown"
                        if action.input and isinstance(action.input, dict):
                            function_name = action.input.get("function_name", "unknown")
                        title = f"{dot} Tool call - {function_name}"
                    else:
                        title = f"{dot} {action.messages}"

                    # Show action with expander
                    with st.expander(title, expanded=False):
                        # Show action details
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("**Input:**")
                            if action.input:
                                if isinstance(action.input, dict):
                                    st.json(action.input)
                                else:
                                    st.text(str(action.input))

                        with col2:
                            st.markdown("**Output:**")
                            if action.output:
                                if isinstance(action.output, dict):
                                    st.json(action.output)
                                else:
                                    st.text(str(action.output))

                        # Show timing info
                        if action.start_time:
                            st.caption(f"Start time: {action.start_time.strftime('%H:%M:%S')}")
                        if action.end_time:
                            duration = (action.end_time - action.start_time).total_seconds()
                            st.caption(f"Duration: {duration:.2f}s")

    def execute_chat(self, user_message: str) -> List[ActionHistory]:
        """Execute chat command and return actions"""
        if not self.cli or not self.cli.chat_commands:
            st.error("Please load configuration first!")
            return []

        try:
            # Clear any existing actions
            initial_actions_count = len(self.cli.actions.actions)
            if initial_actions_count > 0:
                logger.warning(f"Found {initial_actions_count} existing actions, clearing...")

            self.cli.actions.actions.clear()
            if hasattr(self.cli.actions, "_action_history"):
                self.cli.actions._action_history.clear()
            if hasattr(self.cli, "current_actions"):
                self.cli.current_actions = []

            # Execute chat command with subagent support - get directly from URL
            logger.info(f"Executing chat: {user_message} (subagent: {self.current_subagent})")
            self.cli.chat_commands.execute_chat_command(user_message, subagent_name=self.current_subagent)

            # Store session_id in session_state for sidebar access
            session_id = self.get_current_session_id()
            if session_id:
                st.session_state.current_session_id = session_id
                logger.info(f"Stored session_id in session_state: {session_id}")

            # Get all actions from this execution
            new_actions = self.cli.actions.actions.copy()
            logger.info(f"Chat completed with {len(new_actions)} actions")

            # Clear actions to prevent persistence
            self.cli.actions.actions.clear()
            return new_actions

        except Exception as e:
            st.error(f"Error executing chat command: {str(e)}")
            logger.error(f"Chat execution error: {e}")
            import traceback

            logger.error(f"Chat execution traceback: {traceback.format_exc()}")
            return []

    def run(self):
        """Main Streamlit app runner"""
        # Initialize logging for web interface
        if "log_manager_initialized" not in st.session_state:
            st.session_state.log_manager_initialized = True
            logger.info("Web chatbot logging initialized")

        # Page configuration
        st.set_page_config(
            page_title="Datus AI Chat Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={"Get Help": None, "Report a bug": None, "About": None},
        )

        # Hide deploy button and toolbar
        st.set_option("client.toolbarMode", "viewer")

        # Update session_id in session_state at the beginning of each render
        # This ensures sidebar always has the latest session_id
        current_session_id = self.get_current_session_id()
        if current_session_id:
            st.session_state.current_session_id = current_session_id

        # Custom CSS for chat styling
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
            # Only show available subagents when NOT in subagent mode
            self._show_available_subagents()

        # Render sidebar and get config status
        sidebar_state = self.render_sidebar()

        # Main chat interface
        if not sidebar_state["config_loaded"]:
            st.warning("⚠️ Please wait for configuration to load or check the sidebar")
            st.info("Configuration file contains database connections, model settings, etc.")
            return

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show SQL and actions if available
                if "sql" in message and message["sql"]:
                    self.display_sql_with_copy(message["sql"])

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
                    with st.spinner("Processing your query..."):
                        # Execute chat and get actions
                        actions = self.execute_chat(prompt)

                        # Extract SQL and response
                        sql, response = self.extract_sql_and_response(actions)

                        # Display response
                        if response:
                            self.display_markdown_response(response)
                        else:
                            st.markdown(
                                "Sorry, unable to generate a valid response. "
                                "Please check execution details for more information."
                            )

                        # Display SQL if available
                        if sql:
                            self.display_sql_with_copy(sql)

                        # Display action history for current conversation - only once per chat
                        action_render_id = f"{chat_id}_actions"
                        if actions and action_render_id not in st.session_state.rendered_action_ids:
                            self.render_action_history(actions, chat_id)
                            st.session_state.rendered_action_ids.add(action_render_id)

                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": response or "Unable to generate valid response",
                        "sql": sql,
                        "actions": actions,
                        "chat_id": chat_id,
                    }
                    st.session_state.messages.append(assistant_message)

                    # Trigger rerun to update sidebar with new session_id
                    # This ensures the Report Issue button gets the session_id immediately
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


def run_web_interface(args):
    """Launch Streamlit web interface"""
    import os
    import subprocess
    import sys

    try:
        # Get the path to the web chatbot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        web_chatbot_path = os.path.join(current_dir, "web_chatbot.py")

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
