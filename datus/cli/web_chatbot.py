"""
Streamlit Chatbot for Datus Agent

This module provides a web-based chatbot interface using Streamlit,
maximizing reuse of existing Datus CLI components including:
- DatusCLI for real CLI functionality
- ChatCommands for chat processing
- ActionHistoryDisplay for execution visualization
- CollapsibleActionContentGenerator for detail views
"""

from argparse import Namespace
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
        config_path = parse_config_path(config_path)
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Look for namespace configuration
        if "agent" in config and "namespace" in config["agent"]:
            return list(config["agent"]["namespace"].keys())
        elif "namespace" in config:
            return list(config["namespace"].keys())
        else:
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


def get_storage_path_from_config(config_path: str) -> str:
    """Extract storage base_path from configuration file"""
    try:
        import os

        import yaml

        config_path = parse_config_path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        storage_config = config.get("agent", {}).get("storage", {})
        base_path = storage_config.get("base_path", "data")

        # Expand user path if needed
        return os.path.expanduser(base_path)

    except Exception as e:
        logger.warning(f"Failed to read storage path from config: {e}")
        return "data"  # fallback to default


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

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_actions" not in st.session_state:
            st.session_state.current_actions = []
        if "chat_session_initialized" not in st.session_state:
            st.session_state.chat_session_initialized = False
        if "cli_instance" not in st.session_state:
            st.session_state.cli_instance = None
        if "rendered_action_ids" not in st.session_state:
            st.session_state.rendered_action_ids = set()
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None
        if "subagent_name" not in st.session_state:
            st.session_state.subagent_name = None

    @property
    def cli(self) -> DatusCLI:
        """Get CLI instance from session state"""
        return st.session_state.cli_instance

    @cli.setter
    def cli(self, value):
        """Set CLI instance in session state"""
        st.session_state.cli_instance = value

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

            # Note: Removed vector DB cleanup - now using correct storage path to reuse existing DBs

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
                current_subagent = self._get_current_subagent_from_url()
                st.session_state.subagent_name = current_subagent

                # Current subagent info
                if st.session_state.subagent_name:
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

                # Debug section
                st.markdown("---")
                st.subheader("🔍 Debug Info")
                with st.expander("Debug Details", expanded=False):
                    st.write("Query Params:", dict(st.query_params))
                    st.write("Startup Subagent:", st.session_state.get("startup_subagent_name"))
                    st.write("Current Subagent:", st.session_state.get("subagent_name"))

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
            # Read directly from config file to get nodes.chat.model
            import yaml

            config_path = st.session_state.get("startup_config_path", "conf/agent.yml")
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            chat_model = config.get("agent", {}).get("nodes", {}).get("chat", {}).get("model", "")
            if chat_model:
                return chat_model

            # Fallback to first available model
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

    def extract_sql_and_response(self, actions: List[ActionHistory]) -> Tuple[Optional[str], Optional[str]]:
        """Extract SQL and clean response from actions using existing logic"""
        if not actions:
            return None, None

        final_action = actions[-1]

        if (
            final_action.output
            and isinstance(final_action.output, dict)
            and final_action.status == ActionStatus.SUCCESS
        ):
            # Use existing extraction logic from ChatCommands
            sql = final_action.output.get("sql")
            response = final_action.output.get("response")

            # Use existing extraction method
            if self.cli and self.cli.chat_commands:
                extracted_sql, extracted_output = self.cli.chat_commands._extract_sql_and_output_from_content(response)
                sql = sql or extracted_sql
            else:
                extracted_sql, extracted_output = None, response

            # Determine clean output using existing logic
            clean_output = None
            if sql:
                clean_output = extracted_output or response
            elif isinstance(extracted_output, dict):
                clean_output = extracted_output.get("raw_output", str(extracted_output))
            else:
                try:
                    import ast

                    response_dict = ast.literal_eval(response)
                    clean_output = (
                        response_dict.get("raw_output", response) if isinstance(response_dict, dict) else response
                    )
                except (ValueError, SyntaxError):
                    clean_output = response

            return sql, clean_output

        return None, None

    def _get_current_subagent_from_url(self):
        """Get current subagent directly from URL query parameters."""
        try:
            query_params = st.query_params
            return query_params.get("subagent", None)
        except Exception:
            # Fallback for older Streamlit versions
            try:
                query_params = st.experimental_get_query_params()
                if "subagent" in query_params:
                    return query_params["subagent"][0]
            except Exception:
                pass
        return None

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
        """Display SQL with syntax highlighting"""
        if not sql:
            return

        st.markdown("### 🔧 Generated SQL")

        # Display SQL with syntax highlighting (Streamlit has built-in copy functionality)
        st.code(sql, language="sql")

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
            subagent_name = self._get_current_subagent_from_url()
            logger.info(f"Executing chat: {user_message} (subagent: {subagent_name})")
            self.cli.chat_commands.execute_chat_command(user_message, subagent_name=subagent_name)

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

        # Title and description with subagent support - detect directly from URL
        current_subagent = self._get_current_subagent_from_url()
        if current_subagent:
            st.title(f"🤖 Datus AI Chat Assistant - {current_subagent.title()}")
            st.caption(f"Specialized {current_subagent} subagent for SQL generation - Natural Language to SQL")
            # Don't show subagent selection when already in subagent mode
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

                # Skip displaying historical actions to prevent duplication

        # Chat input
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
