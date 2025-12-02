# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
UI components and rendering utilities for web interface.

Handles all display-related functionality including:
- Session items rendering
- Report issue button generation
- Subagent listings
- SQL display with copy functionality
- Action history rendering
"""

import hashlib
from typing import List, Optional

import streamlit as st
import structlog

from datus.cli.action_history_display import ActionContentGenerator
from datus.schemas.action_history import ActionHistory, ActionRole

logger = structlog.get_logger(__name__)


class UIComponents:
    """Manages UI component rendering for the web interface."""

    def __init__(self, server_host: str, server_port: int):
        """
        Initialize UI components.

        Args:
            server_host: Server hostname for generating URLs
            server_port: Server port for generating URLs
        """
        self.server_host = server_host
        self.server_port = server_port

    @staticmethod
    def safe_update_query_params(new_params: dict):
        """Safely update query params while preserving hide_sidebar parameter"""
        current_params = dict(st.query_params)
        hide_sidebar_value = current_params.get("hide_sidebar")

        # Update with new params
        current_params.update(new_params)

        # Restore hide_sidebar if it was present
        if hide_sidebar_value:
            current_params["hide_sidebar"] = hide_sidebar_value

        st.query_params.clear()
        st.query_params.update(current_params)

    def render_session_item(self, info: dict) -> None:
        """Render a single session item in sidebar."""
        sid_short = info["session_id"][:8]
        with st.expander(f"📝 {sid_short}...", expanded=False):
            st.caption(f"**Created:** {info.get('created_at', 'N/A')}")
            st.caption(f"**Messages:** {info.get('message_count', 0)}")
            latest_msg = info.get("latest_user_message", "")
            if latest_msg:
                st.caption(f"**Latest:** {latest_msg[:50]}...")
            if st.button("🔗 Load Session", key=f"load_{info['session_id']}", use_container_width=True):
                self.safe_update_query_params({"session": info["session_id"]})
                st.rerun()

    def generate_report_issue_html(self, session_id: Optional[str] = None) -> str:
        """Generate Report Issue button HTML with JavaScript."""
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

    def show_available_subagents(self, agent_config) -> None:
        """Show available subagents with dynamic routing."""
        if not agent_config or not hasattr(agent_config, "agentic_nodes"):
            return

        # Get all agentic_nodes except 'chat' since it's the default
        agentic_nodes = agent_config.agentic_nodes
        available_subagents = {name: config for name, config in agentic_nodes.items() if name != "chat"}

        if not available_subagents:
            return

        with st.expander("🔧 Access Specialized Subagents", expanded=True):
            st.markdown("**Available specialized subagents:**")

            # Display each available subagent
            for subagent_name, subagent_config in available_subagents.items():
                model_name = subagent_config.get("model", "unknown")
                system_prompt = subagent_config.get("system_prompt", "general")
                tools = subagent_config.get("tools", "")
                workspace_root = subagent_config.get("workspace_root")

                # Create columns for better layout
                col1, col2 = st.columns([3, 1])

                with col1:
                    subagent_url = f"http://{self.server_host}:{self.server_port}/?subagent={subagent_name}"
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
                        UIComponents.safe_update_query_params({"subagent": subagent_name})
                        st.rerun()

            st.markdown("---")
            st.info("💡 **Tip**: Bookmark subagent URLs for direct access!")

    def display_sql_with_copy(self, sql: str, user_message: str, readonly_mode: bool, save_callback) -> None:
        """Display SQL with syntax highlighting and save button."""
        if not sql:
            return

        st.markdown("### 🔧 Generated SQL")

        # Display SQL with syntax highlighting
        st.code(sql, language="sql")

        # Save button (only show if not in readonly mode)
        if not readonly_mode:
            # Create unique ID using current timestamp to ensure uniqueness across duplicate SQLs
            import time

            sql_hash = hashlib.md5(sql.encode()).hexdigest()[:8]
            timestamp = str(time.time()).replace(".", "_")  # Use timestamp for uniqueness
            unique_key = f"save_{sql_hash}_{timestamp}"

            if st.button("👍 Success", key=unique_key, help="Save this query as a success story"):
                save_callback(sql, user_message)

    def display_markdown_response(self, response: str) -> None:
        """Display clean response as formatted markdown."""
        if not response:
            return

        st.markdown("### 💬 AI Response")
        st.markdown(response)

    def render_action_history(self, actions: List[ActionHistory], chat_id: str = None, expanded: bool = False) -> None:
        """Render complete action history with full details.

        Args:
            actions: List of ActionHistory objects to render
            chat_id: Chat ID for the conversation
            expanded: Whether to expand the details by default (True for session page, False for normal page)
        """
        if not actions:
            return

        chat_id = chat_id or "default"

        # Display complete execution history
        with st.expander(f"🔍 View Full Execution Details ({len(actions)} steps)", expanded=expanded):
            st.caption("Complete execution trace with all intermediate steps")

            content_generator = ActionContentGenerator(enable_truncation=False)

            for i, action in enumerate(actions, 1):
                with st.container():
                    # Action header with status indicator
                    dot = content_generator._get_action_dot(action)

                    # Format title based on action type
                    if action.role == ActionRole.TOOL:
                        function_name = "unknown"
                        if action.input and isinstance(action.input, dict):
                            function_name = action.input.get("function_name", "unknown")
                        title = f"Step {i}: {dot} Tool call - {function_name}"
                    else:
                        title = f"Step {i}: {dot} {action.messages}"

                    # Nested expander for each action
                    with st.expander(title, expanded=False):
                        # Two-column layout for input/output
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("**Input:**")
                            if action.input:
                                if isinstance(action.input, dict):
                                    st.json(action.input)
                                else:
                                    st.text(str(action.input))
                            else:
                                st.caption("(no input)")

                        with col2:
                            st.markdown("**Output:**")
                            if action.output:
                                if isinstance(action.output, dict):
                                    st.json(action.output)
                                else:
                                    st.text(str(action.output))
                            else:
                                st.caption("(no output)")

                        # Timing information
                        if action.start_time and action.end_time:
                            duration = (action.end_time - action.start_time).total_seconds()
                            st.caption(
                                f"⏱️ Started: {action.start_time.strftime('%H:%M:%S')} | Duration: {duration:.2f}s"
                            )
                        elif action.start_time:
                            st.caption(f"⏱️ Started: {action.start_time.strftime('%H:%M:%S')}")

                    # Add divider between actions
                    if i < len(actions):
                        st.divider()
