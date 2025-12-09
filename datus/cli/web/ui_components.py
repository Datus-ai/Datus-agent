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
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import structlog
from pandas.core.dtypes.common import is_numeric_dtype

from datus.cli.action_history_display import ActionContentGenerator
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.schemas.action_history import ActionHistory, ActionRole
from datus.schemas.node_models import ExecuteSQLResult
from datus.schemas.visualization import VisualizationInput, VisualizationOutput
from datus.tools.llms_tools.visualization_tool import VisualizationTool

logger = structlog.get_logger(__name__)


def _sql_union_id(sql: str) -> str:
    if not sql:
        return ""

    history = st.session_state.setdefault("_sql_render_history", [])
    pointer = st.session_state.get("_sql_render_pointer", 0)

    if pointer < len(history) and history[pointer]["sql"] == sql:
        sql_id = history[pointer]["id"]
    else:
        sql_hash = hashlib.md5(sql.encode()).hexdigest()[:8]
        sql_id = f"{sql_hash}_{pointer}"
        entry = {"sql": sql, "id": sql_id}
        if pointer < len(history):
            history[pointer] = entry
        else:
            history.append(entry)

    st.session_state["_sql_render_pointer"] = pointer + 1
    return sql_id


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
        cli_instance = st.session_state.get("cli_instance")
        self.agent_config: AgentConfig | None = getattr(cli_instance, "agent_config", None) if cli_instance else None

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

    def display_success_button(self, sql: str, user_message: str, sql_id, save_callback):
        # Create unique ID for this SQL block
        if st.button("👍 Success", key=f"save_{sql_id}", help="Save this query as a success story"):
            save_callback(sql, user_message)

    @staticmethod
    def reset_sql_display_state():
        """Reset SQL render pointer to ensure stable widget keys each rerun."""
        st.session_state["_sql_render_pointer"] = 0

    def _get_visualization_cache(self) -> Dict[str, Dict[str, Any]]:
        """Return visualization recommendations stored in session state."""
        return st.session_state.setdefault("visualization_cache", {})

    def _get_visualization_tool(self) -> VisualizationTool:
        """Return (and cache) a VisualizationTool instance backed by the active LLM."""
        cache_key = "_visualization_tool_instance"
        agent_config = self.agent_config
        if agent_config is None:
            cli_instance = st.session_state.get("cli_instance")
            agent_config = getattr(cli_instance, "agent_config", None) if cli_instance else None

        cached_tool = st.session_state.get(cache_key)
        current_config_id = id(agent_config) if agent_config else None
        cached_config_id = getattr(cached_tool, "_agent_config_id", None) if cached_tool else None

        if cached_tool and current_config_id == cached_config_id:
            return cached_tool

        model = None
        if agent_config:
            try:
                model = LLMBaseModel.create_model(agent_config=agent_config)
            except Exception as exc:
                logger.warning(f"Unable to initialize visualization model: {exc}")

        tool = VisualizationTool(agent_config=agent_config, model=model)
        tool._agent_config_id = current_config_id  # type: ignore[attr-defined]
        st.session_state[cache_key] = tool
        return tool

    def _ensure_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        """Best-effort conversion of incoming result data to a pandas DataFrame."""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if hasattr(data, "to_pandas"):
            try:
                return data.to_pandas()
            except Exception as exc:
                logger.warning(f"Failed to convert data via to_pandas(): {exc}")
        try:
            return pd.DataFrame(data)
        except Exception as exc:
            logger.error(f"Unable to convert result to DataFrame: {exc}")
            return None

    def _apply_chart_defaults(self, state_id: str, df: pd.DataFrame, result: VisualizationOutput) -> None:
        """Seed Streamlit widget defaults based on AI chart recommendation."""
        chart_key = f"chart_type_{state_id}"
        if result.chart_type:
            st.session_state[chart_key] = result.chart_type

        x_key = f"x_col_{state_id}"
        if result.x_col and result.x_col in df.columns:
            st.session_state[x_key] = result.x_col

        y_key = f"y_cols_{state_id}"
        valid_y = [col for col in result.y_cols if col in df.columns]
        if valid_y:
            st.session_state[y_key] = valid_y

    def pre_render_chart(self, state_id: str, data: Any):
        """Callback for AI chart preparation button."""
        cache = self._get_visualization_cache()
        df = self._ensure_dataframe(data)

        if df is None or df.empty:
            cache[state_id] = {
                "success": False,
                "error": "No rows available for visualization",
                "chart_type": "Unknown",
                "x_col": "",
                "y_cols": [],
                "reason": "Dataset is empty",
            }
            return

        try:
            viz_input = VisualizationInput(data=df)
        except Exception as exc:
            logger.error(f"Failed to prepare visualization input: {exc}")
            cache[state_id] = {
                "success": False,
                "error": str(exc),
                "chart_type": "Unknown",
                "x_col": "",
                "y_cols": [],
                "reason": "Unable to interpret dataset",
            }
            return

        tool = self._get_visualization_tool()
        try:
            result = tool.execute(viz_input)
        except Exception as exc:
            logger.error(f"Visualization tool execution failed: {exc}")
            cache[state_id] = {
                "success": False,
                "error": str(exc),
                "chart_type": "Unknown",
                "x_col": "",
                "y_cols": [],
                "reason": "Visualization analysis failed",
            }
            return

        cache[state_id] = result.model_dump()

        if result.success:
            self._apply_chart_defaults(state_id, df, result)

    def display_sql(self, sql, data: ExecuteSQLResult) -> str:
        if not sql:
            return ""
        sql_id = _sql_union_id(sql)
        state_id = hashlib.md5(sql.encode()).hexdigest()
        result_df = data.sql_return if isinstance(data.sql_return, pd.DataFrame) else None
        tab1, tab2, tab3 = st.tabs(["🔧Generated SQL", "📊Execute Result", "📈Chart"])
        with tab1:
            # Display SQL with syntax highlighting
            st.code(sql, language="sql")
        if data.success:
            with tab2:
                if result_df is not None and not result_df.empty:
                    st.dataframe(result_df)
                else:
                    st.warning("No data return")

            with tab3:
                if result_df is not None and not result_df.empty:
                    viz_result = self._get_visualization_cache().get(state_id)

                    if not (viz_result and viz_result.get("success")):
                        st.button(
                            label="Chart by AI",
                            key=f"prepare_chart_{sql_id}",
                            on_click=self.pre_render_chart,
                            args=[state_id, result_df],
                        )

                    if viz_result:
                        if viz_result.get("success"):
                            chart_type = viz_result.get("chart_type", "Bar Chart")
                            reason = viz_result.get("reason") or "AI provided a visualization suggestion."
                            st.caption(f"🤖 AI suggestion: **{chart_type}** — {reason}")
                            self.render_dynamic_chart_in_tab(result_df, state_id, chart_type)
                        else:
                            error_msg = viz_result.get("error") or viz_result.get("reason") or "Unknown error"
                            st.error(f"Chart suggestion failed: {error_msg}")
                else:
                    st.warning("No data return")
        else:
            with tab2:
                st.error("SQL execution failed")
                st.markdown(data.error)
            with tab3:
                st.error("SQL execution failed")
                st.markdown(data.error)
        return sql_id

    def render_dynamic_chart_in_tab(self, df: pd.DataFrame, state_id: str, chart_type: str):
        """
        Render a configurable dynamic chart inside Tab, and the configuration items are located in st.popover.

        Parameters:
            df (pd.DataFrame): DataFrame of SQL query results.
            state_id (str): Stable identifier used for widget state caching.
            chart_type (str): Suggested chart type from visualization tool.
        """
        if df.empty:
            st.error("The data is empty and the chart cannot be generated.")
            return

        # 1. Data column analysis
        all_cols = df.columns.tolist()
        numeric_cols = [col for col in all_cols if is_numeric_dtype(df[col])]

        chart_options = {"Bar Chart": "bar", "Line Chart": "line", "Scatter Plot": "scatter", "Pie Chart": "pie"}
        chart_key = f"chart_type_{state_id}"
        if chart_key not in st.session_state or st.session_state[chart_key] not in chart_options:
            default_chart = chart_type if chart_type in chart_options else list(chart_options.keys())[0]
            st.session_state[chart_key] = default_chart

        # --- Configuration panel (st.popover) ---
        with st.popover("⚙️ Chart Configuration"):
            st.write("**Select chart type and axis mapping**")
            selected_chart_name = st.selectbox("Chart Type", list(chart_options.keys()), key=chart_key)
            chart_func_name = chart_options[selected_chart_name]

            x_key = f"x_col_{state_id}"
            if x_key not in st.session_state or st.session_state[x_key] not in all_cols:
                st.session_state[x_key] = all_cols[0]
            x_col = st.selectbox("X-axis (dimension/grouping)", all_cols, key=x_key)

            y_key = f"y_cols_{state_id}"
            if y_key not in st.session_state:
                st.session_state[y_key] = numeric_cols[:2] if numeric_cols else []
            else:
                st.session_state[y_key] = [col for col in st.session_state[y_key] if col in numeric_cols]
            y_cols = st.multiselect(
                "Y-axis (indicator/value)",
                numeric_cols,
                key=y_key,
            )

        # Dynamic rendering logic
        if not y_cols:
            st.info("Please click the '⚙️ Configure Chart' button above to select at least one Y-axis indicator.")
            return

        # Special treatment for pie charts
        if chart_func_name == "pie":
            if len(y_cols) > 1:
                st.warning("Only one indicator (Y-axis) can be selected for pie charts.")

            y_col_single = y_cols[0]
            fig = px.pie(df, names=x_col, values=y_col_single, title=f"{y_col_single} by {x_col}")

        # Other charts (bar, line, scatter)
        else:
            # Convert wide table to long table (core processing of multiple Y axes)
            if len(y_cols) > 1:
                id_vars = [x_col]

                df_long = pd.melt(
                    df, id_vars=id_vars, value_vars=y_cols, var_name="IndicatorType", value_name="NumericalValue"
                )

                # Plotly unified call (long table after using Melt)
                fig_func = getattr(px, chart_func_name)
                fig = fig_func(
                    df_long,
                    x=x_col,
                    y="NumericalValue",
                    color="IndicatorType",
                    title=f"{selected_chart_name}：{', '.join(y_cols)} vs {x_col}",
                )
                fig.update_layout(barmode="group")
            else:
                # Single Y-axis, no Melt required
                fig_func = getattr(px, chart_func_name)
                fig = fig_func(df, x=x_col, y=y_cols[0], title=f"{selected_chart_name}: {y_cols[0]} vs {x_col}")

        # unified rendering
        st.plotly_chart(fig, config={"width": "content"})

    def display_download(
        self, sql: str, output_md: str, execute_result: ExecuteSQLResult, display_column, sql_id, download_callback
    ):
        if not sql:
            return
        if st.button("⏬ Download", key=f"download_{sql_id}", help="Prepare data for download"):
            download_callback(sql, output_md, execute_result, sql_id, display_column)

    def display_markdown_response(self, response: str) -> None:
        """Display clean response as formatted markdown."""
        if not response:
            return

        st.markdown("### 💬 AI Response")
        st.markdown(response)

    def render_action_item(
        self, chat_id: str, index: int, action: ActionHistory, content_generator: ActionContentGenerator
    ):
        with st.container():
            # Action header with status indicator
            dot = content_generator._get_action_dot(action)

            # Format title based on action type
            if action.role == ActionRole.TOOL:
                function_name = "unknown"
                if action.input and isinstance(action.input, dict):
                    function_name = action.input.get("function_name", "unknown")
                title = f"Step {index}: {dot} Tool call - {function_name}"
            else:
                title = f"Step {index}: {dot} {action.messages}"

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
                    st.caption(f"⏱️ Started: {action.start_time.strftime('%H:%M:%S')} | Duration: {duration:.2f}s")
                elif action.start_time:
                    st.caption(f"⏱️ Started: {action.start_time.strftime('%H:%M:%S')}")

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
                self.render_action_item(chat_id, i, action, content_generator)
