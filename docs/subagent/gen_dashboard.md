# BI Dashboard Node (Legacy)

`GenDashboardAgenticNode` is retained temporarily for code compatibility, but
it is no longer an available built-in or custom subagent. It is not exposed by
the `task` tool, `/agent`, autocomplete, or the agent API.

External BI operations now run directly in the main agent through an installed
BI plugin and its bundled authoring skill. For Superset, configure
`agent.plugins.superset`, then ask the main agent to create or inspect the
assets. The main agent selects and runs the appropriate plugin commands
internally.

Do not use `task(type="gen_dashboard")`, `/gen_dashboard`, or a custom
subagent whose `node_class` is `gen_dashboard`.

See the
[Data Engineering Quickstart](../getting_started/data_engineering_quickstart.md)
for the plugin-based Superset authoring workflow.
