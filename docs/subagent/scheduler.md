# Scheduler Node (Legacy)

`SchedulerAgenticNode` is retained temporarily for code compatibility, but it
is no longer an available built-in or custom subagent. It is not exposed by the
`task` tool, `/agent`, autocomplete, or the agent API.

Scheduling operations now run directly in the main agent through an installed
scheduler plugin and its bundled skill. For Airflow, configure
`agent.plugins.airflow`, then ask the main agent to perform the operation. The
main agent selects and runs the appropriate plugin commands internally.

Do not use `task(type="scheduler")`, `/scheduler`, or a custom subagent whose
`node_class` is `scheduler`.

See the
[Data Engineering Quickstart](../getting_started/data_engineering_quickstart.md)
for the plugin-based Airflow workflow.
