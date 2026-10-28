# BI Dashboard 节点（旧版兼容）

`GenDashboardAgenticNode` 暂时保留用于代码兼容，但已不再是可用的内置或自定义
subagent。`task` 工具、`/agent`、自动补全和 agent API 都不会暴露它。

外部 BI 操作现在由主 agent 直接通过已安装的 BI plugin 及其 bundled authoring
skill 完成。以 Superset 为例，配置 `agent.plugins.superset` 后，直接向主 agent
提出创建或查询资产的任务；主 agent 会在内部选择并运行合适的 plugin 命令。

不要使用 `task(type="gen_dashboard")`、`/gen_dashboard`，也不要创建
`node_class: gen_dashboard` 的自定义 subagent。

Plugin 方式的 Superset authoring 流程参见
[数据工程快速开始](../getting_started/data_engineering_quickstart.zh.md)。
