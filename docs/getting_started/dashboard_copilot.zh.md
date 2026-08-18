# Dashboard Copilot

Dashboard Copilot 从 BI 仪表盘构建项目的 reference SQL 和 metrics。流程由内置 `dashboard-bootstrap` skill 与用户选定的 BI plugin 驱动；Datus Agent 不再硬编码某个 BI 产品的 Dashboard API 或 SQL 编译逻辑。

`/bootstrap-bi` 继续作为兼容快捷入口保留。它只把请求转交给标准 chat/skill pipeline，不再启动旧 Picker、BI streams 或 Dashboard 专用 subagent 生成逻辑。

## 前置条件

开始前需要配置：

1. BI plugin 和 profile。该 plugin 的 bundled export skill 必须能够发现 Dashboard 和 query candidates，并以稳定 ID、SQL 文件、checksum、状态和逐查询脱敏 source identity 导出查询。
2. 将 Dashboard 使用的每个物理数据库配置为 Datus datasource。不需要 profile 级映射；Datus 根据每条查询的真实 BI 连接 identity 匹配，并且每轮只为当前 active datasource 对应的分区生成 metric。
3. 通过 `/model` 选定可用的 LLM。
4. 需要写入 metric 时，项目应使用可写的 Dosi semantic project。MetricFlow 和普通 OSI 项目在该流程中仅支持查询。

Superset plugin 已实现所需导出 contract。其他 BI plugin 只要通过自己的 bundled skill 提供相同能力，就可以接入，无需修改 Datus Agent。

## 启动流程

可以直接使用自然语言：

```text
使用 Superset prod profile，从 World Bank dashboard 构建 reference SQL 和 metrics。
```

也可以使用兼容快捷入口，并把范围作为自由文本附在命令后：

```text
/bootstrap-bi 使用 Superset prod profile 和 World Bank dashboard
```

两种方式进入完全相同的流程。slash command 只要求 Agent 加载 `dashboard-bootstrap`，没有独立的特殊实现。

## 选择与确认

Skill 会引导 Agent 完成：

1. 选择已安装的 BI plugin 和具名 profile。
2. 通过稳定 ID、URL 或无歧义名称选择 Dashboard。
3. 选择用于 reference SQL 的查询。
4. 独立选择用于 metric 初始化的查询。同一查询可以属于任一集合、两个集合或均不选择。
5. 检查 Generation Manifest，其中包含 plugin/profile、Dashboard、所选 query IDs、排除项、datasource 匹配状态、导出模式和歧义。

SQL 中存在聚合函数只是一种推荐信号，不再自动把 Dashboard 查询初始化为 metric。

默认情况下，Agent 展示 Generation Manifest 后会结束当前轮次。用户需要在下一条消息确认或修正；确认前不会导出 SQL，也不会生成 Knowledge Base artifact。

如确实需要跳过这个确认边界，必须显式说明，例如：

```text
/bootstrap-bi 使用 Superset prod profile 和 dashboard 42，自动执行并跳过确认
```

显式 auto-run 不会绕过系统权限确认。

## SQL 导出与 Context 构建

确认后，所选 BI plugin 负责导出 SQL。Datus 会根据已确认 manifest 校验 query identity、状态、文件位置和 checksum。

- 每条已确认的 reference query 都以完整原始 SQL 交给 `gen_sql_summary`。
- 已确认的 metric queries 先按唯一匹配到的 Datus datasource 分区，再按业务域分组，并以完整原始 SQL 交给 `semantic_modeling`。每轮只处理 active datasource 对应的分区。
- `semantic_modeling` 检查在线 schema，更新 Dosi semantic model 和 metrics，执行校验并 reconcile Knowledge Base。
- 一条路径失败不会隐式阻断另一条路径。

Plugin 负责访问 BI 和保证 SQL 忠实性。主 Agent 与 plugin 都不直接手写 reference SQL 索引或 Dosi artifact。

## 结果

最终报告会列出：

- plugin/profile 和 Dashboard identity；
- export directory 和 manifest；
- 成功、失败和跳过的 reference queries；
- 成功、失败和跳过的 semantic domains；
- builtin agents 返回的 reference SQL identifiers、semantic model 文件和 metric names；
- 最小安全重试范围。

新流程构建共享的项目 context，不再自动创建旧 `/bootstrap-bi` pipeline 生成的两个 Dashboard 专用 subagents。现有项目 agents 会正常使用生成后的 reference SQL 与 metric Knowledge Base。

## 失败处理

- Plugin 缺少导出能力：安装或升级兼容的 BI plugin。
- 存在多个匹配 profile 或 Dashboard：使用稳定 profile 名和 Dashboard ID 明确选择。
- SQL failed、partial、未选择或 checksum 不匹配：拒绝该查询，不允许 LLM 猜测重建。
- query source identity 缺失、证据不足或匹配不唯一：reference SQL 可以继续，只停止对应查询的 metric authoring。
- 查询匹配到非 active datasource：对应 metric 分区延后处理，流程不会静默切换 datasource。
- semantic adapter 只读：先迁移到 Dosi，再生成 metrics。

完整流程 contract 参见 [Dashboard Bootstrap](../skills/dashboard_bootstrap.zh.md)。
