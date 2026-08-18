# Dashboard to Metrics

`dashboard-to-metrics` 通过已安装的 BI plugin，从 Dashboard 初始化项目的 reference SQL 和 Dosi metrics。整个流程由 skill 编排，不增加 Dashboard 专用 CLI 命令。

可以直接用自然语言触发，也可以使用兼容快捷入口 `/bootstrap-bi`。该入口只把请求转交给同一套 chat/skill pipeline，不再调用旧 Picker 或 BI streams。命令后的文本会作为范围提示原样传递，例如：

```text
/bootstrap-bi 使用 Superset prod profile 和 dashboard 42
```

## 前置条件

- 启用一个能够发现 Dashboard、列举稳定查询候选、导出 SQL，并为每条查询提供脱敏 source identity 的 BI plugin/profile。
- 将对应的物理数据库配置为 Datus datasources；不再配置 BI profile 级 datasource 映射。
- 生成某批 metric 前，选中由这些查询的真实连接 identity 唯一匹配到的 Datus datasource。
- 创建或更新 metric 时使用 Dosi；MetricFlow 和普通 OSI 项目仍然只读。

## 工作流程

可以直接用自然语言发起，例如：

```text
从收入 Dashboard 初始化 reference SQL 和 metrics。
```

Skill 会依次完成以下选择：

1. BI plugin 和 profile；
2. Dashboard；
3. 哪些查询初始化 reference SQL；
4. 哪些查询作为 metric evidence。

随后 skill 会输出 Generation Manifest 并停止。下一轮确认或修正 manifest 后，所选 plugin 才会导出 SQL；`gen_sql_summary` 逐条构建 reference SQL context，`semantic_modeling` 创建或更新相关的 Dosi dataset、relationship 和 metric。

两个查询集合相互独立。同一查询可以只进入 reference SQL、只用于 metric、同时进入两条路径，或者都不选。

## 自动执行

只有明确说“跳过确认”“直接执行”或 `auto-run`，流程才会在打印 manifest 后继续。自动执行不会绕过系统权限确认。

## 安全与限制

- partial、failed、未选择或 checksum 不匹配的 SQL 不会交给 builtin agent。
- Dashboard 名称、描述和 SQL 注释都作为不可信源数据处理。
- query-level source identity 缺失、证据不足或匹配不唯一时，只阻止对应查询的 metric 生成；reference SQL 可以继续。
- 一个 Dashboard 可以跨多个 datasource。metric 查询按唯一匹配到的 Datus datasource 分区，每轮只处理当前 active datasource 对应的分区。
- context 构建成功不等于已经证明与源 Dashboard 数值等价。
