# 斜杠命令参考

Datus-CLI 中所有可用的斜杠命令，按类别分组。

## 会话

| 命令 | 别名 | 说明 |
|------|------|------|
| `/help` | | 显示所有斜杠命令的帮助信�� |
| `/exit` | `/quit` | 退出 CLI |
| `/clear` | | 清除控制台和聊天会话 |
| `/chat_info` | | 显示当前聊天会话信息 |
| `/compact` | | 通过摘要历史记录压缩聊天会话 |
| `/resume` | | 列出并恢复之前的聊天会话 |
| `/rewind` | | 将当前会话回退到指定的轮次 |

## 元数据

| 命令 | 说明 |
|------|------|
| `/databases` | 列出所有数据库 |
| `/database` | 切换当前数据库 |
| `/tables` | 列出所有表 |
| `/schemas` | 列出所有 schema 或显示 schema 详情 |
| `/schema` | 切换当前 schema |
| `/table_schema` | 显示表字段详情 |
| `/indexes` | 显示表的索引 |

## 上下文

| 命令 | 说明 |
|------|------|
| `/catalog` | 显示数据库目录浏览器 |
| `/subject` | 显示语义模型、指标和参考 |

## Agent

| 命令 | 说明 |
|------|------|
| `/agent` | 打开统一 agent 管理 TUI（Built-in 覆写 / 设为默认）。`/agent <name>` 仍可直接切换默认 agent |
| `/subagent` | 打开统一 agent 管理 TUI（Custom Tab） |
| `/datasource` | 切换当前数据源 |
| `/language` | 固定所有 agentic 节点的响应语言（[Language 命令](language_command.zh.md)） |

## 系统

| 命令 | 别名 | 说明 | 详情 |
|------|------|------|------|
| `/model` | `/models` | 运行时切换 LLM 提供商/模型 | [Model 命令](model_command.zh.md) |
| `/effort` | | 设置 reasoning 强度（off/minimal/low/medium/high） | [Effort 命令](effort_command.zh.md) |
| `/init` | | 端到端初始化项目工作区并生成 `AGENTS.md`；可附加自由文本作为目标/范围提示 | [Init 命令](init_command.zh.md) |
| `/session-summarize` | | 把当前会话总结进各持久化存储（先出 manifest,确认后执行）；可附加自由文本作为关注点提示 | |
| `/memory-organize` | | 审计并整理所有持久化存储（先出整改计划,确认后执行）；可附加自由文本作为关注点提示 | |
| `/mcp` | | 管理 MCP 服务器（列表/添加/删除/检查/调用/过滤） | [MCP 扩展](mcp_extensions.zh.md) |
| `/skill` | | 管理技能和市场 | [Skill 命令](skill_command.zh.md) |
| `/bootstrap-bi` | | 为子 agent 上下文提取 BI 仪表盘资产 | |
| `/services` | | 列出已配置的服务平台及其只读方法 | |
| `/permission` | | 切换当前 CLI / agent 权限配置文件 | |
| `/profile` | | 已废弃的 `/permission` 兼容别名 | |
