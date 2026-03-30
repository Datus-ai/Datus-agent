# Auto Memory

Auto Memory 是 Datus-agent 的持久化记忆系统，使 Agent 能够跨对话自动保留有价值的信息。该机制完全基于文件和 prompt 驱动，无需向量数据库或 embedding。

## 概述

用户与 Agent 交互时，Agent 能够自动识别有价值的信息，并将其持久化到工作目录下的 Markdown 文件中。在后续对话中，这些记忆会自动加载，使 Agent 能够回忆先前的上下文。

**核心特征：**

- **纯文件存储**：记忆以 Markdown 文件形式存储，Agent 通过已有的 `read_file` / `write_file` / `edit_file` 工具管理
- **两层结构**：简洁的 `MEMORY.md` 主文件自动加载，可选的主题子文件按需读取
- **按 subagent 隔离**：每个 subagent 拥有独立的记忆目录
- **零配置**：无需额外设置，符合条件的 Agent 自动启用

## 记忆目录

记忆文件存储在工作目录的 `.datus/memory/` 下，每个 Agent 拥有独立的子目录：

```text
{workspace_root}/
└── .datus/
    └── memory/
        ├── chat/                       # 内置 chat agent
        │   ├── MEMORY.md              # 主文件：自动加载（≤200 行）
        │   ├── patterns.md            # 子文件：按需读取
        │   └── conventions.md
        └── my_custom_agent/           # 自定义 subagent
            ├── MEMORY.md
            └── domain.md
```

> 记忆目录在 Agent 首次写入时自动创建，无需手动创建。

## 哪些 Agent 有记忆

| Agent 类型 | 是否启用记忆 |
|-----------|-------------|
| `chat`（内置主 Agent） | Yes |
| 自定义 subagent | Yes |
| 内置系统 subagent（`gen_sql`、`gen_report` 等） | No |
| `explore` | No |

只有面向用户的交互式 Agent 拥有记忆，执行特定流水线任务的内置系统 subagent 不启用。

## 两层记忆机制

### L1：MEMORY.md（主文件）

- 每次对话开始时**自动加载**到 Agent 上下文
- 上限 **200 行**，超出部分会被截断
- 适合存储简洁的关键信息和 L2 文件链接

### L2：主题子文件

- Agent 通过 `read_file` **按需读取**
- 无行数限制，适合存储详细内容
- 示例：`patterns.md`、`conventions.md`、`domain.md`

**L1 适合存储：**

- 用户偏好和工作习惯
- 关键项目结构和文件路径
- 常用约定和规范
- L2 主题文件的链接

**L2 适合存储：**

- 详细的调试笔记
- 复杂的领域模式和业务规则
- 完整的决策记录

## 使用方式

### 让 Agent 记住信息

直接用自然语言告诉 Agent：

```text
> 记住我偏好使用 DuckDB
> 记住项目使用 snake_case 命名规范
> 记住报表输出格式默认用 Markdown
```

Agent 会将信息写入 `MEMORY.md`，下次对话自动生效。

### 让 Agent 忘记信息

```text
> 忘记我对 DuckDB 的偏好
> 不要再记住命名规范的事
```

Agent 会找到并删除对应的记忆条目。

### 更正记忆

当 Agent 基于记忆给出错误回答时，直接更正即可：

```text
> 不对，我们项目用的是 PostgreSQL，不是 DuckDB
```

Agent 会立即更新记忆中的错误内容。

### 查看当前记忆

记忆文件是普通的 Markdown 文件，可以直接查看或手动编辑：

```bash
cat {workspace_root}/.datus/memory/chat/MEMORY.md
```

也可以让 Agent 读取：

```text
> 读一下你当前的记忆文件
```

## Agent 的记忆行为

Agent 会在以下场景自动利用记忆：

- **新对话开始**：回顾记忆了解用户偏好和先前上下文
- **回答项目问题**：检查记忆中是否有相关决策或约定
- **用户提及以前讨论过的内容**：查找相关记忆条目
- **建议工具、数据库或工作流**：尊重用户已声明的偏好

Agent 会自动判断哪些信息值得保存：

| 应该保存 | 不应保存 |
|---------|---------|
| 跨多次交互确认的稳定模式 | 当前会话的临时任务细节 |
| 关键决策和项目结构 | 未经验证的不完整信息 |
| 用户偏好和工作习惯 | 单次交互中的推测性结论 |
| 常见问题的解决方案 | 进行中的工作状态 |

## 配置

Auto Memory **无需显式配置**，符合条件的 Agent 自动启用。

记忆目录位置跟随 `workspace_root` 设置：

| 优先级 | 来源 |
|--------|------|
| 1 | `agentic_nodes` 中节点级 `workspace_root` |
| 2 | `agent.yml` 中 `storage.workspace_root` |
| 3 | `agent.yml` 中顶层 `workspace_root` |
| 4 | 当前目录（`.`） |

例如，当 `workspace_root` 设置为 `~/my_project` 时，chat agent 的记忆文件位于：

```text
~/my_project/.datus/memory/chat/MEMORY.md
```

## 最佳实践

1. **保持 MEMORY.md 简洁**：控制在 200 行以内，详细内容放到 L2 子文件
2. **按主题组织**：使用语义化的子文件名（如 `db_conventions.md`），而非按时间记录
3. **定期清理**：过时或错误的记忆应及时让 Agent 删除或更正
4. **善用显式请求**：重要信息直接告诉 Agent "记住这个"，确保被持久化
5. **手动编辑也可以**：记忆文件是普通 Markdown，随时可以手动查看和修改
