# Auto Memory

Auto Memory 是 Datus-agent 的持久化记忆系统，使 Agent 能够跨对话自动保留有价值的信息。该机制完全基于文件和 prompt 驱动，无需向量数据库或 embedding。

## 概述

用户与 Agent 交互时，Agent 能够自动识别有价值的信息，并将其持久化到工作目录下的单个 Markdown 文件中。在后续对话中，这份记忆会被完整自动加载，使 Agent 能够回忆先前的上下文。

**核心特征：**

- **纯文件存储**：记忆以单个 Markdown 文件（`MEMORY.md`）形式存储
- **专用工具**：记忆只能通过 `add_memory` / `edit_memory` 写入，通用文件系统工具无法访问记忆目录
- **硬字节上限**：单个扁平文件，上限 **2000 字节**，没有主题子文件、没有索引，因此整份记忆始终能放进 system prompt
- **按 subagent 隔离**：每个 subagent 拥有独立的记忆目录
- **零配置**：无需额外设置，符合条件的 Agent 自动启用

## 记忆目录

记忆存储在工作目录的 `.datus/memory/` 下，每个 Agent 拥有独立的子目录，内含单个 `MEMORY.md`：

```text
{workspace_root}/
└── .datus/
    └── memory/
        ├── chat/                       # 内置 chat agent
        │   └── MEMORY.md              # 单文件，自动加载（≤2000 字节）
        └── my_custom_agent/           # 自定义 subagent
            └── MEMORY.md
```

> 记忆目录在 Agent 首次写入时自动创建，无需手动创建。

## 哪些 Agent 有记忆

| Agent 类型 | 是否启用记忆 |
|-----------|-------------|
| `chat`（内置主 Agent） | Yes |
| 自定义 subagent | Yes |
| 内置系统 subagent（`gen_sql`、`gen_report` 等） | No |
| `explore` | No |

只有面向用户的交互式 Agent 拥有记忆，执行特定流水线任务的内置系统 subagent 不启用；当它们通过 `task` 启动时，会以只读形式 inline 继承父 Agent 的记忆作为背景。

## 单文件记忆

每个 Agent 的记忆是一个扁平的 `MEMORY.md` 文件：

- 每次对话开始时**完整自动加载**到 Agent 上下文
- 上限 **2000 字节**，专用工具会拒绝任何会超限的写入；外部编辑导致超限的文件会在加载时被截断
- 适合存储简洁、稳定的事实：用户偏好、关键项目决策、外部系统引用

没有主题子文件、也没有索引——请保持条目简短，使整份文件始终在上限以内。

## 记忆工具

Agent 通过两个专用工具维护记忆：

| 工具 | 用途 |
|------|------|
| `add_memory(content)` | 向记忆追加一条简洁的事实 |
| `edit_memory(old_string, new_string)` | 更新某条记忆；将 `new_string` 传空字符串即删除该条 |

当 `add_memory` 会使文件超过 2000 字节时，写入会被拒绝并提示先释放空间；Agent 随后用 `edit_memory` 删掉一条过时记忆再重试。

## 使用方式

### 让 Agent 记住信息

直接用自然语言告诉 Agent：

```text
> 记住我偏好使用 DuckDB
> 记住项目使用 snake_case 命名规范
> 记住报表输出格式默认用 Markdown
```

Agent 会通过 `add_memory` 保存信息，下次对话自动生效。

### 让 Agent 忘记信息

```text
> 忘记我对 DuckDB 的偏好
> 不要再记住命名规范的事
```

Agent 会用 `edit_memory` 找到并删除对应的记忆条目。

### 更正记忆

当 Agent 基于记忆给出错误回答时，直接更正即可：

```text
> 不对，我们项目用的是 PostgreSQL，不是 DuckDB
```

Agent 会用 `edit_memory` 立即更新记忆中的错误内容。

### 查看当前记忆

记忆文件是普通的 Markdown 文件，可以直接查看或手动编辑：

```bash
cat {workspace_root}/.datus/memory/chat/MEMORY.md
```

也可以让 Agent 读取：

```text
> 读一下你当前的记忆
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

记忆目录位置跟随解析后的 workspace root：

| 优先级 | 来源 |
|--------|------|
| 1 | `agentic_nodes` 中节点级 `workspace_root` |
| 2 | `agent.yml` 中 `agent.project_root`（默认取启动时的 CWD） |

例如，当 `agent.project_root` 设置为 `~/my_project` 时，chat agent 的记忆文件位于：

```text
~/my_project/.datus/memory/chat/MEMORY.md
```

## 最佳实践

1. **保持条目简洁**：整份文件上限 2000 字节，每条事实一行短句
2. **定期清理**：过时或错误的记忆应及时让 Agent 删除或更正以释放空间
3. **善用显式请求**：重要信息直接告诉 Agent "记住这个"，确保被持久化
4. **手动编辑也可以**：记忆文件是普通 Markdown，随时可手动查看和修改（注意保持在 2000 字节以内）
