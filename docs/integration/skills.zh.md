# Skills

Skills 是 Datus-agent 的技能发现和加载系统，遵循 [agentskills.io](https://agentskills.io) 规范。它通过 SKILL.md 文件实现模块化、按需扩展的能力。

## 快速开始

本教程演示如何使用 **report-generator** 技能与加州学校数据集生成分析报告。

### 步骤 1：创建技能

创建包含 `SKILL.md` 文件的技能目录：

```
~/.datus/skills/
└── report-generator/
    ├── SKILL.md
    └── scripts/
        ├── generate_report.py
        ├── analyze_data.py
        ├── validate.sh
        └── export.sh
```

**SKILL.md** 内容：

```markdown
---
name: report-generator
description: Generate analysis reports from SQL query results with multiple output formats (HTML, Markdown, JSON)
tags: [report, analysis, visualization, export]
version: "1.0.0"
allowed_commands:
  - "python:scripts/*.py"
  - "sh:scripts/*.sh"
---

# Report Generator Skill

This skill generates professional analysis reports from SQL query results.

## Features

- **Multiple Formats**: Export to HTML, Markdown, or JSON
- **Data Analysis**: Automatic statistical analysis and insights

## Usage

### Generate a Report

python scripts/generate_report.py --input results.json --format html --output report.html

Options:
- `--input`: Input data file (JSON or CSV)
- `--format`: Output format (html, markdown, json)
- `--output`: Output file path
- `--title`: Report title (optional)
```

### 步骤 2：在 agent.yml 中配置技能

```yaml
skills:
  directories:
    - ~/.datus/skills
    - ./skills
  warn_duplicates: true

permissions:
  default: allow
  rules:
    # 技能加载需要确认
    - tool: skills
      pattern: "*"
      permission: ask
    # 技能脚本执行需要确认
    - tool: skill_bash
      pattern: "*"
      permission: ask
```

!!! tip
    为 skills 和 skill_bash 使用 `ask` 权限，在执行前需要手动确认，有助于防止意外或危险操作。

### 步骤 3：在聊天会话中使用技能

启动聊天会话并提出问题：

```
> What is the highest eligible free rate for K-12 students in the schools
> in Alameda County? Generate a report using the final result.
```

Agent 将执行以下操作：

1. **加载技能** - 当需要生成报告时，LLM 调用 `load_skill(skill_name="report-generator")` 获取技能指令。

2. **执行 SQL 查询** - 查询加州学校数据库以获取答案。

3. **生成报告** - 执行技能脚本创建报告：

    ```python
    skill_execute_command(
        skill_name="report-generator",
        command="python scripts/generate_report.py --input results.json --format markdown --title 'Alameda County K-12 Free Rate Analysis'"
    )
    ```

![聊天会话展示技能加载和报告生成](../assets/skills1.png)
![聊天会话展示技能加载和报告生成](../assets/skills3.png)

### 步骤 4：查看生成的报告

报告将在技能的工作目录中生成：

![生成的 Markdown 报告展示分析结果](../assets/skill5.png)

## 权限系统

权限系统控制哪些技能和工具可供 Agent 使用。

### 权限级别

| 级别 | 行为 |
|------|------|
| `allow` | 技能可用且可自由使用 |
| `deny` | 技能对 Agent 隐藏（不会出现在提示中） |
| `ask` | 每次使用前需要用户确认 |

### 配置示例

```yaml
permissions:
  default: allow
  rules:
    # 默认允许所有技能
    - tool: skills
      pattern: "*"
      permission: allow

    # 数据库写操作需要确认
    - tool: db_tools
      pattern: "execute_sql"
      permission: ask

    # 隐藏内部/管理技能
    - tool: skills
      pattern: "internal-*"
      permission: deny

    # 潜在危险技能需要确认
    - tool: skills
      pattern: "dangerous-*"
      permission: ask
```

### 模式匹配

模式使用 glob 风格匹配：

- `*` 匹配任意内容
- `report-*` 匹配以 "report-" 开头的技能
- `*-admin` 匹配以 "-admin" 结尾的技能

### 节点特定权限

为特定节点覆盖权限：

```yaml
agentic_nodes:
  chat:
    skills: "report-*, data-*"  # 仅暴露匹配的技能
    permissions:
      rules:
        - tool: skills
          pattern: "admin-*"
          permission: deny
```

## 在 Subagent 中使用技能

默认情况下，**聊天 Subagent 会自动加载所有已发现的技能**。其他 Subagent（报告生成、SQL 生成、指标等）**不会加载任何技能**，除非在 `agent.yml` 中显式配置。

| Subagent 类型 | 默认加载的技能 |
|---------------|---------------|
| Chat | 所有已发现的技能 |
| 其他所有 Subagent（报告、SQL、指标等） | 无 |

### 为自定义 Subagent 启用技能

`agentic_nodes` 中的每个 Subagent 支持三种工具扩展方式，可以混合使用：

| 字段 | 来源 | 描述 |
|------|------|------|
| `tools` | 内置 | Datus 原生工具（如 `db_tools.*`、`context_search_tools.*`、`date_parsing_tools.*`） |
| `mcp` | 第三方 | 外部 MCP 服务器工具，通过 `.mcp.json` 配置（如 `metricflow_mcp`、`filesystem`） |
| `skills` | 用户自定义 | 从 `SKILL.md` 文件发现的技能 — 可在 Markdown 中定义工作流，也可通过自定义脚本扩展能力 |

要在自定义 Subagent 中启用技能，请在 `agent.yml` 的 `agentic_nodes` 部分中为对应 Subagent 添加 `skills` 字段：

```yaml
agentic_nodes:
  # 在单个 Subagent 中混合使用 tools + mcp + skills
  school_report:
    node_class: gen_report
    tools: db_tools.*, context_search_tools.*
    mcp: metricflow_mcp
    skills: "report-*, data-*"
    model: deepseek

  # SQL Subagent，仅使用原生工具和 SQL 技能
  school_sql:
    tools: db_tools.*, date_parsing_tools.*
    skills: "sql-*"
    model: deepseek

  # Chat Subagent，加载所有技能
  school_chat:
    tools: db_tools.*, context_search_tools.*
    skills: "*"
    model: deepseek
```

`skills` 字段接受逗号分隔的 glob 模式列表。只有名称匹配至少一个模式的技能才会对该 Subagent 可用。`node_class` 字段支持两个值：`gen_sql`（默认）和 `gen_report`。

当 Subagent 配置了 `skills` 时：

1. **技能发现** — 系统扫描 `skills.directories`（或默认路径：`~/.datus/skills`、`./skills`）查找所有 `SKILL.md` 文件。
2. **模式过滤** — 仅暴露匹配 Subagent `skills` glob 模式的技能。
3. **权限过滤** — `permissions` 规则进一步过滤哪些技能被允许、拒绝或需要确认。
4. **系统提示注入** — 可用技能以 `<available_skills>` XML 形式附加到 Subagent 的系统提示中，使 LLM 能够调用 `load_skill()` 和 `skill_execute_command()`。

**示例：在 Subagent 中启用报告生成技能**

```yaml
skills:
  directories:
    - ~/.datus/skills

agentic_nodes:
  attribution_report:
    node_class: gen_report
    tools: db_tools.*
    skills: "report-generator"
    model: deepseek
```

通过此配置，`attribution_report` Subagent 将可以访问内置数据库工具和 `report-generator` 技能。LLM 可以调用 `load_skill(skill_name="report-generator")` 获取指令，然后使用 `skill_execute_command()` 运行技能中定义的脚本。

!!! note
    如果 `agent.yml` 中没有全局 `skills:` 部分，系统会自动创建默认的技能管理器，扫描 `~/.datus/skills` 和 `./skills`。

!!! tip
    `skill_execute_command` 工具默认使用 `ask` 权限级别。这意味着在技能脚本执行前用户会收到确认提示，除非在 `permissions` 配置中显式覆盖。

### 在隔离 Subagent 中运行技能

技能也可以通过在 SKILL.md frontmatter 中设置 `context: fork` 来在隔离的 Subagent 上下文中运行：

```markdown
---
name: deep-analysis
description: Perform comprehensive data analysis with multiple iterations
tags: [analysis, research]
context: fork
agent: Explore
---

# Deep Analysis Skill

This skill runs in an isolated Explore subagent for thorough investigation.
```

可用的隔离执行 Subagent 类型：

| Agent 类型 | 用途 |
|------------|------|
| `Explore` | 代码库探索、文件搜索、理解结构 |
| `Plan` | 实现规划、架构决策 |
| `general-purpose` | 多步骤任务、复杂研究 |

### 调用控制

| 字段 | 默认值 | 描述 |
|------|--------|------|
| `disable_model_invocation` | `false` | 如为 true，仅用户可通过 `/skill-name` 调用 |
| `user_invocable` | `true` | 如为 false，从 CLI 菜单隐藏（仅模型调用） |

## SKILL.md 参考

### Frontmatter 字段

| 字段 | 必需 | 描述 |
|------|------|------|
| `name` | 是 | 唯一技能标识符 |
| `description` | 是 | 在可用技能列表中显示的简短描述 |
| `tags` | 否 | 用于分类的标签列表 |
| `version` | 否 | 语义版本字符串 |
| `allowed_commands` | 否 | 允许的脚本模式列表 |
| `context` | 否 | 设为 `"fork"` 以在 subagent 中运行 |
| `agent` | 否 | Subagent 类型：`Explore`、`Plan`、`general-purpose` |
| `disable_model_invocation` | 否 | 如为 true，仅用户可调用 |
| `user_invocable` | 否 | 如为 false，从 CLI 菜单隐藏 |

### 命令模式格式

```
prefix:glob_pattern
```

示例：

- `python:*` - 允许任意 python 命令
- `python:scripts/*.py` - 仅允许 scripts/ 目录中的脚本
- `sh:*.sh` - 允许 shell 脚本
- `python:-c:*` - 允许 python -c 内联代码

### 安全特性

- 命令仅在匹配允许的模式时执行
- 工作目录锁定在技能位置
- 超时强制执行（默认：60 秒）
- 环境变量：`SKILL_NAME`、`SKILL_DIR`

## 故障排除

### 技能未被发现

1. 检查技能目录是否在 `skills.directories` 配置中
2. 验证 SKILL.md 具有有效的 YAML frontmatter（在 `---` 标记之间）
3. `name` 和 `description` 字段都是必需的

### 脚本执行被拒绝

1. 验证命令是否匹配 `allowed_commands` 模式
2. 确保先通过 `load_skill()` 加载了技能
3. 检查模式格式：`prefix:glob_pattern`

### 调试日志

启用调试日志：

```bash
export DATUS_LOG_LEVEL=DEBUG
```

## 技能市场 CLI

Datus 内置了与 AgenticDataStack Town 技能市场交互的 CLI。您可以直接从命令行搜索、安装、发布和管理技能。

### 认证

Town 市场的所有 API 操作都需要认证。在使用市场功能之前，请使用 `login` 命令进行认证。

```bash
# 交互式登录（提示输入邮箱和密码）
datus skill login --marketplace http://datus-marketplace:9000

# 非交互式登录
datus skill login --marketplace http://datus-marketplace:9000 --email user@example.com --password secret

# 登出（清除已保存的令牌）
datus skill logout --marketplace http://datus-marketplace:9000
```

在 REPL 中：
```
datus> .skill login http://datus-marketplace:9000
Email: user@example.com
Password: ****
Login successful! Token saved for http://datus-marketplace:9000
```

令牌保存在 `~/.datus/marketplace_auth.json`，并自动包含在所有后续市场请求中。令牌在 24 小时后过期；重新运行 `login` 以刷新。

### 配置

`agent.yml` 中的市场设置：

```yaml
skills:
  directories:
    - ~/.datus/skills
    - ./skills
  marketplace_url: "http://localhost:9000"  # Town 后端 URL
  auto_sync: false                          # 启动时自动同步推荐技能
  install_dir: "~/.datus/skills"            # 市场技能的安装目录
```

或通过 `--marketplace` 参数为单个命令指定市场 URL：

```bash
datus skill search sql --marketplace http://datus-marketplace:9000
```

### 命令参考

#### `datus skill list`

列出所有本地已安装的技能。

```bash
datus skill list
```

输出：
```
┌──────────────────┬─────────┬─────────────┬─────────────────────────┐
│ Name             │ Version │ Source      │ Tags                    │
├──────────────────┼─────────┼─────────────┼─────────────────────────┤
│ sql-optimization │ 1.0.0   │ marketplace │ sql, optimization       │
│ report-generator │ 1.0.0   │ local       │ report, analysis        │
└──────────────────┴─────────┴─────────────┴─────────────────────────┘
```

#### `datus skill search <query>`

在 Town 市场中搜索技能。

```bash
datus skill search sql
datus skill search optimization
datus skill search --marketplace http://localhost:9000 report
```

输出：
```
Searching for 'sql'...
  sql-optimization v1.0.0 — Optimize SQL queries for better performance
  sql-linting v0.3.0 — Lint SQL queries against best practices
```

#### `datus skill install <name> [version]`

从市场安装技能到本地 `install_dir`。

```bash
# 安装最新版本
datus skill install sql-optimization

# 安装指定版本
datus skill install sql-optimization 1.0.0
```

安装过程：

1. 从 Town 后端下载技能包（`.tar.gz`）
2. 解压到 `~/.datus/skills/<name>/`
3. 在本地注册表中以 `source=marketplace` 注册技能

#### `datus skill publish <path> [--owner <name>]`

将本地技能目录发布到 Town 市场。

```bash
# 从技能目录发布（必须包含 SKILL.md）
datus skill publish ./skills/sql-optimization

# 指定所有者发布
datus skill publish ./skills/sql-optimization --owner "murphy"

# 发布到指定市场
datus skill publish ./skills/sql-optimization --marketplace http://datus-marketplace:9000
```

要求：

- 目录必须包含带有 YAML frontmatter 的有效 `SKILL.md`
- 必需的 frontmatter 字段：`name`、`description`
- 推荐字段：`version`、`tags`、`allowed_commands`、`license`

`SKILL.md` 示例：

```markdown
---
name: sql-optimization
description: Optimize SQL queries for better performance
tags: [sql, optimization, performance]
version: "1.0.0"
license: Apache-2.0
compatibility:
  datus: ">=0.2.0"
allowed_commands:
  - "python:scripts/*.py"
  - "sh:scripts/*.sh"
---

# SQL Optimization Skill
...
```

发布过程：

1. 读取并验证 `SKILL.md` frontmatter
2. 创建技能目录的 `.tar.gz` 包
3. 将技能元数据 POST 到 `POST /api/skills`
4. 上传包到 `POST /api/skills/<name>/<version>/upload`
5. 技能出现在 Town 市场 UI 的 `/skills` 页面

#### `datus skill info <name>`

显示技能详情（检查本地和市场）。

```bash
datus skill info sql-optimization
```

输出：
```
Local: sql-optimization v1.0.0 (marketplace)
  Optimize SQL queries for better performance
Marketplace: sql-optimization v1.0.0
  Owner: murphy  Promoted: True
```

#### `datus skill update`

将所有市场安装的技能更新到最新版本。

```bash
datus skill update
```

此命令检查每个市场安装的技能，如果有更新版本可用则重新下载。

#### `datus skill remove <name>`

从注册表中移除本地已安装的技能。

```bash
datus skill remove sql-optimization
```

### REPL 命令

在交互式 REPL 会话中也可以使用相同的技能操作：

```
datus> .skill list                          # 列出本地技能
datus> .skill search sql                    # 搜索市场
datus> .skill install sql-optimization      # 从市场安装
datus> .skill publish ./skills/my-skill     # 发布到市场
datus> .skill info sql-optimization         # 显示技能详情
datus> .skill update                        # 更新市场技能
datus> .skill remove sql-optimization       # 移除本地技能
```

### 端到端工作流示例

```bash
# 1. 在本地创建技能
mkdir -p ./skills/my-etl-helper/scripts
cat > ./skills/my-etl-helper/SKILL.md << 'EOF'
---
name: my-etl-helper
description: Helper utilities for ETL pipeline development
tags: [etl, pipeline, data-engineering]
version: "1.0.0"
allowed_commands:
  - "python:scripts/*.py"
---

# ETL Helper Skill
Provides utilities for building and testing ETL pipelines.
EOF

# 2. 发布到市场
datus skill publish ./skills/my-etl-helper --owner murphy

# 3. 验证是否出现在市场中
datus skill search etl

# 4. 在另一台机器/agent 上安装
datus skill install my-etl-helper

# 5. 验证本地安装
datus skill list

# 6. 在 Town UI 中查看
open http://localhost:3000/skills
```

### Town 市场 UI

发布后，技能在 Town 前端可见：

- **技能列表** (`/skills`)：浏览所有技能，支持搜索和标签过滤
- **技能详情** (`/skills/<name>`)：查看版本历史、元数据、推荐/删除
- **发布表单**：直接从 Web UI 发布新技能
- **推荐**：将技能标记为"Town 默认"，所有 agent 自动安装
