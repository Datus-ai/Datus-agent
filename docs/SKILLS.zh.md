# AgentSkills 技能系统

AgentSkills 是 Datus-agent 的技能发现和加载系统，遵循 [agentskills.io](https://agentskills.io) 规范。它通过 SKILL.md 文件实现模块化、按需扩展的能力。

## 概述

技能系统提供：
- **技能发现**：自动扫描技能目录中的 SKILL.md 文件
- **权限控制**：统一的 ALLOW/DENY/ASK 权限系统，适用于工具和技能
- **按需加载**：技能仅在需要时通过 `load_skill` 工具加载
- **脚本执行**：为配置了 `allowed_commands` 的技能提供受限的 bash 执行环境

## 快速开始

### 1. 创建技能

创建一个包含 SKILL.md 文件的目录：

```
skills/
└── sql-optimization/
    └── SKILL.md
```

**SKILL.md** 内容：
```markdown
---
name: sql-optimization
description: 提供 SQL 查询优化指导和最佳实践
tags: [sql, optimization, performance]
---

# SQL 优化技能

优化 SQL 查询时，请遵循以下准则：

## 索引使用
- 始终检查查询是否使用了适当的索引
- 需要特定列时避免使用 SELECT *
- 使用 EXPLAIN 分析查询计划

## Join 优化
- 尽可能使用 INNER JOIN 而非子查询
- 确保连接列已建立索引
- 按从小到大的顺序排列表连接
```

### 2. 在 agent.yml 中配置技能

```yaml
skills:
  enabled: true
  directories:
    - skills/
    - ~/.datus/skills/
  warn_on_missing: true

permissions:
  default: allow
  rules:
    - tool: skills
      pattern: "*"
      permission: allow
    - tool: skills
      pattern: "dangerous-*"
      permission: deny
```

### 3. 在工作流中使用技能

技能自动对 agentic 节点可用。LLM 可以使用以下方式加载技能：

```
load_skill(skill_name="sql-optimization")
```

## SKILL.md 格式

### Frontmatter 字段

| 字段 | 必填 | 描述 |
|------|------|------|
| `name` | 否 | 技能标识符（默认为目录名） |
| `description` | 是 | 显示在可用技能列表中的简短描述 |
| `tags` | 否 | 用于分类的标签列表 |
| `version` | 否 | 语义版本字符串 |
| `allowed_commands` | 否 | 允许的 bash 命令模式列表 |
| `disable_model_invocation` | 否 | 如果为 true，技能不会出现在可用技能列表中 |
| `user_invocable` | 否 | 如果为 true，用户可通过 CLI 调用（默认：true） |

### 带脚本的示例

```markdown
---
name: data-profiler
description: 分析数据集并生成统计信息
tags: [data, analysis, profiling]
allowed_commands:
  - "python:scripts/*.py"
  - "python:-c:*"
---

# 数据分析技能

此技能提供数据分析能力。

## 使用方法

运行分析脚本：
```bash
python scripts/profile.py --input data.csv
```
```

## 权限系统

### 权限级别

| 级别 | 行为 |
|------|------|
| `allow` | 工具/技能可用，可自由使用 |
| `deny` | 工具/技能对 LLM 隐藏（永远不会出现在提示词中） |
| `ask` | 每次使用前需要用户确认 |

### 配置

```yaml
permissions:
  default: allow  # 未匹配工具的默认权限
  rules:
    # 允许所有 db_tools
    - tool: db_tools
      pattern: "*"
      permission: allow

    # 危险操作需要确认
    - tool: db_tools
      pattern: "execute_sql"
      permission: ask

    # 拒绝特定技能
    - tool: skills
      pattern: "internal-*"
      permission: deny

nodes:
  chatbot:
    permissions:
      # 节点特定的覆盖配置
      rules:
        - tool: skills
          pattern: "*"
          permission: allow
```

### 模式匹配

模式使用 glob 风格匹配：
- `*` 匹配任何内容
- `execute_*` 匹配以 "execute_" 开头的工具
- `*-admin` 匹配以 "-admin" 结尾的工具

## 脚本执行

技能可以定义 `allowed_commands` 来启用受限的脚本执行。

### 命令模式格式

```
前缀:glob模式
```

示例：
- `python:*` - 允许任何 python 命令
- `python:scripts/*.py` - 仅允许 scripts/ 目录中的脚本
- `sh:*.sh` - 允许 shell 脚本
- `python:-c:*` - 允许带任意参数的 python -c
- `node:*` - 允许任何 node 命令

### 安全特性

- 命令仅在匹配允许的模式时执行
- 工作目录锁定在技能位置
- 可配置超时（默认：60 秒）
- 输出大小限制（默认：50KB）
- 环境隔离，提供 `SKILL_NAME` 和 `SKILL_DIR` 环境变量

### 带脚本的技能示例

```
skills/
└── report-generator/
    ├── SKILL.md
    └── scripts/
        ├── generate.py
        └── validate.py
```

**SKILL.md**：
```markdown
---
name: report-generator
description: 从查询结果生成分析报告
allowed_commands:
  - "python:scripts/*.py"
---

# 报告生成器

使用以下命令生成报告：
```bash
python scripts/generate.py --format html --output report.html
```
```

## API 参考

### SkillManager

技能操作的主要协调器。

```python
from datus.tools.skill_tools import SkillManager, SkillConfig
from datus.tools.permission import PermissionManager, PermissionConfig

# 初始化
skill_config = SkillConfig(directories=["skills/"])
permission_config = PermissionConfig(default_permission="allow")
permission_manager = PermissionManager(global_config=permission_config)

manager = SkillManager(
    skill_config=skill_config,
    permission_manager=permission_manager
)

# 获取节点可用的技能
skills = manager.get_available_skills(node_name="chatbot")

# 加载技能
content, bash_tools = manager.load_skill(
    skill_name="sql-optimization",
    node_name="chatbot"
)

# 生成用于系统提示词的 XML
xml = manager.generate_available_skills_xml(node_name="chatbot")
```

### SkillRegistry

发现和解析 SKILL.md 文件。

```python
from datus.tools.skill_tools import SkillRegistry

registry = SkillRegistry(skill_directories=["skills/"])

# 列出所有发现的技能
for skill in registry.list_skills():
    print(f"{skill.name}: {skill.description}")

# 获取特定技能
skill = registry.get_skill("sql-optimization")

# 加载技能内容
content = registry.load_skill_content("sql-optimization")

# 添加新技能后刷新
registry.refresh()
```

### SkillFuncTool

提供 `load_skill` 原生工具供 LLM 使用。

```python
from datus.tools.skill_tools import SkillFuncTool

func_tool = SkillFuncTool(
    skill_manager=manager,
    node_name="chatbot"
)

# 获取 LLM 可用的工具
tools = func_tool.available_tools()

# 加载技能（由 LLM 调用）
result = func_tool.load_skill(skill_name="sql-optimization")
```

### SkillBashTool

在技能权限范围内执行脚本。

```python
from datus.tools.skill_tools import SkillBashTool

bash_tool = SkillBashTool(
    skill_metadata=skill,
    workspace_root=str(skill.location),
    timeout=60
)

# 执行命令（必须匹配 allowed_commands 模式）
result = bash_tool.execute_command("python scripts/analyze.py --input data.csv")
```

## 与 AgenticNode 集成

技能自动集成到 agentic 节点：

1. **系统提示词注入**：可用技能显示在 `<available_skills>` XML 块中
2. **工具注册**：`load_skill` 工具自动可用
3. **权限过滤**：仅向 LLM 显示有权限的技能

### 自定义技能可用性

在节点配置中：

```yaml
nodes:
  chatbot:
    skill_patterns:
      - "sql-*"
      - "data-*"
    permissions:
      rules:
        - tool: skills
          pattern: "admin-*"
          permission: deny
```

## 最佳实践

### 技能设计

1. **单一职责**：每个技能应专注于一个能力
2. **清晰描述**：编写帮助 LLM 理解何时使用技能的描述
3. **有用的标签**：添加相关标签用于分类
4. **文档使用方法**：在 markdown 内容中包含示例

### 安全性

1. **最小权限**：仅允许必要的命令模式
2. **具体模式**：优先使用 `python:scripts/*.py` 而非 `python:*`
3. **审查脚本**：审计技能目录中的脚本
4. **使用 DENY**：对不需要的节点隐藏敏感技能

### 组织结构

```
skills/
├── sql/
│   ├── optimization/
│   │   └── SKILL.md
│   └── troubleshooting/
│       └── SKILL.md
├── data/
│   ├── profiling/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       └── profile.py
│   └── validation/
│       └── SKILL.md
└── internal/
    └── admin/
        └── SKILL.md  # 对敏感技能使用 permission deny
```

## 故障排除

### 技能未被发现

1. 检查技能目录是否在 `skills.directories` 配置中
2. 验证 SKILL.md 有有效的 YAML frontmatter
3. 检查是否存在 `description` 字段（必填）
4. 启用调试日志查看发现详情

### 技能对节点不可用

1. 检查权限规则是否拒绝了该技能
2. 验证 `disable_model_invocation` 不是 true
3. 如果配置了节点特定的 `skill_patterns`，请检查

### 脚本执行被拒绝

1. 验证命令是否匹配 `allowed_commands` 模式
2. 检查模式格式：`前缀:glob模式`
3. 对于复杂命令，使用更宽松的模式或添加特定模式

### 调试日志

启用调试日志进行故障排除：

```python
import logging
logging.getLogger("datus.tools.skill_tools").setLevel(logging.DEBUG)
logging.getLogger("datus.tools.permission").setLevel(logging.DEBUG)
```
