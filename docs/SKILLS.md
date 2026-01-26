# AgentSkills System

AgentSkills is a skill discovery and loading system for Datus-agent, following the [agentskills.io](https://agentskills.io) specification. It enables modular, on-demand capability expansion through SKILL.md files.

## Overview

The skill system provides:
- **Skill Discovery**: Automatic scanning of skill directories for SKILL.md files
- **Permission Control**: Unified ALLOW/DENY/ASK permission system for tools and skills
- **On-Demand Loading**: Skills are loaded only when needed via the `load_skill` tool
- **Script Execution**: Restricted bash execution for skills with `allowed_commands`

## Quick Start

### 1. Create a Skill

Create a directory with a `SKILL.md` file:

```
skills/
└── sql-optimization/
    └── SKILL.md
```

**SKILL.md** content:
```markdown
---
name: sql-optimization
description: Provides SQL query optimization guidance and best practices
tags: [sql, optimization, performance]
---

# SQL Optimization Skill

When optimizing SQL queries, follow these guidelines:

## Index Usage
- Always check if queries use appropriate indexes
- Avoid SELECT * when specific columns are needed
- Use EXPLAIN to analyze query plans

## Join Optimization
- Prefer INNER JOIN over subqueries when possible
- Ensure join columns are indexed
- Order joins from smallest to largest table
```

### 2. Configure Skills in agent.yml

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

### 3. Use Skills in Workflow

Skills are automatically available to agentic nodes. The LLM can load skills using:

```
load_skill(skill_name="sql-optimization")
```

## SKILL.md Format

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Skill identifier (defaults to directory name) |
| `description` | Yes | Brief description shown in available skills list |
| `tags` | No | List of tags for categorization |
| `version` | No | Semantic version string |
| `allowed_commands` | No | List of permitted bash command patterns |
| `disable_model_invocation` | No | If true, skill won't appear in available skills |
| `user_invocable` | No | If true, user can invoke via CLI (default: true) |

### Example with Scripts

```markdown
---
name: data-profiler
description: Profile datasets and generate statistics
tags: [data, analysis, profiling]
allowed_commands:
  - "python:scripts/*.py"
  - "python:-c:*"
---

# Data Profiler Skill

This skill provides data profiling capabilities.

## Usage

Run the profiler script:
```bash
python scripts/profile.py --input data.csv
```
```

## Permission System

### Permission Levels

| Level | Behavior |
|-------|----------|
| `allow` | Tool/skill is available and can be used freely |
| `deny` | Tool/skill is hidden from LLM (never appears in prompts) |
| `ask` | User confirmation required before each use |

### Configuration

```yaml
permissions:
  default: allow  # Default permission for unmatched tools
  rules:
    # Allow all db_tools
    - tool: db_tools
      pattern: "*"
      permission: allow

    # Deny dangerous operations
    - tool: db_tools
      pattern: "execute_sql"
      permission: ask

    # Deny specific skills
    - tool: skills
      pattern: "internal-*"
      permission: deny

nodes:
  chatbot:
    permissions:
      # Node-specific overrides
      rules:
        - tool: skills
          pattern: "*"
          permission: allow
```

### Pattern Matching

Patterns use glob-style matching:
- `*` matches anything
- `execute_*` matches tools starting with "execute_"
- `*-admin` matches tools ending with "-admin"

## Script Execution

Skills can define `allowed_commands` to enable restricted script execution.

### Command Pattern Format

```
prefix:glob_pattern
```

Examples:
- `python:*` - Allow any python command
- `python:scripts/*.py` - Allow only scripts in scripts/ directory
- `sh:*.sh` - Allow shell scripts
- `python:-c:*` - Allow python -c with any argument
- `node:*` - Allow any node command

### Security Features

- Commands only execute if they match allowed patterns
- Working directory locked to skill location
- Configurable timeout (default: 60 seconds)
- Output size limiting (default: 50KB)
- Environment isolation with `SKILL_NAME` and `SKILL_DIR` variables

### Example Skill with Scripts

```
skills/
└── report-generator/
    ├── SKILL.md
    └── scripts/
        ├── generate.py
        └── validate.py
```

**SKILL.md**:
```markdown
---
name: report-generator
description: Generate analysis reports from query results
allowed_commands:
  - "python:scripts/*.py"
---

# Report Generator

Generate reports using:
```bash
python scripts/generate.py --format html --output report.html
```
```

## API Reference

### SkillManager

Main coordinator for skill operations.

```python
from datus.tools.skill_tools import SkillManager, SkillConfig
from datus.tools.permission import PermissionManager, PermissionConfig

# Initialize
skill_config = SkillConfig(directories=["skills/"])
permission_config = PermissionConfig(default_permission="allow")
permission_manager = PermissionManager(global_config=permission_config)

manager = SkillManager(
    skill_config=skill_config,
    permission_manager=permission_manager
)

# Get available skills for a node
skills = manager.get_available_skills(node_name="chatbot")

# Load a skill
content, bash_tools = manager.load_skill(
    skill_name="sql-optimization",
    node_name="chatbot"
)

# Generate XML for system prompt
xml = manager.generate_available_skills_xml(node_name="chatbot")
```

### SkillRegistry

Discovers and parses SKILL.md files.

```python
from datus.tools.skill_tools import SkillRegistry

registry = SkillRegistry(skill_directories=["skills/"])

# List all discovered skills
for skill in registry.list_skills():
    print(f"{skill.name}: {skill.description}")

# Get specific skill
skill = registry.get_skill("sql-optimization")

# Load skill content
content = registry.load_skill_content("sql-optimization")

# Refresh after adding new skills
registry.refresh()
```

### SkillFuncTool

Provides the `load_skill` native tool for LLM use.

```python
from datus.tools.skill_tools import SkillFuncTool

func_tool = SkillFuncTool(
    skill_manager=manager,
    node_name="chatbot"
)

# Get tools for LLM
tools = func_tool.available_tools()

# Load skill (called by LLM)
result = func_tool.load_skill(skill_name="sql-optimization")
```

### SkillBashTool

Executes scripts within skill permissions.

```python
from datus.tools.skill_tools import SkillBashTool

bash_tool = SkillBashTool(
    skill_metadata=skill,
    workspace_root=str(skill.location),
    timeout=60
)

# Execute command (must match allowed_commands patterns)
result = bash_tool.execute_command("python scripts/analyze.py --input data.csv")
```

## Integration with AgenticNode

Skills are automatically integrated into agentic nodes:

1. **System Prompt Injection**: Available skills appear in `<available_skills>` XML block
2. **Tool Registration**: `load_skill` tool is automatically available
3. **Permission Filtering**: Only permitted skills are shown to LLM

### Customizing Skill Availability

In node configuration:

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

## Best Practices

### Skill Design

1. **Single Responsibility**: Each skill should focus on one capability
2. **Clear Description**: Write descriptions that help LLM understand when to use the skill
3. **Useful Tags**: Add relevant tags for categorization
4. **Document Usage**: Include examples in the markdown content

### Security

1. **Minimal Permissions**: Only allow necessary command patterns
2. **Specific Patterns**: Prefer `python:scripts/*.py` over `python:*`
3. **Review Scripts**: Audit scripts in skill directories
4. **Use DENY**: Hide sensitive skills from nodes that don't need them

### Organization

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
        └── SKILL.md  # Use permission deny for sensitive skills
```

## Troubleshooting

### Skill Not Discovered

1. Check skill directory is in `skills.directories` config
2. Verify SKILL.md has valid YAML frontmatter
3. Check `description` field is present (required)
4. Run with debug logging to see discovery details

### Skill Not Available to Node

1. Check permission rules don't deny the skill
2. Verify `disable_model_invocation` is not true
3. Check node-specific `skill_patterns` if configured

### Script Execution Denied

1. Verify command matches an `allowed_commands` pattern
2. Check pattern format: `prefix:glob_pattern`
3. For complex commands, use more permissive patterns or add specific ones

### Debug Logging

Enable debug logging to troubleshoot:

```python
import logging
logging.getLogger("datus.tools.skill_tools").setLevel(logging.DEBUG)
logging.getLogger("datus.tools.permission").setLevel(logging.DEBUG)
```
