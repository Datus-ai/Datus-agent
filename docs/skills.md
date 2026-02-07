# AgentSkills User Guide

AgentSkills is a skill discovery and loading system for Datus-agent, following the [agentskills.io](https://agentskills.io) specification. It enables modular, on-demand capability expansion through SKILL.md files.

## Quick Start

This tutorial demonstrates how to use the **report-generator** skill with the California Schools dataset to generate analysis reports.

### Step 1: Create a Skill

Create a skill directory with a `SKILL.md` file:

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

**SKILL.md** content:
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

### Step 2: Configure Skills in agent.yml

```yaml
skills:
  directories:
    - ~/.datus/skills
    - ./skills
  warn_duplicates: true

permissions:
  default: allow
  rules:
    - tool: skills
      pattern: "*"
      permission: allow
```

### Step 3: Use the Skill in a Chat Session

Start a chat session and ask your question:

```
> What is the highest eligible free rate for K-12 students in the schools
> in Alameda County? Generate a report using the final result.
```

The agent will:

1. **Load the skill** - When generating a report is needed, the LLM calls `load_skill(skill_name="report-generator")` to get the skill instructions.

2. **Execute SQL query** - Query the California Schools database to find the answer.

3. **Generate report** - Execute the skill's script to create a report:
   ```
   skill_execute_command(
       skill_name="report-generator",
       command="python scripts/generate_report.py --input results.json --format markdown --title 'Alameda County K-12 Free Rate Analysis'"
   )
   ```

![Chat session showing skill loading and report generation](assets/skill_chat_session.png)

### Step 4: View the Generated Report

The report will be generated in the skill's working directory:

![Generated markdown report showing the analysis results](assets/skill_report_output.png)

---

## Permission System

The permission system controls which skills and tools are available to the agent.

### Permission Levels

| Level | Behavior |
|-------|----------|
| `allow` | Skill is available and can be used freely |
| `deny` | Skill is hidden from agent (never appears in prompts) |
| `ask` | User confirmation required before each use |

### Configuration Example

```yaml
permissions:
  default: allow
  rules:
    # Allow all skills by default
    - tool: skills
      pattern: "*"
      permission: allow

    # Require confirmation for database write operations
    - tool: db_tools
      pattern: "execute_sql"
      permission: ask

    # Hide internal/admin skills
    - tool: skills
      pattern: "internal-*"
      permission: deny

    # Require confirmation for potentially dangerous skills
    - tool: skills
      pattern: "dangerous-*"
      permission: ask
```

### Pattern Matching

Patterns use glob-style matching:
- `*` matches anything
- `report-*` matches skills starting with "report-"
- `*-admin` matches skills ending with "-admin"

### Node-Specific Permissions

Override permissions for specific nodes:

```yaml
agentic_nodes:
  chat:
    skills: "report-*, data-*"  # Only expose matching skills
    permissions:
      rules:
        - tool: skills
          pattern: "admin-*"
          permission: deny
```

---

## Using Skills in Subagent

Skills can be configured to run in an isolated subagent context for complex tasks.

### Configure Subagent Execution

Add `context: fork` and specify the `agent` type in the SKILL.md frontmatter:

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

## When to Use
- Complex multi-step analysis
- Tasks requiring extensive exploration
- Investigations that may take multiple turns
```

### Available Subagent Types

| Agent Type | Use Case |
|------------|----------|
| `Explore` | Codebase exploration, file searching, understanding structure |
| `Plan` | Implementation planning, architectural decisions |
| `general-purpose` | Multi-step tasks, complex research |

### Example: Research Skill with Subagent

```markdown
---
name: codebase-research
description: Research codebase patterns and architecture
context: fork
agent: Explore
user_invocable: true
---

# Codebase Research

When invoked, this skill spawns an Explore subagent to:
1. Search for relevant files and patterns
2. Analyze code structure
3. Report findings back to the main conversation
```

### Invocation Control

| Field | Default | Description |
|-------|---------|-------------|
| `disable_model_invocation` | `false` | If true, only user can invoke via `/skill-name` |
| `user_invocable` | `true` | If false, hidden from CLI menu (only model invokes) |

---

## SKILL.md Reference

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier |
| `description` | Yes | Brief description shown in available skills list |
| `tags` | No | List of tags for categorization |
| `version` | No | Semantic version string |
| `allowed_commands` | No | List of permitted script patterns |
| `context` | No | Set to `"fork"` to run in subagent |
| `agent` | No | Subagent type: `Explore`, `Plan`, `general-purpose` |
| `disable_model_invocation` | No | If true, only user can invoke |
| `user_invocable` | No | If false, hidden from CLI menu |

### Command Pattern Format

```
prefix:glob_pattern
```

Examples:
- `python:*` - Allow any python command
- `python:scripts/*.py` - Allow scripts in scripts/ directory only
- `sh:*.sh` - Allow shell scripts
- `python:-c:*` - Allow python -c inline code

### Security Features

- Commands only execute if they match allowed patterns
- Working directory locked to skill location
- Timeout enforcement (default: 60 seconds)
- Environment variables: `SKILL_NAME`, `SKILL_DIR`

---

## Troubleshooting

### Skill Not Discovered

1. Check skill directory is in `skills.directories` config
2. Verify SKILL.md has valid YAML frontmatter (between `---` markers)
3. Both `name` and `description` fields are required

### Script Execution Denied

1. Verify command matches an `allowed_commands` pattern
2. Ensure skill was loaded first via `load_skill()`
3. Check pattern format: `prefix:glob_pattern`

### Debug Logging

Enable debug logging:

```bash
export DATUS_LOG_LEVEL=DEBUG
```

---

## Skill Marketplace CLI

Datus includes a built-in CLI for interacting with the AgenticDataStack Town Skills Marketplace. You can search, install, publish, and manage skills directly from the command line.

### Configuration

Marketplace settings in `agent.yml`:

```yaml
skills:
  directories:
    - ~/.datus/skills
    - ./skills
  marketplace_url: "http://localhost:9000"  # Town backend URL
  auto_sync: false                          # Auto-sync promoted skills on startup
  install_dir: "~/.datus/skills"            # Where marketplace skills are installed
```

Or override the marketplace URL per-command with `--marketplace`:

```bash
datus skill search sql --marketplace http://my-town:9000
```

### Command Reference

#### `datus skill list`

List all locally installed skills.

```bash
datus skill list
```

Output:
```
┌──────────────────┬─────────┬─────────────┬─────────────────────────┐
│ Name             │ Version │ Source      │ Tags                    │
├──────────────────┼─────────┼─────────────┼─────────────────────────┤
│ sql-optimization │ 1.0.0   │ marketplace │ sql, optimization       │
│ report-generator │ 1.0.0   │ local       │ report, analysis        │
└──────────────────┴─────────┴─────────────┴─────────────────────────┘
```

#### `datus skill search <query>`

Search for skills in the Town Marketplace.

```bash
datus skill search sql
datus skill search optimization
datus skill search --marketplace http://localhost:9000 report
```

Output:
```
Searching for 'sql'...
  sql-optimization v1.0.0 — Optimize SQL queries for better performance
  sql-linting v0.3.0 — Lint SQL queries against best practices
```

#### `datus skill install <name> [version]`

Install a skill from the Marketplace to your local `install_dir`.

```bash
# Install latest version
datus skill install sql-optimization

# Install specific version
datus skill install sql-optimization 1.0.0
```

What happens:
1. Downloads the skill bundle (`.tar.gz`) from Town Backend
2. Extracts to `~/.datus/skills/<name>/`
3. Registers the skill with `source=marketplace` in the local registry

#### `datus skill publish <path> [--owner <name>]`

Publish a local skill directory to the Town Marketplace.

```bash
# Publish from a skill directory (must contain SKILL.md)
datus skill publish ./skills/sql-optimization

# Publish with owner
datus skill publish ./skills/sql-optimization --owner "murphy"

# Publish to a specific marketplace
datus skill publish ./skills/sql-optimization --marketplace http://my-town:9000
```

Requirements:
- The directory must contain a valid `SKILL.md` with YAML frontmatter
- Required frontmatter fields: `name`, `description`
- Recommended fields: `version`, `tags`, `allowed_commands`, `license`

Example `SKILL.md`:
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

What happens:
1. Reads and validates `SKILL.md` frontmatter
2. Creates a `.tar.gz` bundle of the skill directory
3. POSTs skill metadata to `POST /api/skills`
4. Uploads the bundle to `POST /api/skills/<name>/<version>/upload`
5. Skill appears in the Town Marketplace UI at `/skills`

#### `datus skill info <name>`

Show details about a skill (checks both local and marketplace).

```bash
datus skill info sql-optimization
```

Output:
```
Local: sql-optimization v1.0.0 (marketplace)
  Optimize SQL queries for better performance
Marketplace: sql-optimization v1.0.0
  Owner: murphy  Promoted: True
```

#### `datus skill update`

Update all marketplace-installed skills to the latest version.

```bash
datus skill update
```

This checks each marketplace-installed skill and re-downloads if a newer version is available.

#### `datus skill remove <name>`

Remove a locally installed skill from the registry.

```bash
datus skill remove sql-optimization
```

### REPL Commands

The same skill operations are available inside the interactive REPL session:

```
datus> .skill list                          # List local skills
datus> .skill search sql                    # Search marketplace
datus> .skill install sql-optimization      # Install from marketplace
datus> .skill publish ./skills/my-skill     # Publish to marketplace
datus> .skill info sql-optimization         # Show skill details
datus> .skill update                        # Update marketplace skills
datus> .skill remove sql-optimization       # Remove local skill
```

### End-to-End Workflow Example

```bash
# 1. Create a skill locally
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

# 2. Publish to marketplace
datus skill publish ./skills/my-etl-helper --owner murphy

# 3. Verify it appears in marketplace
datus skill search etl

# 4. Install on another machine / agent
datus skill install my-etl-helper

# 5. Verify local installation
datus skill list

# 6. View in Town UI
open http://localhost:3000/skills
```

### Town Marketplace UI

After publishing, skills are visible in the Town Frontend:

- **Skills List** (`/skills`): Browse all skills with search and tag filtering
- **Skill Detail** (`/skills/<name>`): View version history, metadata, promote/delete
- **Publish Form**: Publish new skills directly from the web UI
- **Promote**: Mark a skill as "Town Default" so all agents auto-install it
