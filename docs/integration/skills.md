# Skills

Skills is a skill discovery and loading system for Datus-agent, following the [agentskills.io](https://agentskills.io) specification. It enables modular, on-demand capability expansion through SKILL.md files.

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
    # Require confirmation for skill loading
    - tool: skills
      pattern: "*"
      permission: ask
    # Require confirmation for skill script execution
    - tool: skill_bash
      pattern: "*"
      permission: ask
```

!!! tip
    Using `ask` permission for skills and skill_bash requires manual confirmation before execution, which helps prevent accidental or dangerous operations.

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

    ```python
    skill_execute_command(
        skill_name="report-generator",
        command="python scripts/generate_report.py --input results.json --format markdown --title 'Alameda County K-12 Free Rate Analysis'"
    )
    ```

![Chat session showing skill loading and report generation](../assets/skills1.png)
![Chat session showing skill loading and report generation](../assets/skills3.png)

### Step 4: View the Generated Report

The report will be generated in the skill's working directory:

![Generated markdown report showing the analysis results](../assets/skill5.png)

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

## Using Skills in Subagents

By default, the **chat subagent loads all discovered skills** automatically. Other subagents (report generation, SQL generation, metrics, etc.) **do not load any skills** unless explicitly configured in `agent.yml`.

| Subagent Type | Skills Loaded by Default |
|---------------|------------------------|
| Chat | All discovered skills |
| All other subagents (report, SQL, metrics, etc.) | None |

### Enabling Skills for Customized Subagents

Each subagent in `agentic_nodes` supports three types of tool extensions that can be mixed together:

| Field | Source | Description |
|-------|--------|-------------|
| `tools` | Built-in | Datus native tools (e.g. `db_tools.*`, `context_search_tools.*`, `date_parsing_tools.*`) |
| `mcp` | Third-party | External MCP server tools, configured via `.mcp.json` (e.g. `metricflow_mcp`, `filesystem`) |
| `skills` | User-defined | Skills discovered from `SKILL.md` files — can define workflows in Markdown and extend with custom scripts |

To enable skills in a customized subagent, add the `skills` field under the subagent's config in `agentic_nodes` section of `agent.yml`:

```yaml
agentic_nodes:
  # Mixing tools + mcp + skills in a single subagent
  school_report:
    node_class: gen_report
    tools: db_tools.*, context_search_tools.*
    mcp: metricflow_mcp
    skills: "report-*, data-*"
    model: deepseek

  # SQL subagent with only native tools and SQL skills
  school_sql:
    tools: db_tools.*, date_parsing_tools.*
    skills: "sql-*"
    model: deepseek

  # Chat subagent with all skills
  school_chat:
    tools: db_tools.*, context_search_tools.*
    skills: "*"
    model: deepseek
```

The `skills` field accepts a comma-separated list of glob patterns. Only skills whose names match at least one pattern will be available to that subagent. The `node_class` field supports two values: `gen_sql` (default) and `gen_report`.

When a subagent has `skills` configured:

1. **Skill discovery** — The system scans `skills.directories` (or defaults: `~/.datus/skills`, `./skills`) to find all `SKILL.md` files.
2. **Pattern filtering** — Only skills matching the subagent's `skills` glob patterns are exposed.
3. **Permission filtering** — The `permissions` rules further filter which skills are allowed, denied, or require confirmation.
4. **System prompt injection** — Available skills are appended as `<available_skills>` XML to the subagent's system prompt, enabling the LLM to call `load_skill()` and `skill_execute_command()`.

**Example: Enable Report Generation Skill in a Subagent**

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

With this configuration, the `attribution_report` subagent will have access to built-in database tools and the `report-generator` skill. The LLM can call `load_skill(skill_name="report-generator")` to get instructions, then use `skill_execute_command()` to run scripts defined in the skill.

!!! note
    If no global `skills:` section is present in `agent.yml`, the system automatically creates a default skill manager that scans `~/.datus/skills` and `./skills`.

!!! tip
    The `skill_execute_command` tool defaults to `ask` permission level. This means the user will be prompted for confirmation before any skill script executes, unless explicitly overridden in the `permissions` config.

### Running Skills in Isolated Subagent

Skills can also be configured to run in an isolated subagent context by setting `context: fork` in the SKILL.md frontmatter:

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

Available subagent types for isolated execution:

| Agent Type | Use Case |
|------------|----------|
| `Explore` | Codebase exploration, file searching, understanding structure |
| `Plan` | Implementation planning, architectural decisions |
| `general-purpose` | Multi-step tasks, complex research |

### Invocation Control

| Field | Default | Description |
|-------|---------|-------------|
| `disable_model_invocation` | `false` | If true, only user can invoke via `/skill-name` |
| `user_invocable` | `true` | If false, hidden from CLI menu (only model invokes) |

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

## Skill Marketplace CLI

Datus includes a built-in CLI for interacting with the AgenticDataStack Town Skills Marketplace. You can search, install, publish, and manage skills directly from the command line.

### Authentication

The Town Marketplace requires authentication for all API operations. Use the `login` command to authenticate before using marketplace features.

```bash
# Interactive login (prompts for email and password)
datus skill login --marketplace http://datus-marketplace:9000

# Non-interactive login
datus skill login --marketplace http://datus-marketplace:9000 --email user@example.com --password secret

# Logout (clear saved token)
datus skill logout --marketplace http://datus-marketplace:9000
```

In the REPL:
```
datus> /skill login http://datus-marketplace:9000
Email: user@example.com
Password: ****
Login successful! Token saved for http://datus-marketplace:9000
```

Tokens are saved at `~/.datus/marketplace_auth.json` and automatically included in all subsequent marketplace requests. Tokens expire after 24 hours; re-run `login` to refresh.

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
datus skill search sql --marketplace http://datus-marketplace:9000
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
datus skill publish ./skills/sql-optimization --marketplace http://datus-marketplace:9000
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
datus> /skill list                          # List local skills
datus> /skill search sql                    # Search marketplace
datus> /skill install sql-optimization      # Install from marketplace
datus> /skill publish ./skills/my-skill     # Publish to marketplace
datus> /skill info sql-optimization         # Show skill details
datus> /skill update                        # Update marketplace skills
datus> /skill remove sql-optimization       # Remove local skill
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
