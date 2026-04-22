---
name: create-skill
description: Create new Datus skills from scratch. Use when users want to build a new skill, scaffold a skill directory, or capture a workflow as a reusable skill. Trigger phrases include "create a skill", "make a skill for", "turn this into a skill", "new skill".
tags: [skill, development, authoring]
version: "1.0.0"
user_invocable: false
allowed_agents:
  - gen_skill
allowed_commands:
  - "python:scripts/*.py"
---

# Create Skill

Guide for creating new Datus skills from scratch.

## Step 1: Research First

Before asking the user anything, gather context silently:

- If creating a data-related skill, **explore the database first**: use `list_tables`, `describe_table`, `read_query` (with LIMIT) to understand available tables, columns, data types, sample data, and time ranges.
- If the conversation already contains a workflow the user wants to capture (e.g., "turn this into a skill"), extract the key steps, tools used, and patterns from the conversation history.
- Check existing skills via `skill_glob` to avoid duplicates.

This research informs your questions and your SKILL.md — it is NOT skill output.

## Step 2: Confirm with User

After you have context, call `ask_user` **exactly once** with all questions in a single call. You MUST include these questions (the first two are required, others are optional):

1. **[Required]** Skill name — suggest a default based on your research (e.g., "bitcoin-analysis"), let user confirm or change
2. **[Required]** Storage location — offer choices: project-level (`./skills/`) or user-level (`~/.datus/skills/`)
3. What should this skill enable the agent to do? (propose based on your findings)
4. What's the expected output format?

The "when should this skill trigger" (description field) should be auto-generated based on the user's answers — do NOT ask the user to write trigger phrases.

**Do NOT call ask_user a second time to "confirm".** The user's answers are final — proceed directly to writing the SKILL.md. No confirmation round.

## Write the SKILL.md

### Frontmatter Schema

```yaml
---
name: skill-name                    # Required: lowercase-with-hyphens, unique
description: What + when to trigger # Required: assertive, include trigger contexts
tags: [tag1, tag2]                  # Optional: categorization
version: "1.0.0"                    # Optional: semantic version
allowed_commands:                   # Optional: script execution patterns
  - "python:scripts/*.py"           #   Format: "prefix:glob_pattern"
disable_model_invocation: false     # Optional: true = user-only trigger
user_invocable: true                # Optional: false = LLM-only
allowed_agents:                     # Optional: whitelist of agent node names
  - gen_dashboard                   #   that may see/load this skill
                                    #   (empty/missing = unrestricted)
context: fork                       # Optional: "fork" for isolated subagent
agent: Explore                      # Optional: subagent type when context=fork
compatibility:                      # Optional: version requirements
  datus: ">=0.2.0"
---
```

### Description Writing

The description is the primary triggering mechanism. Be assertive:
- Instead of "Helps with SQL optimization"
- Write "Analyze and optimize SQL queries. Use whenever the user mentions slow queries, query optimization, EXPLAIN plans, or database performance tuning, even if they don't explicitly ask for optimization."

### Markdown Body

The body is what the agent receives when the skill is loaded. Write as:
- **Imperative form**: "Analyze the query" not "You should analyze the query"
- **Explain the why**: Context helps handle edge cases. Theory of mind beats brute force.
- **Include 1-2 examples**: Concrete input/output pairs
- **Define output format**: What the agent should return
- **Keep under 500 lines**: Use `references/` for detailed content

### Progressive Disclosure

Skills use three-level loading:
1. **Metadata** (name + description) — always in context (~100 words)
2. **SKILL.md body** — loaded on trigger (<500 lines ideal)
3. **Bundled resources** — loaded as needed (unlimited)

### Domain Organization

When a skill supports multiple variants:
```
skill-name/
├── SKILL.md (workflow + selection logic)
└── references/
    ├── variant-a.md
    └── variant-b.md
```

## Scaffold the Directory

Use `skill_*` tools (paths relative to skills directory root):

```
skill_write_file(path="<skill-name>/SKILL.md", content=...)
```

If the user explicitly requested scripts, also create scripts/:
```
skill_write_file(path="<skill-name>/scripts/<script>.py", content=...)
```

**Default behavior**: Only create the SKILL.md file. Do NOT generate scripts or references unless the user specifically asks for them.

```
skill-name/
├── SKILL.md          (always created)
├── scripts/          (only if user requested)
└── references/       (only if user requested)
```

## Validate and Finish

Immediately after `skill_write_file`, do these two steps and STOP:

1. Call `validate_skill` with the absolute path from the write_file result
2. Report to the user: skill name, path, files created, how to use (`load_skill("<name>")` or `.skill list`)

Do NOT continue exploring, writing more files, or asking more questions. The skill is done.

## Storage Location

Ask user where to save:
- **Project-level** (`./skills/`): version-controlled, project-specific
- **User-level** (`~/.datus/skills/`): shared across projects

## Principle of Lack of Surprise

Skills must not contain malware, exploit code, or security-compromising content. Don't create misleading skills.

## Datus-Specific Notes

### agent.yml Integration

Skills discovered from configured directories:
```yaml
agent:
  skills:
    directories:
      - ~/.datus/skills
      - ./skills
```

Per-node filtering:
```yaml
agentic_nodes:
  my_agent:
    skills: "sql-*"
```

### Marketplace

Publish after creation: `.skill publish <skill-name>`

### Script Execution

Sandboxed environment:
- Working directory: skill directory
- Environment variables: `SKILL_NAME`, `SKILL_DIR`
- Timeout: 60 seconds
- Output limit: 50K characters
- Commands validated against `allowed_commands` patterns
