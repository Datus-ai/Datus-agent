---
name: create-skill
description: Create new Datus skills from scratch. Use when users want to build a new skill, scaffold a skill directory, or capture a workflow as a reusable skill. Trigger phrases include "create a skill", "make a skill for", "turn this into a skill", "new skill".
tags: [skill, development, authoring]
version: "1.0.0"
user_invocable: false
allowed_commands:
  - "python:scripts/*.py"
---

# Create Skill

Guide for creating new Datus skills from scratch.

## Capture Intent

Start by understanding the user's intent. The conversation might already contain a workflow to capture.

Key questions (use `ask_user`):
1. What should this skill enable the agent to do?
2. When should this skill trigger? (user phrases/contexts)
3. What's the expected output format?
4. Should it have script execution capabilities?

Don't ask all at once — start with the most important, follow up based on answers.

## Interview and Research

Probe for: edge cases, input/output formats, dependencies, success criteria.

If creating a data-related skill, use database tools to explore schema — this research informs what you write, it is NOT skill output.

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
skill_create_directory(path="<skill-name>")
skill_write_file(path="<skill-name>/SKILL.md", content=...)
```

If `allowed_commands` configured:
```
skill_create_directory(path="<skill-name>/scripts")
skill_write_file(path="<skill-name>/scripts/<script>.py", content=...)
```

Full structure:
```
skill-name/
├── SKILL.md          (required)
├── scripts/          (if allowed_commands)
├── references/       (if additional docs)
└── assets/           (if templates/icons)
```

## Validate

After writing, call `validate_skill` with the absolute path from write_file result.

Checks: YAML frontmatter, required fields, allowed_commands format, non-empty body.

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
