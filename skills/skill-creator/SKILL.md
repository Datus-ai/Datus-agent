---
name: skill-creator
description: Create, edit, and evaluate Datus skills. Use when users want to create a new skill from scratch, modify or improve an existing skill, scaffold a skill directory structure, or optimize a skill's description for better triggering accuracy. Also use when users say things like "turn this into a skill", "make a skill for", "create a skill", or want to capture a workflow as a reusable skill.
tags: [skill, development, authoring, meta]
version: "1.0.0"
user_invocable: true
allowed_commands:
  - "python:scripts/*.py"
---

# Skill Creator

A skill for creating new Datus skills and iteratively improving them.

## Overview

The skill creation process:

1. Decide what the skill should do and roughly how
2. Write a draft SKILL.md
3. Scaffold the skill directory (scripts/, references/ if needed)
4. Validate the skill structure
5. Iterate based on user feedback

Your job is to figure out where the user is in this process and help them progress. Maybe they want to create a skill from scratch, or maybe they already have a draft and want to improve it.

## Creating a Skill

### Capture Intent

Start by understanding the user's intent. The current conversation might already contain a workflow the user wants to capture (e.g., they say "turn this into a skill"). If so, extract answers from the conversation history first.

Key questions:
1. What should this skill enable the agent to do?
2. When should this skill trigger? (what user phrases/contexts)
3. What's the expected output format?
4. Should it have script execution capabilities?

Use the `ask_user` tool for interactive interviews. Don't ask all questions at once -- start with the most important ones and follow up based on answers.

### Interview and Research

Probe for edge cases, input/output formats, example files, success criteria, and dependencies. If creating a data-related skill, explore the database schema to understand what tables and columns are available.

### Write the SKILL.md

Based on the interview, fill in these components:

**Frontmatter fields:**
- **name**: Lowercase with hyphens. Must be unique. Example: `sql-optimizer`
- **description**: The primary triggering mechanism. Include both what the skill does AND when to use it. Make descriptions assertive -- instead of "Helps with SQL optimization", write "Analyze and optimize SQL queries. Use whenever the user mentions slow queries, query optimization, EXPLAIN plans, index suggestions, or database performance tuning, even if they don't explicitly ask for optimization."
- **tags**: Categorization for discovery. Example: `[sql, performance, database]`
- **version**: Semantic version. Start with `"1.0.0"`
- **allowed_commands**: Script execution patterns. Format: `"prefix:glob_pattern"`. Example: `"python:scripts/*.py"`
- **disable_model_invocation**: Set `true` if only the user should trigger this skill (not the LLM automatically)
- **user_invocable**: Set `false` if this skill is only for programmatic/LLM use
- **context**: Set to `"fork"` to run in an isolated subagent
- **agent**: Subagent type when `context: fork` (e.g., `Explore`, `Plan`)
- **compatibility**: Version requirements. Example: `datus: ">=0.2.0"`

**Markdown body -- the skill instructions:**

This is the core of the skill. When the agent loads the skill, it receives this content as context. Write it as if you're briefing a capable colleague.

### Writing Guide

#### Progressive Disclosure

Skills use a three-level loading system:
1. **Metadata** (name + description) -- Always in context (~100 words)
2. **SKILL.md body** -- Loaded when skill triggers (<500 lines ideal)
3. **Bundled resources** -- Loaded as needed (unlimited)

Keep SKILL.md under 500 lines. If approaching this limit, use `references/` for detailed documentation and add clear pointers from the main file.

#### Writing Patterns

- **Imperative form**: "Analyze the query" not "You should analyze the query"
- **Explain the why**: Instead of rigid MUSTs, explain reasoning so the agent handles edge cases. Theory of mind beats brute force.
- **Examples**: Include 1-2 concrete input/output examples
- **Output format**: Define expected output structure explicitly
- **Domain organization**: When a skill supports multiple variants, organize by domain:

```
skill-name/
+-- SKILL.md (workflow + selection logic)
+-- references/
    +-- variant-a.md
    +-- variant-b.md
```

#### Principle of Lack of Surprise

Skills must not contain malware, exploit code, or security-compromising content. A skill's contents should not surprise the user. Don't create misleading skills or skills designed to facilitate unauthorized access.

### Scaffold the Directory

Create the complete directory structure:

```
skill-name/
+-- SKILL.md          (required)
+-- scripts/          (if allowed_commands configured)
+-- references/       (if additional docs needed)
+-- assets/           (if templates/icons needed)
```

### Validate

After creating the skill, run validation:

```bash
python scripts/quick_validate.py /path/to/skill/SKILL.md
```

This checks:
- YAML frontmatter parses correctly
- Required fields (name, description) are present
- allowed_commands patterns are well-formed

### Storage Location

Ask the user where to save:
- **Project-level** (`./skills/`): Tracked in version control, project-specific
- **User-level** (`~/.datus/skills/`): Available across all projects

## Editing an Existing Skill

When the user wants to modify an existing skill:

1. Load the skill using `load_skill(skill_name="...")`
2. Present the current frontmatter summary and key instruction sections
3. Ask what changes are needed
4. Generate the modified content
5. Show a before/after summary of changes
6. Write the updated file after user confirmation

Focus improvements on:
- **Generalize from feedback**: Skills are used many times across many prompts. Avoid overfitting to specific examples.
- **Keep the prompt lean**: Remove instructions that aren't pulling their weight.
- **Explain the why**: Explain reasoning behind instructions so the model can handle edge cases intelligently.
- **Look for repeated work**: If the agent consistently writes similar helper scripts, bundle them in `scripts/`.

## Description Optimization

The description field is the primary mechanism that determines whether the agent invokes a skill. Tips for effective descriptions:

- Include both what the skill does AND specific trigger contexts
- Be assertive: "Use whenever X" not "Can be used for X"
- Include adjacent keywords the user might use
- Cover edge cases: phrases that should trigger AND phrases that should NOT trigger this skill
- Test mentally: "Would the agent correctly decide to use this skill for [scenario]?"

## Datus-Specific Notes

### agent.yml Integration

Skills are discovered from directories configured in `agent.yml`:
```yaml
agent:
  skills:
    directories:
      - ~/.datus/skills
      - ./skills
```

Per-node skill filtering:
```yaml
agentic_nodes:
  my_agent:
    skills: "sql-*"  # Glob pattern for which skills this node sees
```

### Marketplace

After creating a skill, users can publish to the marketplace:
```
.skill publish <skill-name>
```

### Script Execution

Scripts run in a sandboxed environment:
- Working directory is the skill directory
- Environment variables: `SKILL_NAME`, `SKILL_DIR`
- Timeout: 60 seconds default
- Output size limit: 50K characters
- Commands validated against `allowed_commands` patterns before execution
