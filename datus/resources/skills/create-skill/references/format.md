# Skill Format

SKILL.md begins with YAML frontmatter. Only `name` and `description` are required:

```yaml
---
name: skill-name                    # Lowercase with hyphens; unique
description: What the skill does and when it should be used.
tags: [tag1, tag2]                  # Optional categorization
version: "1.0.0"                    # Optional version
disable_model_invocation: false     # true = user-only invocation
user_invocable: true                # false = hidden from user invocation
allowed_agents:                     # Empty or absent = unrestricted
  - gen_dashboard
context: fork                       # Optional isolated subagent execution
agent: Explore                      # Requires context: fork
compatibility:
  datus: ">=0.2.0"
---
```

The description is used for discovery. State the task and the situations in which the skill should be used; keep detailed rules and field lists in the body.

Use imperative instructions and explain consequential choices. Include inputs, expected output, and examples that clarify behavior. Keep SKILL.md concise; the validator warns when its body exceeds 500 lines, but the appropriate split depends on the content.

For multiple domains or variants, keep selection rules and navigation in the entry point:

```text
skill-name/
├── SKILL.md
└── references/
    ├── variant-a.md
    └── variant-b.md
```

Link supporting files and explain when to read them. Keep a compact skill together when separate files would add navigation without reducing complexity. Check that bundled references exist and agree with the entry point.
