---
name: create-skill
description: Datus skill file format and packaging reference. Consult when creating or revising a skill requires optional frontmatter fields, directory layout, or agent.yml integration details.
tags: [skill, format, authoring]
version: "1.2.0"
user_invocable: false
allowed_agents:
  - gen_skill
---

# Datus Skill Format Reference

Use this reference to resolve Datus format or packaging questions during authoring. The gen_skill system prompt provides the creation, revision, and review workflow.

A skill directory contains `SKILL.md` with YAML frontmatter:

```yaml
---
name: skill-name
description: What the skill does and when to use it.
---
```

Write instructions for a reusable task, including inputs, decisions, expected output, and necessary examples. Keep definitions consistent and record unresolved dependencies rather than inventing them.

Keep the core workflow and navigation in SKILL.md. Move long examples or domain details into supporting files when useful, maintain each rule in one place, and link every bundled resource with instructions for when to read it.

- Read [format.md](references/format.md) for optional frontmatter fields, descriptions, and file organization.
- Read [integration.md](references/integration.md) for skill discovery, agent configuration, and publication commands.

Resolve these paths relative to the `skill_location` returned by `load_skill`.
