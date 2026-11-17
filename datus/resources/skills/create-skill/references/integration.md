# Datus Integration

Project skills live in `.datus/skills/<skill-name>/`; shared user skills live in `~/.datus/skills/<skill-name>/`. Use the location established for the task.

Configured search directories control discovery:

```yaml
agent:
  skills:
    directories:
      - ~/.datus/skills
      - .datus/skills
```

Node skill patterns control which discovered skills appear in that node's available-skills list:

```yaml
agent:
  agentic_nodes:
    my_agent:
      skills: "sql-*"
```

Load a discovered skill with `load_skill("<skill-name>")` or inspect the list with `.skill list`. Resolve its supporting files relative to the location returned by the load operation.

When publication is requested, use `.skill publish <skill-name>` after reviewing and validating the final files.
