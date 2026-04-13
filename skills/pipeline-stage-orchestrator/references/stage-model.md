# Verifiable Stage Model

Each stage should follow this model:

1. `execute`
2. `verify`
3. on failure: `rollback` or `halt`
4. on success: `mark_done`

Recommended stage fields:

- `name`
- `depends_on`
- `execute_action`
- `verify_action`
- `failure_action`
- `success_condition`

This skill is about ordered gating. It is different from task decomposition across multiple agents.

