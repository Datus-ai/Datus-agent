# Database adapters

Datus uses a registry-based adapter system for database connectivity. SQLite and DuckDB ship with the Agent; every server or cloud warehouse is an independently versioned `datus-<type>` package discovered through the `datus.adapters` entry-point group.

Connection profiles are documented under [Datasources](../configuration/datasources.md). That section is the source of truth for accepted fields, defaults, authentication rules, and namespace behavior; this page explains how adapters are installed and loaded.

## Installation and discovery

The `/datasource` manager is the recommended installation path:

```bash
datus
```

Run `/datasource`, choose a database type, and complete its profile. If the adapter is missing, Datus installs `datus-<type>` into the active interpreter, loads its entry point, validates the profile, and tests the connection before saving it.

For managed environments, install the package explicitly and pin it with the rest of the deployment dependencies:

```bash
pip install datus-postgresql
```

Installing an adapter registers four related pieces of behavior:

- its public `type` name, such as `postgresql`;
- a Pydantic connection model that rejects unknown fields and supplies defaults;
- a connector factory plus namespace/URI handlers;
- optional database-specific SQL guidance or skills.

The adapter must be installed in the same Python environment as the `datus` executable. Installing it into a different virtual environment does not make it discoverable.

## Runtime flow

```text
agent.services.datasources.<name>
        ↓
Agent keeps common keys and adapter-specific fields
        ↓
adapter connection model validates types, defaults, and constraints
        ↓
registry creates the connector for <type>
        ↓
SQL execution, metadata discovery, and namespace switching
```

Adapter packages own database-specific configuration. Datus owns the surrounding profile name, `type`, `default`, environment-variable expansion, and current-datasource selection. This boundary is why connection fields are documented per datasource instead of in a shared, approximate table.

## Developing an adapter

Database adapters live in the [`Datus-ai/datus-db-adapters`](https://github.com/Datus-ai/datus-db-adapters) workspace. A new adapter should:

1. depend on `datus-db-core` and reuse `datus-sqlalchemy` when appropriate;
2. define a strict Pydantic configuration model;
3. implement the connector contract and declare accurate namespace capabilities;
4. expose a `datus.adapters` entry point named after the datasource `type`;
5. pass the shared adapter contract and TPC-H tests;
6. add one detailed English and Chinese datasource page to this documentation.

Use the repository's current reference adapter and testing standard rather than copying an older package unchanged.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Adapter is not listed | Install `datus-<type>` into the interpreter that runs `datus`, then reopen `/datasource`. |
| Installation succeeds but loading fails | Import `datus_<type>` directly to expose missing native libraries or dependency conflicts. |
| Profile rejects an extra field | Use only the fields on that datasource's page; adapter models intentionally use strict validation. |
| Metadata hierarchy is wrong | Verify the adapter declares the correct catalog/database/schema capabilities and that the profile uses the documented namespace keys. |

## Next steps

- [Configure datasources](../configuration/datasources.md)
- [Use `/datasource` in the CLI](../cli/other_commands.md#datasource)
- [Configure SQL policy and read-only mode](../configuration/sql_policy.md)
