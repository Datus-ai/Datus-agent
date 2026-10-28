# Develop a Plugin

The recommended way to build a Datus plugin is with the
[`datus-plugin-development`](https://github.com/Datus-ai/Datus-Plugins/blob/main/skills/datus-plugin-development/SKILL.md)
skill from the
[`Datus-Plugins`](https://github.com/Datus-ai/Datus-Plugins) repository.

Give the skill the SDK, API specification, or product documentation you want
to wrap. It will help you choose a useful command surface, generate the plugin,
and verify the result. The skill contains the current plugin contract, so this
page focuses on the workflow instead of repeating implementation details that
can drift out of date.

## Before you start

Prepare:

- The SDK, REST API, OpenAPI specification, CLI documentation, or README you
  want to expose through Datus. A local path or public URL both work.
- A short command name. For example, an Airflow plugin uses `datus airflow`.
- Any scope preferences, such as which operations matter most or which
  administrative actions should be excluded.

You only need to provide the source material and describe what you want the
plugin to do. The development skill handles the implementation.

## Install the development skill

### Codex

Add the Datus plugin marketplace:

```bash
codex plugin marketplace add Datus-ai/Datus-Plugins
```

Open `/plugins`, install **datus-plugin-development**, and start a new session
so the skill is available.

### Claude Code

Add the marketplace and install the development plugin:

```text
/plugin marketplace add Datus-ai/Datus-Plugins
/plugin install datus-plugin-development@datus-plugin
```

## Start a plugin project

Invoke the skill with the documentation source and the command name you want.

In Codex:

```text
$datus-plugin-development ./docs/openapi.yaml, command name: acme
```

In Claude Code:

```text
/datus-plugin-development ./docs/openapi.yaml, command name: acme
```

A URL or a directory containing several related documents can be used in place
of the file path. Add any scope or compatibility requirements to the same
request.

## What happens next

The skill follows a design-first workflow:

1. **Review the source documentation.** It inventories the available
   operations, authentication requirements, configuration, and dependencies.
2. **Propose a design.** You receive a draft covering the proposed commands,
   configuration fields, permission behavior, bundled skills, and anything
   intentionally left out.
3. **Review the scope.** The skill stops before writing code. Confirm the
   proposal, request changes, or bring excluded operations back into scope.
4. **Build the plugin.** After you approve the design, the skill creates the
   working plugin, supporting skills, and tests.
5. **Verify the result.** It runs the relevant tests and checks installation,
   CLI dispatch, bundled skills, and packaging.

This confirmation step is important: it keeps a large vendor API from becoming
an equally large CLI by accident, while leaving the final scope in your hands.

## Try the generated plugin

After implementation, install the local project and inspect its CLI:

```bash
datus plugin install src:./datus-acme-plugin
datus acme --help
```

To build a distributable bundle:

```bash
datus plugin pack ./datus-acme-plugin -o ./dist
```

Use `--with-deps` when the target environment must install the bundle without
access to a package index. See
[Offline installation](introduction.md#offline-install) for the difference
between the two bundle types.

## Source of truth

Use these resources when you need more detail:

- [Development skill source](https://github.com/Datus-ai/Datus-Plugins/blob/main/skills/datus-plugin-development/SKILL.md) —
  the current development workflow and plugin contract.
- [Datus-Plugins README](https://github.com/Datus-ai/Datus-Plugins#develop-your-own-plugin) —
  marketplace installation and repository examples.
- [Contributing to Datus Plugins](https://github.com/Datus-ai/Datus-Plugins/blob/main/CONTRIBUTING.md) —
  workspace setup, tests, pull requests, and releases.
- [Plugin introduction](introduction.md) — installing, configuring, and
  managing plugins as a user.
