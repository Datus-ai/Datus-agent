# Plugins

A plugin connects Datus to an external service or adds a focused command-line
capability without changing Datus itself. Depending on the plugin, it may add:

| Capability | What you get |
|---|---|
| CLI commands | Run the service through `datus <plugin> ...` |
| Skills | Use plugin-provided skills alongside project and user skills |
| Agent context | Let the agent discover the plugin and its configured environments |
| Permission rules | Require confirmation before the agent runs sensitive commands |
| Tool safeguards | Check, rewrite, or reject selected agent tool calls |

Plugins are isolated Python packages. Datus discovers them automatically after
installation, so there is no separate registration step.

Want to build one? Follow [Develop a plugin](development.md).

## Install a plugin

For a published plugin, pass its package name:

```bash
datus plugin install datus-airflow-plugin
datus airflow --help
```

You can also make the source type explicit:

```bash
datus plugin install pip:datus-airflow-plugin
datus plugin install src:./datus-airflow-plugin
datus plugin install whl:./dist/datus_airflow_plugin-0.3.0-py3-none-any.whl
datus plugin install git:https://github.com/acme/datus-airflow-plugin
datus plugin install zip:./dist/datus-airflow-plugin-0.3.0.zip
```

The supported source types are:

| Source | Use it for |
|---|---|
| `pip:` | A package requirement from a package index. This is the default when the prefix is omitted. |
| `src:` | A local plugin project. |
| `whl:` | A wheel already built on disk. |
| `git:` | A Git repository URL. |
| `zip:` | A bundle created by `datus plugin pack` or `datus plugin export`. |

Datus installs the plugin and its dependencies under
`~/.datus/plugins/<name>/`. If the same plugin is already present, add
`--force` to replace it.

If `datus <name>` opens the chat REPL instead of the plugin, check that the
plugin is installed and [active for the current project](#activating-plugins).

## Configure a plugin

Plugin settings live under `agent.plugins.<name>` in `agent.yml`. Each child is
a named **profile**, usually representing an environment such as production or
staging:

```yaml
agent:
  plugins:
    airflow:
      prod:
        default: true
        base_url: https://airflow.example.com
        token: ${AIRFLOW_TOKEN}
      staging:
        base_url: https://airflow-staging.example.com
        token: ${AIRFLOW_STAGING_TOKEN}
```

Use `${ENV_VAR}` references for credentials instead of writing secrets directly
in the file. Datus expands those references before it runs the plugin.

Datus looks for the configuration in this order:

1. The file passed with `--config`.
2. `./conf/agent.yml` in the current project.
3. `~/.datus/conf/agent.yml`.

Configuration changes apply the next time you run the plugin; no restart is
needed.

### Managed API deployments

In a multi-tenant `datus-api` deployment, an `AuthProvider` can supply plugin
settings for each request without writing `agent.yml`. Plugin and skill
discovery stays within that request, so one tenant never falls back to another
tenant's settings or to the server's local project configuration.

`--profile` remains available, while `--config` is rejected because the
provider's configuration is authoritative. When the agent runs a plugin
through Bash, keep the command to one direct `datus <name>` invocation.
Pipelines and redirections are supported, but command substitutions and
wrappers such as `timeout`, `env`, `xargs`, or `sh -c` are rejected.

Many plugins also provide a `<name>-setup` skill. Ask the agent to configure
the plugin and it can collect the required values and create the profile for
you.

### Choose a profile

When a plugin has more than one profile, Datus chooses one in this order:

1. `--profile <name>` on the command line.
2. The profile selected for the current project in `./.datus/config.yml`.
3. A profile marked `default: true`.
4. The only configured profile, when there is just one.

A plugin that needs no configuration can run with an empty profile. If several
profiles remain and none can be selected, Datus asks you to pass `--profile`.

For example:

```bash
datus airflow --profile staging dags list
```

To make a profile the default for one project, select it in
`./.datus/config.yml`:

```yaml
plugins:
  airflow:
    enabled: true
    active_profile: [staging]
```

## Manage plugins

Use `datus plugin` from the terminal:

| Command | Purpose |
|---|---|
| `datus plugin install <source>` | Install a plugin. Add `--force` to replace an existing installation. |
| `datus plugin list` | Show installed plugins, versions, sources, profiles, and project status. |
| `datus plugin info <name>` | Show details for one plugin. |
| `datus plugin upgrade <name>` | Reinstall a plugin from its recorded `pip:`, `git:`, or `src:` source. |
| `datus plugin uninstall <name>` | Remove an installed plugin. |
| `datus plugin enable <name>` | Enable a plugin for the current project. |
| `datus plugin disable <name>` | Disable a plugin for the current project. |
| `datus plugin pack [directory]` | Build a distributable `.zip` from a plugin project. |
| `datus plugin export <name>` | Export an installed plugin as a `.zip`. |

Inside the chat REPL, `/plugins` opens an interactive manager. Use it to browse
plugins, edit profiles, enter secrets as environment-variable references, and
choose which plugins and profiles are active for the current project.

## Activating plugins

Plugins are available to every project by default. Add a `plugins:` section to
`./.datus/config.yml` when a project should use only a selected set:

```yaml
plugins:
  airflow:
    enabled: true
    active_profile: [staging]
  internal-admin:
    enabled: false
```

Once this section exists, it acts as the project's plugin list. Plugins omitted
from the list, or marked `enabled: false`, are not loaded in that project. Their
CLI commands, skills, prompt context, permission rules, and tool safeguards are
all disabled.

The CLI can update the same setting:

```bash
datus plugin enable airflow
datus plugin enable airflow --profile staging
datus plugin disable internal-admin
```

This setting belongs to the current project. It is separate from the
[global plugin switch](#disabling-the-plugin-system).

## Use a plugin with the agent

You can run a plugin directly in a terminal, or let the agent use it as part of
a task:

- Plugin-provided skills appear in `/skill list`.
- A configured plugin can describe its available environments to the agent, so
  the agent knows when the plugin is relevant.
- A setup skill can guide you through creating a profile without exposing
  literal credentials.
- In the chat REPL, `!<plugin> ...` runs a plugin command directly and returns
  the output to the conversation. See [Tool and plugin commands](../cli/execution_command.md).

Plugin context is prepared when a session starts. If you change a profile
during a session, start a new session before expecting the agent to see the
updated environment information.

In managed deployments where the agent cannot edit `agent.yml`, setup skills
are hidden. An administrator must configure the plugin instead.

## Permissions

Commands that the agent runs go through Datus's permission system. A plugin can
mark its own commands as:

- `allow`: the agent may run the command without asking.
- `ask`: the user must confirm first.
- `deny`: the agent may not run the command.

These rules apply only when the agent runs the plugin. Commands you type
directly in a terminal are unaffected.

A plugin can define rules only for its own `datus <name> ...` command. It
cannot grant access to another plugin or to unrelated shell commands. Rules in
your `agent.yml` remain authoritative, and a plugin can never override a user
`deny`.

When an `ask` command is approved with **allow (project)**, Datus remembers that
exact command pattern in the current project's `.datus/config.yml`.

## Offline installation {#offline-install}

Create a bundle on a machine that has network access:

```bash
# From the plugin project directory
datus plugin pack -o ./dist
datus plugin pack --with-deps -o ./dist
```

The default bundle contains the plugin wheel only. The target machine still
needs access to a package index to resolve dependencies.

Use `--with-deps` for a fully offline bundle. It includes the plugin and all
dependency wheels, so build it on the same operating system and Python version
as the target when any dependency contains native code.

Install the resulting file with:

```bash
datus plugin install zip:./dist/datus-airflow-plugin-0.3.0.zip
```

You can also export an installed plugin:

```bash
datus plugin export airflow -o ./dist
```

## Mount plugins from another directory {#plugin-paths}

`agent.plugin_paths` can load a plugin from a directory managed outside
`~/.datus/plugins/`:

```yaml
agent:
  plugin_paths:
    - /opt/shared/datus-plugins/airflow
    - $DATUS_PLUGIN_HOME/internal
```

Each entry must point to one installed plugin directory, not to a parent
directory containing several plugins. This is useful when several projects or
machines share centrally deployed plugins. If a mounted plugin has the same
name as a managed installation, the copy under `~/.datus/plugins/` wins.

## Disable the plugin system {#disabling-the-plugin-system}

Set the global switch in `agent.yml`:

```yaml
agent:
  plugins_enabled: false
```

This disables plugin commands, bundled skills, agent context, permission rules,
and tool safeguards. The default is `true`.

## Next steps

- [Develop a plugin](development.md) with the `datus-plugin-development` skill.
- Learn how [skills](../skills/introduction.md) are discovered and used.
