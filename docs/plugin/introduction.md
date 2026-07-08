# Plugin Introduction

A **plugin** is an installable Python package that extends Datus without
modifying it. Install one into the same Python environment as `datus` and,
depending on what the plugin ships, you get:

| Surface | What it adds |
|---|---|
| CLI subcommand | `datus <plugin> ...` runs the plugin's own command-line interface |
| Skills | plugin-bundled skills appear in `/skill list`, alongside project and user skills |
| Agent awareness | the plugin describes itself and its configured environments in the agent's system prompt, so the model chooses it proactively |
| Bash permissions | the plugin pre-declares which of its subcommands the agent may auto-run and which need confirmation |
| Tool transformers | the plugin can rewrite or deny the agent's tool calls before execution (e.g. enforce SQL scoping policies) |

Plugins are discovered through the `datus.plugins` Python entry-point group on
every invocation — installing or upgrading a plugin requires no restart and no
registration step.

Want to build one? See the [development guide](development.md).

## Installing a plugin

Install the plugin package into the same environment as `datus`:

```bash
pip install datus-plugin-hello
datus hello Ada          # the subcommand is available immediately
```

If `datus <name>` falls through to the REPL instead of running the plugin, the
package is not installed in the environment `datus` runs from.

## Configuration

Plugins are configured under `agent.plugins.<name>` in `agent.yml`, where each
key below `<name>` is a **profile** — one named environment (endpoint,
credentials, options). A plugin can have any number of profiles:

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true              # picked when --profile is omitted (see below)
        greeting: Hi
        token: ${HELLO_TOKEN}      # prefer ${ENV_VAR} for secrets
      staging:
        greeting: Yo
```

Datus resolves the config file in this order: explicit `--config` →
`./conf/agent.yml` (project) → `~/.datus/conf/agent.yml` (user default). Put
the profile in whichever file your datus session actually loads.

`${VAR}` references are expanded from environment variables per profile —
always use them for secrets instead of literal values. Config edits take
effect on the next `datus <plugin>` invocation; no restart is needed.

Some plugins ship a `<name>-setup` skill that writes this configuration for
you — see [Using a plugin with the agent](#using-a-plugin-with-the-agent).

### Which profile runs

When you run `datus <name> ...`, the active profile is resolved in this order:

1. Explicit `--profile <p>` on the command line
   (`datus hello --profile staging ...`).
2. Project pin in `./.datus/config.yml` (see below).
3. The profile flagged `default: true` (more than one is an error).
4. The sole profile, if only one is configured.
5. No `agent.plugins.<name>` section at all → the plugin runs with an empty
   configuration (config-free plugins still work).
6. Multiple profiles with no way to disambiguate → Datus errors and asks you
   to pass `--profile`.

### Pinning a profile per project

To make one project always use a specific profile without typing `--profile`,
pin it in the project's `./.datus/config.yml`:

```yaml
plugins:
  hello: staging
```

## Using a plugin with the agent

Beyond running `datus <name> ...` yourself, plugins integrate with the agent:

- **Skills** — plugin-bundled skills show up in `/skill list` and can be
  invoked like any other skill.
- **Prompt awareness** — a configured plugin lists its environments in the
  agent's system prompt, so the model knows the plugin exists and picks it
  proactively. Ask the agent "which plugins are configured?" to see what it
  knows. The prompt section refreshes at session start; config edits made
  mid-session appear in the next session.
- **Guided setup** — an installed-but-unconfigured plugin typically announces
  itself in the prompt and points the agent at its bundled `<name>-setup`
  skill. Ask the agent to set the plugin up, and it collects the required
  values and writes the profile for you (secrets are referenced as `${VAR}`,
  never written literally).

## Agent bash permissions

When the **agent** (not you) runs a plugin's CLI through its bash tool, the
command goes through Datus' permission layer. Plugins can pre-declare, per
permission profile (`normal` / `auto`), which of their subcommands are safe to
auto-run (`allow`), which require confirmation (`ask`), and which are blocked
(`deny`). Without a declaration, every agent-issued plugin command prompts for
confirmation.

What this means in practice:

- **Plugin declarations are namespace-scoped.** A plugin can only shape rules
  for `datus <its-own-name> ...` — never for `rm`, other plugins, or anything
  else.
- **Your rules always win.** A `deny` rule you write under
  `permissions.bash_commands` in `agent.yml` overrides any plugin `allow`
  (precedence is deny > ask > allow), and plugin declarations never change a
  profile's default posture.
- **`ask` can be relaxed per project.** When the agent hits a plugin-declared
  `ask` subcommand, the confirmation prompt offers **allow (project)** —
  choosing it persists the exact matched pattern to the project's
  `.datus/config.yml` `bash_allow` list, and that subcommand auto-runs from
  then on. The grant never widens beyond the exact pattern, and plugin `deny`
  rules are unaffected.
- **Only the agent is gated.** Typing `datus <name> ...` in a terminal
  yourself is never affected.
- The `dangerous` permission profile ignores all command-level bash rules by
  design, including plugin declarations.

## Disabling the plugin system

`agent.plugins_enabled: false` in `agent.yml` is a master switch that turns
off **all** plugin functionality — `datus <plugin>` dispatch, plugin-bundled
skills, prompt injection (including setup guidance), permission declarations,
and tool transformers. Recommended for API/web deployments where the agent
must not be guided to edit configuration files. The default is `true`.

## Next steps

- [Plugin Development](development.md) — build your own plugin, from a minimal
  `hello` command to the full contract.
- [Skills](../skills/introduction.md) — how skills work, including
  plugin-bundled ones.
