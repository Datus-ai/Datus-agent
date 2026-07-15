# Plugin Introduction

A **plugin** is an installable Python package that extends Datus without
modifying it. `datus plugin install` puts each plugin in its own directory under
`~/.datus/plugins/{name}/` (with its dependencies vendored in), and — depending
on what the plugin ships — you get:

| Surface | What it adds |
|---|---|
| CLI subcommand | `datus <plugin> ...` runs the plugin's own command-line interface |
| Skills | plugin-bundled skills appear in `/skill list`, alongside project and user skills |
| Agent awareness | the plugin describes itself and its configured environments in the agent's system prompt, so the model chooses it proactively |
| Bash permissions | the plugin pre-declares which of its subcommands the agent may auto-run and which need confirmation |
| Tool transformers | the plugin can rewrite or deny the agent's tool calls before execution (e.g. enforce SQL scoping policies) |

Each enabled plugin's directory is appended to `sys.path` at startup, so its
`datus.plugins` entry point is discovered on every invocation — no restart and
no registration step. (A plugin installed the old way, straight into the same
Python environment with plain `pip install`, is still discovered as a fallback.)

Want to build one? See the [development guide](development.md).

## Installing a plugin

`datus plugin install` takes a `{type}:{src}` source and installs the plugin
into `~/.datus/plugins/{name}/`. The type prefix is optional and defaults to
`pip`, so a bare requirement is treated as `pip:`:

```bash
datus plugin install datus-plugin-hello                           # defaults to pip: (a PyPI requirement)
datus plugin install pip:datus-plugin-hello                       # the same, spelled out
datus plugin install src:./datus-plugin-hello                     # a local project directory
datus plugin install whl:./dist/hello-1.0-py3-none-any.whl        # a local wheel file
datus plugin install git:https://github.com/acme/datus-plugin-hello   # a git repository
datus plugin install zip:./dist/datus-plugin-hello-1.0.0.zip      # an offline bundle (see below)

datus hello Ada          # the subcommand is available immediately
```

`pip`, `src`, `whl`, and `git` resolve dependencies from a package index (they
need network); the plugin and its dependencies are vendored into the plugin's
directory via `pip install --target`. `datus plugin install` records how the
plugin was installed in `~/.datus/plugins/{name}/datus-plugin.json`, so
`datus plugin upgrade <name>` can later re-fetch it the same way. If the plugin
is already installed, pass `--force` to replace it.

If `datus <name>` falls through to the REPL instead of running the plugin, it is
not installed, or it is not active for this project (see
[Activating plugins](#activating-plugins)).

### Offline install (air-gapped)

An offline **bundle** is an ordinary `.zip` holding a wheelhouse (the plugin
wheel and, optionally, every dependency wheel). Build one where you *do* have
network with `datus plugin pack`, copy the file across, and install it with
`zip:`:

```bash
# on a networked machine, from the plugin's project directory
datus plugin pack --with-deps -o ./dist     # plugin wheel + all dependency wheels
datus plugin pack -o ./dist                 # plugin wheel only (default)

# on the target machine
datus plugin install zip:./dist/datus-plugin-hello-1.0.0.zip
```

`pack` defaults to **plugin wheel only** — small, but the target machine
resolves dependencies from an index at install time (needs network). Add
`--with-deps` to bundle every dependency wheel so the install is **fully
offline** (`pip install --no-index --find-links`). Every wheel's checksum is
verified against the bundle manifest before anything is installed; if the bundle
was built for a different Python version or platform, install refuses with a
clear message unless you pass `--force` (checksums are still enforced). Because a
with-deps bundle is a same-platform snapshot, build it on a machine matching the
target's OS/Python.

### Exporting an installed plugin

`datus plugin export <name>` reproduces a distributable `.zip` from an already
installed plugin — a `zip:` install returns its retained original bundle
verbatim, while a `pip`/`src`/`whl`/`git` install is re-packed from its recorded
source (needs network).

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
pin it in the project's `./.datus/config.yml` under `plugins.<name>
.active_profile`. When exactly one profile is active it becomes the
`datus <plugin>` default:

```yaml
plugins:
  hello:
    enabled: true
    active_profile:
      - staging
```

## Activating plugins

The `plugins:` section of `./.datus/config.yml` also decides **which installed
plugins are active for the project**. It is optional but authoritative:

- **Omit it entirely** and every installed plugin — and all of its profiles —
  is active. This is the default.
- **Write it** and it becomes a whitelist. Each entry is
  `{enabled: bool, active_profile: [<profile>, ...]}`. A plugin the section
  does not list (or lists with `enabled: false`) is **not loaded** for the
  project: its `datus <plugin>` subcommand is refused, and its bundled skills,
  system-prompt section, tool transformers, and bash rules are all skipped.
  `active_profile` narrows which configured profiles are active (omit it for
  "all profiles").

```yaml
plugins:
  hello:
    enabled: true
    active_profile: [staging]   # only 'staging' is active
  noisy-plugin:
    enabled: false              # installed but off for this project
```

Toggle activation without hand-editing the file:

```bash
datus plugin enable hello                    # activate (all profiles)
datus plugin enable hello --profile staging  # activate, pinned to 'staging'
datus plugin disable noisy-plugin            # deactivate for this project
```

The first `enable`/`disable` in a project seeds the whitelist with every
installed plugin (all enabled) before applying your change, so turning one
plugin off never silently deactivates the others.

This is per-project activation, distinct from the global master switch
[`agent.plugins_enabled`](#disabling-the-plugin-system) which turns the whole
system off everywhere.

## Managing plugins

`datus plugin` is the management entry point (it always works, even for a
deactivated plugin):

| Command | What it does |
|---|---|
| `datus plugin install '[{type}:]{src}'` | Install from `pip:` (default) / `src:` / `whl:` / `git:` / `zip:` into `~/.datus/plugins/` (`--force` replaces an existing install). |
| `datus plugin pack [dir]` | Build a distributable wheelhouse `.zip` from a plugin project directory (default `./`); `--with-deps` bundles dependencies, `-o` sets the output dir. |
| `datus plugin export <name>` | Export an installed plugin as a `.zip` (`-o` sets the output dir). |
| `datus plugin upgrade <name>` | Re-install from the recorded source (pip/git/src). |
| `datus plugin uninstall <name>` | Remove the plugin's `~/.datus/plugins/{name}/` directory. |
| `datus plugin list` | List installed plugins: package, version, source, configured profiles, project activation. |
| `datus plugin info <name>` | Show one plugin's description, profiles, config schema, and activation. |
| `datus plugin enable/disable <name>` | Toggle per-project activation. |

Inside the REPL, `/plugins` opens an interactive manager for the same tasks:
browse installed plugins, create / edit / delete global profiles (the form is
driven by the plugin's config schema — nested schema objects expand into
dotted fields like `s3.secret_access_key`, defaults are pre-filled, empty
fields show their description as a dim placeholder, and secrets are entered
as `${ENV_VAR}` references), and toggle which plugins and profiles are active
for the project.

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
