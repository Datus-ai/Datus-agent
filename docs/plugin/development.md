# Plugin Development

This guide shows how to build a Datus plugin, from a minimal working `hello`
command to the full contract. For what plugins are, how users install and
configure them, and how profiles are resolved, start with the
[introduction](introduction.md).

A plugin is an installable Python package discovered through the
`datus.plugins` entry-point group. The defining constraint: **a plugin never
imports `datus.*` and depends on no shared SDK**. The contract is a small set
of method names that Datus calls by structure (duck typing). Datus is the
*config broker* — it reads `agent.yml`, expands `${VAR}` references, resolves
the active profile, constructs your plugin with a plain `dict`, and calls it.
You just implement the methods.

## Prerequisites

- A Python package you can install into the same environment as `datus`.
- `datus` installed (`pip install datus-agent` or a source checkout).
- Python 3.12+ — a plugin runs inside datus' own interpreter
  (`datus-agent` declares `requires-python >= 3.12`), so your code and
  dependencies must be compatible with it.

## Quickstart: a minimal plugin

A plugin is one class registered under `datus.plugins`. Here is the smallest
useful example — a `hello` command.

**1. Package layout**

```
datus-plugin-hello/
├── pyproject.toml
└── datus_plugin_hello/
    ├── __init__.py
    └── plugin.py
```

**2. The plugin class** (`datus_plugin_hello/plugin.py`)

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional


class HelloPlugin:
    def __init__(self, profile: Optional[Dict[str, Any]] = None) -> None:
        # `profile` is the resolved agent.plugins.hello.<profile> dict
        # (already ${VAR}-expanded by datus). Empty dict is fine.
        self.profile: Dict[str, Any] = profile or {}

    def run_cli(self, argv: List[str]) -> int:
        greeting = self.profile.get("greeting", "Hello")
        name = argv[0] if argv else "world"
        print(f"{greeting}, {name}!")
        return 0
```

**3. Register the entry point** (`pyproject.toml`)

```toml
[project]
name = "datus-plugin-hello"
version = "0.1.0"
dependencies = []                      # note: NOT datus

[project.entry-points."datus.plugins"]
hello = "datus_plugin_hello.plugin:HelloPlugin"
```

The entry-point name (`hello`) alone determines the CLI command
(`datus hello`) and the config key (`agent.plugins.hello`) — the class and
module names are free. Two names are **reserved** and never dispatched to
plugins: `upgrade` and `skill`. A plugin registered under either is silently
unreachable, and names starting with `-` cannot be dispatched at all.

**4. Install and run**

```bash
pip install -e datus-plugin-hello
datus hello Ada          # -> Hello, Ada!
```

That is a complete plugin. Everything below is optional surface area.

## The contract

Datus calls these members **by name** on the class the entry point resolves to.
Your class does not import or subclass anything from Datus.

| Member | Kind | Purpose |
|---|---|---|
| `PluginClass(profile: dict)` | constructor | Datus passes the resolved `agent.plugins.<name>.<profile>` dict (env-expanded) as a **keyword argument** — `PluginClass(profile=...)` — so the parameter must be named `profile`. A config-free plugin may ignore its value. |
| `run_cli(self, argv: list[str]) -> int \| None` | instance method | Runs the subcommand. `argv` is everything after `datus <plugin>`, with Datus' own `--profile` / `--config` already stripped. Return an exit code; `None` means `0`. |
| `skills_dir() -> str \| None` | **optional**, class-level | Returns the bundled skill directory. See [Bundling skills](#bundling-skills). |
| `system_prompt(profiles: dict[str, dict]) -> str \| None` | **optional**, class-level | Returns a markdown block injected into the agent's system prompt. See [System-prompt injection](#system-prompt-injection). |
| `cli_permissions() -> dict \| None` | **optional**, class-level | Declares bash-permission rules for the plugin's own CLI namespace, per permission profile. See [CLI bash permissions](#cli-bash-permissions). |
| `tool_transformers() -> dict \| None` | **optional**, class-level | Declares tool argument transformers that rewrite or deny the agent's tool calls before execution. See [Tool argument transformers](#tool-argument-transformers). |

!!! warning "`skills_dir` and `system_prompt` must be class-reachable"
    Datus resolves both **at startup, without an active profile** (skill
    discovery and prompt building happen before any command runs). Declare them
    as `@classmethod` / `@staticmethod` (or a plain class attribute for
    `skills_dir`) — they must not depend on `__init__`.

## Configuration: what Datus hands you

Users configure your plugin under `agent.plugins.<name>`, where each key below
`<name>` is a **profile** (an environment):

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true
        greeting: Hi
        token: ${HELLO_TOKEN}      # prefer ${ENV_VAR} for secrets
      staging:
        greeting: Yo
```

Datus parses this into `agent.plugins.<name>.<profile> -> dict`, **expands
`${VAR}` per profile**, and injects a `name` key equal to the profile name.
Which profile dict reaches your constructor is decided by Datus — explicit
`--profile`, project pin, `default: true`, sole profile, or an empty dict
when nothing is configured. The full resolution order is documented in the
[introduction](introduction.md#which-profile-runs); you never write any of
that logic. Your constructor simply receives the resolved `dict`.

When testing locally, put your profile in whichever config file your datus
session actually loads (explicit `--config` → `./conf/agent.yml` →
`~/.datus/conf/agent.yml`).

## Implementing `run_cli`

`argv` is the command tail with Datus' global flags removed:

```
datus hello --profile staging greet Ada
                └── stripped ──┘ └── argv = ["greet", "Ada"] ──┘
```

Only `--profile` / `--config` appearing **before the first non-option token**
are consumed as Datus globals; from the first command token onward everything
belongs to the plugin. `datus hello greet --profile staging` therefore passes
`["greet", "--profile", "staging"]` through untouched — your subcommands are
free to define their own `--profile` option.

Return an integer exit code. Suggested conventions:

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | runtime error |
| `2` | usage error |
| `3` | config error |
| `8` | missing optional dependency |

Raising is also fine — Datus catches exceptions from `run_cli` and maps them to
exit code `1` rather than crashing the CLI — but returning an explicit code
gives users clearer signals.

## Recipes: wrapping functions and APIs into a CLI

`run_cli` receives a raw `argv` list, so you are free to route it however you
like. Here are four common patterns, from quickest to richest.

### A. Dict dispatch — a few functions, zero dependencies

The fastest way to expose a handful of functions. Map the first token to a
handler; each handler gets the rest of `argv`.

```python
class ToolboxPlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        if not argv:
            print("usage: datus toolbox <add|upper> ...")
            return 2
        cmd, rest = argv[0], argv[1:]
        handlers = {"add": self._add, "upper": self._upper}
        handler = handlers.get(cmd)
        if handler is None:
            print(f"unknown command: {cmd}")
            return 2
        return handler(rest)

    def _add(self, args):          # datus toolbox add 1 2 3
        print(sum(float(a) for a in args))
        return 0

    def _upper(self, args):        # datus toolbox upper hello
        print(" ".join(args).upper())
        return 0
```

### B. argparse — typed args, flags, auto usage/`-h`

Stdlib, no extra dependency. `argparse` prints usage and raises `SystemExit`
on `-h` or a bad invocation; Datus surfaces that as the exit code (0 for `-h`,
2 for usage errors), which is the conventional CLI behavior.

```python
import argparse

class ToolboxPlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        parser = argparse.ArgumentParser(prog="datus toolbox")
        sub = parser.add_subparsers(dest="cmd", required=True)

        p_add = sub.add_parser("add", help="sum numbers")
        p_add.add_argument("nums", nargs="+", type=float)

        p_grep = sub.add_parser("grep", help="filter lines in a file")
        p_grep.add_argument("pattern")
        p_grep.add_argument("path")
        p_grep.add_argument("-i", "--ignore-case", action="store_true")

        ns = parser.parse_args(argv)      # SystemExit on -h / bad usage
        if ns.cmd == "add":
            print(sum(ns.nums))
            return 0
        if ns.cmd == "grep":
            return self._grep(ns.pattern, ns.path, ns.ignore_case)

    def _grep(self, pattern, path, ignore_case):
        needle = pattern.lower() if ignore_case else pattern
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                hay = line.lower() if ignore_case else line
                if needle in hay:
                    print(line.rstrip())
        return 0
```

### C. Wrapping a REST API

Read the endpoint and credentials from the profile (Datus already expanded
`${VAR}`), then map subcommands to requests. Keep credentials in the profile —
never hard-code them, and never echo them.

```python
import argparse
import json

class PetstorePlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        import requests  # a plugin may depend on its own libraries

        base = self.profile.get("api_base_url")
        if not base:
            print("no api_base_url configured for the profile")
            return 3
        headers = {}
        if self.profile.get("token"):
            headers["Authorization"] = f"Bearer {self.profile['token']}"

        parser = argparse.ArgumentParser(prog="datus petstore")
        sub = parser.add_subparsers(dest="cmd", required=True)
        sub.add_parser("list-pets")
        p_get = sub.add_parser("get-pet")
        p_get.add_argument("id")
        ns = parser.parse_args(argv)

        base = base.rstrip("/")
        if ns.cmd == "list-pets":
            resp = requests.get(f"{base}/pets", headers=headers, timeout=30)
        else:
            resp = requests.get(f"{base}/pets/{ns.id}", headers=headers, timeout=30)

        if resp.status_code >= 400:
            print(f"error {resp.status_code}: {resp.text}")
            return 1
        print(json.dumps(resp.json(), indent=2))
        return 0
```

Corresponding config:

```yaml
agent:
  plugins:
    petstore:
      prod:
        default: true
        api_base_url: https://api.example.com/v1
        token: ${PETSTORE_TOKEN}
```

### D. Typer / Click — richest UX, one extra dependency

For a large command surface, a framework like [Typer](https://typer.tiangolo.com/)
gives you help text, type coercion, and completion. Because Datus constructs
your plugin per-invocation but the Typer app is a module-level object, expose
the active profile through a module global that commands read.

```python
import typer

app = typer.Typer(add_completion=False)
_ACTIVE_PROFILE: dict = {}


@app.command("greet")
def greet(name: str, loud: bool = False):
    greeting = _ACTIVE_PROFILE.get("greeting", "Hello")
    msg = f"{greeting}, {name}!"
    print(msg.upper() if loud else msg)


class GreeterPlugin:
    def __init__(self, profile=None):
        self.profile = profile or {}

    def run_cli(self, argv):
        global _ACTIVE_PROFILE
        _ACTIVE_PROFILE = self.profile
        try:
            # standalone_mode=False stops Click from calling sys.exit itself,
            # so we can return a code and always clear the profile.
            app(args=argv, standalone_mode=False)
            return 0
        except SystemExit as exc:      # -h / usage
            return int(exc.code or 0)
        except typer.Exit as exc:
            return exc.exit_code
        finally:
            _ACTIVE_PROFILE = {}
```

Add `typer` to your package's `dependencies` (your plugin's deps are its own —
just not `datus`).

## Bundling skills

If your package ships a skill directory, expose it via a class-level
`skills_dir()` and Datus will discover the skills at startup (they show up in
`/skill list`, alongside project and user skills).

```python
class HelloPlugin:
    @classmethod
    def skills_dir(cls) -> str:
        from pathlib import Path
        return str(Path(__file__).parent / "skills")
```

Layout and packaging:

```
datus_plugin_hello/
└── skills/
    └── hello/
        └── SKILL.md
```

A minimal `SKILL.md` is YAML frontmatter plus markdown instructions (the
frontmatter follows the [agentskills.io](https://agentskills.io) spec used by
the Skills system):

```markdown
---
name: hello
description: Say hello to someone via the `datus hello` CLI
---

# Hello

Run `datus hello <name>` to greet someone. ...
```

See the [Skills](../skills/introduction.md) docs for the full frontmatter
field reference.

Make sure the skill files are included in the wheel (they are data, not
Python). Hatchling packages every file under the package directory by default,
so nothing extra is needed unless the files are VCS-ignored (then list them
under `[tool.hatch.build.targets.wheel] artifacts`). With setuptools you must
opt in explicitly:

```toml
[tool.setuptools.package-data]
datus_plugin_hello = ["skills/**/*"]
```

After building, verify with `unzip -l dist/*.whl | grep SKILL.md`.

## System-prompt injection

A plugin can tell the agent, up front, what it is and which environments are
configured — so the model chooses it proactively instead of guessing. Expose a
class-level `system_prompt(profiles)`:

```python
class HelloPlugin:
    @classmethod
    def system_prompt(cls, profiles):
        if not profiles:
            # Installed but unconfigured: point the agent at the setup
            # skill instead of disappearing from the prompt.
            return (
                "## Hello (installed, not configured)\n"
                "The `datus hello` CLI is installed but has no environment "
                "configured.\nRun the `hello-setup` skill to configure one."
            )
        envs = "\n".join(
            f"- {name}: {cfg.get('greeting', '?')}"
            for name, cfg in profiles.items()
        )
        return (
            "## Hello\n"
            "Say hello via `datus hello <name>`.\n"
            f"Environments ({len(profiles)}):\n{envs}"
        )
```

Datus passes the plugin's **full** profile mapping (all environments, not just
the active one) and appends the returned markdown to the system prompt of every
agentic node. An installed-but-unconfigured plugin receives `{}` — return a
short "installed, not configured" note pointing at your bundled setup skill
(see below) so the agent can walk the user through configuration. Return
`None` only when there is truly nothing to say.

When at least one plugin contributes a section, Datus prepends its own
`## Plugins` preamble naming the loaded config file and the
`agent.plugins.<plugin>.<profile>` shape — your text never needs to hard-code
config paths.

!!! danger "Never surface secrets"
    The returned text enters the LLM context. Datus hands you the full profile
    dicts — which include `password`, secret keys, access keys — but you must
    emit **only non-secret fields** (endpoints, region, environment names).
    Datus never splices profile values itself; keeping credentials out of the
    prompt is the plugin's responsibility. Use a field allow-list.

## CLI bash permissions

When the **agent** (not a human) runs your CLI through its bash tool — e.g. the
model decides to execute `datus hello greet Ada` — the command goes through
Datus' permission layer. Without a declaration, every such command prompts the
user for confirmation. A class-level `cli_permissions()` lets your plugin
declare, per permission profile, which of its subcommands are safe to auto-run
(`allow`), which must be confirmed (`ask`), and which are blocked (`deny`):

```python
class HelloPlugin:
    @classmethod
    def cli_permissions(cls):
        return {
            "normal": {"allow": ["greet:*"], "ask": ["config set:*"]},
            "auto":   {"allow": ["greet:*", "config set:*"]},
        }
```

Semantics:

- **Patterns are relative to your namespace.** Datus prefixes each pattern
  with `datus <name> `, so `greet:*` becomes `datus hello greet:*`. A plugin
  can never affect commands outside `datus <name>` — not `rm`, not another
  plugin.
- **Pattern syntax** matches `permissions.bash_commands` in `agent.yml`:
  `cmd` is an exact match, `cmd:*` a prefix match, `cmd:glob` a prefix match
  whose first argument must satisfy the glob (e.g. `greet:A*`). A bare `:*`
  covers the whole namespace.
- **Profile keys**: only `normal` and `auto` are accepted. The `dangerous`
  profile ignores all command-level bash rules by design; a `dangerous` key is
  warned about and dropped.
- **Users always win.** A user `deny` rule in `agent.yml` overrides a plugin
  `allow` (deny > ask > allow, regardless of declaration order), and plugin
  declarations can never change a profile's default posture.
- **`ask` rules can be relaxed per project.** When the agent hits one of your
  `ask` subcommands, the confirmation prompt offers "allow (project)" —
  choosing it persists the exact matched pattern (e.g.
  `datus hello config set:*`) to the project's `.datus/config.yml`
  `bash_allow` list, and that subcommand auto-runs from then on. The grant is
  exact-match only: it never widens to the rest of your namespace, and your
  `deny` rules are unaffected. (User-authored `ask` rules from `agent.yml` do
  not get this option — relaxing those belongs in the user's own config.)
- **Scope**: only the agent's bash tool is gated. A human typing
  `datus hello ...` in a terminal is never affected. `plugins_enabled: false`
  disables collection along with the rest of the plugin system (see the
  [introduction](introduction.md#disabling-the-plugin-system)).
- **`--profile` is transparent to matching.** `datus hello --profile prod
  config set x` matches the same rules (and the same project grants) as the
  unqualified form — the leading datus-global flag is normalized away before
  evaluation. `--config <path>` is deliberately *not* normalized: pointing
  datus at a different config file rebinds credentials, so those invocations
  always fall back to a confirmation.
- Malformed declarations (wrong types, unknown keys, empty patterns) are
  logged and skipped — they never break Datus startup.

Declare read-only subcommands as `allow` and state-changing ones as `ask`
under `normal`; promote routine state changes to `allow` under `auto` only
when re-running them is harmless.

## Tool argument transformers

A class-level `tool_transformers()` lets your plugin intercept the **agent's
tool calls** — inspect and rewrite the arguments before the tool executes, or
deny the call outright. The canonical use case is SQL policy enforcement:
append a tenant-scope predicate to every `execute_sql` query, using the
request principal the deployment injects.

```python
class ScopedSqlPlugin:
    @classmethod
    def tool_transformers(cls):
        return {"db_tools.execute_sql": enforce_tenant_scope}


def enforce_tenant_scope(tool_name, args, context):
    tenant_id = (context.get("principal") or {}).get("tenant", {}).get("id")
    if not tenant_id:
        raise PermissionError("missing principal.tenant.id; cannot scope query")
    args["sql"] = add_where_predicate(args["sql"], f"tenant_id = '{tenant_id}'")
    return args
```

Semantics:

- **Declaration shape**: a dict mapping tool patterns to a transformer or a
  list of transformers. Patterns use the proxy syntax — a bare tool name
  (`execute_sql`), or `category.method` with fnmatch globs (`db_tools.*`).
- **Transformer signature**: `transformer(tool_name, args, context) -> dict`,
  sync or async. Return the (possibly modified) argument dict to continue.
  **Raise to deny**: the tool never runs and the model receives your
  exception message as a normal tool failure. Returning anything that is not
  a dict also denies, fail closed.
- **`context`** is a plain dict with `node_name`, `principal` (request-scoped
  caller attributes, empty when the deployment sets none), `project_root`,
  and `agent_config` (the live agent configuration object — read your own
  profile via `context["agent_config"].get_plugin_profile("<name>")`; access
  it duck-typed, never import `datus.*` for it). It is rebuilt on every call,
  so per-request values are always fresh.
- **Coverage**: transformers wrap the agent's `FunctionTool` layer, which
  both execution paths (SDK Runner and the native loop) go through. They do
  **not** cover direct Python invocations of tool methods (e.g.
  reference-template execution) or tools proxied to an external client —
  server-side enforcement that must survive those paths belongs in the tool
  layer itself (see `agent.sql_policy`).
- **Trust model**: transformers run in-process with full access to every
  matched tool call's arguments. They are trusted code, gated by the same
  `plugins_enabled` master switch as the rest of the plugin surface.
- Use a SQL parser or a database-safe query builder when rewriting SQL —
  never string concatenation for policy predicates.
- Malformed declarations (non-dict, non-callable entries, empty patterns)
  are logged and skipped — they never break Datus startup. A declaration
  that collects successfully but fails to apply aborts the agent node
  instead of silently running without enforcement.

## Bundling a setup skill

Editing YAML by hand is the main friction after `pip install`. Ship a
`<name>-setup` skill next to your main skill so the agent can collect the
values and write the profile itself:

```
datus_plugin_hello/
└── skills/
    ├── hello/
    │   └── SKILL.md
    └── hello-setup/
        └── SKILL.md
```

The setup `SKILL.md` should cover, in order:

1. **When to use** — the plugin is unconfigured, or the user wants another
   environment.
2. **Config structure** — a complete YAML template for
   `agent.plugins.<name>.<profile>`, with comments marking required / optional
   / secret fields.
3. **Ask the user** — list the fields that must come from the user (endpoint,
   auth choice, ...). For secrets, instruct the agent to have the user export
   an environment variable and reference it as `${VAR}` in the YAML — never
   write literal secrets to the file.
4. **Write the config** — into the file named by the `## Plugins` prompt
   preamble, marking the first profile `default: true`.
5. **Verify** — a cheap read-only command (e.g. `datus hello version`).
   `datus <plugin>` reloads the config on every invocation, so the profile
   works immediately; the prompt's environment list refreshes next session.

Add a guard note: if the current environment cannot edit the config file
(API / VSCode / web deployment), the agent should tell the user to edit
`agent.yml` on the server instead.

A complete minimal `hello-setup/SKILL.md`:

````markdown
---
name: hello-setup
description: Configure an environment profile for the `datus hello` plugin
---

# Hello Setup

Use this skill when `datus hello` is installed but has no configured
environment, or when the user wants to add another one.

## Config structure

Profiles live under `agent.plugins.hello.<profile>` in the config file named
by the `## Plugins` section of the system prompt:

```yaml
agent:
  plugins:
    hello:
      prod:
        default: true            # mark the first profile as default
        greeting: Hi             # required
        token: ${HELLO_TOKEN}    # secret — reference an env var, never a literal
```

## Steps

1. Ask the user for `greeting` and which environment variable holds the
   token. Have the user export the variable; write `${VAR}` into the YAML —
   never a literal secret.
2. Write the profile into the config file above; mark the first profile
   `default: true`.
3. Verify with a cheap read-only call: `datus hello Ada`.

If this environment cannot edit the config file (API / web deployment), tell
the user to edit `agent.yml` on the server instead.
````

## Verifying your plugin end-to-end

After `pip install -e`, each surface can be checked without restarting
anything (plugins are discovered per invocation):

- **CLI dispatch** — run `datus <name> ...` from any directory. If it falls
  through to the REPL instead, the entry point is missing or misnamed; check
  `pip show -f your-package` for the `entry_points.txt`.
- **Skills** — start `datus` and run `/skill list`; plugin-bundled skills
  appear alongside project and user skills.
- **Prompt injection** — the easiest check is a unit test calling
  `system_prompt()` directly (next section). To confirm it lands in a live
  session, start `datus` and ask the agent "which plugins are configured?" —
  the answer comes from the injected section. Note that config edits take
  effect on the next `datus <plugin>` invocation immediately, but the prompt
  section refreshes only on the next session.

## Testing your plugin

Because Datus is the broker, unit tests construct your plugin with a plain dict
— no `agent.yml`, no Datus imports:

```python
from datus_plugin_hello.plugin import HelloPlugin

def test_run_cli_uses_profile_greeting(capsys):
    rc = HelloPlugin(profile={"name": "prod", "greeting": "Hi"}).run_cli(["Ada"])
    assert rc == 0
    assert "Hi, Ada!" in capsys.readouterr().out

def test_system_prompt_lists_envs_without_secrets():
    text = HelloPlugin.system_prompt({
        "prod": {"name": "prod", "greeting": "Hi", "token": "s3cr3t"},
    })
    assert "## Hello" in text
    assert "s3cr3t" not in text          # secrets must never leak

def test_system_prompt_unconfigured_points_to_setup_skill():
    text = HelloPlugin.system_prompt({})
    assert "not configured" in text
    assert "hello-setup" in text
```

## Constraints checklist

Before publishing, verify:

- [ ] The package does **not** `import datus` anywhere (`grep -rn "import datus" your_pkg/`).
- [ ] The package does **not** depend on `datus` or a shared plugin SDK in `pyproject.toml`.
- [ ] `__init__` accepts the profile as a keyword argument named `profile` (Datus calls `PluginClass(profile=...)`).
- [ ] The entry-point name is not a reserved name (`upgrade`, `skill`) and does not start with `-`.
- [ ] `skills_dir`, `system_prompt`, and `cli_permissions` are class-reachable (`@classmethod` / `@staticmethod` / class attribute).
- [ ] `system_prompt` emits only non-secret fields.
- [ ] `cli_permissions` patterns are namespace-relative (no `datus <name>` prefix — Datus adds it), and state-changing subcommands are `ask` under `normal`.
- [ ] `run_cli` returns an int (or `None`) and does not call `sys.exit()` on the success path.
- [ ] Skill files are packaged into the wheel.
- [ ] The `datus.plugins` entry-point name matches the intended `datus <name>` command and the `agent.plugins.<name>` config key.

## Reference

- **Entry-point group**: `datus.plugins` — one entry per plugin, resolving to a plugin **class**.
- **Contract source of truth**: `datus/plugins/base.py` (documented `DatusPlugin` protocol).
- **Related**: [Plugin Introduction](introduction.md), [Skills](../skills/introduction.md).
