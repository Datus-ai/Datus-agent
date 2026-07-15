# Plugin Development

This guide shows how to build a Datus plugin, from a minimal working `hello`
command to the full contract. For what plugins are, how users install and
configure them, and how profiles are resolved, start with the
[introduction](introduction.md).

A plugin is an installable Python package discovered through the
`datus.plugins` entry-point group. The defining constraints:

- **A plugin never imports `datus.*`** and depends on no shared SDK.
- **The whole contract is one declarative file** — `datus-plugin.yml`, shipped
  inside your package. It names your CLI entry function, your tool
  transformers, your skills directory, your system-prompt template, your bash
  permission rules, and your config schema. The only Python you write are
  plain functions.

Datus is the *config broker* — it reads `agent.yml`, expands `${VAR}`
references, resolves the active profile, and calls your declared `cli`
function with a plain `dict`. Reading the manifest never executes your code:
skills, permissions, prompt sections and config schemas are collected without
importing your package. Only your `cli` function (on `datus <name> ...`) and
your declared tool transformers are imported, lazily.

## Prerequisites

- A Python package you can install into the same environment as `datus`.
- `datus` installed (`pip install datus-agent` or a source checkout).
- Python 3.12+ — a plugin runs inside datus' own interpreter
  (`datus-agent` declares `requires-python >= 3.12`), so your code and
  dependencies must be compatible with it.

## Quickstart: a minimal plugin

**1. Package layout**

```text
datus-plugin-hello/
├── pyproject.toml
└── datus_plugin_hello/
    ├── __init__.py
    ├── datus-plugin.yml      # the manifest — the whole plugin contract
    └── cli.py
```

**2. The manifest** (`datus_plugin_hello/datus-plugin.yml`)

```yaml
manifest_version: 1
description: "Say hello to someone."
cli: datus_plugin_hello.cli:main
```

**3. The CLI function** (`datus_plugin_hello/cli.py`)

```python
from __future__ import annotations


def main(argv: list[str], profile: dict) -> int:
    # `profile` is the resolved agent.plugins.hello.<profile> dict
    # (already ${VAR}-expanded by datus). Empty dict is fine.
    greeting = profile.get("greeting", "Hello")
    name = argv[0] if argv else "world"
    print(f"{greeting}, {name}!")
    return 0
```

**4. Register the entry point** (`pyproject.toml`)

```toml
[project]
name = "datus-plugin-hello"
version = "0.1.0"
dependencies = []                      # note: NOT datus

[project.entry-points."datus.plugins"]
hello = "datus_plugin_hello"           # the PACKAGE name — not a class

[tool.setuptools.package-data]
datus_plugin_hello = ["datus-plugin.yml"]
```

The entry-point value is your **package name** — a pure name → package
mapping, no code reference. The entry-point name (`hello`) alone determines
the CLI command (`datus hello`) and the config key (`agent.plugins.hello`) —
the package name is free. Three names are **reserved** and never dispatched to
plugins: `upgrade`, `skill`, and `plugin`. A plugin registered under any of
them is unreachable (and `datus plugin install` refuses it), and names
starting with `-` cannot be dispatched at all.

The manifest is package data, not Python — make sure it ships in the wheel.
Hatchling packages every file under the package directory by default; with
setuptools use the `[tool.setuptools.package-data]` stanza above. Verify with
`unzip -l dist/*.whl | grep datus-plugin.yml` (both `datus plugin install` and
`datus plugin pack` refuse a package whose manifest is missing).

**5. Install and run**

```bash
datus plugin install src:./datus-plugin-hello   # installs into ~/.datus/plugins/hello/
datus hello Ada          # -> Hello, Ada!
```

For a tight edit-run loop while developing, `pip install -e datus-plugin-hello`
into datus' own environment also works — such plugins are still discovered as a
fallback, without a `~/.datus/plugins/` directory.

That is a complete plugin. Everything below is optional surface area.

## The manifest reference

`datus-plugin.yml` lives at your package root. Only `manifest_version` is
required; every other key is optional:

| Key | Type | Purpose |
|---|---|---|
| `manifest_version` | int, **required** | Must be `1`. A newer version than datus understands rejects the manifest with a "requires newer datus" warning. |
| `description` | string | One-line summary shown by `datus plugin info`. |
| `cli` | code ref | `module.path:function` called as `main(argv, profile)` on `datus <name> ...`. See [Implementing the CLI entry](#implementing-the-cli-entry). Without it, `datus <name>` exits 2. |
| `tool_transformers` | mapping | Tool pattern → code ref (or list of refs) that rewrite or deny the agent's tool calls. See [Tool argument transformers](#tool-argument-transformers). |
| `permissions` | mapping | Bash-permission rules for your own CLI namespace, per permission profile — pure YAML, no code. See [CLI bash permissions](#cli-bash-permissions). |
| `system_prompt` | path | Package-relative path of a Jinja2 template rendered into the agent's system prompt. See [System-prompt template](#system-prompt-template). |
| `skills` | path | Package-relative path of a bundled skill directory. See [Bundling skills](#bundling-skills). |
| `config_schema` | JSON Schema | Inline object schema describing one profile — drives the `/plugins` TUI form and validates profiles before saving. See [Config schema and validation](#config-schema-and-validation). |

A **code ref** is a dotted `module.path:function` string. Paths are relative
to the package directory and may not escape it. Manifest parsing is defensive:
a malformed section is warned about and dropped while the rest of the manifest
stays usable; only a missing/unsupported `manifest_version` (or unreadable
YAML) rejects the manifest as a whole.

The machine-readable contract lives in `datus/plugins/base.py`; this table and
that docstring are kept in sync.

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
Which profile dict reaches your `cli` function is decided by Datus — explicit
`--profile`, project pin, `default: true`, sole profile, or an empty dict
when nothing is configured. The full resolution order is documented in the
[introduction](introduction.md#which-profile-runs); you never write any of
that logic. Your function simply receives the resolved `dict`.

When testing locally, put your profile in whichever config file your datus
session actually loads (explicit `--config` → `./conf/agent.yml` →
`~/.datus/conf/agent.yml`).

## Config schema and validation

Declare a `config_schema` — an inline JSON Schema for **one profile** — and
the `/plugins` TUI renders a proper form instead of free-form key/value
editing, and Datus validates a candidate profile against it before saving:

```yaml
config_schema:
  type: object
  required: [token, s3]
  properties:                    # property order == TUI field order
    token:
      type: string
      description: "API token"
      x-secret: true             # masked in the TUI, stripped from the prompt
    greeting:
      type: string
      description: "Greeting word"
      default: "Hi"
    s3:                          # nested objects expand into dotted form fields
      type: object
      required: [secret_access_key]
      properties:
        region: {type: string, default: us-east-1}
        secret_access_key: {type: string, x-secret: true}
```

Semantics:

- **`x-secret: true`** marks a secret field: the TUI masks it and prompts the
  user to enter a `${ENV_VAR}` reference, and the system-prompt renderer
  strips it (see [System-prompt template](#system-prompt-template)). It is a
  property-level extension keyword — JSON Schema validators ignore it.
- **`required`** membership marks a form field as required; **`default`** is
  pre-filled as the field's initial value. A field left empty (no default,
  nothing typed yet) shows its `description` as a dim placeholder instead.
- **Nested objects** — a `type: object` property with its own `properties`
  expands in the TUI into one field per leaf, named by its dotted path
  (`s3.region`, `s3.secret_access_key`); submitted values are re-assembled
  into the nested profile shape before saving. `x-secret: true` on the object
  marks every leaf secret, and a leaf is form-required only when its whole
  ancestor path is required too. The system-prompt whitelist filters declared
  nested objects recursively under the same rules.
- **Free-form objects pass through wholesale.** A `type: object` property
  **without** its own `properties` (a free-form dict) is *not* filtered per
  key: the TUI keeps it as a single field, and the system-prompt renderer
  passes its entire stored value into the prompt. Only per-key stripping is
  skipped — the field itself still has to be declared — so a secret nested
  inside such a field would reach the LLM. Either declare its sub-`properties`
  (to get recursive filtering) or mark the whole object `x-secret: true`.
- **Validation** runs `jsonschema` on the raw candidate dict (the values the
  user just entered, **before** `${VAR}` expansion). Values containing
  `${ENV_VAR}` placeholders are treated as opaque — pattern/enum/format
  violations on them are suppressed, while a missing `required` field still
  fails. Keep the real runtime validation in your `cli` function.
- **TUI-entered values are strings**, so prefer `type: string` with `pattern`
  / `enum` constraints; reserve other types for hand-written `agent.yml`
  profiles.
- A schema that is itself invalid (rejected by the JSON Schema meta-schema) is
  warned about and treated as absent — the TUI falls back to free-form
  editing.

## Implementing the CLI entry

The manifest's `cli` names a function called as `main(argv, profile)`:

```text
datus hello --profile staging greet Ada
                └── stripped ──┘ └── argv = ["greet", "Ada"] ──┘
```

Only `--profile` / `--config` appearing **before the first non-option token**
are consumed as Datus globals; from the first command token onward everything
belongs to the plugin. `datus hello greet --profile staging` therefore passes
`["greet", "--profile", "staging"]` through untouched — your subcommands are
free to define their own `--profile` option.

Return an integer exit code (`None` means `0`). Suggested conventions:

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | runtime error |
| `2` | usage error |
| `3` | config error |
| `8` | missing optional dependency |

Raising is also fine — Datus catches exceptions from your `cli` function and
maps them to exit code `1` rather than crashing the CLI — but returning an
explicit code gives users clearer signals.

## Recipes: wrapping functions and APIs into a CLI

Your `cli` function receives a raw `argv` list, so you are free to route it
however you like. Here are four common patterns, from quickest to richest.

### A. Dict dispatch — a few functions, zero dependencies

```python
def main(argv, profile):
    if not argv:
        print("usage: datus toolbox <add|upper> ...")
        return 2
    cmd, rest = argv[0], argv[1:]
    handlers = {"add": _add, "upper": _upper}
    handler = handlers.get(cmd)
    if handler is None:
        print(f"unknown command: {cmd}")
        return 2
    return handler(rest)


def _add(args):          # datus toolbox add 1 2 3
    print(sum(float(a) for a in args))
    return 0


def _upper(args):        # datus toolbox upper hello
    print(" ".join(args).upper())
    return 0
```

### B. argparse — typed args, flags, auto usage/`-h`

Stdlib, no extra dependency. `argparse` prints usage and raises `SystemExit`
on `-h` or a bad invocation; Datus surfaces that as the exit code (0 for `-h`,
2 for usage errors), which is the conventional CLI behavior.

```python
import argparse


def main(argv, profile):
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
        return _grep(ns.pattern, ns.path, ns.ignore_case)


def _grep(pattern, path, ignore_case):
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


def main(argv, profile):
    import requests  # a plugin may depend on its own libraries

    base = profile.get("api_base_url")
    if not base:
        print("no api_base_url configured for the profile")
        return 3
    headers = {}
    if profile.get("token"):
        headers["Authorization"] = f"Bearer {profile['token']}"

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
gives you help text, type coercion, and completion. Because Datus calls your
entry function per-invocation but the Typer app is a module-level object,
expose the active profile through a module global that commands read.

```python
import typer

app = typer.Typer(add_completion=False)
_ACTIVE_PROFILE: dict = {}


@app.command("greet")
def greet(name: str, loud: bool = False):
    greeting = _ACTIVE_PROFILE.get("greeting", "Hello")
    msg = f"{greeting}, {name}!"
    print(msg.upper() if loud else msg)


def main(argv, profile):
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = profile
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

Declare a package-relative skills directory in the manifest and Datus
discovers the skills at startup (they show up in `/skill list`, alongside
project and user skills) — no code involved:

```yaml
skills: skills
```

Layout and packaging:

```text
datus_plugin_hello/
├── datus-plugin.yml
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
datus_plugin_hello = ["datus-plugin.yml", "skills/**/*", "prompts/*"]
```

After building, verify with `unzip -l dist/*.whl | grep SKILL.md`.

## System-prompt template

A plugin can tell the agent, up front, what it is and which environments are
configured — so the model chooses it proactively instead of guessing. Declare
a Jinja2 template in the manifest:

```yaml
system_prompt: prompts/system.md.j2
```

```jinja
## Hello
Say hello via `datus hello <name>`.

{% if profiles %}
Environments ({{ profiles | length }}):
{% for name, cfg in profiles.items() %}
- {{ name }}: {{ cfg.get("greeting", "?") }}
{% endfor %}
{% else %}
Installed but not configured — run the `hello-setup` skill to configure an
environment.
{% endif %}
```

The render context:

| Variable | Value |
|---|---|
| `plugin_name` | your entry-point name |
| `profiles` | `dict[str, dict]` — the plugin's profile mapping, **narrowed to the profiles the project activated** (`plugins.<name>.active_profile` in `./.datus/config.yml`) and **secret-stripped** (see below) |
| `config_path` | the loaded agent config file path, or `None` |

An installed-but-unconfigured plugin (or one whose pin matches nothing)
renders with `profiles == {}` — use `{% if profiles %}` to emit a short
"installed, not configured" note pointing at your bundled setup skill instead
of disappearing from the prompt.

When at least one plugin contributes a section, Datus prepends its own
`## Plugins` preamble naming the loaded config file and the
`agent.plugins.<plugin>.<profile>` shape — your template never needs to
hard-code config paths.

!!! note "Secrets are stripped structurally"
    The rendered text enters the LLM context, and profile values are already
    `${VAR}`-expanded (real secrets) at prompt time — so Datus filters the
    profiles **before** your template sees them: only fields declared in your
    `config_schema` and **not** marked `x-secret: true` pass through;
    undeclared fields are dropped too, and declared nested objects (a
    `type: object` with its own `properties`) are filtered recursively under
    the same rules. The one exception is a **free-form** object field
    (declared `type: object` with no `properties`): its whole value passes
    through unfiltered, so mark it `x-secret: true` if it can hold anything
    sensitive. Without a `config_schema`, templates
    receive profile names with empty dicts. A template referencing a stripped
    field fails to render (strict mode) and the section is skipped — it can
    never leak.

Template errors (missing file, syntax error, undefined variable) are logged
and the section is skipped — they never break prompt construction. The
template renders in strict mode (`StrictUndefined`), so a typo shows up in the
log instead of as silently wrong prompt text.

## CLI bash permissions

When the **agent** (not a human) runs your CLI through its bash tool — e.g. the
model decides to execute `datus hello greet Ada` — the command goes through
Datus' permission layer. Without a declaration, every such command prompts the
user for confirmation. The manifest's `permissions` key declares, per
permission profile, which of your subcommands are safe to auto-run (`allow`),
which must be confirmed (`ask`), and which are blocked (`deny`) — pure YAML,
no code:

```yaml
permissions:
  normal:
    allow: ["greet:*"]
    ask: ["config set:*"]
  auto:
    allow: ["greet:*", "config set:*"]
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

The manifest's `tool_transformers` key lets your plugin intercept the
**agent's tool calls** — inspect and rewrite the arguments before the tool
executes, or deny the call outright. The canonical use case is SQL policy
enforcement: append a tenant-scope predicate to every `execute_sql` query,
using the request principal the deployment injects.

```yaml
tool_transformers:
  "db_tools.execute_sql": datus_plugin_scoped_sql.transformers:enforce_tenant_scope
```

```python
# datus_plugin_scoped_sql/transformers.py
def enforce_tenant_scope(tool_name, args, context):
    tenant_id = (context.get("principal") or {}).get("tenant", {}).get("id")
    if not tenant_id:
        raise PermissionError("missing principal.tenant.id; cannot scope query")
    args["sql"] = add_where_predicate(args["sql"], f"tenant_id = '{tenant_id}'")
    return args
```

Semantics:

- **Declaration shape**: a mapping of tool patterns to a code ref or a list of
  code refs. Patterns use the proxy syntax — a bare tool name
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
- **Loading is lazy**: the referenced module is imported when transformers are
  first collected for an agent node, not at manifest load. A ref that fails to
  import (or is not callable) is warned about and skipped — the plugin's
  remaining transformers still apply.
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
- A declaration that collects successfully but fails to apply aborts the
  agent node instead of silently running without enforcement.

## Bundling a setup skill

Editing YAML by hand is the main friction after `pip install`. Ship a
`<name>-setup` skill next to your main skill so the agent can collect the
values and write the profile itself:

```text
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
  through to the REPL instead, the entry point is missing or misnamed, or the
  manifest was rejected (check the log); `datus <name>` printing "declares no
  CLI command" means the manifest loaded but has no `cli` key. Check
  `pip show -f your-package` for the `entry_points.txt` and the bundled
  `datus-plugin.yml`.
- **Skills** — start `datus` and run `/skill list`; plugin-bundled skills
  appear alongside project and user skills.
- **Prompt injection** — render the template in a unit test (next section).
  To confirm it lands in a live session, start `datus` and ask the agent
  "which plugins are configured?" — the answer comes from the injected
  section. Note that config edits take effect on the next `datus <plugin>`
  invocation immediately, but the prompt section refreshes only on the next
  session.

## Testing your plugin

Because Datus is the broker, unit tests call your functions with a plain dict
— no `agent.yml`, no Datus imports. Validate the manifest and template with
plain YAML/Jinja2 tooling:

```python
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from datus_plugin_hello.cli import main

PKG = Path(__file__).parent.parent / "datus_plugin_hello"


def test_cli_uses_profile_greeting(capsys):
    rc = main(["Ada"], {"name": "prod", "greeting": "Hi"})
    assert rc == 0
    assert "Hi, Ada!" in capsys.readouterr().out


def test_manifest_is_valid():
    manifest = yaml.safe_load((PKG / "datus-plugin.yml").read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["cli"] == "datus_plugin_hello.cli:main"
    # every declared path exists in the package
    assert (PKG / manifest["skills"]).is_dir()
    assert (PKG / manifest["system_prompt"]).is_file()


def test_prompt_template_renders_without_secrets():
    env = Environment(loader=FileSystemLoader(PKG), undefined=StrictUndefined)
    template = env.get_template("prompts/system.md.j2")
    # datus strips undeclared / x-secret fields before rendering; emulate that.
    text = template.render(plugin_name="hello", profiles={"prod": {"greeting": "Hi"}}, config_path=None)
    assert "## Hello" in text
    text = template.render(plugin_name="hello", profiles={}, config_path=None)
    assert "hello-setup" in text
```

## Distributing for offline install

Bundle your plugin into a wheelhouse `.zip` with `datus plugin pack`, run from
the plugin's project directory where you have network:

```bash
datus plugin pack -o ./dist               # plugin wheel only (default)
# → ./dist/datus-plugin-hello-1.0.0.zip

datus plugin pack --with-deps -o ./dist   # plugin wheel + every dependency wheel
```

The bundle is a zip of a `datus-bundle.json` manifest plus a `wheels/`
wheelhouse. `pack` verifies the wheel declares a module-ref `datus.plugins`
entry point **and** bundles its `datus-plugin.yml` — a wheel missing either is
refused before any dependency download. Users install it via `datus plugin
install zip:./….zip` (see
[Offline install](introduction.md#offline-install-air-gapped)). Choose the
flavor:

- **Default (plugin wheel only)** — a small bundle; the target machine resolves
  dependencies from a package index at install time (needs network).
- **`--with-deps`** — every transitive dependency wheel is bundled so the install
  is fully offline (`pip install --no-index --find-links`). `pack` uses
  `pip download --only-binary=:all:`, so every dependency must publish a wheel
  (no sdist-only packages). A pure-Python dependency set is portable; a
  dependency with a native extension makes the bundle a **same-platform
  snapshot** — build it on a machine matching the target's OS/Python.
- **A declared `Requires-Python`.** Set it in `pyproject.toml`; `pack` copies it
  into the manifest so install can reject a mismatched interpreter early (with a
  clear message, overridable via `--force`).

## Constraints checklist

Before publishing, verify:

- [ ] The package does **not** `import datus` anywhere (`grep -rn "import datus" your_pkg/`).
- [ ] The package does **not** depend on `datus` or a shared plugin SDK in `pyproject.toml`.
- [ ] The `datus.plugins` entry-point value is the **package name** (no `:Class` part).
- [ ] The entry-point name is not a reserved name (`upgrade`, `skill`, `plugin`) and does not start with `-`.
- [ ] `datus-plugin.yml` sits at the package root, declares `manifest_version: 1`, and ships in the wheel (`unzip -l dist/*.whl`).
- [ ] The `cli` function signature is `main(argv: list[str], profile: dict) -> int | None` and does not call `sys.exit()` on the success path.
- [ ] Secret config fields are marked `x-secret: true` in `config_schema`.
- [ ] The system-prompt template handles `profiles == {}` via `{% if profiles %}`.
- [ ] `permissions` patterns are namespace-relative (no `datus <name>` prefix — Datus adds it), and state-changing subcommands are `ask` under `normal`.
- [ ] Skill files and the prompt template are packaged into the wheel.
- [ ] The entry-point name matches the intended `datus <name>` command and the `agent.plugins.<name>` config key.

## Reference

- **Entry-point group**: `datus.plugins` — one entry per plugin, whose value is the plugin **package name**.
- **Contract source of truth**: `datus/plugins/base.py` (the `datus-plugin.yml` manifest spec).
- **Related**: [Plugin Introduction](introduction.md), [Skills](../skills/introduction.md).
