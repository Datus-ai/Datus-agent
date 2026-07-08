# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The datus-plugin contract.

A *plugin* is an installable package that extends datus with a ``datus
<plugin> ...`` CLI subcommand plus (optionally) a bundled skill directory,
discovered through the ``datus.plugins`` entry-point group. The entry point
resolves to a **plugin class** that datus uses purely by structure — a plugin
package never imports ``datus.*``.

Contract (duck-typed; :class:`DatusPlugin` documents it but is not imported by
plugins):

* ``PluginClass(profile: dict)`` — datus resolves the active profile from
  ``agent.plugins.<name>.<profile>`` (env-expanded) and constructs the plugin
  with it. A config-free plugin may ignore the argument.
* ``instance.run_cli(argv: list[str]) -> int | None`` — run the subcommand;
  ``argv`` is everything after ``datus <plugin>`` with datus' own
  ``--profile`` / ``--config`` already stripped. Return an exit code (``None``
  is treated as ``0``).
* ``PluginClass.skills_dir() -> str | None`` — *optional* classmethod/
  staticmethod/attribute returning the bundled skill directory. Resolved at
  startup for skill discovery, i.e. **without a profile**, so it must be
  reachable at the class level and must not depend on ``__init__``.
* ``PluginClass.system_prompt(profiles: dict[str, dict]) -> str | None`` —
  *optional* classmethod/staticmethod returning a self-describing markdown
  block injected into the agent's system prompt (e.g. "Manage scheduled jobs;
  N environments: ..."). datus passes the plugin's *full* profile mapping
  (all environments, not just the active one; already ``${VAR}``-expanded) and
  appends the returned text verbatim. Like ``skills_dir()`` it is resolved at
  prompt-build time **without a profile instance**, so it must be reachable at
  the class level. The text enters the LLM context, so the plugin must surface
  **only non-secret fields** (never ``password`` / secret / access-key values).
  An installed-but-unconfigured plugin receives ``{}`` — instead of returning
  ``None`` (and disappearing from the prompt), it should return a short
  "installed, not configured" note pointing at its bundled setup skill so the
  agent can walk the user through configuration. datus prepends its own
  ``## Plugins`` preamble naming the loaded config file, so the plugin text
  must not hard-code config paths. Return ``None`` when there is truly nothing
  to say.
* ``PluginClass.cli_permissions() -> dict | None`` — *optional* classmethod/
  staticmethod/attribute declaring bash-permission rules for the plugin's own
  CLI namespace, keyed by permission profile::

      {"normal": {"allow": ["greet:*"], "ask": ["config set:*"]},
       "auto":   {"allow": ["greet:*", "config set:*"]}}

  Patterns are **relative to the plugin namespace** — datus prefixes each with
  ``datus <name> ``, so ``greet:*`` becomes ``datus <name> greet:*`` and a
  plugin can never affect commands outside its own namespace. Pattern syntax
  matches ``permissions.bash_commands`` in agent.yml (``cmd`` exact,
  ``cmd:*`` prefix, ``cmd:glob`` prefix + first-arg glob). Only ``normal`` and
  ``auto`` keys are accepted (``dangerous`` ignores bash rules by design).
  These rules only gate the **agent's** bash tool when it runs ``datus
  <name> ...`` — a human invoking the CLI directly is never gated. A plugin
  rule can never loosen a user ``deny`` from agent.yml, and malformed
  declarations are warned about and skipped, never fatal. Plugin ``ask``
  rules can be relaxed per project: the confirmation prompt offers
  "allow (project)", which persists the exact matched pattern to
  ``.datus/config.yml``'s ``bash_allow`` and auto-runs it from then on
  (exact match only; ``deny`` rules are unaffected). Like the other
  class-level hooks it is resolved **without a profile instance**.

* ``PluginClass.tool_transformers() -> dict | None`` — *optional* classmethod/
  staticmethod/attribute declaring **tool argument transformers**: callables
  that inspect and rewrite (or deny) the agent's tool-call arguments before
  the tool executes. Returns a mapping of tool patterns to a transformer or a
  list of transformers::

      {"db_tools.execute_sql": enforce_sql_scope}
      {"execute_sql": [audit_args, enforce_sql_scope]}

  Pattern syntax matches the proxy layer: bare tool name (``execute_sql``),
  ``category.method`` with fnmatch globs (``db_tools.*``). Each transformer is
  called as ``transformer(tool_name, args, context) -> dict`` (sync or async):
  it returns the possibly-modified argument dict to continue, or raises to
  deny the call — the tool then never runs and the model receives the
  exception message as a normal tool failure (fail closed). ``context`` is a
  plain dict with ``node_name``, ``principal`` (request-scoped, empty when the
  deployment sets none), and ``project_root``. Transformers run in-process
  with full access to every matched tool call's arguments — they are trusted
  code, gated by the same ``agent.plugins_enabled`` master switch as the rest
  of the plugin surface. They only cover LLM-driven tool calls; direct Python
  invocations of tool methods bypass them. Like the other class-level hooks it
  is resolved **without a profile instance**, and a malformed declaration is
  warned about and skipped, never fatal.

Plugins that need configuration should also bundle a ``<name>-setup`` skill
(next to their main skill under ``skills_dir()``) describing the profile YAML
shape, which fields to ask the user for, and how to verify — with secrets
referenced as ``${ENV_VAR}`` placeholders, never literal values.

The whole plugin system is gated by ``agent.plugins_enabled`` (default
``true``): when ``false``, CLI dispatch, skill discovery, and prompt injection
are all disabled — intended for API/web deployments where the agent must not
edit configuration files.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DatusPlugin(Protocol):
    """Structural contract for a datus plugin instance.

    Plugins do not import or subclass this — datus calls the methods by name.
    """

    def run_cli(self, argv: List[str]) -> Optional[int]:  # pragma: no cover - protocol
        ...
