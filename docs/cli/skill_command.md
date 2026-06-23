# Skill Command `/skill`

## Overview

The `/skill` slash command is the unified surface for managing local skills
and interacting with the **Town Skills Marketplace**. A single TUI hosts
tab switching, detail drill-down, search, login, and remove confirmation in
one prompt_toolkit Application; non-interactive subcommands cover the same
operations for scripts.

All marketplace-installed skills live under
`~/.datus/skills/<skill-name>/`; project-local skills in
`./.datus/skills/` take precedence.

---

## Basic Usage

### Interactive TUI

Type `/skill` with no arguments to open the browser:

```text
/skill
```

The TUI has three tabs, switched with **Tab** / **Shift+Tab** or **←/→**:

| Tab | Description |
|-----|-------------|
| **Installed** | Skills already present on disk (local + marketplace sources) |
| **Marketplace** | Skills published in the Town Marketplace |
| **Published** | Skills you have published as the current marketplace user |

Inside a tab:

| Key | Action |
|-----|--------|
| **↑ / ↓** | Move selection |
| **Enter** | Open detail panel for the selected skill |
| **/** | Filter the Marketplace tab by free-text query |
| **i** | Install the highlighted Marketplace skill |
| **u** | Update the highlighted installed marketplace skill |
| **x** | Remove the highlighted installed skill (press twice to confirm) |
| **l** | Open the login form (Marketplace tab) |
| **Esc / q** | Close the picker |

### Subcommand shortcuts

Each subcommand either jumps the TUI to a pre-seeded state or runs
non-interactively (suitable for scripts):

| Command | Behavior |
|---------|----------|
| `/skill list` | Open the TUI on the Installed tab |
| `/skill search <query>` | Open the TUI on the Marketplace tab with `<query>` pre-filled as a filter |
| `/skill login [url]` | Open the login form (with the marketplace URL pre-filled if provided) |
| `/skill logout` | Drop saved marketplace credentials |
| `/skill install <name> [version]` | Non-interactive install; `version` defaults to `latest` |
| `/skill publish <path> [--owner <name>]` | Non-interactive publish from a directory containing `SKILL.md` |
| `/skill info <name>` | Print local + marketplace details as a table |
| `/skill update` | Bulk-upgrade every marketplace-installed skill |
| `/skill remove <name>` | Remove a local skill (asks before deleting files) |
| `/skill help` | Show the command reference table |

---

## Authentication

Marketplace publish / promote actions require a Town account:

1. Run `/skill login` (optionally with a URL: `/skill login http://my-marketplace:9000`).
2. Enter email and password in the form. Credentials are exchanged for a JWT
   and stored locally; the password itself is never persisted.
3. Use `/skill logout` to clear the saved token for the current marketplace.

Tokens are scoped per marketplace URL, so you can authenticate against
multiple marketplaces from the same machine.

---

## Configuration

Skill discovery and the marketplace endpoint are configured in `agent.yml`:

```yaml
skills:
  directories:
    - ~/.datus/skills        # global, shared across projects
    - ./.datus/skills        # project-local; takes precedence
  marketplace_url: "http://localhost:9000"
  install_dir: "~/.datus/skills"
  auto_sync: false
```

Project-local skills (`./.datus/skills/<name>/`) override globally installed
skills with the same name.

---

## Examples

```bash
# Open the picker on Installed
/skill

# Search the marketplace for "sql"
/skill search sql

# Install a specific version
/skill install sql-optimization 1.0.0

# Publish a local skill
/skill publish ./skills/sql-optimization --owner murphy

# Show details (local + marketplace)
/skill info sql-optimization

# Upgrade everything that came from the marketplace
/skill update
```

For more detail on authoring skills, permissions, and the marketplace
workflow, see [Skills Integration](../skills/introduction.md).
