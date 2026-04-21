# Skill Command

!!! warning "Planned Feature"
    The `datus skill` CLI and Town Skills Marketplace integration are planned for a future release. The functionality described in this document is currently under development and not yet available in the latest version.

The `datus skill` command provides a CLI interface for managing skills locally and interacting with the Town Skills Marketplace.

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `login` | Authenticate with the Town Marketplace |
| `logout` | Clear saved authentication token |
| `list` | List all locally installed skills |
| `search <query>` | Search for skills in the Marketplace |
| `install <name> [version]` | Install a skill from the Marketplace |
| `publish <path>` | Publish a local skill to the Marketplace |
| `info <name>` | Show details about a skill |
| `update` | Update all marketplace-installed skills |
| `remove <name>` | Remove a locally installed skill |

## Global Options

| Option | Description |
|--------|-------------|
| `--marketplace <url>` | Override the marketplace URL from `agent.yml` |

## Usage

### Authentication

```bash
# Interactive login
datus skill login --marketplace http://datus-marketplace:9000

# Non-interactive login (use env var to avoid plaintext password in shell history)
DATUS_PASSWORD='***' datus skill login --marketplace http://datus-marketplace:9000 --email user@example.com --password "$DATUS_PASSWORD"

# Logout
datus skill logout --marketplace http://datus-marketplace:9000
```

### List Local Skills

```bash
datus skill list
```

Output:
```
┌──────────────────┬─────────┬─────────────┬─────────────────────────┐
│ Name             │ Version │ Source      │ Tags                    │
├──────────────────┼─────────┼─────────────┼─────────────────────────┤
│ sql-optimization │ 1.0.0   │ marketplace │ sql, optimization       │
│ report-generator │ 1.0.0   │ local       │ report, analysis        │
└──────────────────┴─────────┴─────────────┴─────────────────────────┘
```

### Search Marketplace

```bash
datus skill search sql
datus skill search --marketplace http://localhost:9000 report
```

### Install a Skill

```bash
# Install latest version
datus skill install sql-optimization

# Install specific version
datus skill install sql-optimization 1.0.0
```

### Publish a Skill

```bash
# Publish from a skill directory (must contain SKILL.md)
datus skill publish ./skills/sql-optimization

# Publish with owner
datus skill publish ./skills/sql-optimization --owner "murphy"
```

### Skill Info

```bash
datus skill info sql-optimization
```

### Update Skills

```bash
datus skill update
```

### Remove a Skill

```bash
datus skill remove sql-optimization
```

## REPL Equivalents

All `datus skill` subcommands are available in the REPL as `/skill` commands:

```
datus> /skill list
datus> /skill search sql
datus> /skill install sql-optimization
datus> /skill publish ./skills/my-skill
datus> /skill info sql-optimization
datus> /skill update
datus> /skill remove sql-optimization
```

## Configuration

Marketplace settings can be configured in `agent.yml`:

```yaml
skills:
  directories:
    - ~/.datus/skills
    - ./skills
  marketplace_url: "http://localhost:9000"
  auto_sync: false
  install_dir: "~/.datus/skills"
```

For more details on skill creation, permissions, and the marketplace workflow, see [Skills Integration](../integration/skills.md).
