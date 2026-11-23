"""Materialize the three private nightly configs from explicitly supplied secrets.

Only GitHub Actions masking commands may contain configuration values. Validation
errors identify the input name, never its contents or the YAML parser exception.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

TARGETS = {
    "DATUS_NIGHTLY_AGENT_CONFIG": Path("conf/agent.yml"),
    "DATUS_NIGHTLY_CI_ENV": Path(".env"),
    "DATUS_NIGHTLY_SERVICE_ENV": Path(".nightly.env"),
}
SENSITIVE_KEY = re.compile(r"key|token|password|secret|credential", re.IGNORECASE)
VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ConfigurationError(ValueError):
    """A private configuration failed validation; its message is safe to log."""


def _yaml_secrets(payload: str) -> list[str]:
    name = "DATUS_NIGHTLY_AGENT_CONFIG"
    try:
        parsed = yaml.safe_load(payload)
        tree = yaml.compose(payload)
    except (yaml.YAMLError, RecursionError, ValueError, OverflowError):
        raise ConfigurationError(f"{name} must contain valid Agent YAML") from None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("agent"), dict):
        raise ConfigurationError(f"{name} must contain an agent mapping")
    values = []
    visited = set()

    def walk(node, sensitive=False):
        identity = (id(node), sensitive)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(node, ScalarNode) and sensitive:
            values.append(node.value)
        elif isinstance(node, MappingNode):
            for key, value in node.value:
                inherited = sensitive or (isinstance(key, ScalarNode) and bool(SENSITIVE_KEY.search(key.value)))
                walk(value, inherited)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                walk(value, sensitive)

    walk(tree)
    return values


def _dotenv(name: str, payload: str) -> tuple[str, list[str]]:
    # The existing workflow loads literal NAME=value lines, not shell syntax.
    # Normalize transport line endings and whitespace-only blank lines for it.
    if "\0" in payload or "\r" in payload.replace("\r\n", ""):
        raise ConfigurationError(f"{name} must contain single-line NAME=value assignments")
    values, lines = [], []
    for line in payload.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            lines.append("")
            continue
        lines.append(line)
        if line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not VARIABLE_NAME.fullmatch(key):
            raise ConfigurationError(f"{name} must contain single-line NAME=value assignments")
        values.append(value)
        # Preserve literal file semantics while also protecting quoted values
        # should a later consumer display their unquoted form.
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
            values.append(stripped[1:-1])
    if not values:
        raise ConfigurationError(f"{name} requires at least one NAME=value assignment")
    return "\n".join(lines), values


def _mask(values: list[str]) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for value in dict.fromkeys(values):
        if value:
            escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::add-mask::{escaped}", flush=True)


def prepare(repo_root: Path) -> None:
    payloads = {name: os.environ.get(name, "") for name in TARGETS}
    missing = [name for name, value in payloads.items() if not value.strip()]
    if missing:
        raise ConfigurationError("Missing required nightly secret: " + ", ".join(missing))
    for name, value in payloads.items():
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ConfigurationError(f"{name} must contain UTF-8 text") from None
    masks = _yaml_secrets(payloads["DATUS_NIGHTLY_AGENT_CONFIG"])
    for name in ("DATUS_NIGHTLY_CI_ENV", "DATUS_NIGHTLY_SERVICE_ENV"):
        payloads[name], values = _dotenv(name, payloads[name])
        masks.extend(values)

    root = repo_root.resolve()
    if not root.is_dir():
        raise ConfigurationError("Nightly repository root must be an existing directory")
    conf = root / "conf"
    targets = {name: root / relative for name, relative in TARGETS.items()}
    # Refuse both existing and dangling links before creating any output.
    if conf.is_symlink() or (conf.exists() and not conf.is_dir()):
        raise ConfigurationError("Nightly conf directory must be a regular directory")
    if any(path.is_symlink() or (path.exists() and not path.is_file()) for path in targets.values()):
        raise ConfigurationError("Nightly configuration targets must be regular files")

    _mask(masks)
    conf.mkdir(mode=0o700, exist_ok=True)
    staged = []
    try:
        # Stage every private file before replacing any existing configuration.
        for name, target in targets.items():
            descriptor, temporary = tempfile.mkstemp(prefix=".nightly-config-", dir=target.parent)
            path = Path(temporary)
            staged.append((path, target))
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(payloads[name])
        for path, target in staged:
            path.replace(target)
    finally:
        for path, _ in staged:
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        prepare(args.repo_root)
    except ConfigurationError as error:
        print(f"Nightly configuration error: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("Nightly configuration error: could not write private configuration files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
