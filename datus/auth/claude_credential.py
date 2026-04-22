# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _read_keychain_credentials() -> Optional[dict]:
    """Read Claude Code credentials from macOS Keychain.

    Returns the parsed JSON dict on success, or None on any failure.
    Only attempts on macOS (sys.platform == "darwin").
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.debug("macOS Keychain entry not found for 'Claude Code-credentials'")
            return None
        return json.loads(result.stdout.strip())
    except FileNotFoundError:
        logger.debug("'security' command not found; skipping Keychain lookup")
        return None
    except subprocess.TimeoutExpired:
        logger.debug("Keychain lookup timed out")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse Keychain credentials JSON: {e}")
        return None
    except OSError as e:
        logger.debug(f"Keychain lookup failed: {e}")
        return None


def _extract_oauth_token(data: dict, source_label: str) -> Optional[tuple[str, str]]:
    """Extract and validate OAuth token from a Claude credentials dict."""
    token_data = data.get("claudeAiOauth", {})
    token = token_data.get("accessToken")
    if not token or not token.strip():
        return None
    expires_at = token_data.get("expiresAt")
    if expires_at and int(expires_at) / 1000 < time.time():
        logger.warning(f"Claude subscription token from {source_label} has expired; re-run 'claude setup-token'")
        return None
    logger.debug(f"Using Claude subscription token from {source_label}")
    return token, source_label


def get_claude_subscription_token(api_key_from_config: Optional[str] = None) -> tuple[str, str]:
    """Resolve Claude subscription token by priority.

    Priority:
        1. api_key from config (YAML value or ${CLAUDE_CODE_OAUTH_TOKEN} substitution)
        2. CLAUDE_CODE_OAUTH_TOKEN environment variable
        3. macOS Keychain ("Claude Code-credentials") — only on darwin
        4. ~/.claude/.credentials.json -> claudeAiOauth.accessToken

    Returns:
        (token, source) where source describes where the token was found.
    """
    # Priority 1: config api_key (skip env-substitution placeholders)
    if (
        api_key_from_config
        and api_key_from_config.strip()
        and not api_key_from_config.startswith("<MISSING:")
        and not (api_key_from_config.startswith("${") and api_key_from_config.endswith("}"))
    ):
        logger.debug("Using Claude subscription token from config")
        return api_key_from_config, "config (agent.yml)"

    # Priority 2: CLAUDE_CODE_OAUTH_TOKEN env var
    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token and env_token.strip():
        logger.debug("Using Claude subscription token from CLAUDE_CODE_OAUTH_TOKEN")
        return env_token, "env CLAUDE_CODE_OAUTH_TOKEN"

    # Priority 3: macOS Keychain
    keychain_data = _read_keychain_credentials()
    if keychain_data:
        result = _extract_oauth_token(keychain_data, "macOS Keychain")
        if result:
            return result

    # Priority 4: ~/.claude/.credentials.json
    credentials_path = Path.home() / ".claude" / ".credentials.json"
    if credentials_path.exists():
        try:
            data = json.loads(credentials_path.read_text(encoding="utf-8"))
            file_result = _extract_oauth_token(data, "~/.claude/.credentials.json")
            if file_result:
                return file_result
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse credentials file: {e}")
        except OSError as e:
            logger.warning(f"Could not read credentials file {credentials_path}: {e}")

    raise DatusException(ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_NOT_FOUND)
