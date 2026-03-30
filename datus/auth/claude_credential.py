# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
import time
from pathlib import Path
from typing import Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def get_claude_subscription_token(api_key_from_config: Optional[str] = None) -> tuple[str, str]:
    """Resolve Claude subscription token by priority.

    Priority:
        1. api_key from config (YAML value or ${CLAUDE_CODE_OAUTH_TOKEN} substitution)
        2. CLAUDE_CODE_OAUTH_TOKEN environment variable
        3. ~/.claude/.credentials.json -> claudeAiOauth.accessToken

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

    # Priority 3: ~/.claude/.credentials.json
    credentials_path = Path.home() / ".claude" / ".credentials.json"
    if credentials_path.exists():
        try:
            data = json.loads(credentials_path.read_text(encoding="utf-8"))
            token_data = data.get("claudeAiOauth", {})
            token = token_data.get("accessToken")
            if token and token.strip():
                expires_at = token_data.get("expiresAt")
                if expires_at and int(expires_at) / 1000 < time.time():
                    logger.warning(
                        "Claude subscription token in credentials file has expired; re-run 'claude setup-token'"
                    )
                else:
                    logger.debug("Using Claude subscription token from ~/.claude/.credentials.json")
                    return token, "~/.claude/.credentials.json"
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse credentials file: {e}")
        except OSError as e:
            logger.warning(f"Could not read credentials file {credentials_path}: {e}")

    raise DatusException(ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_NOT_FOUND)
