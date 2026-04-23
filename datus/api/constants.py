"""Constants for Datus API."""

import re

from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS

# Header carrying the caller's user identifier for the open-source default auth.
HEADER_USER_ID = "X-Datus-User-Id"

# Allowed characters for header-provided user_id (also used as SessionManager scope).
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Builtin subagents matching /agent visible list
BUILTIN_SUBAGENTS = SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS
