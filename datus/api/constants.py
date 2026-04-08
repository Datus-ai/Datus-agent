"""Constants for Datus API."""

import re

# Header carrying the caller's user identifier for the open-source default auth.
HEADER_USER_ID = "X-Datus-User-Id"

# Allowed characters for header-provided user_id (also used as SessionManager scope).
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Builtin subagents that come pre-configured
BUILTIN_SUBAGENTS = {
    "gen_sql",
    "gen_report",
    "gen_semantic_model",
    "gen_metrics",
    "gen_sql_summary",
    "gen_ext_knowledge",
}
