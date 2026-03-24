# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OAuth configuration constants for OpenAI Codex authentication."""

import os

# OAuth endpoints
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_CODE_URL = "https://auth.openai.com/deviceauth/usercode"

# Client configuration (override via env var for testing or rotation)
CLIENT_ID = os.environ.get("DATUS_CODEX_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455

# Scopes
SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"

# Codex API endpoint
CODEX_API_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Token refresh interval (8 days in seconds)
TOKEN_REFRESH_INTERVAL_SECONDS = 8 * 24 * 60 * 60

# HTTP request timeout for OAuth calls
HTTP_TIMEOUT = 30.0  # seconds

# Device code polling
DEVICE_CODE_POLL_INTERVAL = 5  # seconds
DEVICE_CODE_TIMEOUT = 900  # 15 minutes
