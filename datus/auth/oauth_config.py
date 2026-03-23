# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OAuth configuration constants for OpenAI Codex authentication."""

# OAuth endpoints
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_CODE_URL = "https://auth.openai.com/deviceauth/usercode"

# Client configuration
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455

# Scopes
SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"

# Codex API endpoint
CODEX_API_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Token refresh interval (8 days in seconds)
TOKEN_REFRESH_INTERVAL_SECONDS = 8 * 24 * 60 * 60

# Device code polling
DEVICE_CODE_POLL_INTERVAL = 5  # seconds
DEVICE_CODE_TIMEOUT = 900  # 15 minutes
