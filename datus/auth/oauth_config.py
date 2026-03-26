# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OAuth configuration constants for OpenAI Codex authentication."""

import os

# OAuth endpoints
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"

# Client configuration — public PKCE client ID (not a secret).
# Security is provided by the PKCE code_verifier/code_challenge, not by hiding the client_id.
# Override via DATUS_CODEX_CLIENT_ID env var if needed.
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
