# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Authentication module for Datus Agent API.
"""

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jwt
import yaml
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

LEGACY_AUTH_UNAVAILABLE = (
    "Legacy JWT authentication is unavailable. Configure auth_clients.yml with client credentials "
    "and a JWT signing key; JWT_SECRET_KEY may override the signing key."
)

security = HTTPBearer(auto_error=False)

_depends_security = Depends(security)


def load_auth_config(config_path: Optional[str] = None) -> Dict:
    """
    Load authentication configuration from auth_clients.yml.

    Configuration is fixed at {agent.home}/conf/auth_clients.yml.
    Configure agent.home in agent.yml to change the root directory.

    Args:
        config_path: Optional explicit path (primarily for testing)
    """
    if config_path is not None:
        yaml_path = Path(config_path).expanduser()
    else:
        from datus.utils.path_manager import get_path_manager

        # Default resolution remains available before AgentConfig is constructed.
        yaml_path = get_path_manager().auth_config_path()
    with yaml_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config:
        raise ValueError("Authentication configuration must be a nonempty mapping")
    return config


class AuthService:
    """Legacy JWT authentication, disabled when explicit configuration is invalid."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.available = False
        self.config = {}
        self.clients = {}
        self.jwt_secret = None
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 2
        try:
            config = load_auth_config(config_path)
            clients = config.get("clients")
            if not isinstance(clients, dict) or not clients:
                raise ValueError("Client credentials are required")
            if any(not isinstance(value, str) or not value.strip() for pair in clients.items() for value in pair):
                raise ValueError("Client identifiers and secrets must be nonempty strings")
            jwt_config = config.get("jwt", {})
            if not isinstance(jwt_config, dict):
                raise ValueError("JWT configuration must be a mapping")
            secret = os.environ.get("JWT_SECRET_KEY", jwt_config.get("secret_key"))
            if not isinstance(secret, str) or not secret.strip():
                raise ValueError("An explicit JWT signing key is required")
            algorithm = jwt_config.get("algorithm", "HS256")
            if (
                not isinstance(algorithm, str)
                or algorithm == "none"
                or algorithm not in jwt.algorithms.get_default_algorithms()
            ):
                raise ValueError("Unsupported JWT signing algorithm")
            hours = jwt_config.get("expiration_hours", 2)
            if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
                raise ValueError("Token lifetime must be positive and finite")
            datetime.now(timezone.utc) + timedelta(hours=hours)
        except (OSError, ValueError, TypeError, OverflowError, yaml.YAMLError):
            # Parser exceptions and paths can contain credentials: log neither.
            logger.warning(LEGACY_AUTH_UNAVAILABLE)
            return
        self.config = config
        self.clients = dict(clients)
        self.jwt_secret = secret
        self.jwt_algorithm = algorithm
        self.jwt_expiration_hours = hours
        self.available = True

    def require_available(self) -> None:
        """Reject legacy requests without preventing unrelated API startup."""
        if not self.available:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LEGACY_AUTH_UNAVAILABLE)

    def validate_client_credentials(self, client_id: str, client_secret: str) -> bool:
        """Validate client credentials."""
        self.require_available()
        return (
            isinstance(client_id, str)
            and isinstance(client_secret, str)
            and client_id in self.clients
            and self.clients[client_id] == client_secret
        )

    def generate_access_token(self, client_id: str) -> Dict[str, Any]:
        """Generate a JWT access token for the client."""
        self.require_available()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=self.jwt_expiration_hours)
        payload = {
            "client_id": client_id,
            "exp": expires_at,
            "iat": datetime.now(timezone.utc),
        }

        access_token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": self.jwt_expiration_hours * 3600,  # Convert to seconds
        }

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token and return the payload."""
        self.require_available()
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            client_id = payload.get("client_id")
            if not isinstance(client_id, str) or not client_id.strip():
                raise jwt.InvalidTokenError("Missing client identity")
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Global auth service instance
auth_service = AuthService()


def require_legacy_auth() -> None:
    """FastAPI dependency for the legacy token endpoint."""
    auth_service.require_available()


def get_current_client(credentials: Optional[HTTPAuthorizationCredentials] = _depends_security) -> str:
    """
    Dependency to get the current authenticated client from the token.
    """
    auth_service.require_available()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    payload = auth_service.validate_token(token)
    client_id = payload.get("client_id")

    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return client_id
