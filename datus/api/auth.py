"""
Authentication module for Datus Agent API.
"""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jwt
import yaml
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Default clients configuration (fallback if config file not found)
DEFAULT_CLIENTS = {"datus_client": "datus_secret_key"}

# Default JWT configuration
DEFAULT_JWT_CONFIG = {"secret_key": "your-secret-key-change-in-production", "algorithm": "HS256", "expiration_hours": 2}

security = HTTPBearer()

_depends_security = Depends(security)


def load_auth_config(config_path: Optional[str] = None) -> Dict:
    """
    Load authentication configuration from auth_clients.yml.

    Args:
        config_path: Path to the config file (searches in order:
            config_path > conf/auth_clients.yml > ~/.datus/conf/auth_clients.yml)
    """
    yaml_path = None
    if config_path:
        yaml_path = Path(config_path).expanduser()

    if not yaml_path and Path("conf/auth_clients.yml").exists():
        yaml_path = Path("conf/auth_clients.yml")

    if not yaml_path:
        home_config = Path.home() / ".datus" / "conf" / "auth_clients.yml"
        if home_config.exists():
            yaml_path = home_config

    # Try to load from the found path
    if yaml_path and yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            print(f"Warning: Failed to load auth config from {yaml_path}: {e}")

    # Return default configuration if no config file found or failed to load
    return {"clients": DEFAULT_CLIENTS, "jwt": DEFAULT_JWT_CONFIG}


class AuthService:
    """Authentication service for handling JWT tokens and client validation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration from file."""
        self.config = load_auth_config()
        self.clients = self.config.get("clients", DEFAULT_CLIENTS)

        # Load JWT configuration
        jwt_config = self.config.get("jwt", DEFAULT_JWT_CONFIG)
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", jwt_config.get("secret_key"))
        self.jwt_algorithm = jwt_config.get("algorithm", "HS256")
        self.jwt_expiration_hours = jwt_config.get("expiration_hours", 2)

    def validate_client_credentials(self, client_id: str, client_secret: str) -> bool:
        """Validate client credentials."""
        return self.clients.get(client_id) == client_secret

    def generate_access_token(self, client_id: str) -> Dict[str, Any]:
        """Generate a JWT access token for the client."""
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

    def validate_token(self, token: str) -> Dict[str, any]:
        """Validate a JWT token and return the payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
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


def get_current_client(credentials: HTTPAuthorizationCredentials = _depends_security) -> str:
    """
    Dependency to get the current authenticated client from the token.
    """
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
