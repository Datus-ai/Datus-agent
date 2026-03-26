#!/usr/bin/env python3
"""Quick test: verify Claude subscription token behavior via ClaudeModel.

Usage:
    export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
    python scripts/test_subscription_models.py

    # Or with direct token:
    python scripts/test_subscription_models.py sk-ant-oat01-...
"""

import os
import sys

# Models to test
MODELS_TO_TEST = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]


def get_token():
    if len(sys.argv) > 1:
        return sys.argv[1]
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        print("ERROR: Pass token as arg or set CLAUDE_CODE_OAUTH_TOKEN")
        sys.exit(1)
    return token


def test_via_claude_model(token: str):
    """Test via our ClaudeModel with subscription auth."""
    from unittest.mock import MagicMock

    from datus.models.claude_model import ClaudeModel

    results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'─' * 60}")
        print(f"Testing: {model_name}")
        print(f"{'─' * 60}")

        try:
            cfg = MagicMock()
            cfg.model = model_name
            cfg.type = "claude"
            cfg.api_key = token
            cfg.base_url = "https://api.anthropic.com"
            cfg.use_native_api = False  # will be forced True by OAuth detection
            cfg.temperature = None
            cfg.top_p = None
            cfg.enable_thinking = False
            cfg.default_headers = {}
            cfg.max_retry = 3
            cfg.retry_interval = 0.0
            cfg.strict_json_schema = True
            cfg.auth_type = "subscription"

            model = ClaudeModel(model_config=cfg)

            print(f"  _is_oauth_token: {model._is_oauth_token}")
            print(f"  use_native_api:  {model.use_native_api}")

            # Check client headers on anthropic_client
            if hasattr(model.anthropic_client, "_client"):
                headers = dict(model.anthropic_client._client.headers)
                print(f"  user-agent:      {headers.get('user-agent', 'N/A')}")
                print(f"  x-app:           {headers.get('x-app', 'N/A')}")
                auth_header = headers.get("authorization", "N/A")
                if auth_header != "N/A":
                    print(f"  authorization:   {auth_header[:40]}...")

            print("  Calling generate('Say hi in 3 words')...")
            response = model.generate("Say hi in 3 words", max_tokens=30)
            print(f"  Response: {response}")
            results[model_name] = True

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results[model_name] = False

    return results


def main():
    token = get_token()
    print(f"Token: {token[:8]}...{token[-4:]}")
    print(f"Models: {', '.join(MODELS_TO_TEST)}")

    results = test_via_claude_model(token)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for model, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {model:40s} {status}")

    if all(results.values()):
        print("\n🎉 Haiku, Sonnet, and Opus all work with subscription token!")
    else:
        failed = [m for m, p in results.items() if not p]
        print(f"\n⚠️  Failed models: {', '.join(failed)}")


if __name__ == "__main__":
    main()
