#!/usr/bin/env python3
"""Diagnostic script: isolate Claude subscription token 401 issue.

Usage:
    # With env var:
    export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
    python scripts/debug_subscription_token.py

    # With direct token:
    python scripts/debug_subscription_token.py sk-ant-oat01-...
"""

import sys


def get_token():
    import os

    if len(sys.argv) > 1:
        return sys.argv[1]
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        print("ERROR: Pass token as arg or set CLAUDE_CODE_OAUTH_TOKEN")
        sys.exit(1)
    return token


def test_1_native_anthropic(token: str, model: str):
    """Test 1: Direct Anthropic SDK (no LiteLLM)."""
    print(f"\n{'=' * 60}")
    print("TEST 1: Native Anthropic SDK (x-api-key header)")
    print(f"{'=' * 60}")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=token)
        resp = client.messages.create(
            model=model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
        )
        print(f"  OK: {resp.content[0].text}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


def test_2_litellm_no_base(token: str, model: str):
    """Test 2: LiteLLM without api_base (lets LiteLLM pick the URL)."""
    print(f"\n{'=' * 60}")
    print("TEST 2: LiteLLM WITHOUT api_base")
    print(f"{'=' * 60}")
    try:
        import litellm

        resp = litellm.completion(
            model=f"anthropic/{model}",
            api_key=token,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
            max_tokens=20,
        )
        print(f"  OK: {resp.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


def test_3_litellm_with_base(token: str, model: str):
    """Test 3: LiteLLM with api_base (how datus currently calls it)."""
    print(f"\n{'=' * 60}")
    print("TEST 3: LiteLLM WITH api_base='https://api.anthropic.com'")
    print(f"{'=' * 60}")
    try:
        import litellm

        resp = litellm.completion(
            model=f"anthropic/{model}",
            api_key=token,
            api_base="https://api.anthropic.com",
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
            max_tokens=20,
        )
        print(f"  OK: {resp.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


def test_4_raw_http(token: str, model: str):
    """Test 4: Raw HTTP with different header combinations."""
    print(f"\n{'=' * 60}")
    print("TEST 4: Raw HTTP requests")
    print(f"{'=' * 60}")

    import json

    import httpx

    body = json.dumps(
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": "Say hi in 3 words"}]}
    )

    # 4a: x-api-key WITHOUT beta headers
    print("  4a: x-api-key (no beta headers)...")
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            content=body,
            timeout=30,
        )
        print(f"      Status: {r.status_code}")
        if r.status_code == 200:
            print(f"      OK: {r.json()['content'][0]['text']}")
        else:
            print(f"      Body: {r.text[:200]}")
    except Exception as e:
        print(f"      Error: {e}")

    # 4b: x-api-key WITH OAuth beta headers (OpenClaw's approach)
    print("  4b: x-api-key + anthropic-beta: claude-code-20250219,oauth-2025-04-20...")
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                "content-type": "application/json",
            },
            content=body,
            timeout=30,
        )
        print(f"      Status: {r.status_code}")
        if r.status_code == 200:
            print(f"      OK: {r.json()['content'][0]['text']}")
        else:
            print(f"      Body: {r.text[:200]}")
    except Exception as e:
        print(f"      Error: {e}")

    # 4c: Authorization: Bearer header
    print("  4c: Authorization: Bearer header...")
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            content=body,
            timeout=30,
        )
        print(f"      Status: {r.status_code}")
        if r.status_code == 200:
            print(f"      OK: {r.json()['content'][0]['text']}")
        else:
            print(f"      Body: {r.text[:200]}")
    except Exception as e:
        print(f"      Error: {e}")


def main():
    token = get_token()
    model = "claude-sonnet-4-20250514"  # Use sonnet (cheaper) for testing

    print(f"Token: {token[:25]}...")
    print(f"Model: {model}")

    r1 = test_1_native_anthropic(token, model)
    r2 = test_2_litellm_no_base(token, model)
    r3 = test_3_litellm_with_base(token, model)
    test_4_raw_http(token, model)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Native Anthropic SDK:        {'PASS' if r1 else 'FAIL'}")
    print(f"  LiteLLM (no api_base):       {'PASS' if r2 else 'FAIL'}")
    print(f"  LiteLLM (with api_base):     {'PASS' if r3 else 'FAIL'}")
    print("  Raw HTTP: see results above")

    if r1 and not r3:
        print("\n  >>> DIAGNOSIS: api_base causes LiteLLM routing issue")
        print("  >>> FIX: Remove api_base from ClaudeModel LiteLLM calls")
    elif not r1 and not r3:
        print("\n  >>> DIAGNOSIS: Token itself is invalid/expired")
        print("  >>> FIX: Run 'claude setup-token' to get a fresh token")
    elif r1 and r2 and r3:
        print("\n  >>> All tests pass - token is valid")


if __name__ == "__main__":
    main()
