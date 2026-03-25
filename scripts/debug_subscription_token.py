#!/usr/bin/env python3
"""Diagnostic script: isolate Claude subscription token 401 issue.

Tests two key hypotheses from OpenClaw analysis:
1. ALL FOUR anthropic-beta headers are required (not just 2)
2. Short model aliases (claude-sonnet-4-6) may be required instead of
   date-pinned names (claude-sonnet-4-20250514)

Usage:
    # With env var:
    export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
    python scripts/debug_subscription_token.py

    # With direct token:
    python scripts/debug_subscription_token.py sk-ant-oat01-...

Note: This is a standalone CLI diagnostic script. It uses print() intentionally
for interactive terminal output (not part of the main datus application).
"""

import sys


def get_token():
    import os

    if len(sys.argv) > 1:
        return sys.argv[1]
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        print("ERROR: Pass token as arg or set CLAUDE_CODE_OAUTH_TOKEN")  # noqa: T201
        sys.exit(1)
    return token


def _redact_token(token: str) -> str:
    """Redact token for safe display — show only type prefix."""
    if token.startswith("sk-ant-oat"):
        return "sk-ant-oat***[REDACTED]"
    return "***[REDACTED]"


# Current beta headers for OAuth tokens
ALL_BETA_HEADERS = ",".join(
    [
        "claude-code-20250219",
        "oauth-2025-04-20",
        "interleaved-thinking-2025-05-14",
        "prompt-caching-scope-2026-01-05",
    ]
)

# OpenClaw uses short aliases, NOT date-pinned names
SHORT_MODEL = "claude-sonnet-4-6"
DATED_MODEL = "claude-sonnet-4-20250514"


def _raw_http_test(label: str, token: str, model: str, headers: dict):
    """Helper: send a raw HTTP request and print the result."""
    import json

    import httpx

    body = json.dumps(
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": "Say hi in 3 words"}]}
    )
    # Redact token in displayed headers
    safe_headers = {
        k: (_redact_token(v) if "sk-ant" in str(v) or "Bearer" in str(v) else v) for k, v in headers.items()
    }
    print(f"  {label}...")  # noqa: T201
    print(f"      model={model}")  # noqa: T201
    print(f"      headers={safe_headers}")  # noqa: T201
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={**headers, "content-type": "application/json"},
            content=body,
            timeout=30,
        )
        print(f"      Status: {r.status_code}")  # noqa: T201
        if r.status_code == 200:
            print(f"      OK: {r.json()['content'][0]['text']}")  # noqa: T201
            return True
        else:
            print(f"      Body: {r.text[:300]}")  # noqa: T201
            return False
    except Exception as e:
        print(f"      Error: {e}")  # noqa: T201
        return False


def test_raw_http(token: str):
    """Raw HTTP tests: systematically test header + model combinations."""
    print(f"\n{'=' * 70}")  # noqa: T201
    print("RAW HTTP TESTS (isolate exact requirements)")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    results = {}

    # Test 1: dated model + NO beta headers
    results["1_dated_no_beta"] = _raw_http_test(
        "Test 1: dated model + NO beta",
        token,
        DATED_MODEL,
        {"x-api-key": token, "anthropic-version": "2023-06-01"},
    )

    # Test 2: dated model + 2 beta headers (our previous attempt)
    results["2_dated_2beta"] = _raw_http_test(
        "Test 2: dated model + 2 beta headers (previous attempt)",
        token,
        DATED_MODEL,
        {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        },
    )

    # Test 3: dated model + ALL 4 beta headers
    results["3_dated_4beta"] = _raw_http_test(
        "Test 3: dated model + ALL 4 beta headers",
        token,
        DATED_MODEL,
        {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ALL_BETA_HEADERS,
        },
    )

    # Test 4: SHORT model + ALL 4 beta headers (OpenClaw's exact approach)
    results["4_short_4beta"] = _raw_http_test(
        "Test 4: SHORT model + ALL 4 beta headers (OpenClaw approach)",
        token,
        SHORT_MODEL,
        {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ALL_BETA_HEADERS,
        },
    )

    # Test 5: SHORT model + 2 beta headers
    results["5_short_2beta"] = _raw_http_test(
        "Test 5: SHORT model + 2 beta headers",
        token,
        SHORT_MODEL,
        {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        },
    )

    # Test 6: SHORT model + NO beta headers
    results["6_short_no_beta"] = _raw_http_test(
        "Test 6: SHORT model + NO beta headers",
        token,
        SHORT_MODEL,
        {"x-api-key": token, "anthropic-version": "2023-06-01"},
    )

    return results


def test_bearer_http(token: str):
    """Raw HTTP with Authorization: Bearer + beta headers (SDK may use this)."""
    print(f"\n{'=' * 70}")  # noqa: T201
    print("BEARER AUTH TESTS (SDK got 403 vs raw x-api-key 401 — SDK auth differs)")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    results = {}

    # Bearer + all 4 beta + short model
    results["bearer_short_4beta"] = _raw_http_test(
        "Test B1: Bearer + ALL 4 beta + SHORT model",
        token,
        SHORT_MODEL,
        {
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ALL_BETA_HEADERS,
        },
    )

    # Bearer + all 4 beta + dated model
    results["bearer_dated_4beta"] = _raw_http_test(
        "Test B2: Bearer + ALL 4 beta + dated model",
        token,
        DATED_MODEL,
        {
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ALL_BETA_HEADERS,
        },
    )

    # Bearer + NO beta + short model
    results["bearer_short_no_beta"] = _raw_http_test(
        "Test B3: Bearer + NO beta + SHORT model",
        token,
        SHORT_MODEL,
        {
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
        },
    )

    # BOTH x-api-key AND Bearer + all 4 beta + short model
    results["both_short_4beta"] = _raw_http_test(
        "Test B4: BOTH x-api-key AND Bearer + ALL 4 beta + SHORT model",
        token,
        SHORT_MODEL,
        {
            "x-api-key": token,
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ALL_BETA_HEADERS,
        },
    )

    return results


def test_native_sdk(token: str):
    """Test with native Anthropic SDK — also dump actual outgoing headers."""
    print(f"\n{'=' * 70}")  # noqa: T201
    print("NATIVE ANTHROPIC SDK TESTS")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    results = {}

    # First, dump what headers the SDK actually sends
    print("\n  Inspecting SDK auth headers...")  # noqa: T201
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=token,
            default_headers={"anthropic-beta": ALL_BETA_HEADERS},
        )
        # Check if there's an auth_token vs api_key distinction
        print(f"      SDK auth_token attr: {getattr(client, 'auth_token', 'N/A')}")  # noqa: T201
    except Exception as e:
        print(f"      Inspect error: {e}")  # noqa: T201

    # SDK test 1: short model + all 4 beta headers
    print(f"\n  SDK Test 1: short model ({SHORT_MODEL}) + all 4 beta headers")  # noqa: T201
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=token,
            default_headers={"anthropic-beta": ALL_BETA_HEADERS},
        )
        resp = client.messages.create(
            model=SHORT_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
        )
        print(f"      OK: {resp.content[0].text}")  # noqa: T201
        results["sdk_short_4beta"] = True
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")  # noqa: T201
        results["sdk_short_4beta"] = False

    # SDK test 2: dated model + all 4 beta headers
    print(f"\n  SDK Test 2: dated model ({DATED_MODEL}) + all 4 beta headers")  # noqa: T201
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=token,
            default_headers={"anthropic-beta": ALL_BETA_HEADERS},
        )
        resp = client.messages.create(
            model=DATED_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
        )
        print(f"      OK: {resp.content[0].text}")  # noqa: T201
        results["sdk_dated_4beta"] = True
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")  # noqa: T201
        results["sdk_dated_4beta"] = False

    # SDK test 3: auth_token (NOT api_key) + all 4 beta + short model
    print("\n  SDK Test 3: auth_token (Bearer) + all 4 beta + SHORT model")  # noqa: T201
    try:
        import anthropic

        client = anthropic.Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": ALL_BETA_HEADERS},
        )
        print(f"      auth_headers: {client.auth_headers}")  # noqa: T201
        resp = client.messages.create(
            model=SHORT_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
        )
        print(f"      OK: {resp.content[0].text}")  # noqa: T201
        results["sdk_auth_token_short"] = True
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")  # noqa: T201
        results["sdk_auth_token_short"] = False

    # SDK test 4: auth_token + all 4 beta + dated model
    print("\n  SDK Test 4: auth_token (Bearer) + all 4 beta + dated model")  # noqa: T201
    try:
        import anthropic

        client = anthropic.Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": ALL_BETA_HEADERS},
        )
        resp = client.messages.create(
            model=DATED_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
        )
        print(f"      OK: {resp.content[0].text}")  # noqa: T201
        results["sdk_auth_token_dated"] = True
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")  # noqa: T201
        results["sdk_auth_token_dated"] = False

    # SDK test 5: auth_token + NO beta + short model
    print("\n  SDK Test 5: auth_token (Bearer) + NO beta + SHORT model")  # noqa: T201
    try:
        import anthropic

        client = anthropic.Anthropic(auth_token=token)
        resp = client.messages.create(
            model=SHORT_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
        )
        print(f"      OK: {resp.content[0].text}")  # noqa: T201
        results["sdk_auth_token_no_beta"] = True
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")  # noqa: T201
        results["sdk_auth_token_no_beta"] = False

    return results


def main():
    token = get_token()

    print(f"Token: {_redact_token(token)}")  # noqa: T201
    print(f"Short model: {SHORT_MODEL}")  # noqa: T201
    print(f"Dated model: {DATED_MODEL}")  # noqa: T201
    print(f"All beta headers: {ALL_BETA_HEADERS}")  # noqa: T201

    raw_results = test_raw_http(token)
    bearer_results = test_bearer_http(token)
    sdk_results = test_native_sdk(token)

    print(f"\n{'=' * 70}")  # noqa: T201
    print("SUMMARY")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201
    print("\nRaw HTTP (x-api-key):")  # noqa: T201
    for key, passed in raw_results.items():
        print(f"  {key:30s} {'PASS' if passed else 'FAIL'}")  # noqa: T201
    print("\nBearer Auth:")  # noqa: T201
    for key, passed in bearer_results.items():
        print(f"  {key:30s} {'PASS' if passed else 'FAIL'}")  # noqa: T201
    print("\nNative SDK:")  # noqa: T201
    for key, passed in sdk_results.items():
        print(f"  {key:30s} {'PASS' if passed else 'FAIL'}")  # noqa: T201

    # Diagnosis
    print(f"\n{'=' * 70}")  # noqa: T201
    print("DIAGNOSIS")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    all_results = {**raw_results, **bearer_results, **sdk_results}

    if bearer_results.get("bearer_short_4beta"):
        print("  >>> Bearer auth + beta headers + short model WORKS")  # noqa: T201
        print("  >>> FIX: Use Authorization: Bearer header for OAuth tokens")  # noqa: T201
    elif bearer_results.get("bearer_dated_4beta"):
        print("  >>> Bearer auth + beta headers WORKS (model name doesn't matter)")  # noqa: T201
        print("  >>> FIX: Use Authorization: Bearer header for OAuth tokens")  # noqa: T201
    elif raw_results.get("4_short_4beta"):
        print("  >>> x-api-key works with short model + all 4 beta headers")  # noqa: T201
    elif not any(all_results.values()):
        print("  >>> ALL tests failed - token may be invalid/expired")  # noqa: T201
        print("  >>> FIX: Run 'claude setup-token' to refresh")  # noqa: T201
    else:
        print("  >>> Mixed results - check individual tests above")  # noqa: T201


if __name__ == "__main__":
    main()
