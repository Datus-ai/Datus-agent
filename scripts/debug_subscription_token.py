#!/usr/bin/env python3
"""Probe Claude setup-token behavior with OpenClaw-aligned requests.

This script focuses on the exact question discussed in the repo notes:
- does a Claude Code setup-token work with Haiku / Sonnet / Opus?
- does it only fail on specific models?
- is the failure caused by request shape rather than auth transport?

The default request shape intentionally mirrors OpenClaw's Anthropic OAuth path:
- Authorization: Bearer <token>
- anthropic-version: 2023-06-01
- anthropic-beta:
  - claude-code-20250219
  - oauth-2025-04-20
  - fine-grained-tool-streaming-2025-05-14
  - interleaved-thinking-2025-05-14
- minimal text-only payload

Usage:
    export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
    python scripts/debug_subscription_token.py

    python scripts/debug_subscription_token.py sk-ant-oat01-...
    python scripts/debug_subscription_token.py --models claude-haiku-3-5 claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

DEFAULT_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]

OPENCLAW_OAUTH_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "fine-grained-tool-streaming-2025-05-14",
    "interleaved-thinking-2025-05-14",
]

OPENCLAW_OAUTH_BETA_HEADER = ",".join(OPENCLAW_OAUTH_BETAS)

CLAUDE_CLIENT_HEADERS = {
    "user-agent": "claude-cli/2.1.75 (external, cli)",
    "x-app": "cli",
    "anthropic-dangerous-direct-browser-access": "true",
}


@dataclass
class ProbeCase:
    label: str
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


def redact_token(token: str) -> str:
    if len(token) <= 16:
        return "***[REDACTED]"
    return f"{token[:12]}...{token[-4:]}"


def get_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_OAUTH_TOKEN")
    if not token:
        print("ERROR: pass token as arg or set CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_OAUTH_TOKEN")  # noqa: T201
        sys.exit(1)
    return token.strip()


def build_cases(token: str, model: str, prompt: str, max_tokens: int) -> list[ProbeCase]:
    minimal_body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    openclaw_oauth_body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "system": [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }
        ],
    }
    return [
        ProbeCase(
            label="openclaw-minimal-bearer",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": OPENCLAW_OAUTH_BETA_HEADER,
            },
            body=minimal_body,
        ),
        ProbeCase(
            label="bearer-with-client-headers",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": OPENCLAW_OAUTH_BETA_HEADER,
                **CLAUDE_CLIENT_HEADERS,
            },
            body=minimal_body,
        ),
        ProbeCase(
            label="bearer-minimal-two-betas",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            },
            body=minimal_body,
        ),
        ProbeCase(
            label="openclaw-beta-endpoint-system",
            url="https://api.anthropic.com/v1/messages?beta=true",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": OPENCLAW_OAUTH_BETA_HEADER,
                **CLAUDE_CLIENT_HEADERS,
            },
            body=openclaw_oauth_body,
        ),
        ProbeCase(
            label="openclaw-beta-endpoint-two-betas-system",
            url="https://api.anthropic.com/v1/messages?beta=true",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                **CLAUDE_CLIENT_HEADERS,
            },
            body=openclaw_oauth_body,
        ),
        ProbeCase(
            label="bearer-no-beta",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
            },
            body=minimal_body,
        ),
        ProbeCase(
            label="x-api-key-openclaw-betas",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": OPENCLAW_OAUTH_BETA_HEADER,
            },
            body=minimal_body,
        ),
    ]


def safe_headers_for_display(headers: dict[str, str]) -> dict[str, str]:
    masked: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower == "authorization" and value.startswith("Bearer "):
            masked[key] = f"Bearer {redact_token(value[7:])}"
        elif "sk-ant" in value:
            masked[key] = redact_token(value)
        else:
            masked[key] = value
    return masked


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), indent=2)


def run_case(client: Any, case: ProbeCase) -> tuple[int | None, str]:
    try:
        response = client.post(
            case.url,
            headers={**case.headers, "content-type": "application/json"},
            json=case.body,
        )
        return response.status_code, response.text
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def print_case_result(model: str, case: ProbeCase, status_code: int | None, response_text: str) -> None:
    print(f"\n[{model}] {case.label}")  # noqa: T201
    print(f"  url: {case.url}")  # noqa: T201
    print(f"  headers: {safe_headers_for_display(case.headers)}")  # noqa: T201
    print(f"  body: {compact_json(case.body)}")  # noqa: T201
    if status_code is None:
        print(f"  transport_error: {response_text}")  # noqa: T201
        return
    print(f"  status: {status_code}")  # noqa: T201
    try:
        parsed = json.loads(response_text)
        formatted = compact_json(parsed)
    except json.JSONDecodeError:
        formatted = response_text
    print(f"  response: {formatted[:4000]}")  # noqa: T201


def infer_summary(label: str, status_code: int | None, response_text: str) -> str:
    if status_code == 200:
        return "ok"
    if status_code is None:
        return "transport_error"
    lower = response_text.lower()
    if "only authorized for use with claude code" in lower:
        return "server_rejected_non_claude_code_use"
    if "extra usage is required for long context requests" in lower:
        return "context1m_or_extra_usage"
    if '"type":"invalid_request_error"' in lower or '"type": "invalid_request_error"' in lower:
        if '"message":"error"' in lower or '"message": "error"' in lower:
            return "generic_invalid_request"
        return "invalid_request_with_detail"
    if status_code in (401, 403):
        return "auth_or_scope"
    if status_code == 429:
        return "rate_limit"
    if status_code == 402:
        return "billing"
    return f"http_{status_code}_{label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", nargs="?", help="Claude setup-token / OAuth token")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model ids to probe",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly ok.",
        help="Prompt to send",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="max_tokens for the probe request",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import httpx

    token = get_token(args.token)

    print("Claude setup-token probe")  # noqa: T201
    print(f"token: {redact_token(token)}")  # noqa: T201
    print(f"models: {', '.join(args.models)}")  # noqa: T201
    print(f"openclaw_oauth_betas: {OPENCLAW_OAUTH_BETA_HEADER}")  # noqa: T201

    summary: list[tuple[str, str, str]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for model in args.models:
            for case in build_cases(token, model, args.prompt, args.max_tokens):
                status_code, response_text = run_case(client, case)
                print_case_result(model, case, status_code, response_text)
                summary.append((model, case.label, infer_summary(case.label, status_code, response_text)))

    print("\nSummary")  # noqa: T201
    for model, label, result in summary:
        print(f"  {model:20s} {label:28s} {result}")  # noqa: T201


if __name__ == "__main__":
    main()
