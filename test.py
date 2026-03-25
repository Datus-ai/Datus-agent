import json
import os

import httpx

t = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
for m in ["claude-haiku-4-5", "claude-sonnet-4-6"]:
    body = json.dumps({"model": m, "max_tokens": 128, "messages": [{"role": "user", "content": "Hi"}]})
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Authorization": f"Bearer {t}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
            "content-type": "application/json",
        },
        content=body,
        timeout=30,
    )
    print(f"{m}: {r.status_code} {r.text[:100]}")
