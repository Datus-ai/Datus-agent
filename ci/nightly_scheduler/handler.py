"""Dispatch the Datus-agent nightly from EventBridge Scheduler exactly once per day.

EventBridge Scheduler fires on time and delivers at least once; this Lambda turns
that into one ``workflow_dispatch`` per Beijing calendar day. Every dispatch
carries ``schedule_key`` (``nightly-YYYY-MM-DD``), which the workflow renders in
its run name, so a retry first looks for an existing run before dispatching.

Credentials come from Secrets Manager at call time and never reach logs, return
values or alerts. Only fixed, non-secret failure codes cross those boundaries.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

# A named logger keeps INFO results visible in Lambda without changing the root level.
LOGGER = logging.getLogger("datus_nightly_scheduler")
LOGGER.setLevel(logging.INFO)

API = "https://api.github.com"
BEIJING = ZoneInfo("Asia/Shanghai")
UTC = dt.timezone.utc
RUN_TITLE_PREFIX = "Run Nightly Tests · "
# A replayed or long-delayed event must not start an old day's nightly.
MAX_EVENT_AGE = dt.timedelta(hours=6)
CONFIRM_ATTEMPTS = 6
CONFIRM_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 20
# Optional confirmation stops while this much Lambda time is still left.
CONFIRM_RESERVE_SECONDS = CONFIRM_INTERVAL_SECONDS + REQUEST_TIMEOUT_SECONDS + 10
TOKEN_KEYS = ("datus-nightly-token", "Token", "token")
FEISHU_KEYS = ("FEISHU_NOTIFY_CI_URL", "webhook")


class DispatchError(Exception):
    """A dispatch failure whose message is a fixed, non-secret code."""


def _config() -> dict:
    return {
        "repository": os.environ.get("REPOSITORY", "Datus-ai/Datus-agent"),
        "workflow": os.environ.get("WORKFLOW_FILE", "run-nightly.yml"),
        "ref": os.environ.get("WORKFLOW_REF", "main"),
        "secret_id": os.environ.get("GITHUB_SECRET_ID", "datus/ci/nightly-github"),
    }


def parse_scheduled_time(value: object) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise DispatchError("scheduled_time_missing")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise DispatchError("scheduled_time_invalid") from None
    if parsed.tzinfo is None:
        raise DispatchError("scheduled_time_invalid")
    return parsed.astimezone(UTC)


def schedule_key(scheduled: dt.datetime) -> str:
    return "nightly-" + scheduled.astimezone(BEIJING).date().isoformat()


def _pick(payload: object, keys: tuple[str, ...]) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_secrets(client, secret_id: str) -> tuple[str, str | None]:
    """Return (GitHub token, optional Feishu webhook) from one secret."""
    try:
        raw = client.get_secret_value(SecretId=secret_id)["SecretString"]
    except Exception:
        raise DispatchError("github_secret_unavailable") from None
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = raw
    token = _pick(payload, TOKEN_KEYS)
    if not token or any(char.isspace() for char in token):
        raise DispatchError("github_token_invalid")
    webhook = _pick(payload, FEISHU_KEYS) if isinstance(payload, dict) else None
    if webhook and urllib.parse.urlsplit(webhook).scheme != "https":
        webhook = None
    return token, webhook


class GitHub:
    def __init__(self, token: str, repository: str, opener=None):
        self._token = token
        self._repository = repository
        self._open = opener or urllib.request.urlopen

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{API}/repos/{self._repository}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "datus-agent-nightly-scheduler",
                **({"Content-Type": "application/json"} if data else {}),
            },
        )
        try:
            response = self._open(request, timeout=REQUEST_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            response = error
        except Exception:
            raise DispatchError("github_transport_uncertain") from None
        try:
            with response:
                status = response.status
                raw = response.read(2 * 1024 * 1024)
        except Exception:
            # A body that times out or resets mid-read is as uncertain as a lost connection.
            raise DispatchError("github_transport_uncertain") from None
        try:
            return status, json.loads(raw) if raw else None
        except ValueError:
            return status, None

    def find_run(self, workflow: str, ref: str, key: str, scheduled: dt.datetime) -> dict | None:
        since = (scheduled - dt.timedelta(days=1)).date().isoformat()
        query = urllib.parse.urlencode(
            {"event": "workflow_dispatch", "branch": ref, "created": f">={since}", "per_page": 100}
        )
        status, payload = self.request("GET", f"/actions/workflows/{workflow}/runs?{query}")
        if status != 200 or not isinstance(payload, dict):
            raise DispatchError(f"github_list_runs_http_{status}")
        title = RUN_TITLE_PREFIX + key
        for run in payload.get("workflow_runs") or []:
            if isinstance(run, dict) and run.get("display_title") == title:
                return {"id": run.get("id"), "url": run.get("html_url"), "status": run.get("status")}
        return None

    def dispatch(self, workflow: str, ref: str, key: str) -> None:
        status, _ = self.request(
            "POST", f"/actions/workflows/{workflow}/dispatches", {"ref": ref, "inputs": {"schedule_key": key}}
        )
        if status != 204:
            raise DispatchError(f"github_dispatch_http_{status}")


def notify_feishu(webhook: str | None, text: str, opener=None) -> bool:
    if not webhook:
        return False
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}, ensure_ascii=False).encode()
    request = urllib.request.Request(webhook, data=payload, method="POST", headers={"Content-Type": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=10) as response:
            result = json.loads(response.read(16384) or b"{}")
        return result.get("code", result.get("StatusCode")) == 0
    except Exception:
        LOGGER.warning("feishu_alert_failed")
        return False


def run(
    event: dict,
    *,
    secrets_client,
    now: dt.datetime,
    opener=None,
    sleep=time.sleep,
    remaining=lambda: float("inf"),
) -> dict:
    """``remaining`` returns the seconds left before the Lambda times out."""
    config = _config()
    scheduled = parse_scheduled_time(event.get("scheduled_time"))
    key = schedule_key(scheduled)
    if now - scheduled > MAX_EVENT_AGE:
        LOGGER.warning("stale_schedule_event key=%s", key)
        return {"status": "stale", "schedule_key": key}

    token, webhook = load_secrets(secrets_client, config["secret_id"])
    github = GitHub(token, config["repository"], opener)
    workflow, ref = config["workflow"], config["ref"]
    try:
        existing = github.find_run(workflow, ref, key, scheduled)
        if existing:
            return {"status": "already_dispatched", "schedule_key": key, "run": existing}
        if event.get("dry_run") is True:
            return {"status": "dry_run", "schedule_key": key, "ref": ref}
        try:
            github.dispatch(workflow, ref, key)
        except DispatchError as error:
            # A lost response may still have created the run; never dispatch blind.
            if str(error) != "github_transport_uncertain":
                raise
            sleep(CONFIRM_INTERVAL_SECONDS)
            existing = github.find_run(workflow, ref, key, scheduled)
            if existing:
                return {"status": "dispatched", "schedule_key": key, "run": existing}
            raise
        # 204 is authoritative. Confirmation only records the run for the logs, so
        # its failures or a shrinking time budget must not fail the invocation.
        for _ in range(CONFIRM_ATTEMPTS):
            if remaining() < CONFIRM_RESERVE_SECONDS:
                break
            sleep(CONFIRM_INTERVAL_SECONDS)
            try:
                existing = github.find_run(workflow, ref, key, scheduled)
            except DispatchError as error:
                LOGGER.warning("dispatch_confirmation_skipped key=%s reason=%s", key, error)
                break
            if existing:
                return {"status": "dispatched", "schedule_key": key, "run": existing}
        return {"status": "dispatched", "schedule_key": key, "run": None}
    except DispatchError as error:
        notify_feishu(
            webhook,
            f"[Datus-agent Nightly] Scheduled dispatch {key} failed: {error}. "
            "EventBridge will retry; check the scheduler Lambda logs if no run starts.",
            opener,
        )
        raise


def handler(event, context):  # pragma: no cover - thin AWS entry point
    import boto3

    result = run(
        event or {},
        secrets_client=boto3.client("secretsmanager"),
        now=dt.datetime.now(UTC),
        remaining=lambda: context.get_remaining_time_in_millis() / 1000,
    )
    LOGGER.info("nightly_scheduler_result %s", json.dumps(result, sort_keys=True))
    return result
