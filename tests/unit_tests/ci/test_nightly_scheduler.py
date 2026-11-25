"""The AWS nightly scheduler dispatches main once per Beijing day and never leaks credentials."""

import datetime as dt
import io
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from ci.nightly_scheduler import deploy, handler

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "run-nightly.yml"
TOKEN = "github_pat_PRIVATE_TOKEN_VALUE"
FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/private-hook-id"
SCHEDULED = "2026-11-23T20:17:00Z"  # 04:17 on 2026-11-24 in Beijing
NOW = dt.datetime(2026, 11, 23, 20, 17, 5, tzinfo=dt.timezone.utc)
KEY = "nightly-2026-11-24"


class FakeSecrets:
    def __init__(self, value=None, error=None):
        self.value = (
            json.dumps({"datus-nightly-token": TOKEN, "FEISHU_NOTIFY_CI_URL": FEISHU}) if value is None else value
        )
        self.error = error
        self.calls = []

    def get_secret_value(self, SecretId):
        self.calls.append(SecretId)
        if self.error:
            raise self.error
        return {"SecretString": self.value}


class Response(io.BytesIO):
    def __init__(self, status, payload=None):
        super().__init__(b"" if payload is None else json.dumps(payload).encode())
        self.status = status


class StalledResponse(Response):
    """Headers arrived, but the body read times out."""

    def read(self, *args):
        raise TimeoutError("The read operation timed out")


class FakeNetwork:
    """Scripted GitHub/Feishu endpoint that records every request."""

    def __init__(self, list_results, dispatch=204, feishu_code=0):
        self.list_results = list(list_results)
        self.dispatch = dispatch
        self.feishu_code = feishu_code
        self.requests = []

    def __call__(self, request, timeout):
        body = request.data.decode() if request.data else None
        self.requests.append((request.get_method(), request.full_url, dict(request.header_items()), body))
        if request.full_url.startswith("https://open.feishu.cn/"):
            return Response(200, {"code": self.feishu_code})
        if request.get_method() == "POST":
            if isinstance(self.dispatch, Exception):
                raise self.dispatch
            if self.dispatch == "stall":
                return StalledResponse(204)
            return Response(self.dispatch)
        runs = self.list_results.pop(0) if len(self.list_results) > 1 else self.list_results[0]
        if runs == "stall":
            return StalledResponse(200, {"workflow_runs": []})
        if isinstance(runs, int):
            return Response(runs, {"message": "error"})
        return Response(200, {"workflow_runs": runs})

    def posts(self):
        return [entry for entry in self.requests if entry[0] == "POST" and "api.github.com" in entry[1]]

    def alerts(self):
        return [entry for entry in self.requests if entry[1] == FEISHU]


def matching_run(key=KEY, run_id=42):
    return {
        "id": run_id,
        "html_url": f"https://github.com/runs/{run_id}",
        "status": "queued",
        "display_title": f"Run Nightly Tests · {key}",
    }


def run(network, event=None, secrets=None, now=NOW, remaining=lambda: float("inf")):
    sleeps = []
    result = handler.run(
        {"scheduled_time": SCHEDULED} if event is None else event,
        secrets_client=secrets or FakeSecrets(),
        now=now,
        opener=network,
        sleep=sleeps.append,
        remaining=remaining,
    )
    return result, sleeps


@pytest.fixture(autouse=True)
def _scheduler_env(monkeypatch):
    for name in ("REPOSITORY", "WORKFLOW_FILE", "WORKFLOW_REF", "GITHUB_SECRET_ID"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("scheduled", "expected"),
    [
        ("2026-11-23T20:17:00Z", "nightly-2026-11-24"),
        ("2026-11-23T15:59:59+00:00", "nightly-2026-11-23"),
        ("2026-11-23T16:00:00+00:00", "nightly-2026-11-24"),
        ("2026-11-24T04:17:00+08:00", "nightly-2026-11-24"),
    ],
)
def test_schedule_key_is_the_beijing_calendar_day(scheduled, expected):
    assert handler.schedule_key(handler.parse_scheduled_time(scheduled)) == expected


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (None, "scheduled_time_missing"),
        ("", "scheduled_time_missing"),
        ("tomorrow", "scheduled_time_invalid"),
        ("2026-11-23T20:17:00", "scheduled_time_invalid"),
    ],
)
def test_unusable_scheduled_time_is_rejected_before_any_request(value, code):
    network = FakeNetwork([[]])
    secrets = FakeSecrets()
    with pytest.raises(handler.DispatchError, match=f"^{code}$"):
        run(network, event={"scheduled_time": value}, secrets=secrets)
    assert network.requests == [] and secrets.calls == []


def test_first_delivery_dispatches_main_with_the_day_key_and_records_the_run():
    network = FakeNetwork([[], [], [matching_run()]])
    result, sleeps = run(network)
    assert result == {
        "status": "dispatched",
        "schedule_key": KEY,
        "run": {"id": 42, "url": "https://github.com/runs/42", "status": "queued"},
    }
    ((method, url, headers, body),) = network.posts()
    assert url == "https://api.github.com/repos/Datus-ai/Datus-agent/actions/workflows/run-nightly.yml/dispatches"
    assert json.loads(body) == {"ref": "main", "inputs": {"schedule_key": KEY}}
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    listing = network.requests[0][1]
    assert "event=workflow_dispatch" in listing and "branch=main" in listing and "created=%3E%3D2026-11-22" in listing
    assert sleeps == [handler.CONFIRM_INTERVAL_SECONDS] * 2


def test_redelivery_for_the_same_day_does_not_dispatch_again():
    other_day = matching_run(key="nightly-2026-11-23", run_id=7)
    network = FakeNetwork([[other_day, matching_run()]])
    result, _ = run(network)
    assert result["status"] == "already_dispatched" and result["run"]["id"] == 42
    assert network.posts() == []


def test_a_previous_day_run_does_not_suppress_today():
    network = FakeNetwork([[matching_run(key="nightly-2026-11-23", run_id=7)], [matching_run()]])
    result, _ = run(network)
    assert result["status"] == "dispatched" and len(network.posts()) == 1


def test_dry_run_checks_access_without_dispatching():
    network = FakeNetwork([[]])
    result, _ = run(network, event={"scheduled_time": SCHEDULED, "dry_run": True})
    assert result == {"status": "dry_run", "schedule_key": KEY, "ref": "main"}
    assert network.posts() == [] and len(network.requests) == 1


def test_stale_event_never_starts_an_old_nightly():
    network = FakeNetwork([[]])
    secrets = FakeSecrets()
    result, _ = run(network, secrets=secrets, now=NOW + dt.timedelta(hours=7))
    assert result == {"status": "stale", "schedule_key": KEY}
    assert network.requests == [] and secrets.calls == []


def test_lost_dispatch_response_is_resolved_by_lookup_not_a_second_post():
    network = FakeNetwork([[], [matching_run()]], dispatch=TimeoutError("read timed out"))
    result, _ = run(network)
    assert result["status"] == "dispatched" and result["run"]["id"] == 42
    assert len(network.posts()) == 1 and network.alerts() == []


def test_unconfirmed_lost_dispatch_fails_for_retry_and_alerts_once():
    network = FakeNetwork([[]], dispatch=ConnectionResetError("reset"))
    with pytest.raises(handler.DispatchError, match="^github_transport_uncertain$"):
        run(network)
    assert len(network.posts()) == 1
    assert len(network.alerts()) == 1


def test_accepted_dispatch_that_is_not_yet_listed_still_counts_as_dispatched():
    network = FakeNetwork([[]])
    result, sleeps = run(network)
    assert result == {"status": "dispatched", "schedule_key": KEY, "run": None}
    assert len(network.posts()) == 1 and len(sleeps) == handler.CONFIRM_ATTEMPTS


def test_failed_confirmation_after_accepted_dispatch_still_succeeds_without_alert():
    # GET 200 (no run) -> POST 204 -> GET 503: the run was created; do not fail or retry.
    network = FakeNetwork([[], 503])
    result, _ = run(network)
    assert result == {"status": "dispatched", "schedule_key": KEY, "run": None}
    assert len(network.posts()) == 1 and network.alerts() == []


def test_confirmation_read_timeout_after_accepted_dispatch_still_succeeds():
    network = FakeNetwork([[], "stall"])
    result, _ = run(network)
    assert result == {"status": "dispatched", "schedule_key": KEY, "run": None}
    assert len(network.posts()) == 1 and network.alerts() == []


def test_dispatch_body_read_timeout_is_resolved_by_lookup():
    network = FakeNetwork([[], [matching_run()]], dispatch="stall")
    result, _ = run(network)
    assert result["status"] == "dispatched" and result["run"]["id"] == 42
    assert len(network.posts()) == 1


def test_listing_read_timeout_before_dispatch_stays_fatal():
    network = FakeNetwork(["stall"])
    with pytest.raises(handler.DispatchError, match="^github_transport_uncertain$"):
        run(network)
    assert network.posts() == [] and len(network.alerts()) == 1


def test_failed_lookup_during_lost_response_recovery_stays_fatal():
    network = FakeNetwork([[], 503], dispatch=TimeoutError("read timed out"))
    with pytest.raises(handler.DispatchError, match="^github_list_runs_http_503$"):
        run(network)
    assert len(network.posts()) == 1 and len(network.alerts()) == 1


@pytest.mark.parametrize("budget", [0, handler.CONFIRM_RESERVE_SECONDS - 1])
def test_confirmation_stops_before_the_lambda_deadline(budget):
    network = FakeNetwork([[]])
    result, sleeps = run(network, remaining=lambda: budget)
    assert result == {"status": "dispatched", "schedule_key": KEY, "run": None}
    assert sleeps == [] and len(network.posts()) == 1


def test_confirmation_uses_the_remaining_budget_as_it_shrinks():
    left = iter([100, 60, handler.CONFIRM_RESERVE_SECONDS - 1])
    network = FakeNetwork([[]])
    result, sleeps = run(network, remaining=lambda: next(left))
    assert result["run"] is None and len(sleeps) == 2


@pytest.mark.parametrize("status", [401, 403, 404, 422, 500])
def test_rejected_dispatch_alerts_feishu_without_secrets_and_raises(status):
    network = FakeNetwork([[]], dispatch=status)
    with pytest.raises(handler.DispatchError, match=f"^github_dispatch_http_{status}$"):
        run(network)
    ((_, _, _, alert),) = network.alerts()
    text = json.loads(alert)["content"]["text"]
    assert KEY in text and f"github_dispatch_http_{status}" in text
    assert TOKEN not in alert and FEISHU not in alert


def test_http_error_raised_by_urlopen_is_read_as_a_status():
    def opener(request, timeout):
        if request.get_method() == "POST":
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b"{}"))
        return FakeNetwork([[]])(request, timeout)

    with pytest.raises(handler.DispatchError, match="^github_dispatch_http_403$"):
        run(opener)


def test_listing_failure_stops_before_dispatch():
    network = FakeNetwork([503])
    with pytest.raises(handler.DispatchError, match="^github_list_runs_http_503$"):
        run(network)
    assert network.posts() == [] and len(network.alerts()) == 1


@pytest.mark.parametrize(
    ("value", "token", "webhook"),
    [
        (json.dumps({"datus-nightly-token": TOKEN, "FEISHU_NOTIFY_CI_URL": FEISHU}), TOKEN, FEISHU),
        (json.dumps({"Token": f" {TOKEN} "}), TOKEN, None),
        (json.dumps({"token": TOKEN, "webhook": "http://insecure.example/hook"}), TOKEN, None),
        (f"{TOKEN}\n", TOKEN, None),
    ],
)
def test_secret_shapes(value, token, webhook):
    assert handler.load_secrets(FakeSecrets(value), "id") == (token, webhook)


@pytest.mark.parametrize(
    ("secrets", "code"),
    [
        (FakeSecrets(error=RuntimeError(f"AccessDenied {TOKEN}")), "github_secret_unavailable"),
        (FakeSecrets(json.dumps({"FEISHU_NOTIFY_CI_URL": FEISHU})), "github_token_invalid"),
        (FakeSecrets(json.dumps({"datus-nightly-token": "two words"})), "github_token_invalid"),
        (FakeSecrets(json.dumps(["list"])), "github_token_invalid"),
    ],
)
def test_unusable_secret_fails_with_a_fixed_code(secrets, code):
    with pytest.raises(handler.DispatchError) as raised:
        handler.load_secrets(secrets, "id")
    assert str(raised.value) == code and raised.value.__cause__ is None


def test_environment_overrides_target(monkeypatch):
    monkeypatch.setenv("REPOSITORY", "Example/repo")
    monkeypatch.setenv("WORKFLOW_FILE", "other.yml")
    monkeypatch.setenv("WORKFLOW_REF", "release")
    secrets = FakeSecrets()
    monkeypatch.setenv("GITHUB_SECRET_ID", "custom/secret")
    network = FakeNetwork([[matching_run()]])
    run(network, secrets=secrets)
    assert secrets.calls == ["custom/secret"]
    assert (
        network.requests[0][1].startswith("https://api.github.com/repos/Example/repo/actions/workflows/other.yml/runs?")
        and "branch=release" in network.requests[0][1]
    )


def test_feishu_alert_reports_receipt():
    assert handler.notify_feishu(FEISHU, "x", FakeNetwork([[]])) is True
    assert handler.notify_feishu(FEISHU, "x", FakeNetwork([[]], feishu_code=9499)) is False
    assert handler.notify_feishu(None, "x", FakeNetwork([[]])) is False

    def broken(request, timeout):
        raise OSError("down")

    assert handler.notify_feishu(FEISHU, "x", broken) is False


def test_template_fires_on_time_and_bounds_retries():
    template = json.loads(deploy.TEMPLATE.read_text())
    params = template["Parameters"]
    assert params["ScheduleExpression"]["Default"] == "cron(17 4 * * ? *)"
    assert params["ScheduleTimezone"]["Default"] == "Asia/Shanghai"
    assert params["ScheduleState"]["Default"] == "DISABLED"
    schedule = template["Resources"]["DailySchedule"]["Properties"]
    assert schedule["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert "GroupName" not in schedule
    target = schedule["Target"]
    assert json.loads(target["Input"]) == {"scheduled_time": "<aws.scheduler.scheduled-time>"}
    assert target["RetryPolicy"]["MaximumEventAgeInSeconds"] <= handler.MAX_EVENT_AGE.total_seconds()
    assert target["DeadLetterConfig"]["Arn"] == {"Fn::GetAtt": ["FailedEvents", "Arn"]}
    async_config = template["Resources"]["FunctionAsyncConfig"]["Properties"]
    assert async_config["MaximumEventAgeInSeconds"] <= handler.MAX_EVENT_AGE.total_seconds()
    assert async_config["DestinationConfig"]["OnFailure"]["Destination"] == {"Fn::GetAtt": ["FailedEvents", "Arn"]}
    function = template["Resources"]["Function"]["Properties"]
    assert function["Handler"] == "index.handler" and function["Code"] == {"ZipFile": deploy.PLACEHOLDER}
    # Mandatory path: listing, dispatch, then one recovery lookup after a lost
    # response. Optional confirmation is bounded by the remaining time instead.
    mandatory = 3 * handler.REQUEST_TIMEOUT_SECONDS + handler.CONFIRM_INTERVAL_SECONDS
    assert function["Timeout"] >= mandatory + handler.CONFIRM_RESERVE_SECONDS + 30


def test_template_grants_only_the_configured_secret():
    template = json.loads(deploy.TEMPLATE.read_text())
    statements = template["Resources"]["FunctionRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    secret = next(s for s in statements if s["Action"] == "secretsmanager:GetSecretValue")
    assert secret["Resource"]["Fn::Sub"].endswith(":secret:${GitHubSecretId}-??????")
    text = deploy.TEMPLATE.read_text()
    assert "github_pat_" not in text and "hook/" not in text


def test_render_inlines_the_reviewed_handler(tmp_path):
    rendered = json.loads(deploy.render())
    assert rendered["Resources"]["Function"]["Properties"]["Code"]["ZipFile"] == deploy.HANDLER.read_text()
    broken = tmp_path / "template.json"
    template = json.loads(deploy.TEMPLATE.read_text())
    template["Resources"]["Function"]["Properties"]["Code"]["ZipFile"] = "print('stale')"
    broken.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="placeholder"):
        deploy.render(template=broken)
    huge = tmp_path / "handler.py"
    huge.write_text("#" * deploy.MAX_TEMPLATE_BYTES)
    with pytest.raises(ValueError, match="size limit"):
        deploy.render(handler=huge)


def test_deploy_only_creates_a_change_set_unless_executed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        deploy.subprocess, "run", lambda command, check: calls.append(command) or type("R", (), {"returncode": 0})
    )
    assert deploy.main(["--alarm-email", "ops@example.com"]) == 0
    assert deploy.main(["--execute", "--schedule-state", "ENABLED", "--github-secret-id", "other/secret"]) == 0
    review, apply = calls
    assert review[-1] == "--no-execute-changeset" and "ScheduleState=DISABLED" in review
    assert "AlarmEmail=ops@example.com" in review
    assert "--no-execute-changeset" not in apply
    assert "ScheduleState=ENABLED" in apply and "GitHubSecretId=other/secret" in apply
    output = tmp_path / "rendered.json"
    assert deploy.main(["--render-only", str(output)]) == 0 and len(calls) == 2
    assert json.loads(output.read_text())["Resources"]["DailySchedule"]


def test_workflow_has_no_github_cron_and_titles_scheduled_runs_for_lookup():
    raw = WORKFLOW.read_text()
    triggers = yaml.load(raw, Loader=yaml.BaseLoader)["on"]
    assert "schedule" not in triggers
    key_input = triggers["workflow_dispatch"]["inputs"]["schedule_key"]
    assert key_input["required"] == "false" and key_input["default"] == ""
    run_name = yaml.safe_load(raw)["run-name"]
    # The handler finds a day's run by this exact rendered title.
    assert f"format('{handler.RUN_TITLE_PREFIX}{{0}}', inputs.schedule_key)" in run_name
    assert run_name.endswith("|| 'Run Nightly Tests' }}")
