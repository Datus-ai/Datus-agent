# Nightly scheduling on AWS EventBridge

The daily Datus-agent nightly is triggered by AWS EventBridge Scheduler instead of
a GitHub `schedule:` cron. GitHub cron runs are best-effort: they routinely start
late under load and can be dropped. EventBridge Scheduler fires at a fixed time
with `FlexibleTimeWindow: OFF`.

```
EventBridge Scheduler (04:17 Asia/Shanghai)
  -> Lambda handler.py   (reads token from Secrets Manager)
    -> POST /actions/workflows/run-nightly.yml/dispatches  ref=main, schedule_key=nightly-YYYY-MM-DD
      -> run-nightly.yml on the self-hosted runner
```

## Once per day

Scheduler and asynchronous Lambda delivery are at-least-once, so every dispatch
carries `schedule_key`, the Beijing calendar day. The workflow renders it in its run
name (`Run Nightly Tests · nightly-YYYY-MM-DD`). Before dispatching, the handler
looks for a `workflow_dispatch` run on `main` with that exact title and does nothing
if one exists.

- A rejected dispatch (HTTP 4xx/5xx) raises, which triggers the bounded Lambda retries.
- A lost response is never retried blindly. The handler checks for the run first and
  fails only when no run is visible.
- An event older than six hours returns `stale` and never starts an old day's run.
- Exhausted retries land in the `<stack>-failed-events` SQS queue. Lambda errors and
  queue depth raise CloudWatch alarms on the `<stack>-alarms` SNS topic. Dispatch
  failures also post a text alert to the Feishu webhook stored in the same secret.

## Credentials

`GitHubSecretId` (default `datus/ci/nightly-github`) holds JSON with
`datus-nightly-token` and, optionally, `FEISHU_NOTIFY_CI_URL`. A plain-text token
also works. The token needs Actions read/write on `Datus-ai/Datus-agent`. Its owner
becomes the run actor of the dispatched workflow.
The Lambda role can read only this secret. Token values never appear in logs,
results or alerts.

## Deploy and enable

```bash
# Create a change set with the schedule disabled; review it in the console.
python ci/nightly_scheduler/deploy.py
# Apply it.
python ci/nightly_scheduler/deploy.py --execute
# Check access without dispatching. Use the deployment Region (deploy.py
# defaults to us-east-1; substitute yours if you passed --region).
aws lambda invoke --region us-east-1 --function-name datus-agent-nightly-scheduler \
  --cli-binary-format raw-in-base64-out \
  --payload "{\"scheduled_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"dry_run\": true}" /dev/stdout
# After the workflow without the GitHub cron is merged to main:
python ci/nightly_scheduler/deploy.py --execute --schedule-state ENABLED
```

Keep the schedule disabled until the cron-free workflow is on `main`, otherwise
both sources trigger a run on the same day.
