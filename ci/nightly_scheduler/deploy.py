"""Render and deploy the Datus-agent nightly EventBridge scheduler stack.

The reviewed ``handler.py`` is inlined into ``template.json`` as the Lambda
``Code.ZipFile`` so the stack needs no S3 bucket. Deploying creates a change set;
pass ``--execute`` to apply it after review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.json"
HANDLER = HERE / "handler.py"
PLACEHOLDER = "__DATUS_NIGHTLY_SCHEDULER_HANDLER__"
# CloudFormation accepts at most 51,200 bytes for a template passed inline.
MAX_TEMPLATE_BYTES = 51_200


def render(template: Path = TEMPLATE, handler: Path = HANDLER) -> str:
    document = json.loads(template.read_text(encoding="utf-8"))
    code = document["Resources"]["Function"]["Properties"]["Code"]
    if code.get("ZipFile") != PLACEHOLDER:
        raise ValueError("Template Lambda code must be the handler placeholder")
    code["ZipFile"] = handler.read_text(encoding="utf-8")
    body = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    if len(body.encode("utf-8")) > MAX_TEMPLATE_BYTES:
        raise ValueError("Rendered template exceeds the CloudFormation inline size limit")
    return body


def deploy_command(args: argparse.Namespace, template_file: Path) -> list[str]:
    overrides = [f"ScheduleState={args.schedule_state}"]
    if args.alarm_email:
        overrides.append(f"AlarmEmail={args.alarm_email}")
    if args.github_secret_id:
        overrides.append(f"GitHubSecretId={args.github_secret_id}")
    command = [
        "aws",
        "cloudformation",
        "deploy",
        "--region",
        args.region,
        "--stack-name",
        args.stack_name,
        "--template-file",
        str(template_file),
        "--capabilities",
        "CAPABILITY_IAM",
        "--no-fail-on-empty-changeset",
        "--parameter-overrides",
        *overrides,
    ]
    if not args.execute:
        command.append("--no-execute-changeset")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-name", default="datus-agent-nightly-scheduler")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--schedule-state", choices=("ENABLED", "DISABLED"), default="DISABLED")
    parser.add_argument("--github-secret-id", default="")
    parser.add_argument("--alarm-email", default="")
    parser.add_argument("--render-only", type=Path, help="Write the rendered template here and stop.")
    parser.add_argument("--execute", action="store_true", help="Apply the change set instead of only creating it.")
    args = parser.parse_args(argv)

    body = render()
    if args.render_only:
        args.render_only.write_text(body, encoding="utf-8")
        return 0
    with tempfile.TemporaryDirectory() as directory:
        rendered = Path(directory) / "template.json"
        rendered.write_text(body, encoding="utf-8")
        return subprocess.run(deploy_command(args, rendered), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
