"""Run the required BI contract against isolated Superset and PostgreSQL services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ci.harness.test_plan import changed_files, impact_plan  # noqa: E402
from ci.p0_plan import check_p0_execution, load_profile  # noqa: E402

GROUP = "P0 BI Product E2E"
COMPOSE_UP_TIMEOUT_SECONDS = 1500
PYTEST_TIMEOUT_SECONDS = 1800
COMPOSE_LOGS_TIMEOUT_SECONDS = 60
COMPOSE_DOWN_TIMEOUT_SECONDS = 120


def required(changed: list[str]) -> bool:
    return GROUP in impact_plan(changed)["nightly_groups"]


def snapshot(root: Path, external: Path) -> dict:
    result = {}
    for name, path in {
        "agent": root,
        "bi_adapters": external / "datus-bi-adapters",
        "plugins": external / "Datus-Plugins",
    }.items():
        sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--binary"])
        if name != "agent":
            if sha != load_profile()["release_sources"][name] or diff:
                raise ValueError(f"BI dependency must be clean and match its pinned SHA: {name}")
        result[name] = {"sha": sha, "patch_sha256": hashlib.sha256(diff).hexdigest()}
        if name == "agent":
            untracked = (
                subprocess.check_output(["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"])
                .decode()
                .split("\0")
            )
            result[name]["untracked_inputs"] = {
                file: hashlib.sha256((path / file).read_bytes()).hexdigest()
                for file in sorted(untracked)
                if file.startswith(("datus/", "tests/", "ci/", ".github/"))
                and Path(file).suffix in {".py", ".yml", ".yaml", ".toml", ".json", ".md", ".txt"}
            }
    return result


def compose_document(source: Path, web_port: int, postgres_port: int) -> dict:
    data = yaml.safe_load(source.read_text())
    for name, service in data["services"].items():
        service.pop("container_name", None)
        port = web_port if name == "superset" else postgres_port
        internal = 8088 if name == "superset" else 5432
        service["ports"] = [f"127.0.0.1:{port}:{internal}"]
        service["volumes"] = [
            str((source.parent / v.split(":", 1)[0]).resolve()) + ":" + v.split(":", 1)[1]
            for v in service.get("volumes", [])
        ]
    return data


def verify(work: Path, sources: dict) -> dict:
    receipt = json.loads((work / "result.json").read_text())
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if receipt.get("sources") != sources or receipt.get("source_id") != identity:
        raise ValueError("BI source identity changed or receipt is stale")
    report = json.loads((work / "execution.json").read_text())
    if report.get("source_id") != identity:
        raise ValueError("BI execution receipt is not bound to these sources")
    coverage = check_p0_execution([work / "execution.json"], {GROUP: load_profile()["groups"][GROUP]})
    if coverage["status"] != "success" or receipt.get("status") != "success":
        raise ValueError(f"BI P0 did not pass: {coverage['errors']}")
    return coverage


def run(work: Path, external: Path) -> None:
    work.mkdir(parents=True, exist_ok=False)
    sources = snapshot(ROOT, external)
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    receipt = {"sources": sources, "source_id": identity, "status": "failed"}
    compose_source = external / "datus-bi-adapters/datus-bi-superset/tests/integration/docker-compose.yml"
    compose_file = work / "compose.yml"
    web_port = int(os.environ.get("SUPERSET_PORT", "0"))
    pg_port = int(os.environ.get("SUPERSET_POSTGRES_PORT", "0"))
    compose_file.write_text(yaml.safe_dump(compose_document(compose_source, web_port, pg_port)))
    project = "datus-bi-gate-" + hashlib.sha256(str(work).encode()).hexdigest()[:12]
    compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    env = dict(
        os.environ,
        DATUS_TEST_LAYER="nightly",
        UV_NO_SYNC="1",
        EXTERNAL_REPOS_ROOT=str(external),
        SUPERSET_URL=f"http://127.0.0.1:{web_port}",
        SUPERSET_POSTGRES_HOST="127.0.0.1",
        SUPERSET_POSTGRES_PORT=str(pg_port),
        DATUS_GATE_SOURCE_ID=identity,
        DATUS_NIGHTLY_TRACE_EXPECTED="1",
        DATUS_NIGHTLY_SUITE_NAME=GROUP,
        DATUS_NIGHTLY_TRACE_REFERENCES_FILE=str(work / "traces.jsonl"),
        DATUS_NIGHTLY_BI_EVIDENCE_DIR=str(work / "cases"),
        DATUS_TEST_HOME=str(work / "home"),
    )
    config = yaml.safe_load((ROOT / "tests/conf/agent.yml").read_text())
    config["agent"]["home"] = str(work / "home")
    config["agent"]["project_root"] = str(work / "workspace")
    # Point test configuration at this job's isolated home without modifying tracked fixtures.
    config_file = work / "agent.yml"
    config_file.write_text(yaml.safe_dump(config))
    env["DATUS_ACCEPTANCE_CONFIG"] = str(config_file)
    nodeids = load_profile()["groups"][GROUP]["nodeids"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *nodeids,
        "-o",
        "addopts=",
        "--tb=short",
        "--no-showlocals",
        "--timeout=600",
        "--timeout-method=thread",
        "-p",
        "ci.pytest_fail_on_skip_plugin",
        "--fail-on-skip",
        "-p",
        "ci.pytest_gate_plugin",
        "--gate-report",
        str(work / "execution.json"),
        "-p",
        "ci.pytest_trace_reference_plugin",
        "--basetemp",
        str(work / "pytest"),
    ]
    try:
        if not env.get("DEEPSEEK_API_KEY"):
            raise ValueError("BI real-model P0 requires DEEPSEEK_API_KEY")
        with (work / "services.log").open("w") as log:
            subprocess.run(
                [*compose, "up", "-d", "--wait", "--wait-timeout", "1200"],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=COMPOSE_UP_TIMEOUT_SECONDS,
            )
        # Docker assigns unused loopback ports when zero is requested, so this
        # gate can run alongside nightly on the same host.
        for service, internal, variable in [
            ("superset", "8088", "SUPERSET_URL"),
            ("postgres", "5432", "SUPERSET_POSTGRES_PORT"),
        ]:
            address = subprocess.check_output([*compose, "port", service, internal], text=True).strip()
            port = int(address.rsplit(":", 1)[1])
            env[variable] = f"http://127.0.0.1:{port}" if service == "superset" else str(port)
        with (work / "pytest.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=PYTEST_TIMEOUT_SECONDS
            )
        receipt["pytest_exit"] = result.returncode
        if result.returncode:
            raise RuntimeError("BI P0 failed; see pytest.log and cases/")
        if snapshot(ROOT, external) != sources:
            raise ValueError("BI sources changed during execution")
        report = json.loads((work / "execution.json").read_text())
        coverage = check_p0_execution([work / "execution.json"], {GROUP: load_profile()["groups"][GROUP]})
        if report.get("source_id") != identity or coverage["status"] != "success":
            raise ValueError("Incomplete BI P0 execution evidence")
        receipt["status"] = "success"
    finally:
        (work / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (work / "services.log").open("a") as log:
            subprocess.run(
                [*compose, "logs", "--no-color", "--tail", "200"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=COMPOSE_LOGS_TIMEOUT_SECONDS,
            )
            subprocess.run(
                [*compose, "down", "-v", "--remove-orphans"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=COMPOSE_DOWN_TIMEOUT_SECONDS,
            )
    verify(work, sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "verify"])
    parser.add_argument("--base-ref")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--external", type=Path, default=ROOT / "external")
    args = parser.parse_args()
    if args.command == "plan":
        value = args.force or required(changed_files(args.base_ref))
        if args.github_output:
            with args.github_output.open("a") as stream:
                stream.write(f"required={str(value).lower()}\n")
        print(json.dumps({"required": value}))
    elif args.command == "run":
        run(args.work.resolve(), args.external.resolve())
    else:
        verify(args.work.resolve(), snapshot(ROOT, args.external.resolve()))


if __name__ == "__main__":
    main()
