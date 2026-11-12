"""Build and validate one immutable Agent/policy/engine combination on a native host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ci.harness.test_plan import changed_files, impact_plan  # noqa: E402
from ci.p0_plan import check_p0_execution, load_profile  # noqa: E402
from ci.release_artifact import distributions, sha256  # noqa: E402
from ci.stage_p0 import stage  # noqa: E402

GROUP = "P0 SQL Policy Plugin Contracts"
CONFIG = ROOT / "ci/harness/sql-policy-host.json"
REQUIRED_STAGES = {
    "toolchain",
    "build-agent",
    "build-osi_engine",
    "build-sql_policies",
    "repair-osi_engine",
    "repair-sql_policies",
    "install-hosts",
    "verify-agent",
    "verify-engine",
    "policy-p0",
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_id(sources: dict) -> str:
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG.read_text())
    if config.get("schema_version") != 1 or not re.fullmatch("[0-9a-f]{40}", config.get("sql_policies", "")):
        raise ValueError("Host configuration requires schema 1 and an immutable policy SHA")
    if not re.fullmatch(r"\d+\.\d+\.\d+", config.get("rust_toolchain", "")):
        raise ValueError("Host configuration requires a pinned Rust toolchain version")
    if not config.get("cold_paths") or not all(isinstance(path, str) and path for path in config["cold_paths"]):
        raise ValueError("Host configuration requires explicit cold-build paths")
    return config


def plan_gate(changed: list[str], *, force: bool = False) -> dict:
    config = load_config()
    required = force or GROUP in impact_plan(changed)["nightly_groups"]
    cold = force or any(
        path == prefix or (prefix.endswith("/") and path.startswith(prefix))
        for path in changed
        for prefix in config["cold_paths"]
    )
    return {"required": required, "cold": required and cold, "sql_policies": config["sql_policies"]}


def revision(root: Path) -> str:
    value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Expected an immutable source SHA: {root}")
    return value


def policy_engine_revision(policy: Path) -> str:
    dependencies = tomllib.loads((policy / "Cargo.toml").read_text())["workspace"]["dependencies"]
    revisions = {
        spec.get("rev")
        for spec in dependencies.values()
        if isinstance(spec, dict)
        and spec.get("git", "").removesuffix(".git") == "https://github.com/Datus-ai/osi-engine"
    }
    if len(revisions) != 1:
        raise ValueError("SQL policy must bind exactly one engine revision")
    value = revisions.pop()
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("SQL policy engine dependency must use a full commit SHA")
    return value


def snapshot(agent: Path, policy: Path, engine: Path) -> dict:
    sources = {}
    for name, root, inputs in (
        ("agent", agent, ("pyproject.toml", "uv.lock")),
        ("sql_policies", policy, ("pyproject.toml", "Cargo.toml", "Cargo.lock")),
        ("osi_engine", engine, ("Cargo.toml", "Cargo.lock", "crates/dosi-py/pyproject.toml")),
    ):
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"])
        if name != "agent" and subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"]
        ):
            raise ValueError(f"External source checkout must remain unmodified: {name}")
        sources[name] = {
            "sha": revision(root),
            "inputs": {path: sha256(root / path) for path in inputs},
            "patch_sha256": hashlib.sha256(diff).hexdigest(),
        }
    if policy_engine_revision(policy) != sources["osi_engine"]["sha"]:
        raise ValueError("Checked-out engine does not match the policy's pinned engine revision")
    return sources


def build_environment(work: Path, mode: str, engine: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("RUSTUP_HOME", str(Path.home() / ".rustup"))
    for name, suffix in (("UV_CACHE_DIR", "uv"), ("CARGO_HOME", "cargo"), ("CARGO_TARGET_DIR", "target")):
        path = work / "cache" / suffix
        if mode == "cold":
            if path.exists() and any(path.iterdir()):
                raise ValueError(f"Cold build cache must start empty: {path}")
            path.mkdir(parents=True, exist_ok=True)
            env[name] = str(path)
        else:
            env.setdefault(name, str(path))
    # Cargo reads the exact engine checkout, without credentials during builds.
    index = int(env.get("GIT_CONFIG_COUNT", "0"))
    env["GIT_CONFIG_COUNT"] = str(index + 2)
    for offset, suffix in enumerate(("", ".git")):
        env[f"GIT_CONFIG_KEY_{index + offset}"] = f"url.{engine.resolve().as_uri()}.insteadOf"
        env[f"GIT_CONFIG_VALUE_{index + offset}"] = "https://github.com/Datus-ai/osi-engine" + suffix
    env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
    env["CARGO_BUILD_JOBS"] = "2"
    env["CARGO_PROFILE_DEV_DEBUG"] = "0"
    env["CARGO_PROFILE_DEV_OPT_LEVEL"] = "0"
    env["UV_NO_SYNC"] = "1"
    env["PYO3_PYTHON"] = sys.executable
    env["LANGFUSE_TRACING_ENABLED"] = "false"
    return env


def one_wheel(root: Path) -> Path:
    wheels = list(root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one built wheel: {root}")
    return wheels[0]


def verify_result(sources: dict, receipt: dict, execution: Path, wheels: dict[str, Path]) -> dict:
    if receipt.get("sources") != sources or receipt.get("source_id") != source_id(sources):
        raise ValueError("Source identity changed after planning/building")
    if set(wheels) != {"agent", "sql_policies", "osi_engine"}:
        raise ValueError("Every source component requires an identified wheel")
    stages = receipt.get("stages", [])
    if {step["name"] for step in stages} != REQUIRED_STAGES or any(step["exit_code"] for step in stages):
        raise ValueError("Every build, installation and runtime verification stage must succeed")
    for name, wheel in wheels.items():
        if receipt.get("artifacts", {}).get(name, {}).get("sha256") != sha256(wheel):
            raise ValueError(f"Built artifact changed: {name}")
    report = json.loads(execution.read_text())
    if report.get("source_id") != receipt["source_id"]:
        raise ValueError("Execution report belongs to a different source combination")
    result = check_p0_execution([execution], {GROUP: load_profile()["groups"][GROUP]})
    if result["status"] != "success":
        raise ValueError("; ".join(result["errors"]))
    return result


def run_gate(agent: Path, policy: Path, engine: Path, work: Path, mode: str, expected: dict) -> None:
    if sys.platform != "linux":
        raise ValueError("The native host gate builds Linux test wheels")
    sources = snapshot(agent, policy, engine)
    if {name: source["sha"] for name, source in sources.items()} != expected:
        raise ValueError("Actual checkouts do not match the workflow's frozen source SHAs")
    work.mkdir(parents=True, exist_ok=True)
    env = build_environment(work, mode, engine)
    receipt = {
        "sources": sources,
        "source_id": source_id(sources),
        "cache_mode": mode,
        "builder": {
            "python": sys.version.split()[0],
            "maturin": metadata.version("maturin"),
            "auditwheel": metadata.version("auditwheel"),
            "engine_features": ["extension-module"],
            "profile": "dev",
        },
        "stages": [],
        "status": "running",
    }
    output = work / "result.json"

    def run(name: str, command: list[str], *, cwd: Path = agent) -> None:
        started = time.monotonic()
        log = work / (name + ".log")
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        receipt["stages"].append(
            {
                "name": name,
                "exit_code": completed.returncode,
                "seconds": round(time.monotonic() - started, 2),
                "command": command,
            }
        )
        write_json(output, receipt)
        if completed.returncode:
            print("\n".join(log.read_text(errors="replace").splitlines()[-30:]), flush=True)
            raise RuntimeError(f"{name} failed; see {log}")
        print(f"{name}: passed", flush=True)

    write_json(output, receipt)
    try:
        run("toolchain", ["rustc", "--version"])
        receipt["rustc"] = (work / "toolchain.log").read_text().strip()
        run("build-agent", ["uv", "build", "--wheel", "--out-dir", str(work / "wheels/agent"), str(agent)])
        for name, root, manifest in (
            ("osi_engine", engine, engine / "crates/dosi-py/Cargo.toml"),
            ("sql_policies", policy, policy / "crates/dosi-policy-py/Cargo.toml"),
        ):
            run(
                "build-" + name,
                [
                    sys.executable,
                    "-m",
                    "maturin",
                    "build",
                    "--profile",
                    "dev",
                    "--locked",
                    "--features",
                    "extension-module",
                    "--interpreter",
                    sys.executable,
                    "--auditwheel",
                    "skip",
                    "--manifest-path",
                    str(manifest),
                    "--out",
                    str(work / "raw-wheels" / name),
                ],
                cwd=root,
            )
            raw = one_wheel(work / "raw-wheels" / name)
            raw_digest = sha256(raw)
            platform_tag = "manylinux_" + platform.libc_ver()[1].replace(".", "_") + "_" + platform.machine()
            run(
                "repair-" + name,
                [
                    sys.executable,
                    "-m",
                    "auditwheel",
                    "repair",
                    "--plat",
                    platform_tag,
                    "--wheel-dir",
                    str(work / "wheels" / name),
                    str(raw),
                ],
            )
            if sha256(raw) != raw_digest:
                raise ValueError("Wheel repair modified the original build artifact")
        wheels = {name: one_wheel(work / "wheels" / name) for name in sources}
        receipt["artifacts"] = {
            name: {"filename": wheel.name, "sha256": sha256(wheel)} for name, wheel in wheels.items()
        }
        write_json(
            work / "wheels/agent/release-artifact.json",
            {"source_sha": sources["agent"]["sha"], "distributions": distributions(work / "wheels/agent")},
        )
        run(
            "install-hosts",
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--reinstall",
                str(wheels["agent"]),
                str(wheels["osi_engine"]),
            ],
        )
        harness = work / "harness"
        stage(agent, harness)
        run(
            "verify-agent",
            [
                sys.executable,
                "-I",
                str(harness / "ci/release_artifact.py"),
                "verify-install",
                "--dist",
                str(work / "wheels/agent"),
                "--source-sha",
                sources["agent"]["sha"],
                "--checkout",
                str(agent),
                "--output",
                str(work / "installed-agent.json"),
            ],
            cwd=harness,
        )
        engine_check = (
            "from importlib import metadata; from pathlib import Path; import sys, dosi_engine; "
            "sys.path.insert(0, sys.argv[1]); from ci.release_artifact import verify_wheel_payload; "
            "root = Path(metadata.distribution('dosi-engine').locate_file('')).resolve(); "
            "assert Path(dosi_engine.__file__).resolve().is_relative_to(root); "
            "verify_wheel_payload(Path(sys.argv[2]), root, 'dosi_engine')"
        )
        run(
            "verify-engine",
            [sys.executable, "-I", "-c", engine_check, str(harness), str(wheels["osi_engine"])],
            cwd=harness,
        )
        env["DATUS_SQL_POLICY_WHEEL"] = str(wheels["sql_policies"])
        env["DATUS_GATE_SOURCE_ID"] = receipt["source_id"]
        env["EXTERNAL_REPOS_ROOT"] = str(policy.parent)
        execution = work / "execution.json"
        run(
            "policy-p0",
            [
                sys.executable,
                "-m",
                "pytest",
                *load_profile()["groups"][GROUP]["nodeids"],
                "-p",
                "ci.pytest_gate_plugin",
                "--gate-report",
                str(execution),
                "-p",
                "ci.pytest_fail_on_skip_plugin",
                "--fail-on-skip",
                "--no-showlocals",
                "--tb=short",
                "-q",
                "-o",
                "log_cli=false",
                "--basetemp=" + str(work / "pytest"),
            ],
            cwd=harness,
        )
        receipt["execution"] = verify_result(snapshot(agent, policy, engine), receipt, execution, wheels)
        receipt["status"] = "success"
    except Exception:
        receipt["status"] = "failure"
        raise
    finally:
        write_json(output, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "resolve", "run"))
    parser.add_argument("--base-ref")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--agent", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--mode", choices=("warm", "cold"), default="warm")
    parser.add_argument("--agent-sha")
    parser.add_argument("--policy-sha")
    parser.add_argument("--engine-sha")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan_gate(changed_files(args.base_ref), force=args.force)
    elif args.command == "resolve":
        result = {
            "agent": revision(args.agent),
            "sql_policies": revision(args.policy),
            "osi_engine": policy_engine_revision(args.policy),
        }
    else:
        run_gate(
            args.agent.resolve(),
            args.policy.resolve(),
            args.engine.resolve(),
            args.work.resolve(),
            args.mode,
            {"agent": args.agent_sha, "sql_policies": args.policy_sha, "osi_engine": args.engine_sha},
        )
        return
    print(json.dumps(result, indent=2))
    if args.github_output:
        with args.github_output.open("a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")


if __name__ == "__main__":
    main()
