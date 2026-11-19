"""Download a verified Policy Release wheel without compiling private sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

REPOSITORY = "Datus-ai/datus-sql-policies"


def request(endpoint: str, *, binary: bool = False) -> bytes:
    command = ["gh", "api", endpoint]
    if binary:
        command += ["-H", "Accept: application/octet-stream"]
    for attempt in range(3):
        try:
            return subprocess.run(command, check=True, capture_output=True, timeout=180).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise RuntimeError(f"GitHub Release request failed: {endpoint}") from None
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def asset_bytes(asset: dict) -> bytes:
    data = request(f"repos/{REPOSITORY}/releases/assets/{asset['id']}", binary=True)
    if asset.get("digest") and asset["digest"] != "sha256:" + checksum(data):
        raise ValueError(f"Release asset digest mismatch: {asset['name']}")
    return data


def platform_name() -> str:
    supported = {("Linux", "x86_64"): "linux-amd64", ("Darwin", "arm64"): "macos-arm64"}
    try:
        return supported[platform.system(), platform.machine()]
    except KeyError:
        raise ValueError("No Policy Release platform for this runner") from None


def select_wheel(release: dict) -> dict:
    supported = set(sys_tags())
    version = Version(release["tag_name"].removeprefix("v"))
    matches = []
    for asset in release["assets"]:
        if not asset["name"].startswith("datus_sql_policies-") or not asset["name"].endswith(".whl"):
            continue
        name, wheel_version, _, tags = parse_wheel_filename(asset["name"])
        if name == "datus-sql-policies" and wheel_version == version and tags & supported:
            matches.append(asset)
    if len(matches) != 1:
        raise ValueError("Release must contain exactly one compatible Policy wheel for this Python/platform")
    return matches[0]


def release_commit(tag: str) -> str:
    reference = json.loads(request(f"repos/{REPOSITORY}/git/ref/tags/{tag}"))["object"]
    for _ in range(5):
        if reference["type"] == "commit":
            return reference["sha"]
        if reference["type"] != "tag":
            break
        reference = json.loads(request(f"repos/{REPOSITORY}/git/tags/{reference['sha']}"))["object"]
    raise ValueError("Policy Release tag does not resolve to a commit")


def prepare(work: Path, *, release: str = "", agent: Path | None = None) -> dict:
    agent = agent or Path.cwd()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise ValueError("Use a fresh Policy Release work directory")
    receipt = {"status": "failure", "requested_release": release or "default"}
    try:
        selected = release or "latest"
        if selected != "latest" and not re.fullmatch(r"v\d+\.\d+\.\d+", selected):
            raise ValueError("Policy Release must be latest or an explicit stable vX.Y.Z tag")
        endpoint = "latest" if selected == "latest" else "tags/" + selected
        resolved = json.loads(request(f"repos/{REPOSITORY}/releases/{endpoint}"))
        tag = resolved["tag_name"]
        if resolved.get("draft") or resolved.get("prerelease") or not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise ValueError("Policy must come from a published stable Release")
        if selected != "latest" and tag != selected:
            raise ValueError("Resolved Policy Release differs from the requested tag")
        target = platform_name()
        wheel = select_wheel(resolved)
        assets = {asset["name"]: asset for asset in resolved["assets"]}
        manifest_name, sums_name = f"BUILD-MANIFEST-{target}.json", f"SHA256SUMS-{target}"
        if manifest_name not in assets or sums_name not in assets:
            raise ValueError("Policy Release is missing its build manifest or checksums")
        manifest_data = asset_bytes(assets[manifest_name])
        sums_data = asset_bytes(assets[sums_name])
        expected = {}
        for line in sums_data.decode().splitlines():
            digest, name = line.split(maxsplit=1)
            if name in expected or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("Malformed or duplicate Policy Release checksums")
            expected[name] = digest
        if expected.get(manifest_name) != checksum(manifest_data):
            raise ValueError("Policy build manifest checksum mismatch")
        manifest = json.loads(manifest_data)
        if (
            manifest.get("development") is not False
            or manifest.get("version") != tag.removeprefix("v")
            or manifest.get("platform") != target
            or not re.fullmatch("[0-9a-f]{40}", manifest.get("source_commit", ""))
            or not re.fullmatch("[0-9a-f]{40}", manifest.get("public_api_revision", ""))
        ):
            raise ValueError("Policy build manifest has inconsistent version/platform/source identity")
        if manifest["source_commit"] != release_commit(tag):
            raise ValueError("Policy build manifest does not match the Release tag's source commit")
        host = Requirement(manifest["hosts"]["dosi_requirement"])
        installed_dosi = metadata.version("dosi-engine")
        if host.name != "dosi-engine" or installed_dosi not in host.specifier:
            raise ValueError(f"Installed dosi-engine {installed_dosi} does not satisfy {host}")
        data = asset_bytes(wheel)
        digest = checksum(data)
        if expected.get(wheel["name"]) != digest:
            raise ValueError("Policy wheel checksum mismatch")
        path = work / wheel["name"]
        path.write_bytes(data)
        receipt.update(
            status="success",
            tag=tag,
            release_id=resolved["id"],
            release_url=resolved["html_url"],
            source_commit=manifest["source_commit"],
            public_api_revision=manifest["public_api_revision"],
            platform=target,
            dosi_version=installed_dosi,
            agent_sha=subprocess.check_output(["git", "-C", str(agent), "rev-parse", "HEAD"], text=True).strip(),
            wheel={"file": wheel["name"], "asset_id": wheel["id"], "sha256": digest, "path": str(path.resolve())},
        )
        return receipt
    except Exception as exc:
        receipt["status"] = "failure"
        receipt["error"] = str(exc)
        raise
    finally:
        (work / "sql-policy-release.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--release", default="")
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    try:
        receipt = prepare(args.work.resolve(), release=args.release)
    except Exception as exc:
        parser.exit(1, f"Policy Release preparation failed: {exc}\n")
    if args.github_env:
        with args.github_env.open("a") as stream:
            stream.write("DATUS_SQL_POLICY_WHEEL=" + receipt["wheel"]["path"] + "\n")
    print(f"Policy {receipt['tag']} ({receipt['source_commit']}): {receipt['wheel']['file']}")


if __name__ == "__main__":
    main()
