"""Bind release validation and publication to the same source and distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distributions(directory: Path) -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.suffix == ".whl" or p.name.endswith(".tar.gz")}


def verify(directory: Path, source_sha: str) -> dict:
    receipt = json.loads((directory / "release-artifact.json").read_text())
    if receipt["source_sha"] != source_sha:
        raise ValueError("Release distributions were built from a different source SHA")
    actual = distributions(directory)
    if not actual or actual != receipt["distributions"] or len([name for name in actual if name.endswith(".whl")]) != 1:
        raise ValueError("Release distribution hashes do not match the validated artifact")
    return receipt


def verify_install(receipt: dict, checkout: Path) -> dict:
    import datus

    module = Path(datus.__file__).resolve()
    if module.is_relative_to((checkout / "datus").resolve()):
        raise ValueError(f"P0 imported the source checkout instead of the candidate wheel: {module}")
    distribution = metadata.distribution("datus-agent")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    wheels = {name: value for name, value in receipt["distributions"].items() if name.endswith(".whl")}
    origin = urlparse(direct.get("url", ""))
    wheel = Path(unquote(origin.path))
    if origin.scheme != "file" or wheel.name not in wheels or direct.get("dir_info", {}).get("editable"):
        raise ValueError("Installed datus-agent does not identify the validated candidate wheel")
    digest = sha256(wheel)
    if digest != wheels[wheel.name] or not module.is_relative_to(Path(distribution.locate_file("datus")).resolve()):
        raise ValueError("Installed datus-agent does not match the validated candidate wheel")
    # uv records local wheel URLs without an archive hash. Check the installed
    # runtime payload, including resources, against the verified wheel itself.
    with zipfile.ZipFile(wheel) as archive:
        payload = [name for name in archive.namelist() if name.startswith("datus/") and not name.endswith("/")]
        if not payload:
            raise ValueError("Candidate wheel contains no datus runtime payload")
        for name in payload:
            installed = Path(distribution.locate_file(name))
            if not installed.is_file() or installed.read_bytes() != archive.read(name):
                raise ValueError(f"Installed candidate payload differs: {name}")
    return {
        **receipt,
        "import_path": str(module),
        "installed_wheel_sha256": digest,
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create", "verify", "verify-install", "requirement"])
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = {"source_sha": args.source_sha, "distributions": distributions(args.dist)}
        (args.dist / "release-artifact.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = verify(args.dist, args.source_sha)
    if args.command == "requirement":
        wheel = next(name for name in result["distributions"] if name.endswith(".whl"))
        print(f"datus-agent @ {(args.dist / wheel).resolve().as_uri()}")
        return
    if args.command == "verify-install":
        if not args.checkout or not args.output:
            parser.error("verify-install requires --checkout and --output")
        result = verify_install(result, args.checkout)
        subprocess.run(["uv", "pip", "check", "--python", sys.executable], check=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            result if args.command != "verify-install" else {k: v for k, v in result.items() if k != "packages"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
