"""Provision an owned CPU vision server; retain pinned, verified model caches.

prepare writes environment.sh for the caller to source and evidence.json for
artifacts. cleanup removes only the container bearing this run's ownership label.
An externally configured endpoint is retained without Docker provisioning.

Pins originate from the official ggml-org Hugging Face file SHA256 metadata and
the official llama.cpp CPU server multi-platform manifest (amd64 and arm64).
The upstream Qwen/Qwen3-VL-2B-Instruct model is Apache-2.0 licensed. Resource
defaults are test budgets, not a claim about the model's minimum requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPOSITORY = "ggml-org/Qwen3-VL-2B-Instruct-GGUF"
REVISION = "ea6a11058182570be6436b9a2e4ee7f7b49f908d"
IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:c2cf6c72d913b6c89a5f0929f2a866e6161e09e7dc187e0f3b0f21e923e26db3"
FILES = {
    "Qwen3-VL-2B-Instruct-Q8_0.gguf": "b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39",
    "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf": "69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b",
}
ALIAS = "nightly-qwen3-vl-2b-instruct-q8"
LABEL = "ai.datus.nightly-vision-owner"


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=300).stdout.strip()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_files(cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    evidence = []
    for name, expected in FILES.items():
        target = cache / name
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        if not target.exists():
            temporary = cache / f".{name}.{uuid.uuid4().hex}.partial"
            try:
                with urlopen(url, timeout=120) as source, temporary.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                if digest(temporary) != expected:
                    raise ValueError(f"Model checksum mismatch: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if digest(target) != expected:
            raise ValueError(f"Cached model checksum mismatch: {name}; remove or repair that cache file")
        evidence.append({"name": name, "sha256": expected, "bytes": target.stat().st_size, "url": url})
    return evidence


def save(work: Path, receipt: dict) -> None:
    (work / "evidence.json").write_text(json.dumps(receipt, indent=2) + "\n")


def cleanup(work: Path) -> None:
    receipt = json.loads((work / "evidence.json").read_text())
    if receipt.get("mode") != "docker" or receipt.get("cleaned_up"):
        return
    container = receipt.get("container_id")
    if container:
        inspected = json.loads(docker("inspect", container))[0]
        if inspected.get("Config", {}).get("Labels", {}).get(LABEL) != receipt["owner"]:
            raise ValueError("Refusing to remove a vision container owned by another run")
        log_error = None
        try:
            logs = subprocess.run(
                ["docker", "logs", container],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            ).stdout
            (work / "server.log").write_text(logs)
        except (subprocess.SubprocessError, OSError) as error:
            log_error = error
        docker("rm", "--force", container)
        if log_error is not None:
            # The container is gone; record that before reporting the lost logs so
            # a later cleanup does not try to inspect a removed container.
            receipt["log_collection_error"] = type(log_error).__name__
            receipt["cleaned_up"] = True
            receipt["cleanup_time"] = time.time()
            save(work, receipt)
            raise log_error
    receipt["cleaned_up"] = True
    receipt["cleanup_time"] = time.time()
    save(work, receipt)


def healthy(base_url: str) -> bool:
    try:
        with urlopen(base_url + "/health", timeout=5) as response:
            return response.status == 200 and json.load(response).get("status") == "ok"
    except (URLError, TimeoutError, ConnectionError, ValueError):
        return False


def environment(work: Path, values: dict[str, str]) -> None:
    target = work / "environment.sh"
    target.touch(mode=0o600)
    target.write_text("".join(f"export {name}={shlex.quote(value)}\n" for name, value in values.items()))


def prepare(work: Path, cache: Path, *, memory: str = "8g", cpus: int = 4, ready_timeout: float = 600) -> dict:
    if not re.fullmatch(r"[1-9][0-9]*[mg]", memory) or cpus < 1 or ready_timeout <= 0:
        raise ValueError("Require positive memory (m/g), CPU count and readiness timeout")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(work.iterdir()):
        raise ValueError("Use a fresh vision work directory")
    receipt = {"status": "failure", "started_at": time.time()}
    save(work, receipt)
    try:
        endpoint = os.environ.get("DATUS_SELF_HOSTED_BASE_URL", "")
        model = os.environ.get("DATUS_SELF_HOSTED_MODEL", "")
        if endpoint or model:
            if not endpoint or not model:
                raise ValueError("External vision requires both DATUS_SELF_HOSTED_BASE_URL and DATUS_SELF_HOSTED_MODEL")
            key_name = os.environ.get("DATUS_SELF_HOSTED_API_KEY_ENV", "LLM_API_KEY")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name) or not os.environ.get(key_name):
                raise ValueError("External vision requires a populated API key environment variable")
            receipt.update(mode="external", model=model, api_key_env=key_name, status="ready")
            environment(
                work,
                {
                    "DATUS_SELF_HOSTED_BASE_URL": endpoint,
                    "DATUS_SELF_HOSTED_MODEL": model,
                    "DATUS_SELF_HOSTED_API_KEY_ENV": key_name,
                },
            )
        else:
            owner = uuid.uuid4().hex
            receipt.update(
                mode="docker",
                owner=owner,
                image=IMAGE,
                model=ALIAS,
                repository=REPOSITORY,
                revision=REVISION,
                model_license="Apache-2.0",
                upstream_model="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
                memory=memory,
                cpus=cpus,
                context_tokens=8192,
                model_files=model_files(cache),
            )
            save(work, receipt)
            docker("pull", IMAGE)
            image = json.loads(docker("image", "inspect", IMAGE))[0]
            receipt["image_identity"] = {key: image.get(key) for key in ("Id", "RepoDigests", "Architecture", "Os")}
            receipt["image_identity"]["labels"] = image.get("Config", {}).get("Labels")
            # Never request a GPU or host-global settings; publish only on loopback.
            container = docker(
                "create",
                "--name",
                "datus-nightly-vision-" + owner,
                "--label",
                LABEL + "=" + owner,
                "--memory",
                memory,
                "--cpus",
                str(cpus),
                "--publish",
                "127.0.0.1::8080",
                "--mount",
                f"type=bind,src={cache.resolve()},dst=/models,readonly",
                IMAGE,
                "-m",
                "/models/" + next(iter(FILES)),
                "--mmproj",
                "/models/" + list(FILES)[1],
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--alias",
                ALIAS,
                "--jinja",
                "--n-gpu-layers",
                "0",
                "--no-mmproj-offload",
                "--ctx-size",
                "8192",
                "--parallel",
                "1",
                "--threads",
                str(cpus),
            )
            receipt["container_id"] = container
            save(work, receipt)
            docker("start", container)
            binding = docker("port", container, "8080/tcp")
            if not re.fullmatch(r"127\.0\.0\.1:[0-9]+", binding):
                raise ValueError("Unexpected Docker port binding")
            base = "http://" + binding
            receipt["base_url"] = base + "/v1"
            deadline = time.monotonic() + ready_timeout
            while not healthy(base):
                state = json.loads(docker("inspect", container))[0]["State"]
                if not state["Running"]:
                    raise RuntimeError(f"Vision container stopped before readiness: {state}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Real vision server did not become ready before deadline")
                time.sleep(2)
            environment(
                work,
                {
                    "DATUS_SELF_HOSTED_BASE_URL": base + "/v1",
                    "DATUS_SELF_HOSTED_MODEL": ALIAS,
                    "DATUS_SELF_HOSTED_API_KEY_ENV": "DATUS_NIGHTLY_LOCAL_VISION_KEY",
                    "DATUS_NIGHTLY_LOCAL_VISION_KEY": "local-loopback-test-key",
                },
            )
            receipt.update(status="ready", ready_at=time.time())
        save(work, receipt)
        return receipt
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        save(work, receipt)
        cleanup(work)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cleanup"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--ready-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.action == "cleanup":
        cleanup(args.work)
    else:
        if args.cache is None:
            parser.error("prepare requires --cache")
        prepare(args.work, args.cache, memory=args.memory, cpus=args.cpus, ready_timeout=args.ready_timeout)


if __name__ == "__main__":
    main()
