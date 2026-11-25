"""Owned vision infrastructure fails closed and retains reproducible evidence."""

import hashlib
import io
import json
import socket
import struct
import subprocess
import threading
from urllib.error import URLError

import pytest

from ci import self_hosted_vision as vision

pytestmark = pytest.mark.component


@pytest.fixture(autouse=True)
def clear_endpoint(monkeypatch):
    for name in ("DATUS_SELF_HOSTED_BASE_URL", "DATUS_SELF_HOSTED_MODEL", "DATUS_SELF_HOSTED_API_KEY_ENV"):
        monkeypatch.delenv(name, raising=False)


def test_verified_cache_never_needs_network_and_corruption_fails(monkeypatch, tmp_path):
    data = b"verified fixture bytes"
    monkeypatch.setattr(vision, "FILES", {"fixture.gguf": hashlib.sha256(data).hexdigest()})
    (tmp_path / "fixture.gguf").write_bytes(data)
    monkeypatch.setattr(vision, "urlopen", lambda *a, **k: pytest.fail("No download for verified cache"))
    evidence = vision.model_files(tmp_path)
    assert evidence[0]["bytes"] == len(data)
    assert vision.REVISION in evidence[0]["url"]
    (tmp_path / "fixture.gguf").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached model checksum mismatch"):
        vision.model_files(tmp_path)


def test_bad_download_is_not_promoted_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "FILES", {"fixture.gguf": "0" * 64})
    monkeypatch.setattr(vision, "urlopen", lambda *a, **k: io.BytesIO(b"wrong bytes"))
    with pytest.raises(ValueError, match="Model checksum mismatch"):
        vision.model_files(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def reset_then_ready(monkeypatch):
    """Consume HTTP headers, send a real TCP RST, then serve a healthy reply."""
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    listener.settimeout(0.1)
    stop = threading.Event()
    requests, errors = [], []

    def serve():
        try:
            while not stop.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(2)
                    headers = b""
                    while b"\r\n\r\n" not in headers:
                        chunk = connection.recv(4096)
                        if not chunk:
                            raise RuntimeError("Health client closed before sending headers")
                        headers += chunk
                    requests.append(headers.split(b"\r\n", 1)[0])
                    if len(requests) == 1:
                        # Abort after receiving the request, rather than producing
                        # the graceful FIN that HTTPServer.shutdown would send.
                        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    else:
                        body = b'{"status":"ok"}'
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: "
                            + str(len(body)).encode()
                            + b"\r\nConnection: close\r\n\r\n"
                            + body
                        )
        except OSError as error:
            if not stop.is_set():
                errors.append(error)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield {"base": f"http://127.0.0.1:{listener.getsockname()[1]}", "requests": requests}
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=3)
        assert not thread.is_alive(), "Owned health server must stop"
        assert errors == []


def test_health_reset_is_not_ready_and_next_http_reply_is_ready(reset_then_ready):
    assert vision.healthy(reset_then_ready["base"]) is False
    assert vision.healthy(reset_then_ready["base"]) is True
    assert reset_then_ready["requests"] == [b"GET /health HTTP/1.1"] * 2


@pytest.fixture
def server(monkeypatch):
    state = {"calls": [], "owner": None, "running": True, "healthy": vision.healthy}

    def docker(*args):
        state["calls"].append(args)
        if args[:2] == ("image", "inspect"):
            return json.dumps([{"Id": "sha256:actual", "Architecture": "arm64", "RepoDigests": [vision.IMAGE]}])
        if args[0] == "create":
            state["owner"] = args[args.index("--label") + 1].split("=", 1)[1]
            return "owned-container"
        if args[0] == "port":
            return "127.0.0.1:45678"
        if args[0] == "inspect":
            return json.dumps(
                [{"State": {"Running": state["running"]}, "Config": {"Labels": {vision.LABEL: state["owner"]}}}]
            )
        return ""

    monkeypatch.setattr(vision, "docker", docker)
    monkeypatch.setattr(vision, "model_files", lambda _: [{"sha256": "verified"}])
    monkeypatch.setattr(vision, "healthy", lambda _: True)
    monkeypatch.setattr(
        vision.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="real server logs")
    )
    return state


def test_real_server_command_resources_evidence_and_owned_cleanup(server, tmp_path):
    work = tmp_path / "work"
    receipt = vision.prepare(work, tmp_path / "cache", memory="8g", cpus=2)
    create = next(call for call in server["calls"] if call[0] == "create")
    assert create[create.index("--publish") + 1] == "127.0.0.1::8080"
    assert create[create.index("--memory") + 1] == "8g"
    assert create[create.index("--cpus") + 1] == "2"
    assert vision.IMAGE in create and "@sha256:" in vision.IMAGE
    assert "--jinja" in create and "--no-mmproj-offload" in create
    assert create[create.index("--n-gpu-layers") + 1] == "0"
    assert "--privileged" not in create and "--sysctl" not in create
    assert receipt["image_identity"]["Architecture"] == "arm64"
    assert receipt["status"] == "ready"
    assert "http://127.0.0.1:45678/v1" in (work / "environment.sh").read_text()
    assert (work / "environment.sh").stat().st_mode & 0o777 == 0o600
    vision.cleanup(work)
    assert ("rm", "--force", "owned-container") in server["calls"]
    assert json.loads((work / "evidence.json").read_text())["cleaned_up"] is True
    assert (work / "server.log").read_text() == "real server logs"
    count = len(server["calls"])
    vision.cleanup(work)
    assert len(server["calls"]) == count


def test_cleanup_refuses_foreign_container(server, tmp_path):
    work = tmp_path / "work"
    vision.prepare(work, tmp_path / "cache")
    server["owner"] = "another-run"
    with pytest.raises(ValueError, match="owned by another run"):
        vision.cleanup(work)
    assert not any(call[0] == "rm" for call in server["calls"])


@pytest.mark.parametrize(
    "log_failure",
    [subprocess.CalledProcessError(1, ["docker", "logs"]), subprocess.TimeoutExpired(["docker", "logs"], 30)],
)
def test_log_collection_failure_still_records_removal_for_a_later_cleanup(server, monkeypatch, tmp_path, log_failure):
    work = tmp_path / "work"
    vision.prepare(work, tmp_path / "cache")

    def failing_logs(*args, **kwargs):
        raise log_failure

    monkeypatch.setattr(vision.subprocess, "run", failing_logs)
    with pytest.raises(type(log_failure)):
        vision.cleanup(work)
    assert ("rm", "--force", "owned-container") in server["calls"]
    evidence = json.loads((work / "evidence.json").read_text())
    assert evidence["cleaned_up"] is True and evidence["log_collection_error"] == type(log_failure).__name__
    count = len(server["calls"])
    vision.cleanup(work)
    assert len(server["calls"]) == count


def test_failed_removal_leaves_cleanup_pending(server, monkeypatch, tmp_path):
    work = tmp_path / "work"
    vision.prepare(work, tmp_path / "cache")
    real_docker = vision.docker

    def docker(*args):
        if args[0] == "rm":
            raise subprocess.CalledProcessError(1, ["docker", *args])
        return real_docker(*args)

    monkeypatch.setattr(vision, "docker", docker)
    with pytest.raises(subprocess.CalledProcessError):
        vision.cleanup(work)
    assert "cleaned_up" not in json.loads((work / "evidence.json").read_text())


def test_readiness_timeout_is_failure_and_cleans_owned_container(server, monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "healthy", lambda _: False)
    ticks = iter([0, 100])
    monkeypatch.setattr(vision.time, "monotonic", lambda: next(ticks))
    work = tmp_path / "work"
    with pytest.raises(TimeoutError, match="did not become ready"):
        vision.prepare(work, tmp_path / "cache", ready_timeout=1)
    evidence = json.loads((work / "evidence.json").read_text())
    assert evidence["status"] == "failure" and evidence["cleaned_up"] is True
    assert not (work / "environment.sh").exists()


def test_external_endpoint_never_creates_or_removes_docker(monkeypatch, tmp_path):
    monkeypatch.setenv("DATUS_SELF_HOSTED_BASE_URL", "https://vision.example/v1")
    monkeypatch.setenv("DATUS_SELF_HOSTED_MODEL", "product-model")
    monkeypatch.setenv("DATUS_SELF_HOSTED_API_KEY_ENV", "VISION_TEST_API_KEY")
    monkeypatch.setenv("VISION_TEST_API_KEY", "secret-not-in-artifacts")
    monkeypatch.setattr(vision, "docker", lambda *a: pytest.fail("External endpoint is unmanaged"))
    result = vision.prepare(tmp_path, tmp_path / "cache")
    vision.cleanup(tmp_path)
    assert result["mode"] == "external"
    assert "secret-not-in-artifacts" not in (tmp_path / "environment.sh").read_text()
    assert "secret-not-in-artifacts" not in (tmp_path / "evidence.json").read_text()


def test_partial_external_configuration_fails_instead_of_skip_or_substitution(monkeypatch, tmp_path):
    monkeypatch.setenv("DATUS_SELF_HOSTED_MODEL", "product-model")
    monkeypatch.setattr(vision, "docker", lambda *a: pytest.fail("Invalid config must fail first"))
    with pytest.raises(ValueError, match="requires both"):
        vision.prepare(tmp_path, tmp_path / "cache")
    assert json.loads((tmp_path / "evidence.json").read_text())["status"] == "failure"


def test_external_missing_key_fails_before_test_can_skip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATUS_SELF_HOSTED_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("DATUS_SELF_HOSTED_MODEL", "configured-model")
    monkeypatch.setenv("DATUS_SELF_HOSTED_API_KEY_ENV", "MISSING_VISION_KEY")
    monkeypatch.delenv("MISSING_VISION_KEY", raising=False)
    with pytest.raises(ValueError, match="populated API key"):
        vision.prepare(tmp_path, tmp_path / "cache")
    assert not (tmp_path / "environment.sh").exists()


def _use_real_health_endpoint(server, reset_then_ready, monkeypatch):
    docker = vision.docker

    def redirected_docker(*args):
        if args[0] == "port":
            return reset_then_ready["base"].removeprefix("http://")
        return docker(*args)

    monkeypatch.setattr(vision, "docker", redirected_docker)
    monkeypatch.setattr(vision, "healthy", server["healthy"])


def test_prepare_recovers_from_health_reset_within_existing_polling_loop(
    server, reset_then_ready, monkeypatch, tmp_path
):
    _use_real_health_endpoint(server, reset_then_ready, monkeypatch)
    work = tmp_path / "work"
    receipt = vision.prepare(work, tmp_path / "cache", ready_timeout=5)
    assert receipt["status"] == "ready"
    assert reset_then_ready["requests"] == [b"GET /health HTTP/1.1"] * 2
    assert not any(call[0] == "rm" for call in server["calls"])
    vision.cleanup(work)
    assert json.loads((work / "evidence.json").read_text())["cleaned_up"] is True


def test_prepare_does_not_retry_reset_after_container_stops(server, reset_then_ready, monkeypatch, tmp_path):
    _use_real_health_endpoint(server, reset_then_ready, monkeypatch)
    server["running"] = False
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="container stopped before readiness"):
        vision.prepare(work, tmp_path / "cache", ready_timeout=5)
    assert reset_then_ready["requests"] == [b"GET /health HTTP/1.1"]
    evidence = json.loads((work / "evidence.json").read_text())
    assert evidence["status"] == "failure" and evidence["cleaned_up"] is True
    assert not (work / "environment.sh").exists()


@pytest.mark.parametrize("error", [URLError("unavailable"), TimeoutError("timeout"), ValueError("bad JSON")])
def test_existing_health_errors_remain_not_ready(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(vision, "urlopen", unavailable)
    assert vision.healthy("http://vision.invalid") is False


def test_health_probe_does_not_swallow_programming_errors(monkeypatch):
    def invalid_call(*args, **kwargs):
        raise TypeError("incorrect health implementation")

    monkeypatch.setattr(vision, "urlopen", invalid_call)
    with pytest.raises(TypeError, match="incorrect health implementation"):
        vision.healthy("http://vision.invalid")
