"""
Regression Tests: Web UI End-to-End (R-13)

Playwright browser tests for the web chatbot interface:
- Page loading and rendering
- Chat input interaction
- Query submission and LLM response
- SQL result display

Prerequisites:
    pip install playwright pytest-playwright
    playwright install chromium
    Set DATUS_TEST_CHATBOT_DIST to the published @datus/web-chatbot dist directory
    with react.production.min.js from react@18.3.1 added alongside the bundle.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.nightly_requirements import import_required
from tests.web_e2e_support import local_chatbot_dist

requests = import_required("requests", reason="requests not installed")
import_required(
    "pytest_playwright",
    reason=("pytest-playwright not installed. Run: pip install pytest-playwright && playwright install chromium"),
)
playwright_sync_api = import_required(
    "playwright.sync_api",
    reason="playwright not installed. Run: pip install pytest-playwright && playwright install chromium",
)

expect = playwright_sync_api.expect

pytestmark = [pytest.mark.regression, pytest.mark.nightly]

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _candidate_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return "<server log file was not created>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) > max_lines:
        lines = [f"... omitted {len(lines) - max_lines} earlier line(s) ...", *lines[-max_lines:]]
    return "\n".join(lines)


def _terminate_process(proc: subprocess.Popen, timeout: int = 5) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


@pytest.fixture(scope="module")
def web_artifacts(tmp_path_factory):
    configured = os.getenv("DATUS_WEB_E2E_ARTIFACT_DIR")
    path = Path(configured) if configured else tmp_path_factory.mktemp("web-e2e-artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def web_server(web_artifacts):
    """Start the FastAPI web chatbot server, wait for ready, yield URL, then terminate."""
    chatbot_dist = local_chatbot_dist()
    log_path = web_artifacts / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = None
    last_failure = "server was not started"

    try:
        for attempt in range(3):
            web_port = _candidate_local_port()
            web_url = f"http://127.0.0.1:{web_port}"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "datus.cli.main",
                    "--web",
                    "--chatbot-dist",
                    str(chatbot_dist),
                    "--port",
                    str(web_port),
                    "--host",
                    "127.0.0.1",
                    "--config",
                    str(PROJECT_ROOT / "tests" / "conf" / "agent.yml"),
                    "--datasource",
                    "ssb_sqlite",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

            for _ in range(30):
                if proc.poll() is not None:
                    last_failure = (
                        f"attempt {attempt + 1}: server exited before becoming ready "
                        f"with code {proc.returncode} at {web_url}"
                    )
                    break
                try:
                    resp = requests.get(f"{web_url}/health", timeout=2)
                    if resp.status_code == 200:
                        yield web_url
                        return
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(1)
            else:
                last_failure = f"attempt {attempt + 1}: server did not start within 30 seconds at {web_url}"

            _terminate_process(proc)
            proc = None

        pytest.fail(f"Failed to launch web server after retries: {last_failure}.\nServer log:\n{_tail(log_path)}")
    finally:
        if proc is not None:
            _terminate_process(proc, timeout=10)
        log_file.close()


@pytest.fixture
def web_page(page, web_server, web_artifacts, request):
    """Persist network, console, screenshot and trace evidence, including on reruns."""
    events = []
    # Browser smoke tests must render using local assets, without external services.
    page.route(
        "**/*", lambda route: route.continue_() if route.request.url.startswith(web_server + "/") else route.abort()
    )
    page.on("request", lambda req: events.append({"event": "request", "url": req.url}))
    page.on("response", lambda resp: events.append({"event": "response", "url": resp.url, "status": resp.status}))
    page.on(
        "requestfailed", lambda req: events.append({"event": "requestfailed", "url": req.url, "error": req.failure})
    )
    page.on("pageerror", lambda error: events.append({"event": "pageerror", "error": str(error)}))
    page.on("console", lambda msg: events.append({"event": "console", "type": msg.type, "text": msg.text}))
    stem = request.node.name
    page.context.tracing.start(screenshots=True, snapshots=True, sources=True)
    try:
        yield page
    finally:
        try:
            page.screenshot(path=str(web_artifacts / f"{stem}.png"), timeout=5_000)
        except playwright_sync_api.Error as exc:
            events.append({"event": "screenshot_error", "error": str(exc)})
        page.context.tracing.stop(path=str(web_artifacts / f"{stem}.zip"))
        (web_artifacts / f"{stem}.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")


@pytest.mark.regression
class TestWebE2E:
    """Playwright end-to-end tests for the web chatbot UI."""

    def test_page_loads(self, web_page, web_server):
        """R13-E01: Page loads successfully with chatbot root."""
        page = web_page
        page.goto(web_server, wait_until="domcontentloaded")
        expect(page.locator("#chatbot-root")).to_be_visible()
        expect(page.locator("#chatbot-root > *").first).to_be_visible(timeout=10_000)

    def test_chat_input_exists(self, web_page, web_server):
        """R13-E03: Chat input is visible and interactable."""
        page = web_page
        page.goto(web_server, wait_until="domcontentloaded")
        # The chatbot component should render an input area
        chat_input = page.locator("#chatbot-root textarea, #chatbot-root input[type='text']")
        expect(chat_input.first).to_be_visible(timeout=10_000)
        expect(chat_input.first).to_be_editable()
        chat_input.first.fill("Web UI smoke test")
        expect(chat_input.first).to_have_value("Web UI smoke test")
