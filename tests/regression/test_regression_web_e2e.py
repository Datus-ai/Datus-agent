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
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.nightly_requirements import import_required

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


def _free_local_port() -> int:
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
def web_server(tmp_path_factory):
    """Start the FastAPI web chatbot server, wait for ready, yield URL, then terminate."""
    log_dir = tmp_path_factory.mktemp("web-e2e-server")
    log_path = log_dir / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = None
    web_url = None

    for _attempt in range(3):
        web_port = _free_local_port()
        candidate_url = f"http://127.0.0.1:{web_port}"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "datus.cli.main",
                "--web",
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
        time.sleep(0.5)
        if proc.poll() is None:
            web_url = candidate_url
            break

    if proc is None or web_url is None:
        log_file.close()
        pytest.fail(f"Failed to launch web server after retries.\nServer log:\n{_tail(log_path)}")

    # Poll until ready (max 30s)
    try:
        for _ in range(30):
            if proc.poll() is not None:
                pytest.fail(
                    f"Web server exited before becoming ready with code {proc.returncode}.\n"
                    f"Server log:\n{_tail(log_path)}"
                )
            try:
                resp = requests.get(f"{web_url}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(1)
        else:
            _terminate_process(proc)
            pytest.fail(f"Web server did not start within 30 seconds at {web_url}.\nServer log:\n{_tail(log_path)}")

        yield web_url
    finally:
        _terminate_process(proc, timeout=10)
        log_file.close()


@pytest.mark.regression
class TestWebE2E:
    """Playwright end-to-end tests for the web chatbot UI."""

    def test_page_loads(self, page, web_server):
        """R13-E01: Page loads successfully with chatbot root."""
        page.goto(web_server)
        page.wait_for_load_state("networkidle")
        expect(page.locator("#chatbot-root")).to_be_visible()

    def test_chat_input_exists(self, page, web_server):
        """R13-E03: Chat input is visible and interactable."""
        page.goto(web_server)
        page.wait_for_load_state("networkidle")
        # The chatbot component should render an input area
        chat_input = page.locator("#chatbot-root textarea, #chatbot-root input[type='text']")
        expect(chat_input.first).to_be_visible(timeout=10_000)
