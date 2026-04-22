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
WEB_PORT = 18501
WEB_URL = f"http://localhost:{WEB_PORT}"


@pytest.fixture(scope="module")
def web_server():
    """Start the FastAPI web chatbot server, wait for ready, yield URL, then terminate."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "datus.cli.main",
            "--web",
            "--port",
            str(WEB_PORT),
            "--host",
            "localhost",
            "--config",
            str(PROJECT_ROOT / "tests" / "conf" / "agent.yml"),
            "--datasource",
            "ssb_sqlite",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until ready (max 30s)
    for _ in range(30):
        try:
            resp = requests.get(f"{WEB_URL}/health", timeout=2)
            if resp.status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("Web server did not start within 30 seconds")

    yield WEB_URL

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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
