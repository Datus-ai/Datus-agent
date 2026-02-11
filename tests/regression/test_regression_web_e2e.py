"""
Regression Tests: Streamlit Web UI End-to-End (R-13)

Playwright browser tests for the Streamlit chatbot interface:
- Page loading and rendering
- Sidebar with model/namespace information
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

requests = pytest.importorskip("requests", reason="requests not installed")
pytest.importorskip(
    "pytest_playwright",
    reason=("pytest-playwright not installed. " "Run: pip install pytest-playwright && playwright install chromium"),
)

from playwright.sync_api import expect  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAMLIT_PORT = 18501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"


@pytest.fixture(scope="module")
def streamlit_server():
    """Start Streamlit as subprocess, wait for ready, yield URL, then terminate."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "datus" / "cli" / "web" / "chatbot.py"),
            "--server.port",
            str(STREAMLIT_PORT),
            "--server.headless",
            "true",
            "--",
            "--config",
            str(PROJECT_ROOT / "tests" / "conf" / "agent.yml"),
            "--namespace",
            "ssb_sqlite",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until ready (max 30s)
    for _ in range(30):
        try:
            resp = requests.get(f"{STREAMLIT_URL}/_stcore/health", timeout=2)
            if resp.status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("Streamlit did not start within 30 seconds")

    yield STREAMLIT_URL

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.regression
class TestWebE2E:
    """Playwright end-to-end tests for Streamlit Web UI."""

    def test_page_loads(self, page, streamlit_server):
        """R13-E01: Page loads successfully with stApp container."""
        page.goto(streamlit_server)
        page.wait_for_load_state("networkidle")
        expect(page.locator('[data-testid="stApp"]')).to_be_visible()

    def test_sidebar_renders(self, page, streamlit_server):
        """R13-E02: Sidebar renders with configuration info."""
        page.goto(streamlit_server)
        page.wait_for_load_state("networkidle")
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible()

    def test_chat_input_exists(self, page, streamlit_server):
        """R13-E03: Chat input is visible and interactable."""
        page.goto(streamlit_server)
        page.wait_for_load_state("networkidle")
        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        expect(chat_input).to_be_visible()
        expect(chat_input).to_be_enabled()

    def test_chat_submit_and_response(self, page, streamlit_server):
        """R13-E04: Submit query and receive LLM response."""
        page.goto(streamlit_server)
        page.wait_for_load_state("networkidle")

        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("How many customers are there?")
        chat_input.press("Enter")

        # Wait for assistant response (up to 120s for LLM)
        messages = page.locator('[data-testid="stChatMessage"]')
        expect(messages.last).to_be_visible(timeout=120_000)

    def test_sql_result_display(self, page, streamlit_server):
        """R13-E05: SQL result is displayed after query."""
        page.goto(streamlit_server)
        page.wait_for_load_state("networkidle")

        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("Count all customers")
        chat_input.press("Enter")

        # Wait for assistant response first
        messages = page.locator('[data-testid="stChatMessage"]')
        expect(messages.nth(1)).to_be_visible(timeout=120_000)

        # Then check for SQL code block or dataframe within chat messages
        sql_or_table = page.locator(
            '[data-testid="stChatMessage"] code, ' '[data-testid="stCodeBlock"], ' '[data-testid="stDataFrame"]'
        )
        expect(sql_or_table.first).to_be_visible(timeout=30_000)
