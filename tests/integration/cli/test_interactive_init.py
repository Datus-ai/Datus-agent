import os
import tempfile

import pytest
from agents import set_tracing_disabled

from datus.cli.interactive_init import InteractiveInit
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
set_tracing_disabled(True)

pytestmark = pytest.mark.nightly


class TestInitKimiConnectivity:
    """Integration test for kimi-k2.5 LLM connectivity during interactive init."""

    @pytest.mark.skipif(not os.getenv("KIMI_API_KEY"), reason="KIMI_API_KEY not available")
    def test_kimi_k25_connectivity_with_param_overrides(self):
        """Verify that interactive init sets temperature=1.0 and top_p=0.95 for kimi-k2.5,
        and the LLM connectivity test succeeds (would fail with default temperature=0.7)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            init.config["agent"]["target"] = "kimi"
            init.config["agent"]["models"]["kimi"] = {
                "type": "kimi",
                "base_url": "https://api.moonshot.cn/v1",
                "api_key": os.getenv("KIMI_API_KEY"),
                "model": "kimi-k2.5",
                "temperature": 1.0,
                "top_p": 0.95,
            }

            success, error_msg = init._test_llm_connectivity()

            assert success is True, (
                f"kimi-k2.5 connectivity should succeed with temperature=1.0, top_p=0.95: {error_msg}"
            )
            logger.info("kimi-k2.5 init connectivity test passed with correct param overrides")

    @pytest.mark.skipif(not os.getenv("KIMI_API_KEY"), reason="KIMI_API_KEY not available")
    def test_kimi_k25_fails_without_param_overrides(self):
        """Confirm that kimi-k2.5 rejects default temperature (0.7), proving the override is necessary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init = InteractiveInit(user_home=tmpdir)

            # Deliberately omit temperature/top_p to trigger the 400 error
            init.config["agent"]["target"] = "kimi"
            init.config["agent"]["models"]["kimi"] = {
                "type": "kimi",
                "base_url": "https://api.moonshot.cn/v1",
                "api_key": os.getenv("KIMI_API_KEY"),
                "model": "kimi-k2.5",
            }

            success, error_msg = init._test_llm_connectivity()

            assert success is False, "kimi-k2.5 should fail without temperature override"
            assert "temperature" in error_msg.lower() or "400" in error_msg, (
                f"Error should mention temperature or 400, got: {error_msg}"
            )
            logger.info(f"kimi-k2.5 correctly rejected default params: {error_msg}")
