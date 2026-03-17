# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import MagicMock, patch

import pytest

import datus.utils.device_utils as device_utils_module
from datus.utils.device_utils import get_device


@pytest.fixture(autouse=True)
def reset_device_cache():
    """Reset the module-level _DEVICE cache before each test."""
    original = device_utils_module._DEVICE
    device_utils_module._DEVICE = None
    yield
    device_utils_module._DEVICE = original


class TestGetDevice:
    def test_returns_cached_device_on_second_call(self):
        device_utils_module._DEVICE = "cpu"
        result = get_device()
        assert result == "cpu"

    def test_macos_arm64_returns_mps(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            result = get_device()
        assert result == "mps"

    def test_macos_non_arm64_returns_cpu(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="x86_64"):
            result = get_device()
        assert result == "cpu"

    def test_linux_nvidia_returns_cuda(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("platform.system", return_value="Linux"), patch("subprocess.run", return_value=mock_result):
            result = get_device()
        assert result == "cuda"

    def test_linux_no_nvidia_no_amd_returns_cpu(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("platform.system", return_value="Linux"), patch("subprocess.run", return_value=mock_result):
            result = get_device()
        assert result == "cpu"

    def test_linux_nvidia_not_found_amd_returns_rocm(self):
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if "nvidia-smi" in cmd:
                raise FileNotFoundError("nvidia-smi not found")
            # rocm-smi
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch("platform.system", return_value="Linux"), patch("subprocess.run", side_effect=side_effect):
            result = get_device()
        assert result == "rocm"

    def test_linux_both_not_found_returns_cpu(self):
        with patch("platform.system", return_value="Linux"), patch(
            "subprocess.run", side_effect=FileNotFoundError("not found")
        ):
            result = get_device()
        assert result == "cpu"

    def test_linux_nvidia_exception_falls_through_to_cpu(self):
        with patch("platform.system", return_value="Linux"), patch(
            "subprocess.run", side_effect=Exception("unexpected error")
        ):
            result = get_device()
        assert result == "cpu"

    def test_windows_nvidia_returns_cuda(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("platform.system", return_value="Windows"), patch("subprocess.run", return_value=mock_result):
            result = get_device()
        assert result == "cuda"

    def test_windows_no_nvidia_amd_returns_rocm(self):
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if "nvidia-smi" in cmd:
                mock_result = MagicMock()
                mock_result.returncode = 1
                return mock_result
            # wmic call for AMD
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "AMD Radeon RX 6800"
            return mock_result

        with patch("platform.system", return_value="Windows"), patch("subprocess.run", side_effect=side_effect):
            result = get_device()
        assert result == "rocm"

    def test_windows_no_gpu_returns_cpu(self):
        def side_effect(cmd, **kwargs):
            mock_result = MagicMock()
            if "nvidia-smi" in cmd:
                mock_result.returncode = 1
            else:
                mock_result.returncode = 0
                mock_result.stdout = "Intel HD Graphics"
            return mock_result

        with patch("platform.system", return_value="Windows"), patch("subprocess.run", side_effect=side_effect):
            result = get_device()
        assert result == "cpu"

    def test_unknown_platform_returns_cpu(self):
        with patch("platform.system", return_value="FreeBSD"):
            result = get_device()
        assert result == "cpu"

    def test_caches_result(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            first = get_device()
            second = get_device()
        assert first == second == "mps"
        # _DEVICE should be set now
        assert device_utils_module._DEVICE == "mps"
