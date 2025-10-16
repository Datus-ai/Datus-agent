# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import platform
import subprocess

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_DEVICE = None


def get_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    system = platform.system()

    if system == "Darwin":  # macOS
        if platform.machine() == "arm64":
            _DEVICE = "mps"  # Apple M1/M2/M3 chip support
            return _DEVICE
    elif system == "Linux":
        # Check for NVIDIA GPU
        try:
            gpu_info = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if gpu_info.returncode == 0:
                _DEVICE = "cuda"
                return _DEVICE
        except FileNotFoundError:
            logger.warn("nvidia-smi not found")
        except Exception as e:
            logger.warn(f"Error getting device: {e}")

        # Check for AMD GPU
        try:
            gpu_info = subprocess.run(["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if gpu_info.returncode == 0:
                _DEVICE = "rocm"
                return _DEVICE
        except FileNotFoundError:
            logger.warn("rocm-smi not found")
        except Exception as e:
            logger.warn(f"Error getting device: {e}")
    elif platform.system() == "Windows":
        # Check for NVIDIA GPU
        try:
            gpu_info = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if gpu_info.returncode == 0:
                _DEVICE = "cuda"
                return _DEVICE
        except Exception as e:
            logger.warn(f"Error getting device: {str(e)}")

        # Check for AMD GPU
        try:
            gpu_info = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if "AMD" in gpu_info.stdout:
                _DEVICE = "rocm"
                return _DEVICE
        except Exception as e:
            logger.warn(f"Error getting device: {str(e)}")

    _DEVICE = "cpu"
    return _DEVICE
