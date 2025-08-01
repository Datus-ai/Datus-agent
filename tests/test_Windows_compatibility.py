import platform
import multiprocessing
from unittest.mock import patch
import pytest

from datus.models.base import LLMBaseModel  # 根据你的实际导入路径调整
from datus.storage.embedding_models import EmbeddingModel


# test datus.models.base
@pytest.mark.acceptance
@pytest.mark.parametrize(
    "platform_name, expected_method",
    [
        ("Windows", "spawn"),  # Windows 平台测试
        ("Linux", "fork"),  # 非Windows（Linux/macOS）测试
        ("Darwin", "fork"),  # macOS 测试
    ],
)
def test_multiprocessing_start_method_base(platform_name, expected_method):
    """
    参数化测试不同平台下的进程启动方法设置：
    - Windows 应使用 'spawn'
    - 非Windows 应使用 'fork'
    """
    with patch("platform.system", return_value=platform_name):
        with patch("multiprocessing.set_start_method") as mock_set:
            # 重新加载模块以触发代码执行
            import importlib
            import datus.models.base
            importlib.reload(datus.models.base)

            # 验证是否调用了正确的启动方法
            mock_set.assert_called_once_with(expected_method, force=True)


# test datus.storage.embedding_models
@pytest.mark.acceptance
@pytest.mark.parametrize(
    "platform_name, expected_method",
    [
        ("Windows", "spawn"),  # Windows 平台测试
        ("Linux", "fork"),  # 非Windows（Linux/macOS）测试
        ("Darwin", "fork"),  # macOS 测试
    ],
)
def test_multiprocessing_start_method_embedding(platform_name, expected_method):
    """
    参数化测试不同平台下的进程启动方法设置：
    - Windows 应使用 'spawn'
    - 非Windows 应使用 'fork'
    """
    with patch("platform.system", return_value=platform_name):
        with patch("multiprocessing.set_start_method") as mock_set:
            # 重新加载模块以触发代码执行
            import importlib
            import datus.storage.embedding_models
            importlib.reload(datus.storage.embedding_models)

            # 验证是否调用了正确的启动方法
            mock_set.assert_called_once_with(expected_method, force=True)