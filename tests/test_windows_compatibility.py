import multiprocessing
import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from datus.models.base import LLMBaseModel  # 根据你的实际导入路径调整
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.constants import DBType
from datus.utils.path_utils import get_files_from_glob_pattern


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


@pytest.mark.unit
def test_detect_toxicology_db(tmp_path):
    """
    专项测试是否能检测到 toxicology.sqlite 文件
    测试场景：
    - 在嵌套目录结构中存在目标文件
    - 使用递归 glob 模式 (**)
    - 验证返回的 URI 格式
    """
    # 1. 准备测试环境
    test_files = [
        "benchmark/bird/dev_20240627/dev_databases/medical/toxicology.sqlite",
        "benchmark/bird/dev_20240627/dev_databases/chemical/untested.sqlite",
        "benchmark/bird/dev_20240627/dev_databases/empty.sqlite",
    ]

    for file in test_files:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    # 2. 执行测试（使用实际业务参数）
    pattern = "benchmark/bird/dev_20240627/dev_databases/**/*.sqlite"
    full_pattern = str(tmp_path / pattern)
    results = get_files_from_glob_pattern(full_pattern, DBType.SQLITE)

    # 3. 验证结果
    toxicology_files = [r for r in results if r["name"] == "toxicology" and r["uri"].endswith("toxicology.sqlite")]

    assert len(toxicology_files) == 1, "应检测到1个toxicology数据库"

    # 4. 验证完整URI格式
    expected_uri = (
        f"{DBType.SQLITE}:///" f"{tmp_path}/benchmark/bird/dev_20240627/dev_databases/medical/toxicology.sqlite"
    ).replace(
        "\\", "/"
    )  # 统一路径分隔符

    assert toxicology_files[0]["uri"] == expected_uri
