import pytest

from datus.storage.metric.store import qualify_name
from datus.tools.metric_tools.search_metric import SearchMetricsTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.fixture
def search_metrics_tool() -> SearchMetricsTool:
    return SearchMetricsTool()


def test_qualify_name_with_none_or_blank():
    first = None
    second = ""
    three = "23456"
    full_name = qualify_name([first, second, three])
    assert "%.%.23456" == full_name
