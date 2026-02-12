from tests.regression.test_regression_llm import PROVIDER_MODELS


def pytest_addoption(parser):
    parser.addoption(
        "--single-model",
        action="store_true",
        default=False,
        help="Only test the first model per provider (skip the rest)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--single-model"):
        return

    # Collect model names to skip (all but the first per provider)
    skip_models = set()
    for spec in PROVIDER_MODELS.values():
        skip_models.update(spec["models"][1:])

    selected = []
    deselected = []
    for item in items:
        if any(f"-{model}]" in item.nodeid for model in skip_models):
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
