import copy

import pytest

from datus.utils.ossie_format import upgrade_document


@pytest.mark.parametrize(
    "payload", ['{"nested":{"x":1,"x":2}}', '{"list":[NaN]}', '{"n":Infinity}', '{"n":1e999}', "[]"]
)
def test_legacy_vendor_payload_rejects_lossy_json(payload):
    model = {
        "version": "0.2.0.dev0",
        "datasets": [{"fields": [{"custom_extensions": [{"vendor_name": "other", "data": payload}]}]}],
    }
    with pytest.raises(ValueError):
        upgrade_document(model)


def test_upgrade_preserves_opaque_fields_and_scoped_metrics_on_repeat():
    opaque = {"custom_extensions": [{"data": "keep unchanged"}]}
    metric = {
        "name": "scoped",
        "expression": {"dialects": [{"dialect": "CUSTOM_DB", "expression": "vendor_fn()"}]},
        "ai_context": copy.deepcopy(opaque),
        "custom_extensions": [{"vendor_name": "OTHER", "data": '{"custom_extensions":[{"data":"keep unchanged"}]}'}],
    }
    model = {"version": "0.2.0.dev0", "datasets": [{"metrics": [metric]}]}
    assert upgrade_document(model)
    assert metric["nonportable_extensions"][0]["data"] == opaque
    assert metric["ai_context"] == opaque
    assert metric["expression"]["dialects"][0] == {"dialect": "CUSTOM_DB", "expression": "vendor_fn()"}
    before = copy.deepcopy(model)
    assert not upgrade_document(model)
    assert model == before


def test_conflicting_extension_names_rejected():
    model = {
        "version": "0.2.0.dev0",
        "datasets": [],
        "metrics": [{"custom_extensions": [], "nonportable_extensions": []}],
    }
    with pytest.raises(ValueError, match="conflict"):
        upgrade_document(model)
