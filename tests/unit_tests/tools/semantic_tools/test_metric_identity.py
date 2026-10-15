# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import copy
import json
import re

import dosi_engine
import pytest

from datus.tools.semantic_tools.metric_identity import ensure_metric_uid, metric_uid, prepare_metric_uids

UID = "tz4a98xxat96iws9zmbrgj3a"


def _metric(uid=UID):
    return {"name": "revenue", "custom_extensions": [{"vendor_name": "DATUS", "data": json.dumps({"uid": uid})}]}


def test_generated_identity_is_stable_and_preserves_extension_metadata():
    metric = {
        "name": "revenue",
        "custom_extensions": [
            {"vendor_name": "OTHER", "data": "opaque metadata"},
            {"vendor_name": "DATUS", "data": '{"v":"1.2","subject_path":["sales"],"unit":"USD"}'},
        ],
    }
    assert ensure_metric_uid(metric)
    assert re.fullmatch(r"[a-z][a-z0-9]{23}", metric_uid(metric))
    payload = json.loads(metric["custom_extensions"][1]["data"])
    assert {key: value for key, value in payload.items() if key != "uid"} == {
        "v": "1.2",
        "subject_path": ["sales"],
        "unit": "USD",
    }
    assert metric["custom_extensions"][0] == {"vendor_name": "OTHER", "data": "opaque metadata"}
    original = copy.deepcopy(metric)
    assert not ensure_metric_uid(metric)
    assert metric == original


@pytest.mark.parametrize("incoming", [None, "", "different", "az4a98xxat96iws9zmbrgj3a"])
def test_existing_identity_wins_over_model_output(incoming):
    candidate = _metric(incoming)
    assert ensure_metric_uid(candidate, _metric())
    assert metric_uid(candidate) == UID


@pytest.mark.parametrize("uid", [123, "a" * 23, "a" * 25, "1" + "a" * 23, "A" + "a" * 23, "a" * 23 + "-"])
def test_invalid_identity_is_rejected(uid):
    with pytest.raises(ValueError, match="24-character CUID2"):
        ensure_metric_uid(_metric(uid))


@pytest.mark.parametrize("data", ["{bad", "[]", "null"])
def test_malformed_extension_is_not_replaced(data):
    candidate = {"name": "revenue", "custom_extensions": [{"vendor_name": "DATUS", "data": data}]}
    original = copy.deepcopy(candidate)
    with pytest.raises(ValueError, match="DATUS extension"):
        ensure_metric_uid(candidate)
    assert candidate == original


def test_multiple_datus_extensions_are_rejected():
    candidate = _metric()
    candidate["custom_extensions"].append({"vendor_name": "datus", "data": "{}"})
    with pytest.raises(ValueError, match="single DATUS"):
        ensure_metric_uid(candidate)


def test_renaming_keeps_identity_and_copying_requires_a_new_identity():
    renamed = _metric()
    renamed["name"] = "net_revenue"
    assert not ensure_metric_uid(renamed)
    document = {"semantic_model": [{"metrics": [_metric(), renamed]}]}
    with pytest.raises(ValueError, match="Duplicate metric uid"):
        prepare_metric_uids(document)
    renamed.pop("custom_extensions")
    assert prepare_metric_uids(document) == ["net_revenue"]
    assert metric_uid(renamed) != UID


@pytest.mark.parametrize("names", [("revenue", "REVENUE"), ("Revenue", " revenue "), ("revenue", "revenue")])
def test_conflicting_names_are_rejected_before_assigning_uids(names):
    document = {"semantic_model": [{"metrics": [{"name": name} for name in names]}]}
    original = copy.deepcopy(document)
    with pytest.raises(ValueError, match="Metric name conflict"):
        prepare_metric_uids(document)
    assert document == original


def test_name_uniqueness_matches_kb_lowercase_rules():
    metrics = [{"name": "straße"}, {"name": "strasse"}]
    document = {"semantic_model": [{"metrics": metrics}]}
    assert prepare_metric_uids(document) == ["straße", "strasse"]
    assert len({metric_uid(metric) for metric in metrics}) == 2


def test_case_only_rename_preserves_existing_uid_when_omitted():
    document = {"semantic_model": [{"metrics": [{"name": "REVENUE"}]}]}
    previous = {"semantic_model": [{"metrics": [_metric()]}]}
    assert prepare_metric_uids(document, previous) == ["REVENUE"]
    assert metric_uid(document["semantic_model"][0]["metrics"][0]) == UID


@pytest.mark.parametrize("ambiguous_side", ["previous", "candidate"])
def test_ambiguous_rename_cannot_reuse_prior_identity(ambiguous_side):
    previous_metrics = [_metric()]
    candidates = [{**_metric(), "name": "net_revenue"}]
    if ambiguous_side == "previous":
        previous_metrics.append({**_metric(), "name": "gross_revenue"})
    else:
        candidates.append({**_metric(), "name": "gross_revenue"})
    previous = {"semantic_model": [{"metrics": previous_metrics}]}
    document = {"semantic_model": [{"metrics": candidates}]}

    assert prepare_metric_uids(document, previous) == [metric["name"] for metric in candidates]
    assert [metric_uid(metric) for metric in candidates] == [""] * len(candidates)
    assert prepare_metric_uids(document) == [metric["name"] for metric in candidates]
    assert len({metric_uid(metric) for metric in candidates}) == len(candidates)
    assert UID not in {metric_uid(metric) for metric in candidates}


def test_native_dosi_accepts_host_identity_metadata():
    metric = {
        "name": "revenue",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.amount)"}]},
    }
    ensure_metric_uid(metric)
    document = {
        "version": "0.2.0.dev0",
        "semantic_model": [
            {
                "name": "sales",
                "datasets": [
                    {
                        "name": "orders",
                        "source": "orders",
                        "fields": [
                            {
                                "name": "amount",
                                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "amount"}]},
                            }
                        ],
                    }
                ],
                "metrics": [metric],
            }
        ],
    }
    result = dosi_engine.validate(json.dumps(document))
    assert result["valid"], result
