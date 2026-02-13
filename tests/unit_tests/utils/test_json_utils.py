import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from datus.utils.json_utils import to_pretty_str, to_str


def test_to_str_serializes_pydantic_model():
    pydantic = pytest.importorskip("pydantic")

    class SampleModel(pydantic.BaseModel):
        created_at: datetime
        amount: Decimal
        tags: set[str]

    model = SampleModel(created_at=datetime(2025, 1, 1, 12, 30), amount=Decimal("12.34"), tags={"alpha", "beta"})
    payload = json.loads(to_str(model))
    assert payload["created_at"] == "2025-01-01T12:30:00"
    assert payload["amount"] == "12.34"
    assert set(payload["tags"]) == {"alpha", "beta"}


def test_to_str_serializes_dataclass_and_uuid():
    @dataclass
    class Example:
        name: str
        identifier: UUID
        location: Path

    instance = Example(name="demo", identifier=uuid4(), location=Path("/tmp/example"))
    payload = json.loads(to_str(instance))
    assert payload["name"] == "demo"
    assert payload["identifier"] == str(instance.identifier)
    assert payload["location"] == "/tmp/example"


def test_to_pretty_str_from_json_text_roundtrip():
    pretty = to_pretty_str('{"foo": 1, "bar": 2}')
    assert "\n" in pretty
    assert json.loads(pretty) == {"foo": 1, "bar": 2}


def test_to_str_returns_raw_for_non_json_bytes():
    assert to_str(b"plain text payload") == "plain text payload"


def test_to_str_coerces_mapping_keys_to_strings():
    payload = json.loads(to_str({1: "value", Path("home"): 2}))
    assert payload == {"1": "value", "home": 2}


def test_to_str_normalizes_pandas_and_numpy_objects():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame([{"value": Decimal("1.5"), "ts": datetime(2024, 1, 1, 8, 30)}])
    array = np.array([1, 2, 3])
    payload = json.loads(to_str({"frame": frame, "array": array}))

    assert payload["array"] == [1, 2, 3]
    assert payload["frame"][0]["value"] == "1.5"
    assert payload["frame"][0]["ts"] == "2024-01-01T08:30:00"
