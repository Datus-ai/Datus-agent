# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/semantic_tools/models.py — Phase 1 data model enhancements."""

from datus.tools.semantic_tools.models import DimensionInfo, SemanticModelInfo


class TestDimensionInfoTypeField:
    """Tests for the new `type` and `is_primary_key` fields on DimensionInfo."""

    def test_basic_construction_unchanged(self):
        """Existing code that only passes name/description should still work."""
        dim = DimensionInfo(name="region", description="Sales region")
        assert dim.name == "region"
        assert dim.description == "Sales region"
        assert dim.type is None
        assert dim.is_primary_key is None

    def test_name_only(self):
        """Minimal construction with just name should work."""
        dim = DimensionInfo(name="status")
        assert dim.name == "status"
        assert dim.description is None
        assert dim.type is None

    def test_with_type_string(self):
        dim = DimensionInfo(name="status", type="string")
        assert dim.type == "string"

    def test_with_type_number(self):
        dim = DimensionInfo(name="amount", type="number")
        assert dim.type == "number"

    def test_with_type_time(self):
        dim = DimensionInfo(name="created_at", type="time")
        assert dim.type == "time"

    def test_with_type_boolean(self):
        dim = DimensionInfo(name="is_active", type="boolean")
        assert dim.type == "boolean"

    def test_with_type_categorical(self):
        """dbt uses 'categorical' as dimension type."""
        dim = DimensionInfo(name="segment", type="categorical")
        assert dim.type == "categorical"

    def test_with_type_geo(self):
        """Cube supports 'geo' dimension type."""
        dim = DimensionInfo(name="location", type="geo")
        assert dim.type == "geo"

    def test_is_primary_key_true(self):
        dim = DimensionInfo(name="id", type="number", is_primary_key=True)
        assert dim.is_primary_key is True

    def test_is_primary_key_false(self):
        dim = DimensionInfo(name="name", is_primary_key=False)
        assert dim.is_primary_key is False

    def test_all_fields_together(self):
        dim = DimensionInfo(
            name="order_id",
            description="Unique order identifier",
            type="number",
            is_primary_key=True,
        )
        assert dim.name == "order_id"
        assert dim.description == "Unique order identifier"
        assert dim.type == "number"
        assert dim.is_primary_key is True

    def test_serialization_roundtrip(self):
        """Serialize to dict and back should preserve all fields."""
        dim = DimensionInfo(name="created_at", description="Creation time", type="time", is_primary_key=False)
        data = dim.model_dump()
        restored = DimensionInfo(**data)
        assert restored == dim

    def test_serialization_omits_none_by_default(self):
        """When new fields are None, they should be present in model_dump but optional."""
        dim = DimensionInfo(name="status")
        data = dim.model_dump()
        assert "type" in data
        assert data["type"] is None
        assert "is_primary_key" in data
        assert data["is_primary_key"] is None

    def test_json_roundtrip(self):
        dim = DimensionInfo(name="region", type="string", is_primary_key=False)
        json_str = dim.model_dump_json()
        restored = DimensionInfo.model_validate_json(json_str)
        assert restored == dim

    def test_backward_compat_from_dict_without_new_fields(self):
        """Constructing from a dict that lacks type/is_primary_key should still work."""
        data = {"name": "old_dimension", "description": "From old code"}
        dim = DimensionInfo(**data)
        assert dim.name == "old_dimension"
        assert dim.type is None
        assert dim.is_primary_key is None


class TestSemanticModelInfo:
    """Tests for the new SemanticModelInfo typed model."""

    def test_minimal_construction(self):
        model = SemanticModelInfo(name="orders")
        assert model.name == "orders"
        assert model.description is None
        assert model.platform_type is None
        assert model.dimensions == []
        assert model.measures == []
        assert model.extra == {}

    def test_full_construction(self):
        dims = [
            DimensionInfo(name="status", type="string"),
            DimensionInfo(name="created_at", type="time", is_primary_key=False),
        ]
        model = SemanticModelInfo(
            name="orders",
            description="Order data cube",
            platform_type="cube",
            dimensions=dims,
            measures=["orders.count", "orders.total_revenue"],
            extra={"connectedComponent": 1, "segments": ["active_orders"]},
        )
        assert model.name == "orders"
        assert model.description == "Order data cube"
        assert model.platform_type == "cube"
        assert len(model.dimensions) == 2
        assert model.dimensions[0].name == "status"
        assert model.dimensions[0].type == "string"
        assert model.measures == ["orders.count", "orders.total_revenue"]
        assert model.extra["connectedComponent"] == 1

    def test_platform_type_cube(self):
        model = SemanticModelInfo(name="orders", platform_type="cube")
        assert model.platform_type == "cube"

    def test_platform_type_view(self):
        model = SemanticModelInfo(name="order_view", platform_type="view")
        assert model.platform_type == "view"

    def test_platform_type_explore(self):
        """Looker uses 'explore' as the model unit."""
        model = SemanticModelInfo(name="order_analysis", platform_type="explore")
        assert model.platform_type == "explore"

    def test_platform_type_semantic_model(self):
        """dbt uses 'semantic_model'."""
        model = SemanticModelInfo(name="fct_orders", platform_type="semantic_model")
        assert model.platform_type == "semantic_model"

    def test_extra_dict_stores_platform_specific_data(self):
        """extra should hold arbitrary platform-specific metadata."""
        model = SemanticModelInfo(
            name="orders",
            extra={
                "joins": [{"name": "customers", "relationship": "many_to_one"}],
                "segments": [{"name": "active", "sql": "status = 'active'"}],
                "connectedComponent": 1,
            },
        )
        assert len(model.extra["joins"]) == 1
        assert model.extra["joins"][0]["relationship"] == "many_to_one"
        assert model.extra["connectedComponent"] == 1

    def test_dimensions_with_typed_dimension_info(self):
        """Dimensions should be List[DimensionInfo] with type info."""
        model = SemanticModelInfo(
            name="orders",
            dimensions=[
                DimensionInfo(name="id", type="number", is_primary_key=True),
                DimensionInfo(name="status", type="string"),
                DimensionInfo(name="created_at", type="time"),
            ],
        )
        assert len(model.dimensions) == 3
        pk_dims = [d for d in model.dimensions if d.is_primary_key]
        assert len(pk_dims) == 1
        assert pk_dims[0].name == "id"
        time_dims = [d for d in model.dimensions if d.type == "time"]
        assert len(time_dims) == 1

    def test_serialization_roundtrip(self):
        model = SemanticModelInfo(
            name="orders",
            description="Order cube",
            platform_type="cube",
            dimensions=[DimensionInfo(name="status", type="string")],
            measures=["orders.count"],
            extra={"key": "value"},
        )
        data = model.model_dump()
        restored = SemanticModelInfo(**data)
        assert restored == model

    def test_json_roundtrip(self):
        model = SemanticModelInfo(
            name="orders",
            dimensions=[DimensionInfo(name="id", type="number", is_primary_key=True)],
            measures=["orders.count"],
        )
        json_str = model.model_dump_json()
        restored = SemanticModelInfo.model_validate_json(json_str)
        assert restored == model

    def test_empty_extra_is_mutable(self):
        """Default empty dict should be a new instance per model (not shared)."""
        m1 = SemanticModelInfo(name="a")
        m2 = SemanticModelInfo(name="b")
        m1.extra["key"] = "value"
        assert "key" not in m2.extra


class TestSemanticAdapterConfig:
    """Tests for the new auth fields on SemanticAdapterConfig."""

    def test_existing_fields_unchanged(self):
        from datus.tools.semantic_tools.config import SemanticAdapterConfig

        config = SemanticAdapterConfig(namespace="test", timeout_seconds=60)
        assert config.namespace == "test"
        assert config.timeout_seconds == 60

    def test_default_values(self):
        from datus.tools.semantic_tools.config import SemanticAdapterConfig

        config = SemanticAdapterConfig()
        assert config.namespace is None
        assert config.timeout_seconds == 30
        assert config.api_base_url is None
        assert config.auth_token is None
        assert config.username is None
        assert config.password is None

    def test_auth_fields(self):
        from datus.tools.semantic_tools.config import SemanticAdapterConfig

        config = SemanticAdapterConfig(
            api_base_url="http://localhost:4000/cubejs-api",
            auth_token="eyJhbGciOiJIUzI1NiJ9.test",
            username="admin",
            password="secret",
        )
        assert config.api_base_url == "http://localhost:4000/cubejs-api"
        assert config.auth_token == "eyJhbGciOiJIUzI1NiJ9.test"
        assert config.username == "admin"
        assert config.password == "secret"

    def test_extra_fields_allowed(self):
        """SemanticAdapterConfig allows extra fields for adapter-specific config."""
        from datus.tools.semantic_tools.config import SemanticAdapterConfig

        config = SemanticAdapterConfig(
            api_base_url="http://localhost:4000",
            custom_field="custom_value",
        )
        assert config.custom_field == "custom_value"

    def test_backward_compat_no_auth(self):
        """Existing code that doesn't pass auth fields should still work."""
        from datus.tools.semantic_tools.config import SemanticAdapterConfig

        config = SemanticAdapterConfig(namespace="prod")
        assert config.namespace == "prod"
        assert config.api_base_url is None
