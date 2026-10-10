# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Saved single-metric queries; parameter bindings are values, never SQL."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datus.schemas.analysis_artifacts import SubjectAssetRef


class ParamRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    param: str = Field(pattern=r"^[a-z_][a-z0-9_]{0,63}$")


class MetricFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "is_null", "is_not_null"] = "eq"
    value: str | int | float | bool | list[str | int | float | bool] | ParamRef | None = None


class MetricQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: SubjectAssetRef
    dimensions: list[str] = Field(default_factory=list, max_length=30)
    time_start: str | ParamRef | None = None
    time_end: str | ParamRef | None = None
    time_granularity: Literal["second", "minute", "hour", "day", "week", "month", "quarter", "year"] | None = None
    filters: list[MetricFilter] = Field(default_factory=list, max_length=30)
    order_by: list[str] = Field(default_factory=list, max_length=30)
    limit: int = Field(default=10000, ge=1, le=100000)
    metric_params: dict[str, str | int | float | bool | ParamRef | None] = Field(default_factory=dict)

    def parameter_names(self) -> set[str]:
        values = [self.time_start, self.time_end, *(f.value for f in self.filters), *self.metric_params.values()]
        return {value.param for value in values if isinstance(value, ParamRef)}


class MetricQueryFile(MetricQueryRequest):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["metric"] = "metric"
    name: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    goal: str = Field(min_length=1)
    datasource: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    metric_detail: dict[str, Any]
    metric_tables: list[str] = Field(default_factory=list)
    captured_at: str
    generated_sql: str = Field(min_length=1)

    @model_validator(mode="after")
    def identity_matches(self) -> MetricQueryFile:
        path = PurePosixPath(self.model_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.model_path:
            raise ValueError("model_path must be a relative project path without traversal")
        if self.metric_detail.get("name") != self.metric.name or self.metric_detail.get("path") != self.metric.path:
            raise ValueError("saved metric detail must match the metric's subject identity")
        return self
