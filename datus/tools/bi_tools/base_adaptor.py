# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AuthParam(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# ---------- Core Models ----------


class ColumnInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    data_type: Optional[str] = None
    description: Optional[str] = None
    table: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class MetricDef(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    expression: str
    table: Optional[str] = None
    description: Optional[str] = None
    origin: str = "dataset"  # dataset | chart | semantic
    extra: Dict[str, Any] = Field(default_factory=dict)


class DimensionDef(BaseModel):  # 从query_context / dataset_query /Metadata API Field
    model_config = ConfigDict(extra="allow")

    name: str
    title: Optional[str] = None
    data_type: Optional[str] = None
    table: Optional[str] = None
    description: Optional[str] = None
    origin: str = "dataset"
    extra: Dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Union[int, str]
    name: str
    dialect: Optional[str]
    description: Optional[str] = None
    tables: Optional[List[str]] = None
    columns: Optional[List[ColumnInfo]] = None
    metrics: Optional[List[MetricDef]] = None
    dimensions: Optional[List[DimensionDef]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: Literal["sql", "semantic"]
    payload: Dict[str, Any] = Field(default_factory=dict)
    sql: Optional[List[str]] = None
    tables: Optional[List[str]] = None
    metrics: Optional[List[MetricDef]] = None
    dimensions: Optional[List[DimensionDef]] = None


class ChartInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Union[int, str]
    name: str
    description: Optional[str] = None
    query: Optional[QuerySpec] = None
    chart_type: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class DashboardInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Union[int, str]
    name: str
    description: Optional[str] = None
    chart_ids: List[Union[int, str]] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


# ---------- Adaptor Interfaces ----------


class BIAdaptorBase(ABC):
    """
    Two-layer design:
      1) Discovery: list dashboards/charts/datasets
      2) Extraction: fetch structured assets for analysis (QuerySpec / DatasetInfo)
    """

    def __init__(
        self, api_base_url: str, auth_params: AuthParam, dialect: str, timeout: Optional[float] = 30.0
    ) -> None:
        self.api_base_url = api_base_url
        self.auth_params = auth_params
        self.dialect = dialect
        self.timeout = timeout

    @classmethod
    def register(cls, platform: str, auth_type: AuthType, display_name: Optional[str] = None) -> None:
        from datus.tools.bi_tools.registry import adaptor_registry

        adaptor_registry.register(platform, cls, auth_type=auth_type, display_name=display_name)

    @abstractmethod
    def platform_name(self) -> str:
        # metabase/superset
        """For routing/logging only; should NOT be exposed to LLM outputs."""
        raise NotImplementedError

    @abstractmethod
    def auth_type(self) -> AuthType:
        raise NotImplementedError

    # ----- Discovery -----

    @abstractmethod
    def parse_dashboard_id(self, dashboard_url: str) -> Union[int, str]:
        raise NotImplementedError

    @abstractmethod
    def get_dashboard_info(self, dashboard_id: Union[int, str]) -> Optional[DashboardInfo]:
        raise NotImplementedError

    @abstractmethod
    def list_charts(self, dashboard_id: Union[int, str]) -> List[ChartInfo]:
        """Return lightweight chart metadata (id/name/type)."""
        raise NotImplementedError

    @abstractmethod
    def list_datasets(self, dashboard_id: Union[int, str]) -> List[DatasetInfo]:
        """Return lightweight dataset metadata (id/name)."""
        raise NotImplementedError

    # ----- Extraction -----

    @abstractmethod
    def get_chart(self, chart_id: Union[int, str], dashboard_id: Union[int, str, None] = None) -> Optional[ChartInfo]:
        """Return ChartInfo with QuerySpec (may or may not include rendered_sql)."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(
        self, dataset_id: Union[int, str], dashboard_id: Union[int, str, None] = None
    ) -> Optional[DatasetInfo]:
        """Return DatasetInfo with columns/metrics/dimensions if available."""
        raise NotImplementedError

    def close(self):
        return


class AuthType(Enum):
    LOGIN = "login"  # username & password
    API_KEY = "api_key"  # api key
