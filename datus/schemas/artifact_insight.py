# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Read-only, versioned explanation of a visual artifact's saved inputs."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from datus.schemas.analysis_artifacts import Insight, QueryBrief, SubjectAssetRef, SuggestedQuestion
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.schemas.gen_visual_dashboard_models import QueryTemplateMetaFile
from datus.schemas.gen_visual_report_models import QueryResultFile
from datus.schemas.key_tables_schema import KeyTablesSchemaFile
from datus.schemas.metric_artifact_query import MetricQueryFile


class ArtifactBlock(BaseModel):
    id: str
    title: str
    kind: str
    query_ids: list[str] = Field(default_factory=list)
    source_path: str


class QueryLineage(BaseModel):
    status: Literal["parsed", "partial", "unavailable"]
    origin: Literal["saved_sql", "sample_parameters", "metric_query", "metric_sample"]
    datasource: str | None = None
    tables: list[str] = Field(default_factory=list)


class InsightQuery(BaseModel):
    source_kind: Literal["sql", "metric"] = "sql"
    metric_query: MetricQueryFile | None = None
    name: str
    goal: str | None = None
    sql: str | None = None
    brief: QueryBrief | None = None
    result: QueryResultFile | None = None
    template: QueryTemplateMetaFile | None = None
    lineage: QueryLineage


class MetricSnapshot(BaseModel):
    origin: Literal["finalization", "metric_execution"] = "finalization"
    ref: SubjectAssetRef
    captured_at: str
    status: Literal["captured", "unavailable"]
    detail: dict[str, Any] | None = None
    tables: list[str] = Field(default_factory=list)
    lineage_status: Literal["resolved", "unavailable"] = "unavailable"


class ReferenceSqlSnapshot(BaseModel):
    ref: SubjectAssetRef
    captured_at: str
    status: Literal["captured", "unavailable"]
    sql: str | None = None
    summary: str | None = None


class ArtifactInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_revision: str
    manifest: ArtifactManifest
    blocks: list[ArtifactBlock] = Field(default_factory=list)
    # Static declarations are not proof that every dynamic component mounted.
    blocks_status: Literal["partial", "unavailable"] = "unavailable"
    queries: list[InsightQuery] = Field(default_factory=list)
    metric_details: list[MetricSnapshot] = Field(default_factory=list)
    reference_sql_details: list[ReferenceSqlSnapshot] = Field(default_factory=list)
    key_tables_schema: KeyTablesSchemaFile | None = None
    insights: list[Insight] = Field(default_factory=list)
    suggested_questions: list[SuggestedQuestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
