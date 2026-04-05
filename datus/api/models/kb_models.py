"""Pydantic v2 models for the bootstrap-kb API."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class KbComponent(str, Enum):
    """Knowledge base components that can be bootstrapped."""

    METADATA = "metadata"
    SEMANTIC_MODEL = "semantic_model"
    METRICS = "metrics"
    EXT_KNOWLEDGE = "ext_knowledge"
    REFERENCE_SQL = "reference_sql"


class BootstrapKbInput(BaseModel):
    """POST body for /api/v1/kb/bootstrap.

    Mirrors the subset of `datus-agent bootstrap-kb` options exposed by the backend API.
    """

    model_config = ConfigDict(use_enum_values=True)

    components: list[KbComponent] = Field(
        ...,
        min_length=1,
        description=(
            "Knowledge base components to initialize. "
            "`metadata` scans live database schema and sample rows; "
            "`semantic_model` derives semantic schema objects from success-story SQLs; "
            "`metrics` derives MetricFlow-style business metrics; "
            "`ext_knowledge` imports or generates business terminology/knowledge entries; "
            "`reference_sql` indexes reusable SQL files."
        ),
    )
    strategy: Literal["overwrite", "check", "incremental"] = Field(
        default="incremental",
        description=(
            "Update strategy. `check` inspects existing data without rebuilding where supported, "
            "`overwrite` clears and rebuilds, and `incremental` appends or updates changed entries."
        ),
    )

    # metadata-specific
    schema_linking_type: str = Field(
        default="full",
        description=(
            "Metadata-only option controlling which object types are indexed from the database. "
            "Expected values follow `datus-agent bootstrap-kb`: `table`, `view`, `mv`, or `full`."
        ),
    )
    catalog: str = Field(
        default="",
        description=(
            "Optional metadata catalog filter. Mainly relevant for engines with a catalog layer, "
            "such as Snowflake or StarRocks."
        ),
    )
    database_name: str = Field(
        default="",
        description=(
            "Optional metadata database filter. Passed through to adapter-specific initialization; "
            "for Snowflake it maps to schema name, while for MySQL/PostgreSQL/StarRocks it is the database name."
        ),
    )

    # semantic_model / metrics source
    success_story: Optional[str] = Field(
        default=None,
        description=(
            "Project-root-relative path to a success-story CSV containing historical question/SQL pairs. "
            "Used by `semantic_model`, `metrics`, and `ext_knowledge` when bootstrapping from success stories."
        ),
    )
    subject_tree: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional predefined hierarchical categories in `domain/layer1/layer2` form, such as "
            "`Sales/Reporting/Daily`. Used by `metrics`, `ext_knowledge`, and `reference_sql`; "
            "if omitted, bootstrap reuses or learns categories from existing KB content."
        ),
    )

    # reference_sql source
    sql_dir: Optional[str] = Field(
        default=None,
        description=(
            "Project-root-relative directory containing `.sql` files for the `reference_sql` component. "
            "Files are scanned recursively and only `SELECT` statements are indexed."
        ),
    )

    # ext_knowledge source
    ext_knowledge: Optional[str] = Field(
        default=None,
        description=(
            "Project-root-relative CSV for the `ext_knowledge` component. Expected columns are "
            "`subject_path`, `name`, `search_text`, and `explanation`. "
            "If omitted, external knowledge can be generated from `success_story` instead."
        ),
    )


class BootstrapKbEvent(BaseModel):
    """SSE event envelope sent to the client."""

    model_config = ConfigDict(use_enum_values=True)

    stream_id: str
    component: str
    stage: str
    message: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[dict] = None
    payload: Optional[dict] = None
    timestamp: str


class BootstrapKbResult(BaseModel):
    """Final summary after all components complete."""

    stream_id: str
    components: dict[str, dict]
