# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Identity contract for visual artifacts only; runtime database keys stay names."""

from pydantic import BaseModel, Field, model_serializer, model_validator


class ArtifactDatasource(BaseModel):
    datasource_id: str | None = Field(default=None, min_length=1)
    datasource: str | None = None  # Legacy input only.

    @model_validator(mode="after")
    def validate_datasource(self):
        if "datasource_id" in self.model_fields_set and not (self.datasource_id or "").strip():
            raise ValueError("datasource_id must be non-empty when present")
        if not self.datasource_id and not (self.datasource or "").strip():
            raise ValueError("query requires datasource_id or a legacy datasource")
        context = getattr(self, "execution_context", None)
        if context:
            key = "datasource_id" if self.datasource_id else "datasource"
            if context.get(key) != getattr(self, key):
                raise ValueError("query execution context must match its datasource")
        return self

    @model_serializer(mode="wrap")
    def serialize_identity(self, handler):
        result = handler(self)
        if self.datasource_id:
            result.pop("datasource", None)
        else:
            result.pop("datasource_id", None)
        return result
