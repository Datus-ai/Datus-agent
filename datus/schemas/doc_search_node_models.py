# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List, Literal

from pydantic import Field, field_validator

from datus.schemas.base import BaseInput, BaseResult


class DocSearchInput(BaseInput):
    """
    Input model for document search node.
    Validates the input parameters for document retrieval.
    """

    keywords: List[str] = Field(..., description="Keywords to search for in documents")
    top_n: int = Field(5, description="Number of documents to return")
    method: Literal["internal", "external", "llm"] = Field("internal", description="Method to use for document search")

    @field_validator("top_n")
    def validate_top_n(cls, v):
        if v <= 0:
            raise ValueError("'top_n' must be a positive integer")
        return v


class DocSearchResult(BaseResult):
    """
    Result model for document search node.
    Contains the retrieved documents.
    """

    docs: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Retrieved documents for each keyword, where key is the keyword and value is a list of document texts"
        ),
    )
    doc_count: int = Field(0, description="Number of documents found")

    @field_validator("doc_count")
    def validate_doc_count(cls, v, values):
        if "docs" in values.data:
            total_docs = sum(len(docs) for docs in values.data["docs"].values())
            if total_docs != v:
                raise ValueError("'doc_count' must match the total number of documents in 'docs'")
        return v
