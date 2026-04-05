"""Generic base models for API responses."""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """Generic result type for API responses with type-safe data field."""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[T] = Field(None, description="Response data when successful")
    errorCode: Optional[str] = Field(None, description="Error code if operation failed")
    errorMessage: Optional[str] = Field(None, description="Error message if operation failed")

    model_config = ConfigDict(arbitrary_types_allowed=True)
