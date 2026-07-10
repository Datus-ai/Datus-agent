# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Backend-independent full-text index definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


@dataclass(frozen=True)
class FtsField:
    """A searchable text field and its index/query configuration."""

    name: str
    boost: float = 1.0
    tokenizer: str = "simple"
    ngram_min_length: int = 2
    ngram_max_length: int = 2

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FTS field name must not be empty")
        if self.boost <= 0:
            raise ValueError("FTS field boost must be positive")
        if self.tokenizer not in {"simple", "raw", "whitespace", "ngram"}:
            raise ValueError(f"Unsupported FTS tokenizer: {self.tokenizer}")
        if self.tokenizer == "ngram" and not (0 < self.ngram_min_length <= self.ngram_max_length):
            raise ValueError("Invalid FTS ngram length range")


@dataclass(frozen=True)
class FtsSpec:
    """The complete FTS contract for one table."""

    fields: tuple[FtsField, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("FTS spec must contain at least one field")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("FTS field names must be unique")
        if self.version < 1:
            raise ValueError("FTS spec version must be positive")

    @classmethod
    def from_names(cls, names: Iterable[str], *, version: int = 1) -> "FtsSpec":
        return cls(tuple(FtsField(name) for name in names), version=version)

    @property
    def columns(self) -> list[str]:
        return [field.name for field in self.fields]

    @property
    def boosts(self) -> list[float]:
        return [field.boost for field in self.fields]


class FtsIndexStatus(StrEnum):
    READY = "ready"
    MISSING = "missing"
    LEGACY = "legacy"
    VERSION_MISMATCH = "version_mismatch"
    UNSUPPORTED = "unsupported"
