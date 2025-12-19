# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
import inspect
from types import FunctionType


def get_public_instance_methods(cls: type) -> dict[str, FunctionType]:
    """
    Get all public, non-static, non-class instance methods of a class.
    """
    methods = {}

    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        # 1. public only
        if name.startswith("_"):
            continue

        # 2. exclude @staticmethod and @classmethod
        descriptor = cls.__dict__.get(name)
        if isinstance(descriptor, (staticmethod, classmethod)):
            continue

        methods[name] = member

    return methods
