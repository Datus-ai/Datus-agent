# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import hashlib
from typing import Set

from datus.storage.reference_template.store import ReferenceTemplateRAG
from datus.utils.exceptions import DatusException, ErrorCode


def gen_reference_template_id(template: str) -> str:
    """Generate MD5 hash ID from template content."""
    return hashlib.md5(template.encode("utf-8")).hexdigest()


def exists_reference_templates(storage: ReferenceTemplateRAG, build_mode: str = "overwrite") -> Set[str]:
    """Get existing reference template IDs based on build mode."""
    valid_modes = {"overwrite", "incremental"}
    if build_mode not in valid_modes:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message_args={"field_name": "build_mode", "except_values": valid_modes, "your_value": build_mode},
        )
    existing_ids = set()
    if build_mode == "overwrite":
        return existing_ids
    if build_mode == "incremental":
        for item in storage.search_all_reference_templates():
            existing_ids.add(str(item["id"]))
    return existing_ids
