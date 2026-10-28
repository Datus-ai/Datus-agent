# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from .benchmark_init import init_snowflake_schema
from .local_init import init_local_schema_async
from .store import SchemaStorage, SchemaValueStorage, SchemaWithValueRAG


def create_metadata_rag(agent_config, sub_agent_name=None, datasource_id=None):
    from datus.storage.kb_retrieval import MetadataFtsRAG, metadata_fts_enabled

    if metadata_fts_enabled(agent_config):
        return MetadataFtsRAG(agent_config, sub_agent_name=sub_agent_name, datasource_id=datasource_id)
    return SchemaWithValueRAG(agent_config, sub_agent_name=sub_agent_name, datasource_id=datasource_id)


__all__ = [
    "SchemaStorage",
    "SchemaValueStorage",
    "SchemaWithValueRAG",
    "create_metadata_rag",
    "init_local_schema_async",
    "init_snowflake_schema",
]
