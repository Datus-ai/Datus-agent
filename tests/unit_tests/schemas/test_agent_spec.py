# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Configuration contracts for generic agents, before any tools or plugins are loaded."""

import json

import pytest
import yaml
from pydantic import ValidationError

from datus.schemas.agent_models import ScopedContext
from datus.schemas.agent_spec import AgentSpec


def test_yaml_selections_round_trip_without_losing_mcp_tool_filters():
    config = yaml.safe_load(
        """
name: sales_insights
instructions: Answer questions using the selected sources.
tools:
  - db_tools.list_tables
mcp:
  - connection_ref: business_docs
    tools:
      - search_documents
skills:
  - sales-guidelines
scoped_context:
  datasource: sales
  tables: orders,customers
  metrics: Finance.Revenue
  sqls: Finance.Reports
  project_files:
    - subject/semantic_models/sales/**
output:
  format: markdown
  instructions: Cite the sources.
"""
    )

    spec = AgentSpec.model_validate(config)
    restored = AgentSpec.model_validate_json(spec.model_dump_json(exclude_unset=True))

    assert restored.model_dump(exclude_unset=True) == config
    assert restored.mcp[0].tools == ["search_documents"]
    assert restored.tools == ["db_tools.list_tables"]


@pytest.mark.parametrize("config", [{"name": "plain"}, {"name": "plain", "tools": []}])
def test_no_declared_tools_selects_no_native_tools(config):
    spec = AgentSpec.model_validate(config)

    assert spec.tools == []
    assert spec.mcp == []


@pytest.mark.parametrize("field", ["tools", "skills"])
@pytest.mark.parametrize("selection", ["db_tools.list_tables", {"mode": "explicit"}, [42], [" "]])
def test_selections_reject_strings_mapping_modes_and_invalid_identifiers(field, selection):
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"name": "plain", field: selection})


def test_mcp_connection_cannot_implicitly_expose_all_tools():
    with pytest.raises(ValidationError, match="tools"):
        AgentSpec.model_validate({"name": "plain", "mcp": [{"connection_ref": "business_docs"}]})


@pytest.mark.parametrize("field", ["endpoint", "api_key", "credential_ref", "command"])
def test_agent_mcp_references_do_not_accept_connection_configuration(field):
    with pytest.raises(ValidationError, match=field):
        AgentSpec.model_validate(
            {
                "name": "plain",
                "mcp": [{"connection_ref": "business_docs", "tools": ["search"], field: "project-owned"}],
            }
        )


def test_repeated_mcp_connection_is_not_silently_overwritten():
    with pytest.raises(ValidationError, match="listed once"):
        AgentSpec.model_validate(
            {
                "name": "plain",
                "mcp": [
                    {"connection_ref": "business_docs", "tools": ["search"]},
                    {"connection_ref": "business_docs", "tools": ["read"]},
                ],
            }
        )


def test_node_config_preserves_new_fields_alongside_existing_routing_settings():
    config = {
        "node_class": "configurable",
        "system_prompt": "sales_insights",
        "permissions": {"tools": "deny"},
        "instructions": "Use the selected business vocabulary.",
        "tools": ["db_tools.list_tables"],
        "skills": ["sales-guidelines"],
        "output": {"format": "json"},
    }

    spec = AgentSpec.from_node_config("sales_insights", config)

    assert spec.name == "sales_insights"
    assert spec.instructions == config["instructions"]
    assert spec.tools == config["tools"]
    assert spec.skills == config["skills"]
    assert json.loads(spec.model_dump_json())["output"]["format"] == "json"


def test_registration_name_cannot_be_replaced_by_the_payload():
    with pytest.raises(ValueError, match="registration name"):
        AgentSpec.from_node_config("sales_insights", {"name": "other_agent"})


def test_user_definition_cannot_select_python_runtime_factories():
    with pytest.raises(ValidationError, match="runtime_factory"):
        AgentSpec.model_validate({"name": "plain", "runtime_factory": "arbitrary.module:create_runtime"})


def test_grouped_scope_round_trips_through_the_existing_context_model():
    scoped_context = {
        "datasource": "warehouse",
        "tables_by_datasource": {"warehouse": "orders,customers", "crm": "customers"},
        "metrics": "Finance.Revenue",
        "sqls": "Finance.Reports",
        "project_files": ["reports/**"],
    }
    spec = AgentSpec(name="research", scoped_context=scoped_context)
    assert isinstance(spec.scoped_context, ScopedContext)
    assert json.loads(spec.model_dump_json(exclude_unset=True))["scoped_context"] == scoped_context


def test_misspelled_scope_fields_cannot_silently_drop_a_business_constraint():
    with pytest.raises(ValidationError, match="Unknown scoped_context fields"):
        AgentSpec(name="research", scoped_context={"project_file": ["private/**"]})
