# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for skills that define db_tools workflows.

Tests skills that guide usage of database tools without their own scripts.
"""


import pytest

from datus.tools.skill_tools import (
    SkillBashTool,
    SkillConfig,
    SkillFuncTool,
    SkillManager,
    SkillRegistry,
)


@pytest.fixture
def workflow_skill_dir(tmp_path):
    """Create a skill defining db_tools workflow."""
    skill_dir = tmp_path / "workflow-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        """---
name: sql-analysis-workflow
description: Guided workflow for SQL data analysis using db_tools
tags:
  - sql
  - workflow
  - analysis
version: 1.0.0
---

# SQL Analysis Workflow

This skill guides you through a structured data analysis workflow using database tools.

## Workflow Steps

### Step 1: Schema Discovery
First, understand the database schema:
```
Use db_tools.list_tables() to see available tables
Use db_tools.describe_table(table_name) for each relevant table
```

### Step 2: Data Exploration
Explore the data with sample queries:
```
Use db_tools.execute_sql("SELECT * FROM {table} LIMIT 10")
```

### Step 3: Analysis Query
Based on the user's question, construct and execute the analysis query:
```
Use db_tools.execute_sql(analysis_query)
```

### Step 4: Result Interpretation
Analyze the results and provide insights:
- Row counts and data distributions
- Key findings related to the user's question
- Recommendations for further analysis

## Example Usage

User: "What are the top 10 customers by revenue?"

Workflow execution:
1. list_tables() -> find customers, orders tables
2. describe_table("customers"), describe_table("orders")
3. execute_sql("SELECT c.name, SUM(o.amount) as revenue FROM customers c
   JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY revenue DESC LIMIT 10")
4. Interpret and present results
"""
    )
    return skill_dir


class TestSkillWorkflowIntegration:
    """Integration tests for skills that define tool workflows."""

    def test_workflow_skill_loaded_without_bash_tool(self, workflow_skill_dir):
        """Test that workflow skill loads but doesn't create SkillBashTool."""
        config = SkillConfig(directories=[str(workflow_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("sql-analysis-workflow")
        assert skill is not None
        assert skill.allowed_commands == []  # No scripts

        # SkillBashTool should return empty tools list
        bash_tool = SkillBashTool(skill_metadata=skill, workspace_root=str(skill.location))
        assert bash_tool.available_tools() == []

    def test_workflow_skill_content_includes_instructions(self, workflow_skill_dir):
        """Test that workflow skill content includes tool usage instructions."""
        config = SkillConfig(directories=[str(workflow_skill_dir.parent)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        content = registry.load_skill_content("sql-analysis-workflow")
        assert content is not None

        # Verify skill content includes tool references
        assert "list_tables()" in content
        assert "describe_table" in content
        assert "execute_sql" in content
        assert "Step 1" in content
        assert "Step 2" in content

    def test_workflow_skill_in_available_skills(self, workflow_skill_dir):
        """Test that workflow skill appears in available skills XML."""
        config = SkillConfig(directories=[str(workflow_skill_dir.parent)])
        manager = SkillManager(config=config)

        xml = manager.generate_available_skills_xml("chatbot")

        # Skill should be listed in available_skills
        assert "sql-analysis-workflow" in xml
        assert "Guided workflow for SQL data analysis" in xml

    def test_workflow_skill_load_via_func_tool(self, workflow_skill_dir):
        """Test loading workflow skill via SkillFuncTool."""
        config = SkillConfig(directories=[str(workflow_skill_dir.parent)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Load the workflow skill
        result = skill_tool.load_skill("sql-analysis-workflow")
        assert result.success == 1

        # Verify skill content includes tool usage instructions
        assert "db_tools.list_tables()" in result.result or "list_tables()" in result.result
        assert "db_tools.describe_table" in result.result or "describe_table" in result.result
        assert "db_tools.execute_sql" in result.result or "execute_sql" in result.result

    def test_workflow_skill_no_bash_tool_created(self, workflow_skill_dir):
        """Test that no bash tool is created for workflow-only skill."""
        config = SkillConfig(directories=[str(workflow_skill_dir.parent)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Load the workflow skill
        skill_tool.load_skill("sql-analysis-workflow")

        # Should not have a bash tool
        bash_tool = skill_tool.get_skill_bash_tool("sql-analysis-workflow")
        assert bash_tool is None

        # All bash tools should be empty
        all_tools = skill_tool.get_all_skill_bash_tools()
        assert len(all_tools) == 0


class TestWorkflowSkillVariations:
    """Tests for various workflow skill configurations."""

    def test_multiple_workflow_skills(self, tmp_path):
        """Test loading multiple workflow skills."""
        # Create first workflow skill
        skill1_dir = tmp_path / "workflow1"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text(
            """---
name: data-exploration
description: Workflow for exploring data
tags:
  - exploration
---
# Data Exploration Workflow
1. Use list_tables()
2. Use describe_table()
"""
        )

        # Create second workflow skill
        skill2_dir = tmp_path / "workflow2"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text(
            """---
name: query-optimization
description: Workflow for query optimization
tags:
  - optimization
---
# Query Optimization Workflow
1. Analyze query plan
2. Add indexes if needed
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Load both skills
        result1 = skill_tool.load_skill("data-exploration")
        result2 = skill_tool.load_skill("query-optimization")

        assert result1.success == 1
        assert result2.success == 1
        assert "list_tables" in result1.result
        assert "query plan" in result2.result.lower()

    def test_workflow_skill_with_examples(self, tmp_path):
        """Test workflow skill with SQL examples."""
        skill_dir = tmp_path / "example-workflow"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: example-workflow
description: Workflow with SQL examples
---
# Example Workflow

## Example Queries

### Count all records
```sql
SELECT COUNT(*) FROM users;
```

### Find top users
```sql
SELECT name, score
FROM users
ORDER BY score DESC
LIMIT 10;
```
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")
        result = skill_tool.load_skill("example-workflow")

        assert result.success == 1
        assert "SELECT COUNT(*)" in result.result
        assert "ORDER BY score DESC" in result.result

    def test_workflow_skill_with_domain_context(self, tmp_path):
        """Test workflow skill with domain-specific context."""
        skill_dir = tmp_path / "ecommerce-workflow"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: ecommerce-analysis
description: E-commerce data analysis workflow
tags:
  - ecommerce
  - sales
---
# E-commerce Analysis Workflow

## Domain Tables
- customers: Customer information
- orders: Order transactions
- products: Product catalog
- order_items: Line items in orders

## Common Metrics
- Revenue: SUM(order_items.price * order_items.quantity)
- AOV: AVG(orders.total)
- Conversion: orders_count / visitors_count

## Analysis Steps
1. Identify relevant tables
2. Apply metric definitions
3. Generate SQL query
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")
        result = skill_tool.load_skill("ecommerce-analysis")

        assert result.success == 1
        assert "customers" in result.result
        assert "Revenue" in result.result
        assert "AOV" in result.result
