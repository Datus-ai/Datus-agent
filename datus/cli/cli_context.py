# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CLI Context for managing recent queries, metrics, and database state.
This class maintains the context for CLI operations independent of workflow.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datus.schemas.node_models import Metric, SQLContext, SqlTask, TableSchema
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class CliContext:
    """CLI context that maintains recent queries, metrics, and database state."""

    # Database connection info
    current_logic_db_name: Optional[str] = None
    current_db_name: Optional[str] = None
    current_catalog: Optional[str] = None
    current_schema: Optional[str] = None

    # Current SQL task
    current_sql_task: Optional[SqlTask] = None

    # Recent history (limited to last 20 items)
    recent_tables: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_metrics: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_sql_contexts: deque = field(default_factory=lambda: deque(maxlen=20))

    def add_table(self, table_schema: TableSchema):
        """Add a table to recent tables, avoiding duplicates."""
        # Remove if already exists to move to front
        table_name = self._get_table_name(table_schema)
        existing_items = [item for item in self.recent_tables if self._get_table_name(item) == table_name]
        for item in existing_items:
            self.recent_tables.remove(item)

        self.recent_tables.appendleft(table_schema)
        logger.debug(f"Added table {table_name} to recent tables")

    def add_tables(self, table_schemas: List[TableSchema]):
        """Add multiple tables to recent tables."""
        for table_schema in table_schemas:
            self.add_table(table_schema)

    def add_metric(self, metric: Metric):
        """Add a metric to recent metrics, avoiding duplicates."""
        # Remove if already exists to move to front
        existing_items = [item for item in self.recent_metrics if item.name == metric.name]
        for item in existing_items:
            self.recent_metrics.remove(item)

        self.recent_metrics.appendleft(metric)
        logger.debug(f"Added metric {metric.name} to recent metrics")

    def add_metrics(self, metrics: List[Metric]):
        """Add multiple metrics to recent metrics."""
        for metric in metrics:
            self.add_metric(metric)

    def add_sql_context(self, sql_context: SQLContext):
        """Add an SQL context to recent executions."""
        # SQLContext is a Pydantic model, so we can't add arbitrary fields
        # Just store it directly
        self.recent_sql_contexts.appendleft(sql_context)
        logger.debug(f"Added SQL context to recent executions: {sql_context.sql_query[:100]}...")

    def get_recent_tables(self) -> List[TableSchema]:
        """Get list of recent tables."""
        return list(self.recent_tables)

    def get_recent_metrics(self) -> List[Metric]:
        """Get list of recent metrics."""
        return list(self.recent_metrics)

    def get_recent_sql_contexts(self) -> List[SQLContext]:
        """Get list of recent SQL contexts."""
        return list(self.recent_sql_contexts)

    def get_last_sql_context(self) -> Optional[SQLContext]:
        """Get the most recent SQL context."""
        return self.recent_sql_contexts[0] if self.recent_sql_contexts else None

    def get_last_sql(self) -> Optional[str]:
        """Get the most recent SQL query."""
        last_context = self.get_last_sql_context()
        return last_context.sql_query if last_context else None

    def update_database_context(
        self, db_name: str = None, catalog: str = None, schema: str = None, db_logic_name: str = None
    ):
        """Update current database context."""
        if db_name is not None:
            self.current_db_name = db_name
            logger.debug(f"Updated current database: {db_name}")
        if catalog is not None:
            self.current_catalog = catalog
            logger.debug(f"Updated current catalog: {catalog}")
        if schema is not None:
            self.current_schema = schema
            logger.debug(f"Updated current schema: {schema}")
        if db_logic_name:
            self.current_logic_db_name = db_logic_name
            logger.debug(f"Updated current logic db name: {db_logic_name}")

    def set_current_sql_task(self, sql_task: SqlTask):
        """Set the current SQL task."""
        self.current_sql_task = sql_task
        logger.debug(f"Updated current SQL task: {sql_task.task[:100]}...")

    def get_or_create_sql_task(self, task_text: str = None, database_type: str = None, prompt_callback=None) -> SqlTask:
        """Get current SQL task or create a new one."""
        import uuid

        if self.current_sql_task and not task_text:
            return self.current_sql_task

        # Ensure we have a valid task text
        if not task_text and self.current_sql_task:
            task_text = self.current_sql_task.task

        # If still no task text, prompt user for input
        if not task_text and prompt_callback:
            task_text = prompt_callback("Enter task description", default="")

        # Fail if no task text provided
        if not task_text:
            raise ValueError("Task description is required but not provided")

        # Create new SQL task
        sql_task = SqlTask(
            id=str(uuid.uuid4()),
            database_type=database_type or "sqlite",
            task=task_text,
            database_name=self.current_db_name or "",
            output_dir="output",
            external_knowledge="",
        )

        self.current_sql_task = sql_task
        return sql_task

    def clear_history(self):
        """Clear all recent history."""
        self.recent_tables.clear()
        self.recent_metrics.clear()
        self.recent_sql_contexts.clear()
        logger.info("Cleared all CLI context history")

    def clear_tables(self):
        """Clear recent tables only."""
        self.recent_tables.clear()
        logger.info("Cleared recent tables")

    def clear_metrics(self):
        """Clear recent metrics only."""
        self.recent_metrics.clear()
        logger.info("Cleared recent metrics")

    def clear_sql_contexts(self):
        """Clear recent SQL contexts only."""
        self.recent_sql_contexts.clear()
        logger.info("Cleared recent SQL contexts")

    def _get_table_name(self, table_schema: TableSchema) -> str:
        """Get full table name for comparison."""
        return f"{table_schema.catalog_name}.{table_schema.database_name}.{table_schema.table_name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for debugging/logging."""
        return {
            "current_db_name": self.current_db_name,
            "current_catalog": self.current_catalog,
            "current_schema": self.current_schema,
            "recent_tables_count": len(self.recent_tables),
            "recent_metrics_count": len(self.recent_metrics),
            "recent_sql_contexts_count": len(self.recent_sql_contexts),
            "recent_tables": [self._get_table_name(t) for t in list(self.recent_tables)[:5]],  # Show first 5
            "recent_metrics": [m.name for m in list(self.recent_metrics)[:5]],  # Show first 5
        }

    def get_context_summary(self) -> str:
        """Get a summary of current context for display."""
        lines = []

        # Database info
        db_info = []
        if self.current_catalog:
            db_info.append(f"catalog: {self.current_catalog}")
        if self.current_db_name:
            db_info.append(f"database: {self.current_db_name}")
        if self.current_schema:
            db_info.append(f"schema: {self.current_schema}")

        if db_info:
            lines.append(f"Current: {', '.join(db_info)}")

        # SQL Task info
        if self.current_sql_task:
            task_preview = (
                self.current_sql_task.task[:50] + "..."
                if len(self.current_sql_task.task) > 50
                else self.current_sql_task.task
            )
            lines.append(f"Task: {task_preview}")

        # Recent items counts
        counts = []
        if self.recent_tables:
            counts.append(f"{len(self.recent_tables)} tables")
        if self.recent_metrics:
            counts.append(f"{len(self.recent_metrics)} metrics")
        if self.recent_sql_contexts:
            counts.append(f"{len(self.recent_sql_contexts)} SQL queries")

        if counts:
            lines.append(f"Recent: {', '.join(counts)}")

        return "; ".join(lines) if lines else "No context available"
