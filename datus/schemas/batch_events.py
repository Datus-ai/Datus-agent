# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified event system for batch processing tasks.

This module provides a generic event model that can be used across different
batch processing scenarios like reference_sql_init, metrics_init, etc.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class BatchStage(str, Enum):
    """Lifecycle stages for batch processing tasks.

    Task Level:
        TASK_STARTED: Task has started
        TASK_VALIDATED: Items validated, total count confirmed
        TASK_PROCESSING: Task is processing items
        TASK_COMPLETED: Task completed successfully
        TASK_FAILED: Task failed

    Group Level (for file/batch grouping):
        GROUP_STARTED: Group processing started
        GROUP_COMPLETED: Group processing completed

    Item Level:
        ITEM_STARTED: Single item processing started
        ITEM_PROCESSING: Item is being processed (with action details)
        ITEM_COMPLETED: Item completed successfully
        ITEM_FAILED: Item processing failed
    """

    # Task level stages
    TASK_STARTED = "task_started"
    TASK_VALIDATED = "task_validated"
    TASK_PROCESSING = "task_processing"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # Group level stages
    GROUP_STARTED = "group_started"
    GROUP_COMPLETED = "group_completed"

    # Item level stages
    ITEM_STARTED = "item_started"
    ITEM_PROCESSING = "item_processing"
    ITEM_COMPLETED = "item_completed"
    ITEM_FAILED = "item_failed"


class BatchEvent(BaseModel):
    """Unified event model for batch processing tasks.

    This model provides a flexible structure that can represent various
    types of events in batch processing scenarios.

    Examples:
        # Task started event
        BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.TASK_STARTED,
            total_items=10,
        )

        # Item processing event with action
        BatchEvent(
            biz_name="metrics_init",
            stage=BatchStage.ITEM_PROCESSING,
            item_id="row_5",
            action_name="gen_semantic_model",
            status="processing",
        )

        # Item failed event
        BatchEvent(
            biz_name="reference_sql_init",
            stage=BatchStage.ITEM_FAILED,
            item_id="sql_123",
            error="Failed to generate SQL summary",
            payload={"filepath": "/path/to/file.sql"},
        )
    """

    # Business identification
    biz_name: str = Field(..., description="Business name (e.g., 'reference_sql_init', 'metrics_init')")

    # Lifecycle stage
    stage: BatchStage = Field(..., description="Current lifecycle stage of the event")

    # Optional identifiers for tracking hierarchy
    task_id: Optional[str] = Field(default=None, description="Unique identifier for the batch task")
    group_id: Optional[str] = Field(default=None, description="Group identifier (e.g., file path, batch id)")
    item_id: Optional[str] = Field(default=None, description="Item identifier (e.g., sql_id, row index)")

    # Action details for ITEM_PROCESSING stage
    action_name: Optional[str] = Field(default=None, description="Action name (e.g., 'gen_sql_summary')")
    status: Optional[str] = Field(default=None, description="Action status (e.g., 'processing', 'success')")

    # Messages and errors
    message: Optional[str] = Field(default=None, description="Human-readable message")
    error: Optional[str] = Field(default=None, description="Error message if applicable")
    exception_type: Optional[str] = Field(default=None, description="Exception type name if error occurred")

    # Progress tracking
    total_items: Optional[int] = Field(default=None, description="Total number of items in the batch/group")
    completed_items: Optional[int] = Field(default=None, description="Number of successfully completed items")
    failed_items: Optional[int] = Field(default=None, description="Number of failed items")

    # Flexible payload for additional data
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Additional event-specific data")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="When the event occurred")

    class Config:
        use_enum_values = True


# Type alias for emit callback
BatchEventEmitter = Callable[["BatchEvent"], None]


class BatchEventHelper:
    """Helper class for creating and emitting batch events.

    Provides convenient methods for emitting events at different lifecycle stages.
    Emits BatchEvent objects directly for type-safe event handling.
    """

    def __init__(self, biz_name: str, emit: Optional[Callable[["BatchEvent"], None]] = None):
        """Initialize the helper.

        Args:
            biz_name: The business name for events (e.g., 'reference_sql_init')
            emit: Optional emit callback that accepts BatchEvent
        """
        self.biz_name = biz_name
        self._emit = emit

    def emit(self, event: BatchEvent) -> None:
        """Emit an event using the configured callback.

        Args:
            event: The event to emit
        """
        if self._emit is None:
            return
        try:
            self._emit(event)
        except Exception as exc:
            logger.debug(f"emit callback failed: {exc}")

    # =========================================================================
    # Task Level Events
    # =========================================================================

    def task_started(
        self,
        total_items: Optional[int] = None,
        task_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit task started event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.TASK_STARTED,
            task_id=task_id,
            total_items=total_items,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def task_validated(
        self,
        total_items: int,
        valid_items: Optional[int] = None,
        invalid_items: Optional[int] = None,
        task_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit task validated event (items validated, count confirmed)."""
        extra_payload = dict(payload) if payload else {}
        if valid_items is not None:
            extra_payload["valid_items"] = valid_items
        if invalid_items is not None:
            extra_payload["invalid_items"] = invalid_items

        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.TASK_VALIDATED,
            task_id=task_id,
            total_items=total_items,
            payload=extra_payload if extra_payload else None,
        )
        self.emit(event)
        return event

    def task_processing(
        self,
        total_items: int,
        task_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit task processing event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.TASK_PROCESSING,
            task_id=task_id,
            total_items=total_items,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def task_completed(
        self,
        total_items: int,
        completed_items: int,
        failed_items: int = 0,
        task_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit task completed event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.TASK_COMPLETED,
            task_id=task_id,
            total_items=total_items,
            completed_items=completed_items,
            failed_items=failed_items,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def task_failed(
        self,
        error: str,
        task_id: Optional[str] = None,
        exception_type: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit task failed event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.TASK_FAILED,
            task_id=task_id,
            error=error,
            exception_type=exception_type,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    # =========================================================================
    # Group Level Events
    # =========================================================================

    def group_started(
        self,
        group_id: str,
        total_items: Optional[int] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit group started event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.GROUP_STARTED,
            group_id=group_id,
            total_items=total_items,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def group_completed(
        self,
        group_id: str,
        total_items: Optional[int] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit group completed event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.GROUP_COMPLETED,
            group_id=group_id,
            total_items=total_items,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    # =========================================================================
    # Item Level Events
    # =========================================================================

    def item_started(
        self,
        item_id: str,
        group_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit item started event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.ITEM_STARTED,
            group_id=group_id,
            item_id=item_id,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def item_processing(
        self,
        item_id: str,
        action_name: str,
        status: Optional[str] = None,
        group_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit item processing event with action details."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.ITEM_PROCESSING,
            group_id=group_id,
            item_id=item_id,
            action_name=action_name,
            status=status,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def item_completed(
        self,
        item_id: str,
        group_id: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit item completed event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.ITEM_COMPLETED,
            group_id=group_id,
            item_id=item_id,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event

    def item_failed(
        self,
        item_id: str,
        error: str,
        group_id: Optional[str] = None,
        exception_type: Optional[str] = None,
        **payload: Any,
    ) -> BatchEvent:
        """Emit item failed event."""
        event = BatchEvent(
            biz_name=self.biz_name,
            stage=BatchStage.ITEM_FAILED,
            group_id=group_id,
            item_id=item_id,
            error=error,
            exception_type=exception_type,
            payload=payload if payload else None,
        )
        self.emit(event)
        return event
