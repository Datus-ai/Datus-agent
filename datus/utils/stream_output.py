# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified streaming output component for CLI operations.

Provides a consistent UI for displaying:
- Progress bar
- Current task/file being processed
- Rolling message window (keeps last N lines)
- LLM output (maximized visibility)
"""

from collections import deque
from contextlib import contextmanager
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.text import Text


class StreamOutputManager:
    """Unified streaming output manager for displaying task progress and large model output"""

    def __init__(
        self,
        console: Console,
        max_message_lines: int = 10,
        show_progress: bool = True,
        title: str = "Processing",
    ):
        """
        Initialize output manager

        Args:
            console: Rich console instance
            max_message_lines: Maximum number of lines in the message scrolling window (default 10 lines)
            show_progress: Whether to display a progress bar
            title: Progress bar title
        """
        self.console = console
        self.max_message_lines = max_message_lines
        self.show_progress = show_progress
        self.title = title

        # Progress bar will be created in start() based on total_items
        self.progress: Optional[Progress] = None

        # Message Queue (keep last N rows, auto-scroll)
        self.messages = deque(maxlen=max_message_lines)

        # Current task information
        self.current_task = ""
        self.current_file = ""
        self.task_number = 0

        # Live display
        self.live: Optional[Live] = None
        self.progress_task: Optional[TaskID] = None
        self._is_running = False

    def _create_progress(self, total_items: int) -> Progress:
        """
        Create progress bar based on total items count.

        For single item (total_items <= 1), use spinner-only style without count.
        For multiple items, use full progress bar with percentage and count.
        """
        if total_items <= 1:
            # Single task mode: spinner + description only
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=False,
            )
        else:
            # Multi-task mode: full progress bar with count
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                console=self.console,
                transient=False,
            )

    def start(self, total_items: int, description: Optional[str] = None):
        """
        Start the output manager

        Args:
            total_items: Total number of tasks
            description: Progress bar description (optional)
        """
        if self._is_running:
            return

        # Create progress bar based on total_items count
        self.progress = self._create_progress(total_items)

        desc = description or self.title
        self.progress_task = self.progress.add_task(desc, total=total_items)
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,  # Increase the refresh rate to better display LLM output
            transient=False,
        )
        self.live.start()
        self._is_running = True

    def stop(self):
        """Stop the output manager"""
        if self.live and self._is_running:
            self.live.stop()
            self.live = None
            self._is_running = False

    def update_progress(self, advance: int = 1, description: Optional[str] = None):
        """
        Update the progress bar

        Args:
            advance: Progress increments
            description: New description (optional)
        """
        if self.progress is not None and self.progress_task is not None:
            self.progress.update(self.progress_task, advance=advance)
            if description:
                self.progress.update(self.progress_task, description=description)
        self._refresh()

    def set_progress(self, completed: int, description: Optional[str] = None):
        """
        Set the absolute value of the progress bar

        Args:
            completed: Completed quantity
            description: New description (optional)
        """
        if self.progress is not None and self.progress_task is not None:
            self.progress.update(self.progress_task, completed=completed)
            if description:
                self.progress.update(self.progress_task, description=description)
        self._refresh()

    def start_file(self, filepath: str, total_items: Optional[int] = None):
        """
        Start working on the file

        Args:
            filepath: File path
            total_items: Total number of items in the file (optional)
        """
        self.current_file = filepath
        self.messages.clear()
        self.task_number = 0
        if total_items:
            self.add_message(f"Processing {total_items} items...", style="cyan")
        self._refresh()

    def complete_file(self, filepath: str):
        """
        Complete file processing

        Args:
            filepath: File path
        """
        self.current_file = ""
        self._refresh()

    def start_task(self, task_description: str):
        """
        Start working on the task

        Args:
            task_description: Mission description
        """
        self.task_number += 1
        self.current_task = f"[{self.task_number}] {task_description}"
        self._refresh()

    def add_message(self, message: str, style: str = ""):
        """
        Add a message to a scrolling window (auto-scroll, keep only the most recent N lines)

        Args:
            message: Message content (supports multiple lines)
            style: Rich Style (Optional)
        """
        if not message:
            return

        # Handle multi-line messages
        lines = str(message).strip().splitlines()
        for line in lines:
            if line.strip():
                self.messages.append((line, style))

        self._refresh()

    def add_llm_output(self, output: str):
        """
        Add LLM output (priority display, use special styles)

        Args:
            output: LLM output content
        """
        self.add_message(output, style="white")

    def complete_task(self, success: bool = True, message: str = ""):
        """
        Complete the current task

        Args:
            success: Whether it was successful or not
            message: Complete the message
        """
        if message:
            icon = "✓" if success else "✗"
            style = "green" if success else "red"
            self.add_message(f"{icon} {message}", style=style)
        self.current_task = ""
        self._refresh()

    def error(self, message: str):
        """
        Display an error message

        Args:
            message: Error message
        """
        self.add_message(f"✗ {message}", style="bold red")

    def warning(self, message: str):
        """
        Display a warning message

        Args:
            message: Warning message
        """
        self.add_message(f"⚠ {message}", style="yellow")

    def success(self, message: str):
        """
        Displays a success message

        Args:
            message: Success message
        """
        self.add_message(f"✓ {message}", style="green")

    def _render(self):
        """Render the entire output interface"""
        components = []

        # 1. Progress bar (fixed at the top)
        if self.show_progress and self.progress is not None and self.progress_task is not None:
            components.append(self.progress)
            components.append("")  # Empty line separation

        # 2. Current file (if any)
        if self.current_file:
            file_panel = Panel(
                Text(self.current_file, style="bold cyan"),
                title="📁 Current File",
                border_style="cyan",
                padding=(0, 1),
            )
            components.append(file_panel)

        # 3. Current Task (if any)
        if self.current_task:
            components.append(Text(f"→ {self.current_task}", style="bold yellow"))

        # 4. Message scrolling area (display up to N lines, auto-scrolling)
        if self.messages:
            message_lines = []
            for msg, style in self.messages:
                # Add indentation to make the output clearer
                text = Text(f"  {msg}", style=style or "dim")
                message_lines.append(text)

            # Use Panel to wrap your message, providing a border and title
            messages_panel = Panel(
                Group(*message_lines),
                title="💬 Output ",
                border_style="blue",
                padding=(0, 1),
            )
            components.append(messages_panel)

        return Group(*components)

    def _refresh(self):
        """Refresh the display"""
        if self.live and self._is_running:
            self.live.update(self._render())

    @contextmanager
    def task_context(self, task_description: str):
        """
        Task context manager that automatically handles start and finish

        Usage:
            with output_mgr.task_context("Processing item"):
                # do work
                output_mgr.add_message("Step 1 done")
                output_mgr.add_message("Step 2 done")
            # Automatically marked as complete
        """
        self.start_task(task_description)
        try:
            yield self
            self.complete_task(success=True)
        except Exception as e:
            self.complete_task(success=False, message=str(e))
            raise

    @contextmanager
    def file_context(self, filepath: str, total_items: Optional[int] = None):
        """
        File processing context manager

        Usage:
            with output_mgr.file_context("data.sql", total_items=10):
                for item in items:
                    with output_mgr.task_context(f"Processing {item}"):
                        # do work
        """
        self.start_file(filepath, total_items)
        try:
            yield self
            self.complete_file(filepath)
        except Exception:
            self.complete_file(filepath)
            raise


def create_stream_output_manager(
    console: Console,
    max_message_lines: int = 10,
    show_progress: bool = True,
    title: str = "Processing",
) -> StreamOutputManager:
    """
    Create a factory function for the Streaming Output Manager

    Args:
        console: Rich console instance
        max_message_lines: The maximum number of lines in the message scroll window
        show_progress: Whether a progress bar is displayed
        title: Progress bar title

    Returns:
        StreamOutputManager instance
    """
    return StreamOutputManager(
        console=console,
        max_message_lines=max_message_lines,
        show_progress=show_progress,
        title=title,
    )
