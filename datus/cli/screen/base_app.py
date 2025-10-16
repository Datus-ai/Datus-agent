# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
from contextlib import suppress
from typing import Type

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from textual.app import App
from textual.driver import Driver
from textual.message import Message
from textual.screen import Screen
from textual.types import CSSPathType
from textual.worker import WorkerFailed

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class BaseApp(App):
    def __init__(
        self,
        driver_class: Type[Driver] | None = None,
        css_path: CSSPathType | None = None,
        watch_css: bool = False,
        ansi_color: bool = False,
    ):
        super().__init__(driver_class=driver_class, css_path=css_path, watch_css=watch_css, ansi_color=ansi_color)
        self._app_ready = False
        self._error_console = Console(stderr=True, force_terminal=True)

    def on_mount(self) -> None:
        """App mount handler"""
        self._app_ready = True

    def on_unmount(self) -> None:
        """App unmount handler"""
        self._app_ready = False

    async def _on_message(self, message: Message) -> None:
        """
        Override to catch all message handling exceptions
        This includes Screen mount/unmount and other events
        """
        try:
            await super()._on_message(message)
        except Exception as e:
            self._handle_exception(e, message)

    def push_screen(self, screen: Screen | str, *args, **kwargs):
        """
        Override push_screen to wrap screens with exception handling
        """
        # If screen is a class or instance, wrap it
        if isinstance(screen, type) and issubclass(screen, Screen):
            # It's a Screen class, instantiate with wrapper
            screen_instance = screen(*args, **kwargs)
            self._wrap_screen(screen_instance)
            return super().push_screen(screen_instance)
        elif isinstance(screen, Screen):
            # It's a Screen instance, wrap it
            self._wrap_screen(screen)
            return super().push_screen(screen, *args, **kwargs)
        else:
            # It's a string (screen name), just pass through
            return super().push_screen(screen, *args, **kwargs)

    def _wrap_screen(self, screen: Screen):
        """
        Wrap screen methods with exception handling
        """
        # Wrap critical screen methods
        if hasattr(screen, "on_mount"):
            original_on_mount = screen.on_mount
            screen.on_mount = self._create_wrapped_method(original_on_mount, screen, "on_mount", is_critical=True)

        # Wrap other event handlers
        for attr_name in dir(screen):
            if attr_name.startswith("on_") and attr_name != "on_mount":
                attr = getattr(screen, attr_name)
                if callable(attr) and not hasattr(attr, "_wrapped"):
                    wrapped = self._create_wrapped_method(attr, screen, attr_name)
                    wrapped._wrapped = True
                    setattr(screen, attr_name, wrapped)

    def _create_wrapped_method(self, method, screen, method_name, is_critical=False):
        """
        Create a wrapped version of a screen method
        """
        from functools import wraps

        @wraps(method)
        def sync_wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                self._handle_screen_exception(e, screen, method_name, is_critical)

        @wraps(method)
        async def async_wrapper(*args, **kwargs):
            try:
                return await method(*args, **kwargs)
            except Exception as e:
                self._handle_screen_exception(e, screen, method_name, is_critical)

        # Check if method is async
        import asyncio

        if asyncio.iscoroutinefunction(method):
            return async_wrapper
        else:
            return sync_wrapper

    def _handle_screen_exception(self, error: Exception, screen, context, is_critical=False):
        """
        Handle exception from a screen

        Args:
            error: The exception
            screen: The screen instance or class
            context: Context (method name or message)
            is_critical: Whether this is a critical error (like on_mount)
        """
        # Get screen info
        if isinstance(screen, Screen):
            screen_name = screen.__class__.__name__
        else:
            screen_name = str(screen)

        if isinstance(context, Message):
            context_str = f"handling {context.__class__.__name__}"
        else:
            context_str = str(context)

        # Log the error
        logger.error(f"Exception in screen '{screen_name}' during {context_str}: {error}")

        # Format error message
        error_msg = self._format_error_message(error)
        detailed_msg = f"Screen: {screen_name}\nContext: {context_str}\n{error_msg}"

        # Display error
        self._display_screen_error(detailed_msg, error, is_critical)

        # For critical errors (like on_mount), we might need to take action
        if is_critical:
            self._handle_critical_screen_error(screen, error)

    def _display_screen_error(self, message: str, error: Exception, is_critical: bool):
        """
        Display screen error using multiple strategies
        """
        # Strategy 1: Try notification if app is ready
        if self._app_ready and not is_critical:
            try:
                self.notify(message=str(error)[:200], title="❌ Screen Error", severity="error", timeout=8)
            except Exception as e:
                logging.debug(f"Exception in screen error: {e}", exc_info=True)

        # Strategy 2: Always show in console
        self._error_console.print(
            Panel(
                Text(message, style="red bold"),
                title=f"[red bold]❌ {'CRITICAL ' if is_critical else ''}Screen Error[/red bold]",
                border_style="red",
                expand=False,
                padding=(1, 2),
            )
        )

    def _handle_critical_screen_error(self, screen, error):
        """
        Handle critical screen errors (like on_mount failures)
        """
        self._error_console.print("[yellow]Critical screen error detected. Attempting recovery...[/yellow]")

        try:
            # Try to pop the broken screen
            if self.screen == screen:
                self.pop_screen()
                self._error_console.print("[green]✓ Recovered by removing broken screen[/green]")

                # Show error in the UI
                self.notify(f"Screen failed to load: {error}", title="Screen Error", severity="error", timeout=10)
        except Exception as recovery_error:
            self._error_console.print(
                f"[red]Recovery failed: {recovery_error}[/red]\n"
                "[yellow]Application may be unstable. Consider restarting.[/yellow]"
            )

    def _handle_exception(self, error: Exception, message: Message = None):
        """
        General exception handler
        """
        # Log
        logger.error(f"Unhandled exception: {error}")

        # Handle WorkerFailed
        if isinstance(error, WorkerFailed):
            error = error.error if hasattr(error, "error") else error

        # Display error
        self._display_error(error)

    def _display_error(self, error: Exception):
        """
        Display general errors
        """
        error_msg = self._format_error_message(error)

        # Try UI notification
        if self._app_ready:
            try:
                self.notify(message=error_msg[:200], title="❌ Error", severity="error", timeout=8)
            except Exception as e:
                logging.debug(f"Exception in screen error: {e}", exc_info=True)

        # Console output
        self._error_console.print(
            Panel(
                Text(error_msg, style="red bold"),
                title="[red bold]❌ Application Error[/red bold]",
                border_style="red",
                expand=False,
            )
        )

    def _format_error_message(self, error: Exception) -> str:
        """
        Format error message
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # Truncate if too long
        max_length = 500
        if len(error_msg) > max_length:
            error_msg = error_msg[:max_length] + "..."

        return f"{error_type}: {error_msg}"

    async def action_quit(self) -> None:
        """Override quit action"""
        self._app_ready = False
        with suppress(Exception):
            await super().action_quit()
