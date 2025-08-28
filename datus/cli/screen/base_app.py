import sys
import traceback
from threading import current_thread, main_thread
from typing import Type

from textual.app import App
from textual.driver import Driver
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

    def on_mount(self) -> None:
        self._app_ready = True

    def on_unmount(self) -> None:
        self._app_ready = False

    def _handle_exception(self, error: Exception) -> None:
        logger.error(f"CLI Execution Exceptions: {traceback.format_exception(error)}")
        original_error = error
        if isinstance(error, WorkerFailed):
            error = original_error.error
        self._safe_notify_error(error, original_error)

    def _safe_notify_error(self, error: Exception, original_error: Exception = None) -> None:
        try:
            # 检查是否可以安全地调用 notify
            if not self._can_notify():
                # 如果不能使用 notify，使用备用方案
                self._fallback_error_display(error)
                return

            # 如果在非主线程中，使用 call_from_thread
            if current_thread() != main_thread():
                self.call_from_thread(
                    self.notify, message=str(error), title="Error Occurred", severity="error", timeout=5
                )
            else:
                # 在主线程中直接调用
                self.notify(message=str(error), title="Error Occurred", severity="error", timeout=5)

        except Exception as notify_error:
            logger.error(f"Failed to show notification: {notify_error}")
            self._fallback_error_display(error)

    def _can_notify(self) -> bool:
        """检查是否可以安全地调用 notify"""
        try:
            if not self._app_ready:
                return False

            if not hasattr(self, "screen") or self.screen is None:
                return False

            if hasattr(self, "_running") and not self._running:
                return False

            return True

        except Exception:
            return False

    def _fallback_error_display(self, error: Exception) -> None:
        error_msg = f"\n{'='*50}\nERROR: {error}\n{'='*50}\n"

        try:
            print(error_msg, file=sys.stderr)
        except Exception:
            pass

        try:
            if hasattr(self, "console"):
                self.console.print(f"[red bold]{error_msg}[/red bold]")
        except Exception:
            pass

        logger.error(f"Fallback error display: {error}")

    async def action_quit(self) -> None:
        self._app_ready = False
        await super().action_quit()
