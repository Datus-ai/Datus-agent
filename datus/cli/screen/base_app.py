import traceback
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

    def _handle_exception(self, error: Exception) -> None:
        logger.error(f"CLI Execution Exceptions: {traceback.format_exception(error)}")
        if isinstance(error, WorkerFailed):
            error = error.error
        self._notify_error(error)

    def _notify_error(self, error: Exception):
        try:
            self.notify(message=str(error), title="Error Occurred", severity="error", timeout=5)
        except Exception as notify_error:
            logger.error(f"Failed to show notification: {notify_error}")
            raise
