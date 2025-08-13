from textual.app import App
from textual.worker import WorkerFailed

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class BaseApp(App):
    def __init__(self):
        super().__init__()

    def _handle_exception(self, error: Exception) -> None:
        logger.error(f"CLI Execution Exceptions: {error}")
        if isinstance(error, WorkerFailed):
            error = error.error
        self._notify_error(error)

    def _notify_error(self, error: Exception):
        try:
            self.notify(message=str(error), title="Error Occurred", severity="error", timeout=5)
        except Exception as notify_error:
            logger.error(f"Failed to show notification: {notify_error}")
