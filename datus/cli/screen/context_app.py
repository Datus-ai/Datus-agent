from enum import Enum
from typing import Dict

from .base_app import BaseApp
from .catalogs_screen import CatalogsScreen
from .context_screen import CatalogScreen, MetricsScreen, TableScreen, WorkloadContextScreen


class ScreenType(str, Enum):
    """Enum for screen types."""

    CATALOG = "catalog"
    TABLE = "table"
    METRICS = "metrics"
    WORKFLOW_CONTEXT = "workflow_context"
    CATALOGS = "catalogs"


class ContextApp(BaseApp):
    """App for displaying context screens."""

    def __init__(self, screen_type: ScreenType, title: str, data: Dict, inject_callback=None):
        """
        Initialize the context app.

        Args:
            screen_type: Type of screen to display (catalog, table, metrics, workflow)
            title: Title of the screen
            data: Data to display in the screen
            inject_callback: Callback for injecting data into the workflow
        """
        super().__init__()
        self.screen_type = screen_type
        self.title = title
        self.data = data
        self.inject_callback = inject_callback

    def on_mount(self):
        """Mount the appropriate screen based on type."""
        if self.screen_type == ScreenType.CATALOG:
            self.push_screen(CatalogScreen(self.title, self.data, self.inject_callback))
        elif self.screen_type == ScreenType.TABLE:
            self.push_screen(TableScreen(self.title, self.data, self.inject_callback))
        elif self.screen_type == ScreenType.METRICS:
            self.push_screen(MetricsScreen(self.title, self.data, self.inject_callback))
        elif self.screen_type == ScreenType.WORKFLOW_CONTEXT:
            self.push_screen(WorkloadContextScreen(self.title, self.data, self.inject_callback))
        elif self.screen_type == ScreenType.CATALOGS:
            self.push_screen(CatalogsScreen(self.title, self.data, self.inject_callback))


def show_catalog_screen(title: str, data: Dict, inject_callback=None):
    """
    Show a catalog screen.

    Args:
        title: Title of the screen
        data: Catalog data to display
        inject_callback: Callback for injecting data into the workflow
    """
    app = ContextApp(ScreenType.CATALOG, title, data, inject_callback)
    app.run()


def show_table_screen(title: str, data: Dict, inject_callback=None):
    """
    Show a table screen.

    Args:
        title: Title of the screen
        data: Table data to display
        inject_callback: Callback for injecting data into the workflow
    """
    app = ContextApp(ScreenType.TABLE, title, data, inject_callback)
    app.run()


def show_metrics_screen(title: str, data: Dict):
    """
    Show a metrics screen.

    Args:
        title: Title of the screen
        data: Metrics data to display
    """
    app = ContextApp(ScreenType.METRICS, title, data)
    app.run()


# Define run_in_process at module level so it can be pickled for multiprocessing
# def run_in_process(context_type, title, data):
#    try:
#        app = ContextApp(context_type, title, data)
#        app.run()
#    except Exception as e:
#        import traceback
#        traceback.print_exc()
#        print(f"Error in Textual app: {e}")


def show_workflow_context_screen(title: str, data: Dict, run_new_loop=True):
    """
    Show a workflow context screen that displays all context types.

    Args:
        title: Title of the screen
        data: Workflow context data to display
    """
    _show_screen(ScreenType.WORKFLOW_CONTEXT, title, data, run_new_loop)


def _show_screen(screen_type: ScreenType, title: str, data: Dict, inject_callback=None, run_new_loop=True):
    app = ContextApp(screen_type=screen_type, title=title, data=data, inject_callback=inject_callback)
    if run_new_loop:
        # Rich already runs in a separate loop, so we need to create a new one
        import asyncio

        try:
            event_loop = asyncio.get_event_loop()
        except RuntimeError:
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
        app.run(loop=event_loop)
    else:
        app.run()


def show_catalogs_screen(title: str, data: Dict, inject_callback=None, run_new_loop=True):
    """
    Show a catalogs screen.

    Args:
        title: Title of the screen
        data: Catalogs data to display
    """
    _show_screen(ScreenType.CATALOGS, title, data, inject_callback=inject_callback, run_new_loop=run_new_loop)
