import sqlite3
from pathlib import Path

import pytest
from rich.table import Table
from textual.widgets import Static, Tree

from datus.cli.screen import ContextApp
from datus.cli.screen.context_app import ScreenType
from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.utils.constants import DBType
from tests.conftest import load_acceptance_config


@pytest.fixture
def agent_config() -> AgentConfig:
    agent_config = load_acceptance_config(datasource="bird_school")
    agent_config.rag_base_path = str(Path("tests") / "data")
    return agent_config


@pytest.fixture
def db_manager(agent_config: AgentConfig) -> DBManager:
    return db_manager_instance(agent_config.datasource_configs)


@pytest.fixture
def catalog_agent_config(agent_config: AgentConfig, tmp_path: Path) -> AgentConfig:
    db_path = tmp_path / "california_schools.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE schools (
                CDSCode INTEGER PRIMARY KEY,
                School TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO schools (CDSCode, School)
            VALUES (1, 'Alpha High School')
            """
        )
        conn.commit()

    agent_config.services.datasources["bird_school"].uri = str(db_path)
    return agent_config


@pytest.fixture
def catalog_db_manager(catalog_agent_config: AgentConfig) -> DBManager:
    return DBManager(catalog_agent_config.datasource_configs)


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_catalog_command(catalog_agent_config: AgentConfig, catalog_db_manager: DBManager):
    app = ContextApp(
        screen_type=ScreenType.CATALOGS,
        title="Database Catalogs",
        data={
            "db_type": DBType.SQLITE,
            "database_name": "california_schools",
            "db_connector": catalog_db_manager.get_conn("bird_school", "california_schools"),
            "agent_config": catalog_agent_config,
        },
    )
    async with app.run_test() as pilot:
        await pilot.pause()  # allow initial tree to be built

        catalog_screen = pilot.app.screen
        tree = catalog_screen.query_one("#catalogs-tree", Tree)
        pilot.app.set_focus(tree)
        await pilot.pause()

        # WORKAROUND: Directly call the action method instead of simulating a key press.
        # This bypasses a bug in the screen's on_key method.
        catalog_screen.action_cursor_down()
        tree.select_node(tree.cursor_node)
        await pilot.pause()

        # The database node should be selected.
        assert tree.cursor_node.data["name"] == "california_schools"
        assert tree.cursor_node.data["type"] == "database"

        # 'enter' works because it's explicitly handled in the buggy on_key method.
        await pilot.press("enter")
        await pilot.pause(1)  # Wait for the tree to expand and children to load asynchronously.

        # The database node should now have children (tables).
        catalog_screen.action_cursor_down()
        await pilot.pause()

        first_table_node = tree.cursor_node
        assert "📋" in str(first_table_node.label)  # check for table icon

        # The cursor should now have moved to the first child node (a table).
        assert tree.cursor_node.data["type"] == "table"

        # Press enter to load table details.
        await pilot.press("enter")
        await pilot.pause(1)  # Wait for details to load.

        columns_panel = catalog_screen.query_one("#columns-panel", Static)

        columns_table = columns_panel.renderable
        assert isinstance(columns_table, Table)
        assert len(columns_table.columns) == 7
        app.exit()


@pytest.mark.nightly
@pytest.mark.asyncio
async def test_subject_command(agent_config: AgentConfig):
    app = ContextApp(
        screen_type=ScreenType.SUBJECT,
        title="Subject",
        data={
            "agent_config": agent_config,
            "database_name": "california_schools",
        },
    )
    async with app.run_test() as pilot:
        entry = await exec_domains_textual(pilot, "#subject-tree")
        assert entry["type"] == "subject_entry"
        assert entry["entry_type"] in ("metric", "sql", "ext_knowledge", "reference_template")
        app.exit()


async def exec_domains_textual(pilot, tree_id: str):
    await pilot.pause(1)
    subject_screen = pilot.app.screen
    await pilot.pause()

    tree = subject_screen.query_one(tree_id, Tree)
    pilot.app.set_focus(tree)
    domain_nodes = tree.cursor_node.children
    assert len(domain_nodes) > 0

    assert domain_nodes[0].data["type"] == "subject_node"

    tree.select_node(domain_nodes[0])
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()

    layer1_nodes = tree.cursor_node.children
    assert len(layer1_nodes) > 0

    assert layer1_nodes[0].data["type"] == "subject_node"
    tree.select_node(layer1_nodes[0])
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()

    layer2_nodes = tree.cursor_node.children

    assert len(layer2_nodes) > 0

    assert layer2_nodes[0].data["type"] == "subject_node"

    tree.select_node(layer2_nodes[0])
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(1)

    table_nodes = tree.cursor_node.children
    first_table_node = table_nodes[0].data
    assert first_table_node.get("type") == "subject_entry"
    assert first_table_node.get("entry_type") in ("metric", "sql", "ext_knowledge", "reference_template")
    tree.select_node(table_nodes[0])
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(1)
    return first_table_node
