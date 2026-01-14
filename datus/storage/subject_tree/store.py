"""SQLite-based storage for hierarchical subject taxonomy tree.

This module implements a tree structure using the adjacency list model,
following the pattern established by TaskStore for consistency.
"""
import fnmatch
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
from lancedb.pydantic import LanceModel

from datus.storage import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.lancedb_conditions import and_, in_, like
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

SUBJECT_ID_COLUMN_NAME = "subject_node_id"
SUBJECT_PATH_COLUMN_NAME = "subject_path"
NAME_COLUMN_NAME = "name"
CREATED_AT_COLUMN_NAME = "created_at"


class SubjectTreeStore:
    """SQLite-based storage for hierarchical subject taxonomy tree.

    Implements adjacency list model for tree structure.
    Follows the pattern established by TaskStore for consistency.

    Attributes:
        db_path: Directory path for database storage
        db_file: Full path to the SQLite database file
    """

    def __init__(self, db_path: str):
        """Initialize SubjectTreeStore.

        Args:
            db_path: Directory path for database storage
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.db_file = os.path.join(db_path, "subject_tree.db")
        self._ensure_table()
        logger.info(f"SubjectTreeStore initialized at {self.db_file}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        """Create subject_nodes table and indices if not exists."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create main table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS subject_nodes (
                    node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(parent_id, name)
                )
            """
            )

            # Create indices
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_subject_parent_id
                ON subject_nodes(parent_id)
            """
            )

            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_subject_parent_name
                ON subject_nodes(parent_id, name)
            """
            )

            conn.commit()
            logger.debug("Subject nodes table and indices created/verified")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create subject_nodes table: {e}")
            raise
        finally:
            conn.close()

    # ========== CRUD Operations ==========

    def create_node(self, parent_id: Optional[int], name: str, description: str = "") -> Dict[str, Any]:
        """Create a new subject node.

        Args:
            parent_id: Parent node ID (None for root nodes)
            name: Node name (must be unique under same parent)
            description: Optional description

        Returns:
            Created node dict with all fields

        Raises:
            ValueError: If validation fails
            sqlite3.IntegrityError: If duplicate name under same parent
        """
        # Validate name
        if not name or not name.strip():
            raise ValueError("Node name cannot be empty")

        name = name.strip()

        # Validate parent exists (if provided)
        if parent_id is not None:
            parent = self.get_node(parent_id)
            if not parent:
                raise ValueError(f"Parent node {parent_id} not found")

        # Create node
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO subject_nodes
                (parent_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (parent_id, name, description, now, now),
            )

            node_id = cursor.lastrowid
            conn.commit()

            # Get created node to get full path for logging
            created_node = self.get_node(node_id)
            logger.info(f"Created node: {self.get_full_path(node_id)} (node_id={node_id})")

            return created_node

        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Node with name '{name}' already exists under parent {parent_id}")
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create node: {e}")
            raise
        finally:
            conn.close()

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            Node dict or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM subject_nodes WHERE node_id = ?
            """,
                (node_id,),
            )

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

        finally:
            conn.close()

    def get_node_by_path(self, path: List[str]) -> Optional[Dict[str, Any]]:
        """Get node by path components.

        Traverses the tree by following the path components.

        Args:
            path: Path components (e.g., ['Finance', 'Revenue', 'Q1'])

        Returns:
            Node dict or None if not found
        """
        if not path:
            return None

        parent_id = None

        for component in path:
            node = self._find_child_by_name(parent_id, component)
            if not node:
                return None
            parent_id = node["node_id"]

        return self.get_node(parent_id)

    def update_node(
        self,
        node_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> bool:
        """Update node fields.

        When changing parent_id, validates:
        - No cycles created

        Args:
            node_id: Node ID to update
            name: New name (optional)
            description: New description (optional)
            parent_id: New parent ID (optional, use -1 to set to None)

        Returns:
            True if updated, False if node not found

        Raises:
            ValueError: If validation fails
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found for update")
            return False

        # Prepare update values
        update_fields = []
        update_values = []

        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Node name cannot be empty")
            update_fields.append("name = ?")
            update_values.append(name)

        if description is not None:
            update_fields.append("description = ?")
            update_values.append(description)

        # Handle parent_id change
        if parent_id is not None:
            # -1 means set to None (root level)
            if parent_id == -1:
                parent_id = None

            # Validate no cycle
            if parent_id is not None and not self.validate_no_cycle(node_id, parent_id):
                raise ValueError(f"Moving node {node_id} to parent {parent_id} would create a cycle")

            # Validate parent exists (if provided)
            if parent_id is not None:
                parent = self.get_node(parent_id)
                if not parent:
                    raise ValueError(f"Parent node {parent_id} not found")

            # Update parent
            update_fields.append("parent_id = ?")
            update_values.append(parent_id)

        if not update_fields:
            logger.debug(f"No fields to update for node {node_id}")
            return True

        # Always update updated_at
        update_fields.append("updated_at = ?")
        update_values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Add node_id for WHERE clause
        update_values.append(node_id)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Execute update
            cursor.execute(
                f"""
                UPDATE subject_nodes
                SET {', '.join(update_fields)}
                WHERE node_id = ?
            """,
                update_values,
            )

            conn.commit()
            logger.info(f"Updated node {node_id}")
            return True

        except sqlite3.IntegrityError as e:
            conn.rollback()
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Node with name '{name}' already exists under parent {parent_id}")
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update node {node_id}: {e}")
            raise
        finally:
            conn.close()

    def delete_node(self, node_id: int, cascade: bool = True) -> bool:
        """Delete node.

        Args:
            node_id: Node to delete
            cascade: If True, delete all descendants (via CASCADE constraint)
                    If False, fail if node has children

        Returns:
            True if deleted, False if node not found

        Raises:
            ValueError: If cascade=False and node has children
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found for deletion")
            return False

        children = self.get_children(node_id)
        # Check for children if not cascading
        if not cascade:
            if children:
                raise ValueError(
                    f"Cannot delete node {node_id}: has {len(children)} children. "
                    f"Use cascade=True to delete children as well."
                )

        descendants = self.get_descendants(node_id)

        # Get path before deletion for logging
        node_path = self.get_full_path(node_id)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for descendant in descendants:
                child_path = self.get_full_path(descendant["node_id"])
                cursor.execute("DELETE FROM subject_nodes WHERE node_id = ?", (descendant["node_id"],))
                logger.info(f"Deleted child node {descendant['node_id']} ({child_path})")
            cursor.execute("DELETE FROM subject_nodes WHERE node_id = ?", (node_id,))
            conn.commit()
            logger.info(f"Deleted node {node_id} ({node_path})" + (" with cascade" if cascade else ""))

            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete node {node_id}: {e}")
            raise
        finally:
            conn.close()

    # ========== Tree Traversal ==========

    def get_children(self, parent_id: Optional[int]) -> List[Dict[str, Any]]:
        """Get direct children of a node.

        Args:
            parent_id: Parent node ID (None for root nodes)

        Returns:
            List of child nodes
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if parent_id is None:
                cursor.execute(
                    """
                    SELECT * FROM subject_nodes
                    WHERE parent_id IS NULL
                    ORDER BY name
                """
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM subject_nodes
                    WHERE parent_id = ?
                    ORDER BY name
                """,
                    (parent_id,),
                )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        finally:
            conn.close()

    def get_descendants(self, node_id: int) -> List[Dict[str, Any]]:
        """Get all descendants (recursive) of a node.

        Args:
            node_id: Root node ID

        Returns:
            List of all descendant nodes (depth-first order)
        """
        descendants = []
        children = self.get_children(node_id)

        for child in children:
            descendants.append(child)
            # Recursively get grandchildren
            descendants.extend(self.get_descendants(child["node_id"]))

        return descendants

    def get_ancestors(self, node_id: int) -> List[Dict[str, Any]]:
        """Get all ancestors from node to root.

        Args:
            node_id: Node ID

        Returns:
            List of ancestor nodes from immediate parent to root
        """
        ancestors = []
        current = self.get_node(node_id)

        while current and current["parent_id"] is not None:
            parent = self.get_node(current["parent_id"])
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors

    def get_matched_children_id(self, subject_path: List[str] = None, descendant: bool = True) -> Optional[List[int]]:
        """Get all node IDs for a subject path including its descendants.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            descendant: collect all child nodes

        Returns:
            List of node IDs including the target path and all its descendants,
            or None if the path doesn't exist
        """
        if not subject_path:
            subject_path = ["*"]

        tree = self.get_tree_structure()
        result: List[int] = []

        def collect_all(node: Dict):
            result.append(node["node_id"])
            if descendant:
                for child in node.get("children", {}).values():
                    collect_all(child)

        def dfs(nodes: Dict, level: int):
            if level == len(subject_path):
                return

            pat = subject_path[level]

            for key, node in nodes.items():
                if fnmatch.fnmatch(key, pat):
                    if level == len(subject_path) - 1:
                        collect_all(node)
                    else:
                        dfs(node.get("children", {}), level + 1)

        dfs(tree, 0)
        return result

    def get_full_path(self, node_id: int) -> List[str]:
        """Get full path by traversing ancestors.

        Args:
            node_id: Node ID

        Returns:
            Path components like ['Finance', 'Revenue', 'Q1'] or empty list if not found
        """
        node = self.get_node(node_id)
        if not node:
            return []

        # Build path from ancestors
        ancestors = self.get_ancestors(node_id)
        ancestors.reverse()  # Root to parent order

        # Build path components
        components = [a["name"] for a in ancestors]
        components.append(node["name"])

        return components

    # ========== Tree Building ==========

    def get_tree_structure(self, root_id: Optional[int] = None) -> Dict[str, Any]:
        """Build nested tree structure with node_id, name, and children.

        Args:
            root_id: Root node ID (None for entire forest)

        Returns:
            Nested dict structure where each node has node_id, name, and children.
            If root_id is None, returns all root trees merged.
            Example: {
                "Finance": {"node_id": 1, "name": "Finance", "children": {
                    "Revenue": {"node_id": 2, "name": "Revenue", "children": {
                        "Q1": {"node_id": 3, "name": "Q1", "children": {}},
                        "Q2": {"node_id": 4, "name": "Q2", "children": {}}
                    }}
                }}
            }
        """
        roots = []
        if root_id is None:
            # Get all root nodes
            roots = self.get_children(None)
            if not roots:
                return {}
        else:
            root = self.get_node(root_id)
            if not root:
                return {}
            roots = [root]

        result = {}
        for root in roots:
            root_dict = self._build_path_tree_recursive(root)
            result[root_dict["name"]] = root_dict
        return result

    def get_simple_tree_structure(self, root_id: Optional[int] = None) -> Dict[str, Any]:
        """Build nested tree structure with node names as keys.

        Args:
            root_id: Root node ID (None for entire forest)

        Returns:
            Nested dict with node names as keys.
            If root_id is None, returns all root trees merged.
            Example: {"Finance": {"Revenue": {"Q1": {}, "Q2": {}}}}
        """
        tree_structure = self.get_tree_structure(root_id)

        def dfs(nodes: Dict) -> Dict:
            result = {}
            for key, node in nodes.items():
                children = node.get("children", {})
                result[key] = dfs(children)
            return result

        return dfs(tree_structure)

    def _build_path_tree_recursive(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build path tree structure with node_id, name, and children."""
        children = self.get_children(node["node_id"])

        if children:
            child_dict = {}
            for child in children:
                child_node = self._build_path_tree_recursive(child)
                child_dict[child_node["name"]] = child_node
        else:
            child_dict = {}

        return {"node_id": node["node_id"], "name": node["name"], "children": child_dict}

    def find_or_create_path(self, path_components: List[str]) -> int:
        """Find or create nodes along a path.

        Args:
            path_components: List of node names from root to leaf
                           Example: ['Finance', 'Revenue', 'Q1']

        Returns:
            Leaf node ID (last component)

        Raises:
            ValueError: If path is empty or would exceed max depth

        Example:
            node_id = store.find_or_create_path(['Finance', 'Revenue', 'Q1'])
            # Creates Finance -> Revenue -> Q1 if not exists
            # Returns Q1's node_id
        """
        if not path_components:
            raise ValueError("Path components cannot be empty")

        parent_id = None

        for component in path_components:
            # Try to find existing node
            node = self._find_child_by_name(parent_id, component)

            if node:
                # Node exists, continue
                parent_id = node["node_id"]
            else:
                # Create new node
                created = self.create_node(parent_id=parent_id, name=component, description="")
                parent_id = created["node_id"]

        return parent_id

    def _find_child_by_name(self, parent_id: Optional[int], name: str) -> Optional[Dict[str, Any]]:
        """Find a child node by name under a specific parent."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if parent_id is None:
                cursor.execute(
                    """
                    SELECT * FROM subject_nodes
                    WHERE parent_id IS NULL AND name = ?
                """,
                    (name,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM subject_nodes
                    WHERE parent_id = ? AND name = ?
                """,
                    (parent_id, name),
                )

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

        finally:
            conn.close()

    def rename(self, old_path: List[str], new_path: List[str]) -> bool:
        """Rename a subject node by moving it to a new path.

        This method can either:
        1. Rename a node (keeping the same parent but changing the name)
        2. Move a node to a different parent
        3. Both rename and move a node

        Args:
            old_path: Current path of the node to rename (e.g., ['Finance', 'Revenue', 'Q1'])
            new_path: New path for the node (e.g., ['Finance', 'Revenue', 'Quarter1'])

        Returns:
            True if renamed successfully, False if old_path not found

        Raises:
            ValueError: If validation fails (e.g., new path would create cycle,
                       duplicate name under parent, or old_path doesn't exist)

        Examples:
            # Rename Q1 to Quarter1 (same parent)
            store.rename(['Finance', 'Revenue', 'Q1'], ['Finance', 'Revenue', 'Quarter1'])

            # Move Q1 to different parent
            store.rename(['Finance', 'Revenue', 'Q1'], ['Finance', 'Costs', 'Q1'])

            # Both rename and move
            store.rename(['Finance', 'Revenue', 'Q1'], ['Finance', 'Costs', 'Quarter1'])
        """
        if not old_path:
            raise ValueError("Old path cannot be empty")

        if not new_path:
            raise ValueError("New path cannot be empty")

        # Find the node to rename
        old_node = self.get_node_by_path(old_path)
        if not old_node:
            raise ValueError(f"Node not found at path: {'/'.join(old_path)}")

        # Extract new parent path and new name
        new_parent_path = new_path[:-1] if len(new_path) > 1 else []
        new_name = new_path[-1]

        # Find the new parent node
        new_parent_id = None
        if new_parent_path:
            new_parent = self.get_node_by_path(new_parent_path)
            if not new_parent:
                raise ValueError(f"New parent not found at path: {'/'.join(new_parent_path)}")
            new_parent_id = new_parent["node_id"]

        # Validate new name
        if not new_name or not new_name.strip():
            raise ValueError("New name cannot be empty")
        new_name = new_name.strip()

        # Check if this would create a cycle when moving
        if new_parent_id is not None and not self.validate_no_cycle(old_node["node_id"], new_parent_id):
            raise ValueError("Moving node to new parent would create a cycle")

        # Check for duplicate name under new parent (unless we're keeping the same node)
        if new_parent_id != old_node["parent_id"] or new_name != old_node["name"]:
            existing_node = self._find_child_by_name(new_parent_id, new_name)
            if existing_node and existing_node["node_id"] != old_node["node_id"]:
                raise ValueError(f"Node with name '{new_name}' already exists under parent {new_parent_id}")

        # Use update_node to perform the rename/move
        # For parent_id, we need to be explicit about the change since None means "move to root"
        name_changed = new_name != old_node["name"]
        parent_changed = new_parent_id != old_node["parent_id"]

        success = self.update_node(
            node_id=old_node["node_id"],
            name=new_name if name_changed else None,
            parent_id=-1 if parent_changed and new_parent_id is None else (new_parent_id if parent_changed else None),
        )

        if success:
            logger.info(f"Renamed node from {'/'.join(old_path)} to {'/'.join(new_path)}")

        return success

    def validate_no_cycle(self, node_id: int, new_parent_id: int) -> bool:
        """Ensure moving node to new_parent_id doesn't create cycle.

        Args:
            node_id: Node to move
            new_parent_id: Proposed new parent

        Returns:
            True if no cycle, False if cycle would be created
        """
        # Check if new_parent_id is node_id itself
        if node_id == new_parent_id:
            return False

        # Check if new_parent_id is a descendant of node_id
        descendants = self.get_descendants(node_id)
        descendant_ids = {d["node_id"] for d in descendants}

        return new_parent_id not in descendant_ids


def base_schema_columns() -> List:
    return [
        pa.field(NAME_COLUMN_NAME, pa.string()),
        pa.field(SUBJECT_ID_COLUMN_NAME, pa.int64()),
        pa.field(CREATED_AT_COLUMN_NAME, pa.string()),
    ]


class BaseSubjectEmbeddingStore(BaseEmbeddingStore):
    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: EmbeddingModel,
        on_duplicate_columns: str = "vector",
        schema: Optional[Union[pa.Schema, LanceModel]] = None,
        vector_source_name: str = "definition",
        vector_column_name: str = "vector",
    ):
        super().__init__(
            db_path, table_name, embedding_model, on_duplicate_columns, schema, vector_source_name, vector_column_name
        )

        # Initialize SubjectTreeStore for managing subject hierarchy
        self.subject_tree = SubjectTreeStore(db_path)

    def batch_store(
        self,
        items: List[Dict[str, Any]],
    ) -> None:
        """Generic batch processing with subject_path conversion.

        Args:
            items: List of items to store
        """
        if not items:
            return

        # Process all items to convert subject_path to subject_node_id
        batch_data = []
        for item in items:
            subject_path = item.get("subject_path", [])

            # Validate required fields (will be overridden by subclasses if needed)
            if not subject_path:
                logger.warning(f"Skipping item with missing subject_path: {item}")
                continue

            # Find or create the subject tree path to get node_id
            try:
                subject_node_id = self.subject_tree.find_or_create_path(subject_path)

                # Create new item dict without subject_path field and with subject_node_id
                processed_item = item.copy()
                processed_item[SUBJECT_ID_COLUMN_NAME] = subject_node_id
                processed_item.pop(SUBJECT_PATH_COLUMN_NAME, None)

                # Auto-generate timestamp if needed
                if CREATED_AT_COLUMN_NAME not in processed_item:
                    processed_item[CREATED_AT_COLUMN_NAME] = self._get_current_timestamp()

                batch_data.append(processed_item)

            except Exception as e:
                logger.error(f"Failed to process item with subject_path '{subject_path}': {str(e)}")
                continue

        # Store the batch using the parent class method
        if batch_data:
            self.store_batch(batch_data)
            logger.info(f"Successfully stored {len(batch_data)} items in batch")

    def batch_upsert(
        self,
        items: List[Dict[str, Any]],
        on_column: str = "id",
    ) -> None:
        """Generic batch upsert with subject_path conversion (update if key exists, insert if not).

        Args:
            items: List of items to upsert
            on_column: Column name to match for deduplication (default: "id")
        """
        if not items:
            return

        # Process all items to convert subject_path to subject_node_id
        batch_data = []
        for item in items:
            subject_path = item.get("subject_path", [])

            # Validate required fields
            if not subject_path:
                logger.warning(f"Skipping item with missing subject_path: {item}")
                continue

            # Find or create the subject tree path to get node_id
            try:
                subject_node_id = self.subject_tree.find_or_create_path(subject_path)

                # Create new item dict without subject_path field and with subject_node_id
                processed_item = item.copy()
                processed_item[SUBJECT_ID_COLUMN_NAME] = subject_node_id
                processed_item.pop(SUBJECT_PATH_COLUMN_NAME, None)

                # Auto-generate timestamp if needed
                if CREATED_AT_COLUMN_NAME not in processed_item:
                    processed_item[CREATED_AT_COLUMN_NAME] = self._get_current_timestamp()

                batch_data.append(processed_item)

            except Exception as e:
                logger.error(f"Failed to process item with subject_path '{subject_path}': {str(e)}")
                continue

        # Upsert the batch using the parent class method
        if batch_data:
            self.upsert_batch(batch_data, on_column=on_column)
            logger.info(f"Successfully upserted {len(batch_data)} items in batch")

    def search_with_subject_filter(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
        selected_fields: Optional[List[str]] = None,
        name_field: Optional[str] = "name",
        additional_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Generic search with subject filtering supporting both exact path and parent+name patterns.

        Args:
            query_text: Query text for vector search
            subject_path: Subject hierarchy path
            top_n: Number of results to return
            selected_fields: List of fields to return
            name_field: Field name for parent+name matching (e.g., "search_text", "name")
            additional_conditions: Additional filter conditions

        Returns:
            List of matching items with subject_path enriched
        """
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        # Set up path filters for both exact path and parent+name matching
        path_filter = [(subject_path, "")]
        if subject_path and len(subject_path) > 1:
            path_filter.append((subject_path[:-1], subject_path[-1]))

        results = []
        for path, name in path_filter:
            conditions = additional_conditions.copy() if additional_conditions else []

            # Convert path to include all descendant node_ids if provided
            if path:
                # Get all node IDs including the path and its descendants
                node_ids = self.subject_tree.get_matched_children_id(path, False if name else True)
                if node_ids:
                    subject_condition = in_(SUBJECT_ID_COLUMN_NAME, node_ids)
                    conditions.append(subject_condition)
                else:
                    continue

            # Add name field condition if provided (for parent+name matching)
            if name and name_field:
                name_condition = like(name_field, name)
                conditions.append(name_condition)

            # Build where clause
            where = None if len(conditions) == 0 else and_(*conditions)
            if selected_fields and SUBJECT_ID_COLUMN_NAME not in selected_fields:
                selected_fields = [SUBJECT_ID_COLUMN_NAME] + selected_fields
            # Perform search
            if query_text:
                search_result = self.search(
                    query_txt=query_text,
                    select_fields=selected_fields,
                    top_n=top_n,
                    where=where,
                )
            else:
                search_result = self._search_all(where=where, select_fields=selected_fields)

            # Enrich with subject_path - this adds subject_path field to all results
            result_list = search_result.to_pylist()
            result_list = self._enrich_with_subject_path(result_list)
            if len(result_list) > 0:
                results.extend(result_list)

        # Ensure all results have subject_path field (it's added during enrichment)
        return results

    def create_subject_index(self) -> None:
        """Create scalar index on subject_node_id field."""
        self._ensure_table_ready()
        try:
            self.table.create_scalar_index(SUBJECT_ID_COLUMN_NAME, replace=True)
            logger.info(f"Created scalar index on subject_node_id for {self.table_name}")
        except Exception as e:
            logger.warning(f"Failed to create scalar index for {self.table_name}: {str(e)}")

    def get_subject_tree_flat(self) -> List[str]:
        """Get flat list of all subject paths from the tree structure."""

        def flatten_tree(tree: Dict[str, Any], prefix: str = "") -> List[str]:
            paths = []
            for name, children in tree.items():
                current_path = f"{prefix}/{name}" if prefix else name
                paths.append(current_path)
                if children:
                    paths.extend(flatten_tree(children, current_path))
            return paths

        structure = self.subject_tree.get_simple_tree_structure()
        return flatten_tree(structure)

    def get_subject_tree_str(self) -> str:
        """Get subject tree structure as indented string.

        Returns a string representation of the tree structure with 2-space indentation.

        Returns:
            String representation of the tree structure

        Example:
            Finance
              Revenue
                Q1
                Q2
              Costs
        """

        def tree_to_str(tree: Dict[str, Any], indent: int = 0) -> str:
            lines = []
            for name, children in sorted(tree.items()):
                # Add current node with indentation (2 spaces per level)
                lines.append("  " * indent + name)
                # Recursively add children
                if children:
                    lines.append(tree_to_str(children, indent + 1))
            return "\n".join(lines)

        structure = self.subject_tree.get_simple_tree_structure()
        if not structure:
            return ""
        return tree_to_str(structure)

    def _enrich_with_subject_path(self, results: List[Dict]) -> List[Dict]:
        """Enrich results with subject_path from SubjectTreeStore.

        Args:
            results: PyArrow table with subject_node_id

        Returns:
            PyArrow table with subject_path field added
        """
        if len(results) == 0:
            return results

        for result in results:
            node_id = result.get(SUBJECT_ID_COLUMN_NAME)
            if node_id:
                result[SUBJECT_PATH_COLUMN_NAME] = self.subject_tree.get_full_path(node_id)
            else:
                result[SUBJECT_PATH_COLUMN_NAME] = []
            result.pop(SUBJECT_ID_COLUMN_NAME, None)

        return results

    def rename(self, old_path: List[str], new_path: List[str]) -> bool:
        """Rename a subject node or a storage entry.

        This method handles multiple scenarios:

        1. Subject rename: old_path is a subject node path
           - Renames/moves the subject node in the tree
           - Example: ['Finance', 'Revenue', 'Q1'] -> ['Finance', 'Revenue', 'Quarter1']

        2. Storage entry rename/move: old_path is subject_path + '/' + name
           a. Rename only (subject_path unchanged, name changed):
              - Example: ['Finance', 'Revenue', 'old_metric'] -> ['Finance', 'Revenue', 'new_metric']
           b. Move only (subject_path changed, name unchanged):
              - Example: ['Finance', 'Revenue', 'metric'] -> ['Finance', 'Costs', 'metric']
              - Updates subject_node_id to point to new parent
           c. Both rename and move:
              - Example: ['Finance', 'Revenue', 'old'] -> ['Finance', 'Costs', 'new']
              - Performs both operations above

        Args:
            old_path: Current path (e.g., ['Finance', 'Revenue', 'Q1'] or ['Finance', 'Revenue', 'metric_name'])
            new_path: New path (e.g., ['Finance', 'Revenue', 'Quarter1'] or ['Finance', 'Costs', 'new_metric'])

        Returns:
            True if renamed successfully

        Raises:
            ValueError: If validation fails

        Examples:
            # Case 1: Rename a subject node
            store.rename(['Finance', 'Revenue', 'Q1'], ['Finance', 'Revenue', 'Quarter1'])

            # Case 2a: Rename a storage entry (same parent)
            store.rename(['Finance', 'Revenue', 'old_metric'], ['Finance', 'Revenue', 'new_metric'])

            # Case 2b: Move a storage entry (same name, different parent)
            store.rename(['Finance', 'Revenue', 'metric'], ['Finance', 'Costs', 'metric'])

            # Case 2c: Rename and move a storage entry
            store.rename(['Finance', 'Revenue', 'old'], ['Finance', 'Costs', 'new'])
        """
        if not old_path or not new_path:
            raise ValueError("Paths cannot be empty")

        # Case 1: Check if old_path exists in subject_tree (subject node)
        old_node = self.subject_tree.get_node_by_path(old_path)
        if old_node:
            logger.info(f"Renaming subject node from {'/'.join(old_path)} to {'/'.join(new_path)}")
            return self.subject_tree.rename(old_path, new_path)

        # Case 2: old_path is a storage entry (subject_path + '/' + name)
        if len(old_path) < 2:
            raise ValueError(
                f"Storage entry path must have at least 2 components (subject_path + name), "
                f"got: {'/'.join(old_path)}"
            )

        # Extract components
        old_parent_path = old_path[:-1]
        new_parent_path = new_path[:-1]
        old_name = old_path[-1]
        new_name = new_path[-1]

        # Get old subject_node_id
        if old_parent_path:
            old_parent_node = self.subject_tree.get_node_by_path(old_parent_path)
            if not old_parent_node:
                raise ValueError(f"Subject path not found: {'/'.join(old_parent_path)}")
            old_subject_node_id = old_parent_node["node_id"]
        else:
            raise ValueError("Storage entries must have a subject path")

        # Prepare update values
        update_values = {}
        name_changed = old_name != new_name
        parent_changed = old_parent_path != new_parent_path
        new_subject_node_id = None  # Initialize for conflict check

        # Case 2a: Name changed - update name field
        if name_changed:
            update_values["name"] = new_name

        # Case 2b/2c: Parent changed - update subject_node_id
        if parent_changed:
            if new_parent_path:
                new_parent_node = self.subject_tree.get_node_by_path(new_parent_path)
                if not new_parent_node:
                    raise ValueError(f"New subject path not found: {'/'.join(new_parent_path)}")
                new_subject_node_id = new_parent_node["node_id"]
            else:
                raise ValueError("Storage entries must have a subject path")

            update_values[SUBJECT_ID_COLUMN_NAME] = new_subject_node_id

        # Check if there's anything to update
        if not update_values:
            logger.info(f"No changes needed for path: {'/'.join(old_path)}")
            return True

        # Build where clause to find the storage entry
        self._ensure_table_ready()
        from datus.storage.lancedb_conditions import and_, build_where, eq

        where_condition = and_(eq(SUBJECT_ID_COLUMN_NAME, old_subject_node_id), eq("name", old_name))

        # Check for conflicts when both name and parent are changing
        if name_changed and parent_changed:
            # Check if target (new_subject_node_id + new_name) already exists
            conflict_condition = and_(eq(SUBJECT_ID_COLUMN_NAME, new_subject_node_id), eq("name", new_name))
            conflict_where = build_where(conflict_condition)
            existing_count = self.table.count_rows(conflict_where)
            if existing_count > 0:
                raise ValueError(
                    f"Storage entry with name '{new_name}' already exists "
                    f"under subject path: {'/'.join(new_parent_path)}"
                )

        # Perform the update
        self.update(where=where_condition, update_values=update_values)

        # Log what was done
        if name_changed and parent_changed:
            logger.info(f"Renamed and moved storage entry from '{'/'.join(old_path)}' to '{'/'.join(new_path)}'")
        elif name_changed:
            logger.info(f"Renamed storage entry from '{'/'.join(old_path)}' to '{'/'.join(new_path)}'")
        elif parent_changed:
            logger.info(f"Moved storage entry from '{'/'.join(old_path)}' to '{'/'.join(new_path)}'")

        return True

    def update_entry(self, subject_path: List[str], name: str, update_values: Dict[str, Any]) -> bool:
        """Update fields for a storage entry, excluding subject_node_id and name.

        This method allows updating any fields of an entry except subject_node_id and name.
        To change subject_path or name, use the rename() method instead.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the entry to update
            update_values: Dictionary of field names and new values to update

        Returns:
            True if update successful

        Raises:
            ValueError: If trying to update subject_node_id or name,
                       if entry not found, or if validation fails

        Examples:
            # Update description field
            store.update_entry(
                subject_path=['Finance', 'Revenue'],
                name='sales_metric',
                update_values={'description': 'Updated sales metrics for Q1'}
            )

            # Update multiple fields at once
            store.update_entry(
                subject_path=['Finance', 'Revenue'],
                name='sales_metric',
                update_values={
                    'description': 'New description',
                    'data_type': 'numeric',
                    'unit': 'USD'
                }
            )
        """
        if not subject_path:
            raise ValueError("subject_path cannot be empty")

        if not name or not name.strip():
            raise ValueError("name cannot be empty")

        if not update_values:
            raise ValueError("update_values cannot be empty")

        # Check for forbidden fields
        forbidden_fields = {SUBJECT_ID_COLUMN_NAME, NAME_COLUMN_NAME}
        forbidden_in_update = forbidden_fields.intersection(update_values.keys())
        if forbidden_in_update:
            raise ValueError(
                f"Cannot update fields: {', '.join(forbidden_in_update)}. "
                f"Use rename() method to change subject_path or name."
            )

        # Find subject_node_id from subject_path
        subject_node = self.subject_tree.get_node_by_path(subject_path)
        if not subject_node:
            raise ValueError(f"Subject path not found: {'/'.join(subject_path)}")

        subject_node_id = subject_node["node_id"]

        # Build where clause to locate the entry
        self._ensure_table_ready()
        from datus.storage.lancedb_conditions import and_, build_where, eq

        where_condition = and_(eq(SUBJECT_ID_COLUMN_NAME, subject_node_id), eq(NAME_COLUMN_NAME, name.strip()))

        # Check if entry exists
        where_clause = build_where(where_condition)
        count = self.table.count_rows(where_clause)
        if count == 0:
            raise ValueError(f"Entry not found: name='{name}' under subject_path={'/'.join(subject_path)}")

        # Perform update
        self.update(where=where_condition, update_values=update_values)

        logger.info(
            f"Updated entry '{name}' under subject_path={'/'.join(subject_path)} "
            f"with fields: {', '.join(update_values.keys())}"
        )

        return True

    def list_entries(
        self, node_id: int, name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get storage entries by subject node ID and entry name.

        This is a generic method for retrieving entries from storage using
        subject_node_id and name fields. Commonly used for fetching specific
        entries like metrics or SQL by their location in the subject tree.

        Args:
            node_id: Subject node ID (parent node in the subject tree)
            name: Entry name (e.g., metric name or SQL name)
            limit: Maximum number of results to return (default: None)

        Returns:
            List of entries matching the criteria, enriched with subject_path

        Examples:
            # Fetch a specific metric under a subject node
            metrics = store.get_entries_by_node_id_and_name(
                node_id=42,
                name='revenue_total'
            )

            # Fetch a specific SQL entry
            sql_entries = store.get_entries_by_node_id_and_name(
                node_id=42,
                name='daily_sales_query'
            )
        """
        try:
            from datus.storage.lancedb_conditions import eq

            # Ensure table is ready
            self._ensure_table_ready()

            # Build where clause
            conditions = [eq(SUBJECT_ID_COLUMN_NAME, node_id)]

            if name:
                conditions.append(eq(NAME_COLUMN_NAME, name))

            where_clause = None if len(conditions) == 0 else and_(*conditions)
            # Execute search
            results = self._search_all(where=where_clause, limit=limit).to_pylist()

            # Enrich with subject_path
            return self._enrich_with_subject_path(results)

        except Exception as e:
            logger.error(f"Failed to fetch entries by node_id={node_id}: {e}")
            return []
