"""RDB-backed storage for hierarchical subject taxonomy tree.

This module implements a tree structure using the adjacency list model,
using the pluggable RDB backend abstraction.
"""

import fnmatch
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pyarrow as pa
from datus_storage_base.conditions import and_, in_, like
from datus_storage_base.rdb.base import ColumnDef, IndexDef, TableDefinition, UniqueViolationError, WhereOp

from datus.storage import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

SUBJECT_ID_COLUMN_NAME = "subject_node_id"
SUBJECT_PATH_COLUMN_NAME = "subject_path"
NAME_COLUMN_NAME = "name"
CREATED_AT_COLUMN_NAME = "created_at"
ROOT_PARENT_ID = -1  # Used instead of NULL to ensure UNIQUE constraint works for root nodes

_SUBJECT_NODES_TABLE = TableDefinition(
    table_name="subject_nodes",
    columns=[
        ColumnDef(name="node_id", col_type="INTEGER", primary_key=True, autoincrement=True),
        ColumnDef(name="parent_id", col_type="INTEGER"),
        ColumnDef(name="name", col_type="TEXT", nullable=False),
        ColumnDef(name="description", col_type="TEXT", default=""),
        ColumnDef(name="created_at", col_type="TEXT", nullable=False),
        ColumnDef(name="updated_at", col_type="TEXT", nullable=False),
        ColumnDef(name="datasource_id", col_type="TEXT", default=""),
    ],
    indices=[
        IndexDef(name="idx_subject_parent_id", columns=["parent_id"]),
        IndexDef(name="idx_subject_parent_name_ds", columns=["parent_id", "name", "datasource_id"], unique=True),
        IndexDef(name="idx_subject_datasource_id", columns=["datasource_id"]),
    ],
    constraints=["UNIQUE(parent_id, name, datasource_id)"],
)


@dataclass
class SubjectNodeRecord:
    """Typed record for the subject_nodes table."""

    node_id: Optional[int] = None
    parent_id: int = -1  # ROOT_PARENT_ID
    name: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    datasource_id: str = ""


def _node_to_dict(record: SubjectNodeRecord) -> Dict[str, Any]:
    """Convert a SubjectNodeRecord to dict, mapping ROOT_PARENT_ID to None."""
    return {
        "node_id": record.node_id,
        "parent_id": None if record.parent_id == ROOT_PARENT_ID else record.parent_id,
        "name": record.name,
        "description": record.description,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "datasource_id": record.datasource_id,
    }


class SubjectTreeStore:
    """RDB-backed storage for hierarchical subject taxonomy tree.

    Implements adjacency list model for tree structure.
    """

    def __init__(self, namespace: str = ""):
        """Initialize SubjectTreeStore.

        Args:
            namespace: Namespace for RDB path isolation.  In PHYSICAL mode
                each namespace gets its own SQLite file; in LOGICAL mode
                all share one file.

        Reads ``table_prefix``, ``extra_fields``, and ``scope_indices`` from
        the storage registry defaults (set via ``configure_storage_defaults()``).
        """
        from datus.storage.backend_holder import create_rdb_for_store
        from datus.storage.registry import get_storage_defaults

        defaults = get_storage_defaults()
        table_prefix = defaults.get("table_prefix", "")
        scope_indices = defaults.get("scope_indices", [])
        extra_pa_fields = defaults.get("extra_fields")

        # Build table definition with optional prefix, extra columns, and scope indices
        table_def = TableDefinition(
            table_name=f"{table_prefix}{_SUBJECT_NODES_TABLE.table_name}"
            if table_prefix
            else _SUBJECT_NODES_TABLE.table_name,
            columns=list(_SUBJECT_NODES_TABLE.columns),
            indices=list(_SUBJECT_NODES_TABLE.indices),
            constraints=list(_SUBJECT_NODES_TABLE.constraints),
        )
        if extra_pa_fields:
            extra_cols = [ColumnDef(name=f.name, col_type="TEXT", default="") for f in extra_pa_fields]
            table_def.columns = table_def.columns + extra_cols
        if scope_indices:
            existing_idx_names = {idx.name for idx in table_def.indices}
            for col in scope_indices:
                idx_name = f"idx_subject_{col}"
                if idx_name not in existing_idx_names:
                    table_def.indices = table_def.indices + [IndexDef(name=idx_name, columns=[col])]

        self._rdb = create_rdb_for_store("subject_tree", namespace=namespace)
        self._table = self._rdb.ensure_table(table_def)
        self._migrate_null_parents()

        logger.info("SubjectTreeStore initialized")

    def _migrate_null_parents(self):
        """Migrate existing NULL parent_id values to ROOT_PARENT_ID (-1)."""
        self._table.update(
            {"parent_id": ROOT_PARENT_ID},
            where=[("parent_id", WhereOp.IS_NULL, None)],
        )

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
        """
        if not name or not name.strip():
            raise ValueError("Node name cannot be empty")

        name = name.strip()

        if parent_id is not None:
            parent = self.get_node(parent_id)
            if not parent:
                raise ValueError(f"Parent node {parent_id} not found")

        existing_node = self._find_child_by_name(parent_id, name)
        if existing_node:
            raise ValueError(f"Node with name '{name}' already exists under parent {parent_id}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_parent_id = ROOT_PARENT_ID if parent_id is None else parent_id

        try:
            record = SubjectNodeRecord(
                parent_id=db_parent_id,
                name=name,
                description=description,
                created_at=now,
                updated_at=now,
            )
            node_id = self._table.insert(record)

            created_node = self.get_node(node_id)
            logger.info(f"Created node: {self.get_full_path(node_id)} (node_id={node_id})")
            return created_node

        except UniqueViolationError as e:
            raise ValueError(f"Node with name '{name}' already exists under parent {parent_id}") from e
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            Node dict or None if not found
        """
        rows = self._table.query(SubjectNodeRecord, where={"node_id": node_id})
        if rows:
            return _node_to_dict(rows[0])
        return None

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
        """Update node fields."""
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found for update")
            return False

        data: Dict[str, Any] = {}

        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Node name cannot be empty")
            data["name"] = name

        if description is not None:
            data["description"] = description

        if parent_id is not None:
            if parent_id != ROOT_PARENT_ID and not self.validate_no_cycle(node_id, parent_id):
                raise ValueError(f"Moving node {node_id} to parent {parent_id} would create a cycle")
            if parent_id != ROOT_PARENT_ID:
                parent = self.get_node(parent_id)
                if not parent:
                    raise ValueError(f"Parent node {parent_id} not found")
            data["parent_id"] = parent_id

        if not data:
            logger.debug(f"No fields to update for node {node_id}")
            return True

        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self._table.update(data, where={"node_id": node_id})
            logger.info(f"Updated node {node_id}")
            return True
        except UniqueViolationError as e:
            raise ValueError(f"Node with name '{name}' already exists under parent {parent_id}") from e
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            raise

    def delete_node(self, node_id: int, cascade: bool = True) -> bool:
        """Delete node."""
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found for deletion")
            return False

        children = self.get_children(node_id)
        if not cascade:
            if children:
                raise ValueError(
                    f"Cannot delete node {node_id}: has {len(children)} children. "
                    f"Use cascade=True to delete children as well."
                )

        descendants = self.get_descendants(node_id)
        node_path = self.get_full_path(node_id)

        try:
            with self._rdb.transaction():
                for descendant in descendants:
                    child_path = self.get_full_path(descendant["node_id"])
                    self._table.delete(where={"node_id": descendant["node_id"]})
                    logger.info(f"Deleted child node {descendant['node_id']} ({child_path})")
                self._table.delete(where={"node_id": node_id})
            logger.info(f"Deleted node {node_id} ({node_path})" + (" with cascade" if cascade else ""))
            return True
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            raise

    # ========== Tree Traversal ==========

    def get_children(self, parent_id: Optional[int]) -> List[Dict[str, Any]]:
        """Get direct children of a node."""
        db_parent_id = ROOT_PARENT_ID if parent_id is None else parent_id
        where = {"parent_id": db_parent_id}
        rows = self._table.query(
            SubjectNodeRecord,
            where=where,
            order_by=["name"],
        )
        return [_node_to_dict(row) for row in rows]

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

        Handles race conditions in parallel writes: if create_node raises
        ValueError due to a duplicate, falls back to finding the existing node.

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
                # Create new node, handle race condition with parallel writes
                try:
                    created = self.create_node(parent_id=parent_id, name=component, description="")
                    parent_id = created["node_id"]
                except ValueError:
                    # Race condition: another thread/process created the node between
                    # our _find_child_by_name check and create_node call
                    existing_node = self._find_child_by_name(parent_id, component)
                    if existing_node:
                        parent_id = existing_node["node_id"]
                    else:
                        raise

        return parent_id

    def _find_child_by_name(self, parent_id: Optional[int], name: str) -> Optional[Dict[str, Any]]:
        """Find a child node by name under a specific parent."""
        db_parent_id = ROOT_PARENT_ID if parent_id is None else parent_id
        where = {"parent_id": db_parent_id, "name": name}
        rows = self._table.query(
            SubjectNodeRecord,
            where=where,
        )
        if rows:
            return _node_to_dict(rows[0])
        return None

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
        table_name: str,
        embedding_model: EmbeddingModel,
        on_duplicate_columns: str = "vector",
        schema: Optional[pa.Schema] = None,
        vector_source_name: str = "definition",
        vector_column_name: str = "vector",
        unique_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            table_name=table_name,
            embedding_model=embedding_model,
            on_duplicate_columns=on_duplicate_columns,
            schema=schema,
            vector_source_name=vector_source_name,
            vector_column_name=vector_column_name,
            unique_columns=unique_columns,
            **kwargs,
        )

        # Get per-namespace SubjectTreeStore from registry
        from datus.storage.backend_holder import get_current_namespace
        from datus.storage.registry import get_subject_tree_store

        self.subject_tree = get_subject_tree_store(namespace=get_current_namespace())

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
        self._create_scalar_index(SUBJECT_ID_COLUMN_NAME)

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
                f"Storage entry path must have at least 2 components (subject_path + name), got: {'/'.join(old_path)}"
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
        from datus_storage_base.conditions import and_, eq

        where_condition = and_(eq(SUBJECT_ID_COLUMN_NAME, old_subject_node_id), eq("name", old_name))

        # Check for conflicts when both name and parent are changing
        if name_changed and parent_changed:
            # Check if target (new_subject_node_id + new_name) already exists
            conflict_condition = and_(eq(SUBJECT_ID_COLUMN_NAME, new_subject_node_id), eq("name", new_name))
            existing_count = self._count_rows(conflict_condition)
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

    def update_entry(
        self, subject_path: List[str], name: str, update_values: Dict[str, Any], extra_conditions: Optional[List] = None
    ) -> bool:
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
        from datus_storage_base.conditions import and_, eq

        conditions = [eq(SUBJECT_ID_COLUMN_NAME, subject_node_id), eq(NAME_COLUMN_NAME, name.strip())]
        if extra_conditions:
            conditions.extend(extra_conditions)
        where_condition = and_(*conditions)

        # Check if entry exists
        count = self._count_rows(where_condition)
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
            from datus_storage_base.conditions import eq

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

    def delete_entry(self, subject_path: List[str], name: str, extra_conditions: Optional[List] = None) -> bool:
        """Delete entry by subject_path and name from vector store.

        This is a generic method for deleting entries from storage using
        subject_path and name fields. Subclasses may override this method
        to add additional logic (e.g., yaml file handling for metrics).

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the entry to delete
            extra_conditions: Additional filter conditions

        Returns:
            True if deleted successfully, False if entry not found

        Raises:
            ValueError: If subject_path is empty or name is empty
        """
        if not subject_path:
            raise ValueError("subject_path cannot be empty")

        if not name or not name.strip():
            raise ValueError("name cannot be empty")

        name = name.strip()

        # Find subject_node_id from subject_path
        subject_node = self.subject_tree.get_node_by_path(subject_path)
        if not subject_node:
            logger.warning(f"Subject path not found: {'/'.join(subject_path)}")
            return False

        subject_node_id = subject_node["node_id"]

        # Build where clause to locate the entry
        self._ensure_table_ready()
        from datus_storage_base.conditions import and_, eq

        conditions = [eq(SUBJECT_ID_COLUMN_NAME, subject_node_id), eq(NAME_COLUMN_NAME, name)]
        if extra_conditions:
            conditions.extend(extra_conditions)
        where_condition = and_(*conditions)

        # Check if entry exists
        count = self._count_rows(where_condition)
        if count == 0:
            logger.warning(f"Entry not found: name='{name}' under subject_path={'/'.join(subject_path)}")
            return False

        # Delete the entry
        self._delete_rows(where_condition)

        logger.info(f"Deleted entry '{name}' under subject_path={'/'.join(subject_path)}")

        return True
