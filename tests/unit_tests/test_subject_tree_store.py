"""Unit tests for SubjectTreeStore."""

import os
import shutil
import tempfile

import pytest

from datus.storage.subject_tree.store import SubjectTreeStore


class TestSubjectTreeStore:
    """Test cases for SubjectTreeStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a SubjectTreeStore instance for testing."""
        return SubjectTreeStore(temp_dir)

    # ========== CRUD Operations Tests ==========

    def test_create_root_node(self, store):
        """Test creating root node."""
        node = store.create_node(None, "Finance", "财务领域")

        assert node["node_id"] is not None
        assert node["name"] == "Finance"
        assert node["description"] == "财务领域"
        assert node["parent_id"] is None
        assert store.get_full_path(node["node_id"]) == ["Finance"]
        assert node["created_at"] is not None
        assert node["updated_at"] is not None

    def test_create_child_node(self, store):
        """Test creating child node."""
        root = store.create_node(None, "Finance", "财务领域")
        child = store.create_node(root["node_id"], "Revenue", "收入管理")

        assert child["parent_id"] == root["node_id"]
        assert store.get_full_path(child["node_id"]) == ["Finance", "Revenue"]
        assert child["name"] == "Revenue"

    def test_create_deep_hierarchy(self, store):
        """Test creating multi-level hierarchy."""
        # Create: Finance -> Revenue -> Q1 -> Jan
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")
        jan = store.create_node(q1["node_id"], "Jan")

        assert store.get_full_path(jan["node_id"]) == ["Finance", "Revenue", "Q1", "Jan"]

    def test_duplicate_name_same_parent_fails(self, store):
        """Test that duplicate names under same parent fail."""
        root = store.create_node(None, "Finance")
        store.create_node(root["node_id"], "Revenue")

        with pytest.raises(ValueError, match="already exists"):
            store.create_node(root["node_id"], "Revenue")

    def test_duplicate_name_different_parent_succeeds(self, store):
        """Test that duplicate names under different parents succeed."""
        finance = store.create_node(None, "Finance")
        sales = store.create_node(None, "Sales")

        rev1 = store.create_node(finance["node_id"], "Revenue")
        rev2 = store.create_node(sales["node_id"], "Revenue")

        assert rev1["node_id"] != rev2["node_id"]
        assert store.get_full_path(rev1["node_id"]) == ["Finance", "Revenue"]
        assert store.get_full_path(rev2["node_id"]) == ["Sales", "Revenue"]

    def test_empty_name_fails(self, store):
        """Test that empty name fails."""
        with pytest.raises(ValueError, match="cannot be empty"):
            store.create_node(None, "")

        with pytest.raises(ValueError, match="cannot be empty"):
            store.create_node(None, "   ")

    def test_create_arbitrary_depth(self, store):
        """Test creating arbitrarily deep hierarchy (no depth limit)."""
        parent_id = None

        # Create 20 levels to test unlimited depth
        for i in range(20):
            node = store.create_node(parent_id, f"Level{i}")
            parent_id = node["node_id"]

        # Should succeed - no depth limit
        assert parent_id is not None

    def test_get_node(self, store):
        """Test getting node by ID."""
        created = store.create_node(None, "Finance", "财务领域")

        retrieved = store.get_node(created["node_id"])

        assert retrieved is not None
        assert retrieved["node_id"] == created["node_id"]
        assert retrieved["name"] == "Finance"
        assert retrieved["description"] == "财务领域"

    def test_get_node_not_found(self, store):
        """Test getting non-existent node."""
        node = store.get_node(99999)
        assert node is None

    def test_get_node_by_path(self, store):
        """Test getting node by full path."""
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        # Get by path
        retrieved = store.get_node_by_path(["Finance", "Revenue", "Q1"])

        assert retrieved is not None
        assert retrieved["node_id"] == q1["node_id"]
        assert retrieved["name"] == "Q1"

    def test_get_node_by_path_not_found(self, store):
        """Test getting non-existent path."""
        node = store.get_node_by_path(["NonExistent", "Path"])
        assert node is None

    def test_update_node_name(self, store):
        """Test updating node name."""
        node = store.create_node(None, "Finance", "财务")

        success = store.update_node(node["node_id"], name="Financial")

        assert success is True

        updated = store.get_node(node["node_id"])
        assert updated["name"] == "Financial"
        assert store.get_full_path(updated["node_id"]) == ["Financial"]

    def test_update_node_description(self, store):
        """Test updating node description."""
        node = store.create_node(None, "Finance", "旧描述")

        success = store.update_node(node["node_id"], description="新描述")

        assert success is True

        updated = store.get_node(node["node_id"])
        assert updated["description"] == "新描述"
        assert updated["name"] == "Finance"  # Name unchanged

    def test_update_child_name_updates_paths(self, store):
        """Test that updating child name updates full_path for descendants."""
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        # Update Revenue to Income
        store.update_node(revenue["node_id"], name="Income")

        # Check paths updated
        updated_revenue = store.get_node(revenue["node_id"])
        updated_q1 = store.get_node(q1["node_id"])

        assert store.get_full_path(updated_revenue["node_id"]) == ["Finance", "Income"]
        assert store.get_full_path(updated_q1["node_id"]) == ["Finance", "Income", "Q1"]

    def test_update_node_parent(self, store):
        """Test moving node to different parent."""
        finance = store.create_node(None, "Finance")
        sales = store.create_node(None, "Sales")
        revenue = store.create_node(finance["node_id"], "Revenue")

        # Move Revenue from Finance to Sales
        success = store.update_node(revenue["node_id"], parent_id=sales["node_id"])

        assert success is True

        updated = store.get_node(revenue["node_id"])
        assert updated["parent_id"] == sales["node_id"]
        assert store.get_full_path(updated["node_id"]) == ["Sales", "Revenue"]

    def test_update_node_parent_to_root(self, store):
        """Test moving node to root level."""
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")

        # Move Revenue to root (parent_id = -1 means None)
        success = store.update_node(revenue["node_id"], parent_id=-1)

        assert success is True

        updated = store.get_node(revenue["node_id"])
        assert updated["parent_id"] is None
        assert store.get_full_path(updated["node_id"]) == ["Revenue"]

    def test_update_parent_cycle_detection(self, store):
        """Test that moving node to its own descendant fails."""
        # Create: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        # Try to move Finance under Q1 (would create cycle)
        with pytest.raises(ValueError, match="cycle"):
            store.update_node(finance["node_id"], parent_id=q1["node_id"])

        # Try to move node to itself
        with pytest.raises(ValueError, match="cycle"):
            store.update_node(finance["node_id"], parent_id=finance["node_id"])

    def test_update_node_not_found(self, store):
        """Test updating non-existent node."""
        success = store.update_node(99999, name="NewName")
        assert success is False

    def test_delete_node(self, store):
        """Test deleting a node."""
        node = store.create_node(None, "Finance")

        success = store.delete_node(node["node_id"])

        assert success is True
        assert store.get_node(node["node_id"]) is None

    def test_delete_node_cascade(self, store):
        """Test cascade deletion of node with children."""
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        # Delete Revenue should cascade to Q1
        success = store.delete_node(revenue["node_id"], cascade=True)

        assert success is True
        assert store.get_node(revenue["node_id"]) is None
        assert store.get_node(q1["node_id"]) is None
        assert store.get_node(finance["node_id"]) is not None

    def test_delete_node_with_children_no_cascade_fails(self, store):
        """Test that deleting node with children fails without cascade."""
        finance = store.create_node(None, "Finance")
        store.create_node(finance["node_id"], "Revenue")

        with pytest.raises(ValueError, match="has .* children"):
            store.delete_node(finance["node_id"], cascade=False)

    def test_delete_node_not_found(self, store):
        """Test deleting non-existent node."""
        success = store.delete_node(99999)
        assert success is False

    # ========== Tree Traversal Tests ==========

    def test_get_children(self, store):
        """Test getting child nodes."""
        root = store.create_node(None, "Finance")
        store.create_node(root["node_id"], "Revenue")
        store.create_node(root["node_id"], "Expense")

        children = store.get_children(root["node_id"])

        assert len(children) == 2
        names = {c["name"] for c in children}
        assert names == {"Revenue", "Expense"}

    def test_get_children_root_level(self, store):
        """Test getting root level nodes."""
        store.create_node(None, "Finance")
        store.create_node(None, "Sales")

        roots = store.get_children(None)

        assert len(roots) == 2
        names = {r["name"] for r in roots}
        assert names == {"Finance", "Sales"}

    def test_get_children_no_children(self, store):
        """Test getting children of leaf node."""
        leaf = store.create_node(None, "Finance")

        children = store.get_children(leaf["node_id"])

        assert len(children) == 0

    def test_get_descendants(self, store):
        """Test recursive descendants query."""
        # Build: Finance -> Revenue -> Q1, Q2
        #                -> Expense -> Fixed, Variable
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        expense = store.create_node(finance["node_id"], "Expense")
        store.create_node(revenue["node_id"], "Q1")
        store.create_node(revenue["node_id"], "Q2")
        store.create_node(expense["node_id"], "Fixed")
        store.create_node(expense["node_id"], "Variable")

        descendants = store.get_descendants(finance["node_id"])

        assert len(descendants) == 6
        names = {d["name"] for d in descendants}
        assert names == {"Revenue", "Expense", "Q1", "Q2", "Fixed", "Variable"}

    def test_get_descendants_leaf_node(self, store):
        """Test descendants of leaf node."""
        leaf = store.create_node(None, "Finance")

        descendants = store.get_descendants(leaf["node_id"])

        assert len(descendants) == 0

    def test_get_ancestors(self, store):
        """Test getting ancestors."""
        # Build: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        ancestors = store.get_ancestors(q1["node_id"])

        assert len(ancestors) == 2
        # Should be in order from immediate parent to root
        assert ancestors[0]["name"] == "Revenue"
        assert ancestors[1]["name"] == "Finance"

    def test_get_ancestors_root_node(self, store):
        """Test ancestors of root node."""
        root = store.create_node(None, "Finance")

        ancestors = store.get_ancestors(root["node_id"])

        assert len(ancestors) == 0

    def test_get_full_path(self, store):
        """Test getting full path."""
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        path = store.get_full_path(q1["node_id"])

        assert path == ["Finance", "Revenue", "Q1"]

    def test_get_full_path_not_found(self, store):
        """Test getting path for non-existent node."""
        path = store.get_full_path(99999)
        assert path == []

    # Leaf nodes and level-based queries removed as level field is no longer available

    # ========== Tree Building Tests ==========

    def test_get_tree_structure(self, store):
        """Test building nested tree structure."""
        # Build: Finance -> Revenue -> Q1, Q2
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(revenue["node_id"], "Q1")
        store.create_node(revenue["node_id"], "Q2")

        tree = store.get_simple_tree_structure(finance["node_id"])

        assert tree["Finance"] is not None
        assert len(tree["Finance"]) == 1
        assert tree["Finance"]["Revenue"] is not None
        assert len(tree["Finance"]["Revenue"]) == 2

        child_names = {c for c in tree["Finance"]["Revenue"]}
        assert child_names == {"Q1", "Q2"}

    def test_get_tree_structure_empty(self, store):
        """Test getting tree structure when empty."""
        tree = store.get_simple_tree_structure()
        assert tree == {}

    def test_find_or_create_path(self, store):
        """Test path-based node creation."""
        path = ["Finance", "Revenue", "Q1"]

        node_id = store.find_or_create_path(path)

        node = store.get_node(node_id)
        assert node["name"] == "Q1"
        assert store.get_full_path(node["node_id"]) == ["Finance", "Revenue", "Q1"]

    def test_find_or_create_path_idempotent(self, store):
        """Test that finding existing path returns same node."""
        path = ["Finance", "Revenue", "Q1"]

        node_id1 = store.find_or_create_path(path)
        node_id2 = store.find_or_create_path(path)

        assert node_id1 == node_id2

    def test_find_or_create_path_partial_exists(self, store):
        """Test creating path when partial path exists."""
        # Create Finance -> Revenue
        finance = store.create_node(None, "Finance")
        store.create_node(finance["node_id"], "Revenue")

        # Create Finance -> Revenue -> Q1 (Revenue exists)
        path = ["Finance", "Revenue", "Q1"]
        node_id = store.find_or_create_path(path)

        node = store.get_node(node_id)
        assert node["name"] == "Q1"
        assert store.get_full_path(node["node_id"]) == ["Finance", "Revenue", "Q1"]

    def test_find_or_create_path_empty_fails(self, store):
        """Test that empty path fails."""
        with pytest.raises(ValueError, match="cannot be empty"):
            store.find_or_create_path([])

    def test_find_or_create_path_arbitrary_depth_succeeds(self, store):
        """Test that arbitrarily deep paths succeed (no depth limit)."""
        # Create a very deep path (50 levels)
        path = [f"Level{i}" for i in range(50)]

        # Should succeed - no depth limit
        node_id = store.find_or_create_path(path)
        assert node_id is not None

        node = store.get_node(node_id)
        assert node["name"] == "Level49"
        assert store.get_full_path(node["node_id"]) == path

    def test_validate_no_cycle(self, store):
        """Test cycle validation."""
        # Build: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        # Valid moves
        assert store.validate_no_cycle(q1["node_id"], finance["node_id"]) is True
        assert store.validate_no_cycle(revenue["node_id"], finance["node_id"]) is True

        # Invalid moves (would create cycle)
        assert store.validate_no_cycle(finance["node_id"], revenue["node_id"]) is False
        assert store.validate_no_cycle(finance["node_id"], q1["node_id"]) is False
        assert store.validate_no_cycle(revenue["node_id"], q1["node_id"]) is False

        # Moving to self
        assert store.validate_no_cycle(finance["node_id"], finance["node_id"]) is False

    def test_find_node(self, store):
        # Build: Finance -> Revenue -> Q1, Q2
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(revenue["node_id"], "Q1")
        store.create_node(revenue["node_id"], "Q2")

        children_id = store.get_matched_children_id(["Finance"])
        assert children_id == [1, 2, 3, 4]
        children_id = store.get_matched_children_id()
        assert children_id == [1, 2, 3, 4]
        children_id = store.get_matched_children_id(["Finance", "Revenue", "Q1"])
        assert children_id == [3]
        children_id = store.get_matched_children_id(["Finance", "Revenue", "Q*"])
        assert children_id == [3, 4]

    # ========== Integration Tests ==========

    def test_complex_tree_operations(self, store):
        """Test complex tree manipulations."""
        # Build initial tree
        finance = store.create_node(None, "Finance", "财务领域")
        revenue = store.create_node(finance["node_id"], "Revenue", "收入")
        expense = store.create_node(finance["node_id"], "Expense", "支出")
        q1 = store.create_node(revenue["node_id"], "Q1")
        q2 = store.create_node(revenue["node_id"], "Q2")

        # Verify structure
        assert len(store.get_children(finance["node_id"])) == 2
        assert len(store.get_descendants(finance["node_id"])) == 4

        # Rename Revenue to Income
        store.update_node(revenue["node_id"], name="Income")
        updated_q1 = store.get_node(q1["node_id"])
        assert store.get_full_path(updated_q1["node_id"]) == ["Finance", "Income", "Q1"]

        # Move Q2 from Revenue to Expense
        store.update_node(q2["node_id"], parent_id=expense["node_id"])
        updated_q2 = store.get_node(q2["node_id"])
        assert store.get_full_path(updated_q2["node_id"]) == ["Finance", "Expense", "Q2"]

        # Delete Income (cascade Q1)
        store.delete_node(revenue["node_id"], cascade=True)
        assert store.get_node(revenue["node_id"]) is None
        assert store.get_node(q1["node_id"]) is None
        assert store.get_node(q2["node_id"]) is not None  # Still exists under Expense

        # Verify final state
        remaining = store.get_descendants(finance["node_id"])
        assert len(remaining) == 2  # Expense and Q2
        assert {n["name"] for n in remaining} == {"Expense", "Q2"}

    # ========== Rename Method Tests ==========

    def test_rename_node_same_parent(self, store):
        """Test renaming a node while keeping the same parent."""
        # Create: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")

        # Rename Q1 to Quarter1 (same parent)
        success = store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Revenue", "Quarter1"])

        assert success is True

        # Verify the rename
        old_node = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_node = store.get_node_by_path(["Finance", "Revenue", "Quarter1"])

        assert old_node is None  # Old path no longer exists
        assert new_node is not None  # New path exists
        assert new_node["name"] == "Quarter1"
        assert new_node["node_id"] == q1["node_id"]
        assert store.get_full_path(new_node["node_id"]) == ["Finance", "Revenue", "Quarter1"]

    def test_rename_node_different_parent(self, store):
        """Test moving a node to a different parent (keeping same name)."""
        # Create: Finance -> Revenue -> Q1
        #         Finance -> Expense
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(finance["node_id"], "Expense")
        store.create_node(revenue["node_id"], "Q1")

        # Move Q1 from Revenue to Expense
        success = store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Expense", "Q1"])

        assert success is True

        # Verify the move
        old_location = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_location = store.get_node_by_path(["Finance", "Expense", "Q1"])

        assert old_location is None  # No longer at old location
        assert new_location is not None  # Now at new location
        assert new_location["name"] == "Q1"
        assert store.get_full_path(new_location["node_id"]) == ["Finance", "Expense", "Q1"]

    def test_rename_node_both_parent_and_name(self, store):
        """Test renaming and moving a node simultaneously."""
        # Create: Finance -> Revenue -> Q1
        #         Finance -> Expense
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(finance["node_id"], "Expense")
        store.create_node(revenue["node_id"], "Q1")

        # Move Q1 from Revenue to Expense and rename to FirstQuarter
        success = store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Expense", "FirstQuarter"])

        assert success is True

        # Verify the rename and move
        old_location = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_location = store.get_node_by_path(["Finance", "Expense", "FirstQuarter"])

        assert old_location is None  # No longer at old location
        assert new_location is not None  # Now at new location
        assert new_location["name"] == "FirstQuarter"
        assert store.get_full_path(new_location["node_id"]) == ["Finance", "Expense", "FirstQuarter"]

    def test_rename_root_node(self, store):
        """Test renaming a root node."""
        finance = store.create_node(None, "Finance")
        child = store.create_node(finance["node_id"], "Revenue")

        # Rename Finance to Financial
        success = store.rename(["Finance"], ["Financial"])

        assert success is True

        # Verify the rename
        old_node = store.get_node_by_path(["Finance"])
        new_node = store.get_node_by_path(["Financial"])

        assert old_node is None  # Old name no longer exists
        assert new_node is not None  # New name exists
        assert new_node["name"] == "Financial"

        # Verify child path is updated
        updated_child = store.get_node(child["node_id"])
        assert store.get_full_path(updated_child["node_id"]) == ["Financial", "Revenue"]

    def test_rename_to_root_level(self, store):
        """Test moving a node to root level."""
        # Create: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(revenue["node_id"], "Q1")

        # Move Q1 to root level (no parent)
        success = store.rename(["Finance", "Revenue", "Q1"], ["Q1"])

        assert success is True

        # Verify the move to root
        old_location = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_location = store.get_node_by_path(["Q1"])

        assert old_location is None  # No longer at old location
        assert new_location is not None  # Now at root level
        assert new_location["name"] == "Q1"
        assert new_location["parent_id"] is None
        assert store.get_full_path(new_location["node_id"]) == ["Q1"]

    def test_rename_empty_old_path_fails(self, store):
        """Test that rename fails with empty old path."""
        with pytest.raises(ValueError, match="Old path cannot be empty"):
            store.rename([], ["NewName"])

    def test_rename_empty_new_path_fails(self, store):
        """Test that rename fails with empty new path."""
        store.create_node(None, "Finance")

        with pytest.raises(ValueError, match="New path cannot be empty"):
            store.rename(["Finance"], [])

    def test_rename_nonexistent_node_fails(self, store):
        """Test that rename fails for non-existent node."""
        with pytest.raises(ValueError, match="Node not found at path"):
            store.rename(["NonExistent", "Path"], ["New", "Path"])

    def test_rename_nonexistent_parent_fails(self, store):
        """Test that rename fails when new parent doesn't exist."""
        # Create: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(revenue["node_id"], "Q1")

        # Try to move Q1 under non-existent parent
        with pytest.raises(ValueError, match="New parent not found at path"):
            store.rename(["Finance", "Revenue", "Q1"], ["NonExistent", "Q1"])

    def test_rename_empty_name_fails(self, store):
        """Test that rename fails with empty new name."""
        store.create_node(None, "Finance")

        with pytest.raises(ValueError, match="New name cannot be empty"):
            store.rename(["Finance"], [""])

        with pytest.raises(ValueError, match="New name cannot be empty"):
            store.rename(["Finance"], ["   "])

    def test_rename_duplicate_name_same_parent_fails(self, store):
        """Test that rename fails when target name already exists under same parent."""
        # Create: Finance -> Revenue, Expense
        finance = store.create_node(None, "Finance")
        store.create_node(finance["node_id"], "Revenue")
        store.create_node(finance["node_id"], "Expense")

        # Try to rename Revenue to Expense (duplicate under Finance)
        with pytest.raises(ValueError, match="already exists under parent"):
            store.rename(["Finance", "Revenue"], ["Finance", "Expense"])

    def test_rename_duplicate_name_different_parent_fails(self, store):
        """Test that rename fails when target name already exists under new parent."""
        # Create: Finance -> Revenue -> Q1
        #         Finance -> Expense -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        expense = store.create_node(finance["node_id"], "Expense")
        store.create_node(revenue["node_id"], "Q1")
        store.create_node(expense["node_id"], "Q1")

        # Try to move Q1 from Revenue to Expense (Q1 already exists there)
        with pytest.raises(ValueError, match="already exists under parent"):
            store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Expense", "Q1"])

    def test_rename_creates_cycle_fails(self, store):
        """Test that rename fails when it would create a cycle."""
        # Create: Finance -> Revenue -> Q1
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(revenue["node_id"], "Q1")

        # Try to move Finance under Q1 (would create cycle)
        with pytest.raises(ValueError, match="Moving node to new parent would create a cycle"):
            store.rename(["Finance"], ["Finance", "Revenue", "Q1", "Finance"])

    def test_rename_self_to_descendant_fails(self, store):
        """Test that moving a node under its own descendant fails."""
        # Create: Finance -> Revenue -> Q1 -> Jan
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")
        store.create_node(q1["node_id"], "Jan")

        # Try to move Revenue under Jan (cycle)
        with pytest.raises(ValueError, match="Moving node to new parent would create a cycle"):
            store.rename(["Finance", "Revenue"], ["Finance", "Revenue", "Q1", "Jan", "Revenue"])

    def test_rename_same_path_no_changes(self, store):
        """Test that renaming to the same path succeeds but makes no changes."""
        finance = store.create_node(None, "Finance")
        store.create_node(finance["node_id"], "Revenue")

        # Rename to same path
        success = store.rename(["Finance", "Revenue"], ["Finance", "Revenue"])

        assert success is True

        # Verify node is unchanged
        node = store.get_node_by_path(["Finance", "Revenue"])
        assert node is not None
        assert node["name"] == "Revenue"

    def test_rename_same_name_different_parent(self, store):
        """Test moving a node to different parent but keeping same name."""
        # Create: Finance -> Revenue -> Q1
        #         Finance -> Expense
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(finance["node_id"], "Expense")
        store.create_node(revenue["node_id"], "Q1")

        # Move Q1 to Expense (same name, different parent)
        success = store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Expense", "Q1"])

        assert success is True

        # Verify the move
        old_location = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_location = store.get_node_by_path(["Finance", "Expense", "Q1"])

        assert old_location is None
        assert new_location is not None
        assert new_location["name"] == "Q1"

    def test_rename_with_descendants(self, store):
        """Test renaming a node that has descendants."""
        # Create: Finance -> Revenue -> Q1 -> Jan, Feb
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        q1 = store.create_node(revenue["node_id"], "Q1")
        jan = store.create_node(q1["node_id"], "Jan")
        feb = store.create_node(q1["node_id"], "Feb")

        # Rename Q1 to Quarter1
        success = store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Revenue", "Quarter1"])

        assert success is True

        # Verify the rename and descendants are still accessible
        old_q1 = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_q1 = store.get_node_by_path(["Finance", "Revenue", "Quarter1"])

        assert old_q1 is None
        assert new_q1 is not None
        assert new_q1["name"] == "Quarter1"

        # Verify descendants can still be accessed through new path
        jan_updated = store.get_node_by_path(["Finance", "Revenue", "Quarter1", "Jan"])
        feb_updated = store.get_node_by_path(["Finance", "Revenue", "Quarter1", "Feb"])

        assert jan_updated is not None
        assert feb_updated is not None
        assert jan_updated["node_id"] == jan["node_id"]
        assert feb_updated["node_id"] == feb["node_id"]

    def test_rename_move_with_descendants(self, store):
        """Test moving a node with descendants to a different parent."""
        # Create: Finance -> Revenue -> Q1 -> Jan, Feb
        #         Finance -> Expense
        finance = store.create_node(None, "Finance")
        revenue = store.create_node(finance["node_id"], "Revenue")
        store.create_node(finance["node_id"], "Expense")
        q1 = store.create_node(revenue["node_id"], "Q1")
        jan = store.create_node(q1["node_id"], "Jan")
        feb = store.create_node(q1["node_id"], "Feb")

        # Move Q1 (with Jan, Feb) from Revenue to Expense
        success = store.rename(["Finance", "Revenue", "Q1"], ["Finance", "Expense", "Q1"])

        assert success is True

        # Verify the move and descendants are still accessible
        old_location = store.get_node_by_path(["Finance", "Revenue", "Q1"])
        new_location = store.get_node_by_path(["Finance", "Expense", "Q1"])

        assert old_location is None
        assert new_location is not None

        # Verify descendants can still be accessed through new parent
        jan_updated = store.get_node_by_path(["Finance", "Expense", "Q1", "Jan"])
        feb_updated = store.get_node_by_path(["Finance", "Expense", "Q1", "Feb"])

        assert jan_updated is not None
        assert feb_updated is not None
        assert jan_updated["node_id"] == jan["node_id"]
        assert feb_updated["node_id"] == feb["node_id"]

    def test_rename_preserves_description_and_timestamps(self, store):
        """Test that rename preserves description and updates timestamp."""
        finance = store.create_node(None, "Finance", "财务领域")
        revenue = store.create_node(finance["node_id"], "Revenue", "收入管理")

        # Get original timestamps
        original_created = revenue["created_at"]
        original_updated = revenue["updated_at"]

        # Rename Revenue to Income
        import time

        time.sleep(1.1)  # Ensure timestamp difference (more than 1 second)
        success = store.rename(["Finance", "Revenue"], ["Finance", "Income"])

        assert success is True

        # Verify preservation
        updated_node = store.get_node_by_path(["Finance", "Income"])
        assert updated_node is not None
        assert updated_node["name"] == "Income"
        assert updated_node["description"] == "收入管理"  # Description preserved
        assert updated_node["created_at"] == original_created  # Created time preserved
        assert updated_node["updated_at"] > original_updated  # Updated time changed
