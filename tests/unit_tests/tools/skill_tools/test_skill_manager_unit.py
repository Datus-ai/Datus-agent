# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for skill_manager.py covering diff lines."""

from pathlib import Path
from unittest.mock import MagicMock

from datus.tools.skill_tools.skill_config import SkillConfig, SkillMetadata
from datus.tools.skill_tools.skill_manager import SkillManager


def _make_skill(name="test-skill", **kwargs):
    defaults = dict(description="A test skill", location=Path("/tmp/test"), tags=["sql"])
    defaults.update(kwargs)
    return SkillMetadata(name=name, **defaults)


class TestSkillManagerInit:
    def test_default_init(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        SkillManager(registry=registry)
        registry.scan_directories.assert_called_once()

    def test_custom_config(self):
        config = SkillConfig(marketplace_url="http://custom:8080")
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        manager = SkillManager(config=config, registry=registry)
        assert manager.config.marketplace_url == "http://custom:8080"


class TestGetAvailableSkills:
    def test_no_patterns(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill()]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("test-node")
        assert len(skills) == 1

    def test_with_patterns(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 2
        registry.list_skills.return_value = [_make_skill("sql-opt"), _make_skill("data-clean")]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("test-node", patterns=["sql-*"])
        assert len(skills) == 1
        assert skills[0].name == "sql-opt"

    def test_wildcard_pattern(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 2
        registry.list_skills.return_value = [_make_skill("a"), _make_skill("b")]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("test-node", patterns=["*"])
        assert len(skills) == 2

    def test_model_invocation_filter(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        s = _make_skill(disable_model_invocation=True)
        registry.list_skills.return_value = [s]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("test-node")
        assert len(skills) == 0

    def test_allowed_agents_hides_from_other_agent(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(allowed_agents=["gen_dashboard"])]
        manager = SkillManager(registry=registry)
        assert manager.get_available_skills("chat") == []

    def test_allowed_agents_exposes_to_listed_agent(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(allowed_agents=["gen_dashboard"])]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("gen_dashboard")
        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_empty_allowed_agents_visible_to_everyone(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(allowed_agents=[])]
        manager = SkillManager(registry=registry)
        assert len(manager.get_available_skills("chat")) == 1
        assert len(manager.get_available_skills("gen_dashboard")) == 1

    def test_allowed_agents_matches_via_node_class(self):
        """Alias mismatch should still pass when the canonical class matches."""
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(allowed_agents=["gen_dashboard"])]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("my_dashboard", node_class="gen_dashboard")
        assert len(skills) == 1

    def test_allowed_agents_hidden_when_both_identifiers_miss(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(allowed_agents=["gen_dashboard"])]
        manager = SkillManager(registry=registry)
        skills = manager.get_available_skills("my_table", node_class="gen_table")
        assert skills == []


class TestLoadSkill:
    def test_load_success(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill()
        registry.load_skill_content.return_value = "# Skill Content"
        manager = SkillManager(registry=registry)
        ok, msg, content = manager.load_skill("test-skill", "node")
        assert ok is True
        assert content == "# Skill Content"

    def test_load_not_found(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        registry.get_skill.return_value = None
        manager = SkillManager(registry=registry)
        ok, msg, content = manager.load_skill("nonexistent", "node")
        assert ok is False
        assert content is None

    def test_load_content_fails(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill()
        registry.load_skill_content.return_value = None
        manager = SkillManager(registry=registry)
        ok, msg, content = manager.load_skill("test-skill", "node")
        assert ok is False

    def test_load_with_deny_permission(self):
        from datus.tools.permission.permission_config import PermissionLevel

        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill()
        perm_mgr = MagicMock()
        perm_mgr.check_permission.return_value = PermissionLevel.DENY
        manager = SkillManager(registry=registry, permission_manager=perm_mgr)
        ok, msg, content = manager.load_skill("test-skill", "node")
        assert ok is False
        assert "denied" in msg.lower()

    def test_load_with_ask_permission(self):
        from datus.tools.permission.permission_config import PermissionLevel

        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill()
        perm_mgr = MagicMock()
        perm_mgr.check_permission.return_value = PermissionLevel.ASK
        manager = SkillManager(registry=registry, permission_manager=perm_mgr)
        ok, msg, content = manager.load_skill("test-skill", "node")
        assert ok is False
        assert msg == "ASK_PERMISSION"

    def test_load_rejected_for_disallowed_agent(self):
        """Default load enforces ``allowed_agents`` as a hard reject."""
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill(allowed_agents=["gen_table"])
        manager = SkillManager(registry=registry)
        ok, msg, content = manager.load_skill("test-skill", "chat")
        assert ok is False
        assert content is None
        assert "chat" in msg
        registry.load_skill_content.assert_not_called()

    def test_load_scope_check_honours_node_class(self):
        """Alias mismatch with whitelisted class should still pass."""
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill(allowed_agents=["gen_dashboard"])
        registry.load_skill_content.return_value = "# OK"
        manager = SkillManager(registry=registry)
        ok, _msg, content = manager.load_skill(
            "test-skill",
            "my_dashboard",
            node_class="gen_dashboard",
        )
        assert ok is True
        assert content == "# OK"

    def test_load_scope_bypass_for_authoring_agent(self):
        """``check_scope=False`` bypasses the hard reject (skill-editing workflow)."""
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill(allowed_agents=["gen_table"])
        registry.load_skill_content.return_value = "# Scoped"
        manager = SkillManager(registry=registry)
        ok, _msg, content = manager.load_skill(
            "test-skill",
            "gen_skill",
            check_scope=False,
        )
        assert ok is True
        assert content == "# Scoped"

    def test_load_scope_rejection_runs_before_permission_check(self):
        """Scope reject must fire before the permission manager is consulted."""
        from datus.tools.permission.permission_config import PermissionLevel

        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill(allowed_agents=["gen_table"])
        perm_mgr = MagicMock()
        perm_mgr.check_permission.return_value = PermissionLevel.ALLOW
        manager = SkillManager(registry=registry, permission_manager=perm_mgr)

        ok, _msg, content = manager.load_skill("test-skill", "chat")
        assert ok is False
        assert content is None
        perm_mgr.check_permission.assert_not_called()

    def test_load_allowed_agent_succeeds(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill(allowed_agents=["gen_table"])
        registry.load_skill_content.return_value = "# Content"
        manager = SkillManager(registry=registry)
        ok, msg, content = manager.load_skill("test-skill", "gen_table")
        assert ok is True
        assert content == "# Content"


class TestGenerateSkillsXml:
    def test_generate_xml(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill()]
        manager = SkillManager(registry=registry)
        xml = manager.generate_available_skills_xml("node")
        assert "<available_skills>" in xml
        assert "test-skill" in xml

    def test_generate_xml_empty(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        registry.list_skills.return_value = []
        manager = SkillManager(registry=registry)
        xml = manager.generate_available_skills_xml("node")
        assert xml == ""


class TestMarketplaceOperations:
    def test_get_marketplace_client_cached(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        manager = SkillManager(registry=registry)
        c1 = manager._get_marketplace_client()
        c2 = manager._get_marketplace_client()
        assert c1 is c2

    def test_search_marketplace_error(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=registry)
        results = manager.search_marketplace(query="test")
        assert results == []

    def test_install_from_marketplace_error(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=registry)
        ok, msg = manager.install_from_marketplace("test")
        assert ok is False

    def test_publish_to_marketplace_error(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=registry)
        ok, msg = manager.publish_to_marketplace("/nonexistent")
        assert ok is False

    def test_publish_resolves_skill_name(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        skill_meta = _make_skill()
        registry.get_skill.return_value = skill_meta
        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=registry)
        # Will fail because marketplace is unreachable, but tests resolution path
        ok, msg = manager.publish_to_marketplace("test-skill")
        assert ok is False

    def test_sync_promoted_skills_error(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        config = SkillConfig(marketplace_url="http://nonexistent:9999")
        manager = SkillManager(config=config, registry=registry)
        synced = manager.sync_promoted_skills()
        assert synced == []


class TestGenerateSkillsXmlContent:
    def test_generate_xml_includes_description(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(description="SQL query optimization techniques")]
        manager = SkillManager(registry=registry)
        xml = manager.generate_available_skills_xml("node")
        assert "<description>" in xml
        assert "SQL query optimization techniques" in xml

    def test_generate_xml_includes_tags(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill(tags=["sql", "performance"])]
        manager = SkillManager(registry=registry)
        xml = manager.generate_available_skills_xml("node")
        assert "<tags>sql, performance</tags>" in xml


class TestUtilityMethods:
    def test_get_skill(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.get_skill.return_value = _make_skill()
        manager = SkillManager(registry=registry)
        assert manager.get_skill("test-skill") is not None

    def test_refresh(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        manager = SkillManager(registry=registry)
        manager.refresh()
        registry.refresh.assert_called_once()

    def test_list_all_skills(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 1
        registry.list_skills.return_value = [_make_skill()]
        manager = SkillManager(registry=registry)
        assert len(manager.list_all_skills()) == 1

    def test_parse_skill_patterns(self):
        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        manager = SkillManager(registry=registry)
        assert manager.parse_skill_patterns("") == []
        assert manager.parse_skill_patterns("sql-*, data-*") == ["sql-*", "data-*"]
        assert manager.parse_skill_patterns("  sql-*  ,  data-*  ") == ["sql-*", "data-*"]

    def test_check_skill_permission_no_manager(self):
        from datus.tools.permission.permission_config import PermissionLevel

        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        manager = SkillManager(registry=registry)
        assert manager.check_skill_permission("test", "node") == PermissionLevel.ALLOW

    def test_check_skill_permission_with_manager(self):
        from datus.tools.permission.permission_config import PermissionLevel

        registry = MagicMock()
        registry.get_skill_count.return_value = 0
        perm_mgr = MagicMock()
        perm_mgr.check_permission.return_value = PermissionLevel.DENY
        manager = SkillManager(registry=registry, permission_manager=perm_mgr)
        assert manager.check_skill_permission("test", "node") == PermissionLevel.DENY
