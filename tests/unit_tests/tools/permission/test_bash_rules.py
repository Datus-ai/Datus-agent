# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for fine-grained bash command permission rules.

Tests pattern matching semantics, the deny-first decision order, the safety
ceiling for wrappers/metacharacters, session bucketing, and ruleset merging.
"""

import shlex

import pytest

from datus.tools.permission.bash_rules import (
    BashClassifierConfig,
    BashCommandRules,
    BashDecisionSource,
    command_matches_pattern,
    evaluate_bash_command,
    session_bucket_for,
)
from datus.tools.permission.permission_config import PermissionLevel


def argv(command: str):
    return shlex.split(command)


class TestCommandMatchesPattern:
    """Tests for the three pattern forms: exact, prefix:*, prefix:glob."""

    def test_exact_match(self):
        """Exact pattern (no colon) matches only the identical command."""
        assert command_matches_pattern(argv("git status"), "git status")
        assert not command_matches_pattern(argv("git status --short"), "git status")
        assert not command_matches_pattern(argv("git"), "git status")

    def test_prefix_star_matches_prefix_and_more(self):
        """prefix:* matches the bare prefix and anything following it."""
        assert command_matches_pattern(argv("git log"), "git log:*")
        assert command_matches_pattern(argv("git log --oneline -5"), "git log:*")
        assert not command_matches_pattern(argv("git logs"), "git log:*")
        assert not command_matches_pattern(argv("git"), "git log:*")

    def test_multi_word_prefix(self):
        """Multi-word prefixes work (uv run pytest:*)."""
        assert command_matches_pattern(argv("uv run pytest tests/ -k foo"), "uv run pytest:*")
        assert not command_matches_pattern(argv("uv run python evil.py"), "uv run pytest:*")

    def test_prefix_glob_restricts_first_arg(self):
        """prefix:glob requires the first remainder token to match the glob."""
        assert command_matches_pattern(argv("python scripts/etl.py"), "python:scripts/*.py")
        assert not command_matches_pattern(argv("python -c 'print(1)'"), "python:scripts/*.py")
        assert not command_matches_pattern(argv("python other/x.py"), "python:scripts/*.py")
        # bare prefix does not satisfy a non-* glob
        assert not command_matches_pattern(argv("python"), "python:scripts/*.py")

    def test_prefix_glob_matches_joined_remainder(self):
        """The joined remainder string may satisfy the glob as a whole."""
        assert command_matches_pattern(argv("npm run build --prod"), "npm run:build --prod")

    def test_unanchored_matches_at_any_offset(self):
        """anchor=False finds the prefix anywhere in argv (deny-rule mode)."""
        assert command_matches_pattern(argv("xargs rm -rf build"), "rm:*", anchor=False)
        assert command_matches_pattern(argv("find . -exec rm {}"), "rm:*", anchor=False)
        assert not command_matches_pattern(argv("xargs rm -rf build"), "rm:*", anchor=True)

    def test_unanchored_does_not_match_inside_quoted_token(self):
        """A quoted argument containing the word is one token and must not match."""
        assert not command_matches_pattern(argv("git commit -m 'rm important'"), "rm:*", anchor=False)

    def test_word_boundary(self):
        """ls:* must not match lsof (token equality via fnmatch, not startswith)."""
        assert not command_matches_pattern(argv("lsof -i :8080"), "ls:*")


class TestEvaluateDecisionOrder:
    """Tests for the deny -> safety -> ask -> allow -> default order."""

    def test_deny_beats_allow(self):
        rules = BashCommandRules(allow=["git:*"], deny=["git push:*"])
        decision = evaluate_bash_command("git push origin main", rules)
        assert decision.level == PermissionLevel.DENY
        assert decision.source == BashDecisionSource.DENY_RULE
        assert decision.matched_pattern == "git push:*"

    def test_ask_beats_allow(self):
        rules = BashCommandRules(allow=["docker:*"], ask=["docker push:*"])
        decision = evaluate_bash_command("docker push img:latest", rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.ASK_RULE

    def test_allow_rule(self):
        rules = BashCommandRules(allow=["git log:*"])
        decision = evaluate_bash_command("git log --oneline", rules)
        assert decision.level == PermissionLevel.ALLOW
        assert decision.source == BashDecisionSource.ALLOW_RULE
        assert decision.matched_pattern == "git log:*"

    def test_default_ask_when_nothing_matches(self):
        rules = BashCommandRules(allow=["git status"])
        decision = evaluate_bash_command("cargo build --release", rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.DEFAULT

    def test_default_can_be_allow(self):
        rules = BashCommandRules(default=PermissionLevel.ALLOW)
        decision = evaluate_bash_command("cargo build", rules)
        assert decision.level == PermissionLevel.ALLOW
        assert decision.source == BashDecisionSource.DEFAULT

    def test_unanchored_deny_catches_xargs(self):
        rules = BashCommandRules(deny=["rm:*"])
        decision = evaluate_bash_command("xargs rm -rf build", rules)
        assert decision.level == PermissionLevel.DENY
        assert decision.matched_pattern == "rm:*"

    def test_deny_beats_wrapper_safety(self):
        """A denied command inside a wrapper is DENY, not the wrapper's ASK."""
        rules = BashCommandRules(deny=["rm:*"])
        decision = evaluate_bash_command("sudo rm -rf /", rules)
        assert decision.level == PermissionLevel.DENY
        assert decision.source == BashDecisionSource.DENY_RULE


class TestSafetyCeiling:
    """Wrappers and shell metacharacters never auto-allow."""

    @pytest.mark.parametrize(
        "command",
        [
            "bash -c 'git status'",
            "sh script.sh",
            "sudo ls",
            "env FOO=1 ls",
            "xargs echo",
            "eval echo hi",
            "timeout 5 git status",
        ],
    )
    def test_wrapper_forces_ask(self, command):
        rules = BashCommandRules(allow=["git status", "ls:*", "echo:*", "sh:*", "bash:*"])
        decision = evaluate_bash_command(command, rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    @pytest.mark.parametrize(
        "command",
        [
            "python -c 'print(1)'",
            "python3 -c 'print(1)'",
            "python3.12 -uc 'print(1)'",  # short-option cluster still contains -c
            "perl -e 'unlink foo'",
            "ruby -e 'puts 1'",
            "node --eval 'process.exit()'",
            "node -p '1+1'",
            "php -r 'echo 1;'",
        ],
    )
    def test_interpreter_inline_code_forces_ask(self, command):
        """``python -c`` / ``perl -e`` … execute a string the rules cannot see,
        so even a blanket interpreter allow rule never auto-runs them."""
        rules = BashCommandRules(allow=["python:*", "python3:*", "python3.12:*", "perl:*", "ruby:*", "node:*", "php:*"])
        decision = evaluate_bash_command(command, rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    def test_interpreter_script_path_still_allowed(self):
        """Interpreters are NOT blanket wrappers: the documented
        ``python:scripts/*.py`` allow form must keep working."""
        rules = BashCommandRules(allow=["python:scripts/*.py"])
        decision = evaluate_bash_command("python scripts/etl.py", rules)
        assert decision.level == PermissionLevel.ALLOW
        assert decision.matched_pattern == "python:scripts/*.py"

    def test_interpreter_script_own_dash_c_arg_not_flagged(self):
        """A ``-c`` AFTER the script path belongs to the script, not the
        interpreter — option scanning stops at the first non-option token."""
        rules = BashCommandRules(allow=["python:*"])
        decision = evaluate_bash_command("python tool.py -c config.yml", rules)
        assert decision.level == PermissionLevel.ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "git status && rm -rf /",
            "ls || rm x",  # logical OR is not a pipeline
            "ls |& grep foo",  # stderr pipe is not a simple pipeline
            "ls |",  # trailing empty pipeline segment
            "echo hi > /etc/passwd",
            "echo `whoami`",
            "echo $(id)",
            "echo ${HOME}",
            "ls; rm x",
        ],
    )
    def test_metacharacters_force_ask(self, command):
        rules = BashCommandRules(allow=["git status", "ls:*", "echo:*", "grep:*", "rm:*"])
        decision = evaluate_bash_command(command, rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    def test_unparseable_command(self):
        rules = BashCommandRules(allow=["echo:*"])
        decision = evaluate_bash_command('echo "unclosed', rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.UNPARSEABLE
        assert decision.safety_forced is True

    def test_empty_command(self):
        decision = evaluate_bash_command("   ", BashCommandRules())
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.UNPARSEABLE

    def test_plain_command_is_not_safety_forced(self):
        decision = evaluate_bash_command("cargo build", BashCommandRules())
        assert decision.safety_forced is False


class TestSessionBuckets:
    """Bucket keys scope 'always allow' grants."""

    def test_matched_pattern_is_bucket(self):
        rules = BashCommandRules(ask=["docker:*"])
        decision = evaluate_bash_command("docker ps", rules)
        assert decision.bucket == "docker:*"

    def test_group_command_buckets_on_two_tokens(self):
        assert session_bucket_for(argv("git push origin main"), None) == "git push"
        assert session_bucket_for(argv("docker compose up"), None) == "docker compose"

    def test_plain_command_buckets_on_first_token(self):
        assert session_bucket_for(argv("ls -la"), None) == "ls"

    def test_group_command_with_flag_first_buckets_on_one_token(self):
        assert session_bucket_for(argv("git -C /tmp status"), None) == "git"

    def test_default_decision_bucket(self):
        decision = evaluate_bash_command("git push origin main", BashCommandRules())
        assert decision.bucket == "git push"

    def test_datus_buckets_per_plugin_namespace(self):
        # ``datus`` is a group command: approving one plugin's namespace must
        # not green-light another plugin's.
        assert session_bucket_for(argv("datus hello greet world"), None) == "datus hello"
        assert session_bucket_for(argv("datus other doit"), None) == "datus other"
        assert session_bucket_for(argv("datus --help"), None) == "datus"


class TestBashCommandRulesModel:
    """Tests for from_dict / merge_with / is_empty."""

    def test_from_dict_none_and_empty(self):
        assert BashCommandRules.from_dict(None) is None
        assert BashCommandRules.from_dict({}) is None

    def test_from_dict_parses_all_sections(self):
        rules = BashCommandRules.from_dict(
            {
                "allow": ["git log:*"],
                "deny": ["rm:*"],
                "ask": ["docker:*"],
                "default": "allow",
                "classifier": {"enabled": True, "model": "gpt-x", "confidence_threshold": 0.9},
            }
        )
        assert rules.allow == ["git log:*"]
        assert rules.deny == ["rm:*"]
        assert rules.ask == ["docker:*"]
        assert PermissionLevel(rules.default) == PermissionLevel.ALLOW
        assert rules.classifier.enabled is True
        assert rules.classifier.model == "gpt-x"
        assert rules.classifier.confidence_threshold == 0.9

    def test_from_dict_malformed_raises(self):
        """Malformed sections raise so agent_config's fail-closed fallback fires."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BashCommandRules.from_dict({"allow": "not-a-list"})
        with pytest.raises(ValueError):
            BashCommandRules.from_dict({"default": "bogus"})

    def test_merge_with_concatenates_lists(self):
        base = BashCommandRules(allow=["git log:*"], deny=["rm:*"])
        override = BashCommandRules(allow=["make:*"], ask=["docker:*"])
        merged = base.merge_with(override)
        assert merged.allow == ["git log:*", "make:*"]
        assert merged.deny == ["rm:*"]
        assert merged.ask == ["docker:*"]

    def test_merge_with_none_returns_self(self):
        base = BashCommandRules(allow=["git log:*"])
        assert base.merge_with(None) is base

    def test_merge_default_only_when_explicit(self):
        base = BashCommandRules(default=PermissionLevel.ALLOW)
        # override did not set default explicitly -> base's kept
        merged = base.merge_with(BashCommandRules(allow=["x:*"]))
        assert PermissionLevel(merged.default) == PermissionLevel.ALLOW
        # override set default explicitly -> override wins
        merged = base.merge_with(BashCommandRules(default=PermissionLevel.ASK))
        assert PermissionLevel(merged.default) == PermissionLevel.ASK

    def test_merge_classifier_only_when_explicit(self):
        base = BashCommandRules(classifier=BashClassifierConfig(enabled=True))
        merged = base.merge_with(BashCommandRules(allow=["x:*"]))
        assert merged.classifier.enabled is True
        merged = base.merge_with(BashCommandRules(classifier=BashClassifierConfig(enabled=False)))
        assert merged.classifier.enabled is False

    def test_is_empty(self):
        assert BashCommandRules().is_empty()
        assert not BashCommandRules(allow=["ls:*"]).is_empty()


class TestSplitPipeline:
    """split_pipeline: top-level unquoted | segmentation."""

    def test_no_pipe_returns_single(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("git status") == ["git status"]

    def test_simple_pipeline(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("cat a | grep b | wc -l") == ["cat a", "grep b", "wc -l"]

    def test_pipe_in_double_quotes_not_split(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline('grep "a|b" file') == ['grep "a|b" file']

    def test_pipe_in_single_quotes_not_split(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("awk '{print $1|$2}'") == ["awk '{print $1|$2}'"]

    def test_escaped_pipe_not_split(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("echo a\\|b") == ["echo a\\|b"]

    def test_logical_or_returns_none(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("a || b") is None

    def test_stderr_pipe_returns_none(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("a |& b") is None

    def test_empty_segment_returns_none(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("ls |") is None
        assert split_pipeline("| ls") is None
        assert split_pipeline("a || b") is None

    def test_unbalanced_quotes_returns_none(self):
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline('echo "unclosed | grep') is None


class TestPipelineEvaluation:
    """Per-segment judging + aggregation for pipelines."""

    def test_all_allow_segments_auto_allow(self):
        rules = BashCommandRules(allow=["cat:*", "grep:*", "wc:*"])
        d = evaluate_bash_command("cat log | grep err | wc -l", rules)
        assert d.level == PermissionLevel.ALLOW
        assert d.source == BashDecisionSource.ALLOW_RULE

    def test_deny_segment_blocks_whole_pipeline(self):
        rules = BashCommandRules(allow=["cat:*"], deny=["rm:*"])
        d = evaluate_bash_command("cat x | rm -rf y", rules)
        assert d.level == PermissionLevel.DENY
        assert d.matched_pattern == "rm:*"

    def test_deny_segment_via_unanchored_wrapper(self):
        rules = BashCommandRules(allow=["ls:*"], deny=["rm:*"])
        d = evaluate_bash_command("ls | xargs rm", rules)
        assert d.level == PermissionLevel.DENY
        assert d.matched_pattern == "rm:*"

    def test_wrapper_segment_forces_safety_ask(self):
        rules = BashCommandRules(allow=["ls:*", "echo:*"])
        d = evaluate_bash_command("ls | xargs echo", rules)
        assert d.level == PermissionLevel.ASK
        assert d.source == BashDecisionSource.SAFETY
        assert d.safety_forced is True

    def test_unmatched_segment_asks_default(self):
        rules = BashCommandRules(allow=["cat:*"])
        d = evaluate_bash_command("cat x | frobnicate", rules)
        assert d.level == PermissionLevel.ASK
        assert d.source == BashDecisionSource.DEFAULT
        assert d.bucket == "frobnicate"

    def test_ask_rule_segment_outranks_default_segment(self):
        """The pipeline's source must be ASK_RULE so a permissive profile's
        default-fallback in the hook can't swallow the ask-rule segment."""
        rules = BashCommandRules(allow=["cat:*"], ask=["docker:*"])
        d = evaluate_bash_command("frobnicate | docker ps | cat", rules)
        assert d.level == PermissionLevel.ASK
        assert d.source == BashDecisionSource.ASK_RULE
        assert d.bucket == "docker:*"

    def test_deny_outranks_wrapper_and_ask(self):
        rules = BashCommandRules(allow=["cat:*"], ask=["docker:*"], deny=["rm:*"])
        d = evaluate_bash_command("docker ps | xargs rm | cat", rules)
        assert d.level == PermissionLevel.DENY
        assert d.matched_pattern == "rm:*"

    def test_metachar_inside_segment_still_safety(self):
        """A pipeline segment carrying other metachars is safety-forced."""
        rules = BashCommandRules(allow=["cat:*", "grep:*"])
        d = evaluate_bash_command("cat x | grep y > out.txt", rules)
        assert d.level == PermissionLevel.ASK
        assert d.source == BashDecisionSource.SAFETY
        assert d.safety_forced is True

    def test_ask_pipeline_carries_all_non_allow_segments(self):
        """The hook's project-grant bypass must see EVERY non-allow segment,
        not just the representative one."""
        rules = BashCommandRules(allow=["cat:*"], ask=["docker:*"])
        d = evaluate_bash_command("frobnicate | docker ps | cat x", rules)
        assert d.segment_ask_patterns == (
            (BashDecisionSource.DEFAULT, None),
            (BashDecisionSource.ASK_RULE, "docker:*"),
        )

    def test_single_command_has_no_segment_patterns(self):
        rules = BashCommandRules(ask=["docker:*"])
        d = evaluate_bash_command("docker ps", rules)
        assert d.segment_ask_patterns is None


class TestDatusProfileFlagNormalization:
    """Leading ``--profile`` datus globals must not defeat plugin rules."""

    RULES = BashCommandRules(
        allow=["datus hello greet:*"],
        ask=["datus hello config set:*"],
        deny=["datus hello config wipe:*"],
    )

    def test_deny_matches_profile_qualified_raw_argv(self):
        """Deny rules additionally match the RAW (pre-normalization) argv so a
        user can fence off a specific plugin profile even though ask/allow
        matching sees through the ``--profile`` flag."""
        rules = BashCommandRules(allow=["datus hello greet:*"], deny=["datus hello --profile prod:*"])
        d = evaluate_bash_command("datus hello --profile prod greet Ada", rules)
        assert d.level == PermissionLevel.DENY
        assert d.matched_pattern == "datus hello --profile prod:*"
        # Other profiles are unaffected by the profile-scoped deny.
        d = evaluate_bash_command("datus hello --profile dev greet Ada", rules)
        assert d.level == PermissionLevel.ALLOW

    def test_profile_space_form_matches_allow(self):
        d = evaluate_bash_command("datus hello --profile prod greet Ada", self.RULES)
        assert d.level == PermissionLevel.ALLOW
        assert d.matched_pattern == "datus hello greet:*"

    def test_profile_equals_form_matches_ask_with_pattern_bucket(self):
        d = evaluate_bash_command("datus hello --profile=prod config set k v", self.RULES)
        assert d.level == PermissionLevel.ASK
        assert d.matched_pattern == "datus hello config set:*"
        assert d.bucket == "datus hello config set:*"

    def test_profile_flag_cannot_dodge_deny(self):
        d = evaluate_bash_command("datus hello --profile prod config wipe all", self.RULES)
        assert d.level == PermissionLevel.DENY
        assert d.matched_pattern == "datus hello config wipe:*"

    def test_repeated_profile_flags_all_stripped(self):
        d = evaluate_bash_command("datus hello --profile a --profile=b greet Ada", self.RULES)
        assert d.level == PermissionLevel.ALLOW

    def test_config_flag_is_not_stripped(self):
        # ``--config`` rebinds credentials/endpoints; commands carrying it
        # fall through to the default decision instead of matching rules.
        d = evaluate_bash_command("datus hello --config /tmp/x.yml config set k v", self.RULES)
        assert d.level == PermissionLevel.ASK
        assert d.matched_pattern is None
        assert d.bucket == "datus hello"

    def test_subcommand_position_profile_belongs_to_plugin(self):
        # From the first command token onward flags belong to the plugin;
        # ``greet:*`` covers the remainder, no stripping involved.
        d = evaluate_bash_command("datus hello config set --profile x", self.RULES)
        assert d.level == PermissionLevel.ASK
        assert d.matched_pattern == "datus hello config set:*"

    def test_trailing_profile_without_value_left_alone(self):
        d = evaluate_bash_command("datus hello --profile", self.RULES)
        assert d.level == PermissionLevel.ASK
        assert d.matched_pattern is None

    def test_non_datus_commands_untouched(self):
        rules = BashCommandRules(allow=["git greet:*"])
        d = evaluate_bash_command("git --profile prod greet", rules)
        assert d.level == PermissionLevel.ASK  # no normalization outside datus
