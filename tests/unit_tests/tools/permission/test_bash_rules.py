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
    contains_shell_metachars,
    evaluate_bash_command,
    session_bucket_for,
    split_command_chain,
    split_pipeline,
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
            "ls |& grep foo",  # stderr pipe is not a simple pipe
            "ls |",  # trailing empty segment
            "ls & rm x",  # background `&` is not a sequencer
            "ls &",
            "ls;; rm x",  # empty segment between `;;`
            "echo hi > /etc/passwd",
            "echo `whoami`",
            "echo $(id)",
            "echo ${HOME}",
            "ls\nrm x",  # newline is not segmented
        ],
    )
    def test_metacharacters_force_ask(self, command):
        rules = BashCommandRules(allow=["git status", "ls:*", "echo:*", "grep:*", "rm:*"])
        decision = evaluate_bash_command(command, rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    @pytest.mark.parametrize(
        "command",
        [
            "for f in *; do echo $f; done",
            "if ls; then echo ok; fi",
            "while ls; do echo x; done",
            "(cd /tmp; ls)",
            "{ ls; echo done; }",
        ],
    )
    def test_compound_constructs_force_ask(self, command):
        """Loops/conditionals/subshells are not decomposable into sub-commands.

        Without this guard ``do rm $f`` would be judged — and persisted — as an
        ordinary ``rm`` command.
        """
        rules = BashCommandRules(allow=["ls:*", "echo:*", "cd:*", "rm:*"])
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


class TestBenignRedirectionExemption:
    """``2>&1`` and a ``/dev/null`` target do not force a confirmation.

    Neither form can express anything the argv already can't: fd duplication
    rewires the command's own streams and names no path, and a ``/dev/null``
    target discards by definition. They are also two of the most common shell
    idioms there are, so prompting for them trains users to click through
    prompts. Every other redirection still hits the ceiling because it names a
    path an argv-matched allow rule cannot vouch for.
    """

    RULES = BashCommandRules(allow=["ls:*", "cat:*", "head:*", "grep:*", "echo:*"])

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la 2>&1",
            "ls -la 1>&2",
            "ls -la >&2",
            "cat f <&0",
            "ls -la 2>&-",
            "ls -la 2>/dev/null",
            "ls -la > /dev/null",
            "ls -la >> /dev/null",
            "ls -la 2>> /dev/null",
            "ls -la &>/dev/null",
            "ls -la &>> /dev/null",
        ],
    )
    def test_benign_redirection_auto_allows(self, command):
        decision = evaluate_bash_command(command, self.RULES)
        assert decision.level == PermissionLevel.ALLOW
        assert decision.safety_forced is False

    def test_exempt_redirection_inside_a_chain_is_judged_per_sub_command(self):
        """The motivating case. Bailing out of segmentation on the ``&`` of
        ``2>&1`` sent the whole string to the ceiling and hid ``head`` from the
        per-sub-command prompt."""
        assert split_command_chain("ls -la /tmp 2>&1 | head -20") == ["ls -la /tmp 2>&1", "head -20"]

        decision = evaluate_bash_command("ls -la /tmp 2>&1 | head -20", self.RULES)
        assert decision.level == PermissionLevel.ALLOW

    @pytest.mark.parametrize(
        "command",
        [
            "ls > out.txt",  # names an arbitrary path
            "ls >> out.txt",
            "ls 2> err.txt",
            "ls &> out.txt",  # both streams, still an arbitrary path
            "ls < in.txt",  # input redirection is not exempt
            "ls > /dev/nullx",  # near-miss on the device name
            "ls > /dev/stdout",  # only /dev/null is exempt
        ],
    )
    def test_other_redirections_still_force_ask(self, command):
        decision = evaluate_bash_command(command, self.RULES)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    @pytest.mark.parametrize("command", ["ls &", "sleep 5 &", "ls 2>&1 &", "ls & rm x"])
    def test_background_ampersand_is_not_mistaken_for_a_redirection(self, command):
        """The exemption keys on the ``&`` sitting next to ``<``/``>``; a
        trailing job-control ``&`` must still stop segmentation."""
        assert split_command_chain(command) is None

        decision = evaluate_bash_command(command, self.RULES)
        assert decision.level == PermissionLevel.ASK
        assert decision.safety_forced is True

    @pytest.mark.parametrize(
        "command",
        [
            "echo $(id) 2>/dev/null",
            "echo `whoami` 2>&1",
            "ls\nrm x 2>&1",
        ],
    )
    def test_exemption_does_not_rescue_other_metacharacters(self, command):
        """Stripping the redirection must not strip anything else with it."""
        decision = evaluate_bash_command(command, self.RULES)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    def test_exempt_redirection_does_not_change_a_chain_verdict(self):
        """A non-allow-listed sub-command still blocks the chain, and the
        redirection changes nothing about it: the redirected and bare forms of
        the same chain must be judged identically."""
        redirected = evaluate_bash_command("ls; rm x 2>/dev/null", self.RULES)
        bare = evaluate_bash_command("ls; rm x", self.RULES)

        assert redirected.level == PermissionLevel.ASK
        assert redirected.level == bare.level
        assert redirected.source == bare.source
        assert redirected.safety_forced == bare.safety_forced

    def test_wrapper_ceiling_still_beats_the_exemption(self):
        """Ordering guard: the wrapper rule fires before the redirection check,
        so a shell wrapper cannot smuggle a command past the exemption."""
        rules = BashCommandRules(allow=["sh:*", "ls:*"])
        decision = evaluate_bash_command('sh -c "rm -rf / 2>&1"', rules)
        assert decision.level == PermissionLevel.ASK
        assert decision.source == BashDecisionSource.SAFETY
        assert decision.safety_forced is True

    def test_deny_rule_still_beats_the_exemption(self):
        rules = BashCommandRules(allow=["ls:*"], deny=["ls:/secret*"])
        decision = evaluate_bash_command("ls /secret 2>/dev/null", rules)
        assert decision.level == PermissionLevel.DENY

    def test_dangerous_command_is_unaffected_by_its_redirection(self):
        """``rm -rf /`` was never safety-forced — no metacharacters. Adding
        ``2>/dev/null`` used to force it, which made the redirection look like
        the dangerous part. Both forms now agree; neither auto-allows."""
        rules = BashCommandRules(allow=["ls:*"])
        bare = evaluate_bash_command("rm -rf /", rules)
        redirected = evaluate_bash_command("rm -rf / 2>/dev/null", rules)

        assert bare.level == PermissionLevel.ASK
        assert redirected.level == PermissionLevel.ASK
        assert redirected.safety_forced == bare.safety_forced

    def test_anchored_allow_rule_does_not_match_the_redirection_token(self):
        """Known limitation, pinned deliberately: ``shlex`` keeps ``2>&1`` as an
        argv token, so an exact (unglobbed) allow rule no longer matches. The
        command is an ordinary ASK, not a safety-forced one."""
        rules = BashCommandRules(allow=["git status"])
        decision = evaluate_bash_command("git status 2>&1", rules)

        assert decision.level == PermissionLevel.ASK
        assert decision.safety_forced is False
        assert evaluate_bash_command("git status", rules).level == PermissionLevel.ALLOW

    def test_execution_layer_whitelist_still_rejects_every_redirection(self):
        """The exemption is permission-layer only. ``BashTool`` shares
        ``contains_shell_metachars`` as the skills whitelist — a separate trust
        boundary that must keep rejecting redirection outright."""
        assert contains_shell_metachars("ls -la 2>&1") is True
        assert contains_shell_metachars("ls -la 2>/dev/null") is True
        assert split_pipeline("ls 2>&1") == ["ls 2>&1"]


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
        """The prompt and the grant writers must see EVERY non-allow segment,
        not just the representative one."""
        rules = BashCommandRules(allow=["cat:*"], ask=["docker:*"])
        d = evaluate_bash_command("frobnicate | docker ps | cat x", rules)
        assert [(s.command, s.source, s.matched_pattern, s.bucket) for s in d.ask_segments] == [
            ("frobnicate", BashDecisionSource.DEFAULT, None, "frobnicate"),
            ("docker ps", BashDecisionSource.ASK_RULE, "docker:*", "docker:*"),
        ]

    def test_single_command_carries_itself_as_one_ask_segment(self):
        """Callers get one code path: a plain ASK still populates ask_segments."""
        rules = BashCommandRules(ask=["docker:*"])
        d = evaluate_bash_command("docker ps", rules)
        assert len(d.ask_segments) == 1
        seg = d.ask_segments[0]
        assert seg.command == "docker ps"
        assert seg.source == BashDecisionSource.ASK_RULE
        assert seg.matched_pattern == "docker:*"
        assert seg.bucket == "docker:*"
        assert seg.safety_forced is False

    def test_allow_and_deny_decisions_carry_no_ask_segments(self):
        rules = BashCommandRules(allow=["ls:*"], deny=["rm:*"])
        assert evaluate_bash_command("ls -la", rules).ask_segments == ()
        assert evaluate_bash_command("rm -rf x", rules).ask_segments == ()


class TestCommandChainSplitting:
    """split_command_chain: top-level unquoted |, &&, ||, ; segmentation."""

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("git status", ["git status"]),
            ("a && b", ["a", "b"]),
            ("a || b", ["a", "b"]),
            ("a; b", ["a", "b"]),
            ("a | b", ["a", "b"]),
            ("a && b || c; d | e", ["a", "b", "c", "d", "e"]),
            ('git commit -m "fix; bug"', ['git commit -m "fix; bug"']),
            ("grep 'a && b' file", ["grep 'a && b' file"]),
            ("echo a\\;b", ["echo a\\;b"]),
            (r"find . -name '*.py' -exec ls {} \;", [r"find . -name '*.py' -exec ls {} \;"]),
        ],
    )
    def test_segments(self, command, expected):
        from datus.tools.permission.bash_rules import split_command_chain

        assert split_command_chain(command) == expected

    @pytest.mark.parametrize(
        "command",
        [
            "a |& b",  # stderr pipe
            "ls &",  # background
            "a & b",  # background
            "ls &&",  # empty trailing segment
            "; ls",  # empty leading segment
            "a;; b",  # empty middle segment
            'echo "unclosed && ls',  # unbalanced quotes
        ],
    )
    def test_unsegmentable_returns_none(self, command):
        from datus.tools.permission.bash_rules import split_command_chain

        assert split_command_chain(command) is None

    def test_split_pipeline_still_only_splits_pipes(self):
        """The execution-layer whitelist (BashTool) must not be widened."""
        from datus.tools.permission.bash_rules import split_pipeline

        assert split_pipeline("a && b") == ["a && b"]
        assert split_pipeline("a; b") == ["a; b"]
        assert split_pipeline("a || b") is None
        assert split_pipeline("a | b | c") == ["a", "b", "c"]


class TestChainEvaluation:
    """Per-sub-command judging for &&, ||, ; chains."""

    @pytest.mark.parametrize("op", ["&&", "||", ";"])
    def test_all_allow_sub_commands_auto_allow(self, op):
        rules = BashCommandRules(allow=["git status", "ls:*"])
        d = evaluate_bash_command(f"git status {op} ls -la", rules)
        assert d.level == PermissionLevel.ALLOW
        assert d.source == BashDecisionSource.ALLOW_RULE

    @pytest.mark.parametrize("op", ["&&", "||", ";"])
    def test_deny_sub_command_blocks_whole_chain(self, op):
        rules = BashCommandRules(allow=["git status"], deny=["rm:*"])
        d = evaluate_bash_command(f"git status {op} rm -rf /", rules)
        assert d.level == PermissionLevel.DENY
        assert d.matched_pattern == "rm:*"

    def test_ask_chain_lists_every_non_allow_sub_command(self):
        rules = BashCommandRules(allow=["git fetch"], ask=["npm:*"])
        d = evaluate_bash_command("git fetch && rm -rf build && npm ci", rules)
        assert d.level == PermissionLevel.ASK
        assert [(s.command, s.source) for s in d.ask_segments] == [
            ("rm -rf build", BashDecisionSource.DEFAULT),
            ("npm ci", BashDecisionSource.ASK_RULE),
        ]

    def test_repeated_sub_commands_are_all_reported(self):
        """Display fidelity: the user sees each occurrence, not a deduped one."""
        rules = BashCommandRules()
        d = evaluate_bash_command("rm a; rm b", rules)
        assert [s.command for s in d.ask_segments] == ["rm a", "rm b"]

    def test_safety_forced_sub_command_still_lists_the_others(self):
        """The old short-circuit hid sibling sub-commands from the prompt."""
        rules = BashCommandRules(allow=["git fetch"])
        d = evaluate_bash_command("git fetch && sudo ls && npm ci", rules)
        assert d.level == PermissionLevel.ASK
        assert d.source == BashDecisionSource.SAFETY
        assert d.safety_forced is True
        assert [s.command for s in d.ask_segments] == ["sudo ls", "npm ci"]
        assert [s.safety_forced for s in d.ask_segments] == [True, False]

    def test_mixed_pipe_and_sequence_operators(self):
        rules = BashCommandRules(allow=["cat:*", "grep:*"])
        d = evaluate_bash_command("cat log | grep err && frobnicate", rules)
        assert d.level == PermissionLevel.ASK
        assert [s.command for s in d.ask_segments] == ["frobnicate"]


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
