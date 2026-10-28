# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for the OS-level bash sandbox helpers.

Covers ``SandboxSettings`` parsing, policy resolution (realpath, dedup,
filtering), Seatbelt profile generation, bwrap argv generation, mechanism
detection/caching and argv wrapping. All tests are deterministic and never
spawn a real sandbox.
"""

import os
import sys

import pytest

from datus.tools.func_tool import bash_sandbox
from datus.tools.func_tool.bash_sandbox import (
    MECHANISM_BWRAP,
    MECHANISM_SEATBELT,
    SandboxPolicy,
    SandboxSettings,
    SandboxUnavailableError,
    build_bwrap_prefix,
    build_policy,
    build_seatbelt_profile,
    wrap_argv,
)


@pytest.fixture(autouse=True)
def reset_mechanism_cache():
    bash_sandbox._reset_detection_cache()
    yield
    bash_sandbox._reset_detection_cache()


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def make_policy(workspace, readable=(), writable_extra=()):
    return SandboxPolicy(
        cwd=str(workspace),
        writable_roots=(str(workspace), *writable_extra),
        readable_roots=tuple(readable),
    )


class TestSandboxSettingsFromDict:
    def test_none_and_non_dict_yield_defaults(self):
        for raw in (None, "yes", 42, ["enabled"]):
            settings = SandboxSettings.from_dict(raw)
            assert settings.enabled is False
            assert settings.allow_read == []
            assert settings.allow_write == []

    def test_enabled_coercion_variants(self):
        assert SandboxSettings.from_dict({"enabled": True}).enabled is True
        assert SandboxSettings.from_dict({"enabled": "true"}).enabled is True
        assert SandboxSettings.from_dict({"enabled": "ON"}).enabled is True
        assert SandboxSettings.from_dict({"enabled": 1}).enabled is True
        assert SandboxSettings.from_dict({"enabled": "false"}).enabled is False
        assert SandboxSettings.from_dict({"enabled": 0}).enabled is False
        # Unparseable values fall back to the default (False).
        assert SandboxSettings.from_dict({"enabled": "maybe"}).enabled is False
        assert SandboxSettings.from_dict({"enabled": None}).enabled is False

    def test_path_lists_filter_invalid_entries(self):
        settings = SandboxSettings.from_dict(
            {
                "allow_read": ["/a", "", 42, None, "/b"],
                "allow_write": "/single",
            }
        )
        assert settings.allow_read == ["/a", "/b"]
        assert settings.allow_write == ["/single"]

    def test_non_list_path_values_yield_empty(self):
        settings = SandboxSettings.from_dict({"allow_read": {"dir": "/a"}, "allow_write": 3})
        assert settings.allow_read == []
        assert settings.allow_write == []

    def test_mode_defaults_to_normal(self):
        settings = SandboxSettings.from_dict({"enabled": True})
        assert settings.mode == bash_sandbox.MODE_NORMAL
        assert settings.is_strict is False

    def test_mode_strict_parsed_case_insensitive(self):
        assert SandboxSettings.from_dict({"mode": "strict"}).is_strict is True
        assert SandboxSettings.from_dict({"mode": " STRICT "}).is_strict is True
        assert SandboxSettings.from_dict({"mode": "normal"}).is_strict is False

    def test_invalid_mode_falls_back_to_normal(self):
        for bad in ("paranoid", 3, ["strict"], {}):
            assert SandboxSettings.from_dict({"mode": bad}).mode == bash_sandbox.MODE_NORMAL

    def test_deny_network_parsed(self):
        assert SandboxSettings.from_dict({"deny_network": True}).deny_network is True
        assert SandboxSettings.from_dict({"deny_network": "true"}).deny_network is True
        assert SandboxSettings.from_dict({}).deny_network is False


class TestStrictEnv:
    def test_keeps_allowlist_and_locale_only(self):
        base = {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "C",
            "LC_CTYPE": "UTF-8",
            "OPENAI_API_KEY": "sk-secret",
            "DATABASE_PASSWORD": "hunter2",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "TMPDIR": "/tmp/x",
        }
        result = bash_sandbox.strict_env(base)
        assert result == {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "C",
            "LC_CTYPE": "UTF-8",
            "TMPDIR": "/tmp/x",
        }

    def test_empty_base_yields_empty(self):
        assert bash_sandbox.strict_env({}) == {}


class TestBuildPolicy:
    def test_workspace_and_tmp_in_writable(self, workspace):
        policy = build_policy(SandboxSettings(), workspace)
        assert str(workspace.resolve()) in policy.writable_roots
        assert os.path.realpath("/tmp") in policy.writable_roots
        assert policy.cwd == os.path.realpath(str(workspace))

    def test_symlinks_resolved_to_realpath(self, tmp_path, workspace):
        real_dir = tmp_path / "real_target"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)
        policy = build_policy(SandboxSettings(allow_write=[str(link)]), workspace)
        assert str(real_dir.resolve()) in policy.writable_roots
        assert str(link) not in policy.writable_roots

    def test_missing_dirs_filtered_out(self, tmp_path, workspace):
        missing = tmp_path / "does_not_exist"
        policy = build_policy(
            SandboxSettings(allow_read=[str(missing)], allow_write=[str(missing)]),
            workspace,
            dynamic_write_dirs=[missing],
        )
        assert str(missing) not in policy.writable_roots
        assert str(missing) not in policy.readable_roots

    def test_dynamic_write_dirs_and_extra_read_dirs_merged(self, tmp_path, workspace):
        output_dir = tmp_path / "session_out"
        output_dir.mkdir()
        datus_home = tmp_path / "datus_home"
        datus_home.mkdir()
        policy = build_policy(
            SandboxSettings(),
            workspace,
            dynamic_write_dirs=[output_dir],
            extra_read_dirs=[str(datus_home)],
        )
        assert str(output_dir.resolve()) in policy.writable_roots
        assert str(datus_home.resolve()) in policy.readable_roots

    def test_readable_excludes_writable_duplicates(self, tmp_path, workspace):
        shared = tmp_path / "shared"
        shared.mkdir()
        policy = build_policy(
            SandboxSettings(allow_read=[str(shared)], allow_write=[str(shared)]),
            workspace,
        )
        assert str(shared.resolve()) in policy.writable_roots
        assert str(shared.resolve()) not in policy.readable_roots

    def test_duplicates_deduped_preserving_order(self, workspace):
        policy = build_policy(
            SandboxSettings(allow_write=[str(workspace), str(workspace)]),
            workspace,
        )
        assert policy.writable_roots.count(str(workspace.resolve())) == 1
        assert policy.writable_roots[0] == str(workspace.resolve())

    def test_tilde_expansion(self, tmp_path, workspace, monkeypatch):
        home = tmp_path / "home"
        (home / "extra").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        policy = build_policy(SandboxSettings(allow_read=["~/extra"]), workspace)
        assert str((home / "extra").resolve()) in policy.readable_roots

    def test_python_prefixes_readable(self, workspace):
        policy = build_policy(SandboxSettings(), workspace)
        combined = set(policy.readable_roots) | set(policy.writable_roots)
        assert os.path.realpath(sys.prefix) in combined
        assert os.path.realpath(sys.base_prefix) in combined

    def test_deny_network_propagates_to_policy(self, workspace):
        assert build_policy(SandboxSettings(), workspace).deny_network is False
        assert build_policy(SandboxSettings(deny_network=True), workspace).deny_network is True


class TestBuildPolicyStrict:
    """Strict mode drops caller-injected dirs but keeps operator allowlists."""

    def test_extra_read_dirs_ignored(self, tmp_path, workspace):
        datus_home = tmp_path / "datus_home"
        datus_home.mkdir()
        policy = build_policy(
            SandboxSettings(mode=bash_sandbox.MODE_STRICT),
            workspace,
            extra_read_dirs=[str(datus_home)],
        )
        assert str(datus_home.resolve()) not in policy.readable_roots

    def test_dynamic_write_dirs_ignored(self, tmp_path, workspace):
        session_dir = tmp_path / "session_out"
        session_dir.mkdir()
        policy = build_policy(
            SandboxSettings(mode=bash_sandbox.MODE_STRICT),
            workspace,
            dynamic_write_dirs=[session_dir],
        )
        assert str(session_dir.resolve()) not in policy.writable_roots

    def test_explicit_allowlists_still_apply(self, tmp_path, workspace):
        shared_read = tmp_path / "shared_read"
        shared_read.mkdir()
        shared_write = tmp_path / "shared_write"
        shared_write.mkdir()
        policy = build_policy(
            SandboxSettings(
                mode=bash_sandbox.MODE_STRICT,
                allow_read=[str(shared_read)],
                allow_write=[str(shared_write)],
            ),
            workspace,
        )
        assert str(shared_read.resolve()) in policy.readable_roots
        assert str(shared_write.resolve()) in policy.writable_roots

    def test_validated_runtime_read_dirs_survive_strict(self, tmp_path, workspace):
        plugin_dir = tmp_path / "tenant_plugin"
        plugin_dir.mkdir()
        policy = build_policy(
            SandboxSettings(mode=bash_sandbox.MODE_STRICT),
            workspace,
            runtime_read_dirs=[str(plugin_dir)],
        )
        assert str(plugin_dir.resolve()) in policy.readable_roots

    def test_workspace_tmp_and_python_env_survive(self, workspace):
        policy = build_policy(SandboxSettings(mode=bash_sandbox.MODE_STRICT), workspace)
        assert str(workspace.resolve()) in policy.writable_roots
        assert os.path.realpath("/tmp") in policy.writable_roots
        combined = set(policy.readable_roots) | set(policy.writable_roots)
        assert os.path.realpath(sys.prefix) in combined


class TestSeatbeltProfile:
    def test_profile_structure(self, workspace):
        profile = build_seatbelt_profile(make_policy(workspace))
        assert profile.startswith("(version 1)\n(allow default)\n")
        assert "(deny file-write*" in profile
        assert "(deny file-read-data" in profile
        # Regression guard: denying file-read* would also deny metadata and
        # break stat/PATH lookup/dyld everywhere.
        assert "(deny file-read*" not in profile

    def test_writable_roots_exempted_from_both_denies(self, workspace):
        profile = build_seatbelt_profile(make_policy(workspace))
        assert profile.count(f'(subpath "{workspace}")') == 2

    def test_readable_roots_only_in_read_deny(self, tmp_path, workspace):
        readable = tmp_path / "readonly_root"
        readable.mkdir()
        profile = build_seatbelt_profile(make_policy(workspace, readable=[str(readable)]))
        assert profile.count(f'(subpath "{readable}")') == 1
        write_block = profile.split("(deny file-read-data")[0]
        assert str(readable) not in write_block

    def test_system_roots_and_devices_present(self, workspace):
        profile = build_seatbelt_profile(make_policy(workspace))
        for root in ("/usr", "/bin", "/System", "/Library", "/private/etc"):
            assert f'(subpath "{root}")' in profile
        assert '(literal "/dev/null")' in profile
        assert '(literal "/dev/urandom")' in profile
        assert '(subpath "/dev/fd")' in profile

    def test_root_dir_data_stays_readable(self, workspace):
        # Regression guard: dyld reads the ``/`` directory itself during
        # cryptex resolution at process start; without this literal every
        # sandboxed process dies with SIGABRT before main().
        profile = build_seatbelt_profile(make_policy(workspace))
        read_block = profile.split("(deny file-read-data")[1]
        assert '(literal "/")' in read_block

    def test_path_quoting_escapes_special_chars(self, tmp_path):
        weird = tmp_path / 'dir with "quotes"'
        weird.mkdir()
        policy = make_policy(weird)
        profile = build_seatbelt_profile(policy)
        assert '\\"quotes\\"' in profile

    def test_backslash_escaped(self):
        assert bash_sandbox._sb_quote("a\\b") == '"a\\\\b"'

    def test_deny_network_appends_network_deny(self, workspace):
        policy = SandboxPolicy(
            cwd=str(workspace),
            writable_roots=(str(workspace),),
            readable_roots=(),
            deny_network=True,
        )
        assert build_seatbelt_profile(policy).rstrip().endswith("(deny network*)")

    def test_no_network_deny_by_default(self, workspace):
        assert "(deny network*)" not in build_seatbelt_profile(make_policy(workspace))


class TestBwrapPrefix:
    @pytest.fixture(autouse=True)
    def fake_bwrap(self, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "_find_bwrap", lambda: "/usr/bin/bwrap")
        # Deterministic system-dir layout regardless of the host OS.
        monkeypatch.setattr(
            bash_sandbox,
            "_mount_args_for_system_dir",
            lambda p: ["--ro-bind", p, p] if p in ("/usr", "/etc") else [],
        )

    def test_basic_structure(self, workspace):
        args = build_bwrap_prefix(make_policy(workspace))
        assert args[0] == "/usr/bin/bwrap"
        assert "--die-with-parent" in args
        assert "--unshare-net" not in args
        assert ["--proc", "/proc"] == args[args.index("--proc") : args.index("--proc") + 2]
        assert ["--dev", "/dev"] == args[args.index("--dev") : args.index("--dev") + 2]
        assert args[-2:] == ["--chdir", str(workspace)]

    def test_tmpfs_for_tmp_dirs(self, workspace):
        args = build_bwrap_prefix(make_policy(workspace))
        tmpfs_targets = [args[i + 1] for i, a in enumerate(args) if a == "--tmpfs"]
        assert tmpfs_targets == ["/tmp", "/var/tmp"]

    def test_writable_tmp_roots_skipped(self, workspace):
        args = build_bwrap_prefix(make_policy(workspace, writable_extra=("/tmp", "/var/tmp")))
        bind_targets = [args[i + 1] for i, a in enumerate(args) if a == "--bind"]
        assert bind_targets == [str(workspace)]

    def test_readonly_binds_before_writable_binds(self, tmp_path, workspace):
        readable = tmp_path / "ro_root"
        readable.mkdir()
        args = build_bwrap_prefix(make_policy(workspace, readable=[str(readable)]))
        joined = " ".join(args)
        ro_pair = f"--ro-bind {readable} {readable}"
        rw_pair = f"--bind {workspace} {workspace}"
        assert ro_pair in joined
        assert rw_pair in joined
        assert joined.index(ro_pair) < joined.index(rw_pair)

    def test_workspace_under_tmp_still_bound(self, workspace):
        nested = "/tmp/pytest-of-user/ws"
        policy = SandboxPolicy(cwd=nested, writable_roots=(nested,), readable_roots=())
        args = build_bwrap_prefix(policy)
        tmpfs_pos = args.index("--tmpfs")
        bind_pos = args.index("--bind")
        assert tmpfs_pos < bind_pos
        assert args[bind_pos + 1 : bind_pos + 3] == [nested, nested]

    def test_missing_bwrap_raises(self, workspace, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "_find_bwrap", lambda: None)
        with pytest.raises(SandboxUnavailableError):
            build_bwrap_prefix(make_policy(workspace))

    def test_deny_network_adds_unshare_net(self, workspace):
        policy = SandboxPolicy(
            cwd=str(workspace),
            writable_roots=(str(workspace),),
            readable_roots=(),
            deny_network=True,
        )
        args = build_bwrap_prefix(policy)
        assert "--unshare-net" in args

    def test_no_unshare_net_by_default(self, workspace):
        assert "--unshare-net" not in build_bwrap_prefix(make_policy(workspace))


class TestMountArgsForSystemDir:
    def test_symlink_becomes_symlink_arg(self, tmp_path):
        target = tmp_path / "usr_bin"
        target.mkdir()
        link = tmp_path / "bin"
        link.symlink_to("usr_bin")
        assert bash_sandbox._mount_args_for_system_dir(str(link)) == ["--symlink", "usr_bin", str(link)]

    def test_real_dir_becomes_ro_bind(self, tmp_path):
        real = tmp_path / "usr"
        real.mkdir()
        assert bash_sandbox._mount_args_for_system_dir(str(real)) == ["--ro-bind", str(real), str(real)]

    def test_missing_path_yields_nothing(self, tmp_path):
        assert bash_sandbox._mount_args_for_system_dir(str(tmp_path / "nope")) == []


class TestDetectMechanism:
    def test_darwin_with_working_sandbox_exec(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(bash_sandbox, "_find_sandbox_exec", lambda: "/usr/bin/sandbox-exec")
        monkeypatch.setattr(bash_sandbox, "_probe", lambda argv: True)
        assert bash_sandbox.detect_mechanism() == MECHANISM_SEATBELT
        assert bash_sandbox.is_available() is True

    def test_darwin_probe_failure(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(bash_sandbox, "_find_sandbox_exec", lambda: "/usr/bin/sandbox-exec")
        monkeypatch.setattr(bash_sandbox, "_probe", lambda argv: False)
        assert bash_sandbox.detect_mechanism() is None

    def test_darwin_missing_sandbox_exec(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(bash_sandbox, "_find_sandbox_exec", lambda: None)
        monkeypatch.setattr(bash_sandbox, "_probe", lambda argv: True)
        assert bash_sandbox.detect_mechanism() is None

    def test_linux_with_working_bwrap(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(bash_sandbox, "_find_bwrap", lambda: "/usr/bin/bwrap")
        monkeypatch.setattr(bash_sandbox, "_probe", lambda argv: True)
        assert bash_sandbox.detect_mechanism() == MECHANISM_BWRAP

    def test_linux_bwrap_probe_failure(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(bash_sandbox, "_find_bwrap", lambda: "/usr/bin/bwrap")
        monkeypatch.setattr(bash_sandbox, "_probe", lambda argv: False)
        assert bash_sandbox.detect_mechanism() is None

    def test_windows_unsupported(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert bash_sandbox.detect_mechanism() is None
        assert bash_sandbox.is_available() is False

    def test_result_cached_across_calls(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        calls = []

        def probe(argv):
            calls.append(argv)
            return True

        monkeypatch.setattr(bash_sandbox, "_find_bwrap", lambda: "/usr/bin/bwrap")
        monkeypatch.setattr(bash_sandbox, "_probe", probe)
        assert bash_sandbox.detect_mechanism() == MECHANISM_BWRAP
        assert bash_sandbox.detect_mechanism() == MECHANISM_BWRAP
        assert len(calls) == 1

    def test_unavailable_reason_mentions_platform_tool(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert "sandbox-exec" in bash_sandbox.unavailable_reason()
        monkeypatch.setattr(sys, "platform", "linux")
        assert "bwrap" in bash_sandbox.unavailable_reason()
        monkeypatch.setattr(sys, "platform", "win32")
        assert "win32" in bash_sandbox.unavailable_reason()


class TestWrapArgv:
    def test_seatbelt_wrapping(self, workspace, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "detect_mechanism", lambda: MECHANISM_SEATBELT)
        monkeypatch.setattr(bash_sandbox, "_find_sandbox_exec", lambda: "/usr/bin/sandbox-exec")
        argv = ["/bin/bash", "-c", "echo hi"]
        wrapped = wrap_argv(argv, make_policy(workspace))
        assert wrapped[0] == "/usr/bin/sandbox-exec"
        assert wrapped[1] == "-p"
        assert "(deny file-write*" in wrapped[2]
        assert wrapped[3:] == argv

    def test_bwrap_wrapping(self, workspace, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "detect_mechanism", lambda: MECHANISM_BWRAP)
        monkeypatch.setattr(bash_sandbox, "_find_bwrap", lambda: "/usr/bin/bwrap")
        monkeypatch.setattr(bash_sandbox, "_mount_args_for_system_dir", lambda p: [])
        argv = ["/bin/bash", "-c", "echo hi"]
        wrapped = wrap_argv(argv, make_policy(workspace))
        assert wrapped[0] == "/usr/bin/bwrap"
        assert wrapped[-3:] == argv

    def test_unavailable_raises(self, workspace, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "detect_mechanism", lambda: None)
        with pytest.raises(SandboxUnavailableError):
            wrap_argv(["/bin/bash", "-c", "true"], make_policy(workspace))


class TestPolicyImmutability:
    def test_policy_is_frozen(self, workspace):
        policy = make_policy(workspace)
        with pytest.raises(AttributeError):
            policy.cwd = "/elsewhere"
