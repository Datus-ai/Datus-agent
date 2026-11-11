"""A permission profile scoped to one running chat and its interactive descendants."""

import asyncio
import weakref
from contextlib import ExitStack
from typing import Any

from datus.tools.permission.profiles import build_user_overrides, get_profile


class LivePermissionProfile:
    def __init__(self, agent_config: Any, profile: str):
        self.agent_config = agent_config
        self.profile = profile
        self.changed = asyncio.Event()
        self._managers: weakref.WeakSet = weakref.WeakSet()

    def register(self, manager: Any) -> None:
        if manager is None:
            return
        if manager.active_profile != self.profile:
            raw = getattr(self.agent_config, "_raw_permissions", {}) or {}
            overrides = build_user_overrides(self.profile, {k: v for k, v in raw.items() if k != "profile"})
            manager.switch_profile(self.profile, user_overrides=overrides)
        manager.live_profile = self
        self._managers.add(manager)

    def switch(self, profile: str) -> None:
        get_profile(profile)  # Validate even when there are no managers yet.
        if profile == self.profile:
            return
        raw = getattr(self.agent_config, "_raw_permissions", {}) or {}
        overrides = build_user_overrides(profile, {k: v for k, v in raw.items() if k != "profile"})
        managers = list(self._managers)
        # Publish all managers together; a failed rebuild restores exact configs
        # and session grants, including the manager that raised mid-switch.
        with ExitStack() as locks:
            for manager in managers:
                locks.enter_context(manager._state_lock)
            snapshots = [(m, m.global_config, m.active_profile, dict(m._session_approvals)) for m in managers]
            try:
                for manager in managers:
                    manager.switch_profile(profile, user_overrides=overrides)
            except Exception:
                for manager, config, active, approvals in snapshots:
                    manager.global_config = config
                    manager.active_profile = active
                    manager._session_approvals.clear()
                    manager._session_approvals.update(approvals)
                raise
        self.profile = profile
        changed, self.changed = self.changed, asyncio.Event()
        changed.set()
