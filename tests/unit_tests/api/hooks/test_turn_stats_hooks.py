"""Tests for datus.api.hooks.turn_stats_hooks — registration surface."""

from datus.api.hooks import TurnStatsHook, get_turn_stats_hook, set_turn_stats_hook


class _Hook:
    def capture_context(self, http_request, stream_request, user_id):
        return {"k": "v"}

    async def on_turn_finished(self, event):
        return None


def test_set_and_clear_hook():
    hook = _Hook()
    set_turn_stats_hook(hook)
    try:
        assert get_turn_stats_hook() is hook
        assert isinstance(hook, TurnStatsHook)
    finally:
        set_turn_stats_hook(None)
    assert get_turn_stats_hook() is None
