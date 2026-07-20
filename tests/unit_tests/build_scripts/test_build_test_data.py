from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_build_test_data_uses_official_agent_entrypoint():
    build_script = (_REPO_ROOT / "build_scripts" / "build_test_data.sh").read_text(encoding="utf-8")

    assert 'uv run datus-agent "$@"' in build_script
