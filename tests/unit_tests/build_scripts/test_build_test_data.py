import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_build_test_data_uses_official_agent_entrypoint():
    build_script = (_REPO_ROOT / "build_scripts" / "build_test_data.sh").read_text(encoding="utf-8")

    assert 'uv run datus-agent "$@"' in build_script


def _bootstrap_result_failed(output: str) -> bool:
    build_script = (_REPO_ROOT / "build_scripts" / "build_test_data.sh").read_text(encoding="utf-8")
    start = build_script.index("bootstrap_result_failed() {")
    end = build_script.index("\n}\n", start) + len("\n}\n")
    function_source = build_script[start:end]
    result = subprocess.run(
        ["bash", "-c", f"{function_source}\nbootstrap_result_failed"],
        check=False,
        input=output,
        text=True,
    )
    return result.returncode == 0


def test_build_test_data_ignores_retried_nested_failure_when_final_result_succeeds():
    output = """tool result: {'metadata': {'status': 'failed', 'error': 'expected dry-run retry'}}
Final Result: {'status': 'success', 'message': 'semantic_modeling bootstrap completed'}"""

    assert not _bootstrap_result_failed(output)


def test_build_test_data_detects_failed_final_result():
    output = "Final Result: {'status': 'failed', 'message': 'bootstrap failed'}"

    assert _bootstrap_result_failed(output)
