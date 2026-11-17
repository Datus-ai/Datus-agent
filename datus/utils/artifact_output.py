"""Host-supplied artifact output, independent of the project's context and KB."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator


def artifact_output_root(config: Any) -> Path:
    return Path(getattr(config, "artifact_output_root", None) or config.project_root).expanduser().resolve()


def artifact_mutation_guard(config: Any, path: Path) -> None:
    """Protect query/model files while allowing artifact presentation edits."""
    root = artifact_output_root(config)
    try:
        parts = path.resolve().relative_to(root).parts
    except ValueError:
        raise ValueError("Artifact writes must stay inside the supplied output root") from None
    if len(parts) < 4 or parts[0] not in {"reports", "dashboards"} or parts[2] not in {"render", "analysis"}:
        raise ValueError("Use dedicated artifact tools for manifests, queries and models")
    if parts[2] == "render" and path.suffix not in {".jsx", ".js", ".css"}:
        raise ValueError("Render files must be JSX, JavaScript or CSS")


def session_user_scope(config: Any, user_scope: str | None) -> str | None:
    """A host may supply a session directory already isolated for this caller."""
    return None if getattr(config, "session_dir_scoped", False) else user_scope


_UNBOUND = object()
_request_slug_factory: ContextVar[Any] = ContextVar("artifact_request_slug_factory", default=_UNBOUND)


@contextmanager
def artifact_request_scope(config: Any = None, *, factory: Any = _UNBOUND) -> Iterator[None]:
    """Capture the host's allocator for this request and its background turns."""
    allocator = getattr(config, "artifact_slug_factory", None) if factory is _UNBOUND else factory
    token = _request_slug_factory.set(allocator)
    try:
        yield
    finally:
        _request_slug_factory.reset(token)


def current_artifact_slug_factory(config: Any) -> Any:
    factory = _request_slug_factory.get()
    # An explicit None disables an allocator retained by a cached facade.
    return getattr(config, "artifact_slug_factory", None) if factory is _UNBOUND else factory
