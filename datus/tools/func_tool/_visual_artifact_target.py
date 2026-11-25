"""Resolve the target of visual authoring calls and record successful mutations."""

from __future__ import annotations

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from xml.etree.ElementTree import ParseError, fromstring

from agents import FunctionTool

from datus.schemas.artifact_manifest import ARTIFACT_SLUG_RE, ArtifactKind, ArtifactManifest
from datus.tools.func_tool._visual_artifact_helpers import (
    _atomic_write_text,
    append_intent_section,
    artifact_write_lock,
    utc_now_iso,
)
from datus.tools.func_tool.base import FuncToolResult


def studio_edit_targets(message: str, kind: str) -> list[str]:
    """Recover the target carried by Studio's chart-card attachments or renderer repair messages."""
    # The renderer includes this standalone line before a stack trace in its repair request.
    targets = set(re.findall(rf"^{kind} slug: ([a-z0-9_]{{1,80}})$", message, re.MULTILINE))
    for match in re.finditer(r"<artifact-chart-card\s+[^<>]*?/>", message):
        try:
            attrs = fromstring(match.group()).attrib
        except ParseError:
            continue
        other_kind = "dashboard" if kind == "report" else "report"
        slug = attrs.get(f"{kind}-slug")
        if slug and f"{other_kind}-slug" not in attrs and ARTIFACT_SLUG_RE.fullmatch(slug):
            targets.add(slug)
    return sorted(targets)


def selected_visual_slug(user_input, kind: str) -> str | None:
    targets = set(studio_edit_targets(getattr(user_input, "user_message", "") or "", kind))
    inherited = getattr(user_input, "studio_edit_targets", None)
    if isinstance(inherited, dict):
        targets.update(inherited.get(kind, ()))
    explicit = getattr(user_input, f"{kind}_slug", None)
    if isinstance(explicit, str) and explicit:
        targets.add(explicit)
    if len(targets) > 1:
        raise ValueError(
            f"Choose one {kind} to edit per turn. Please confirm the target: {', '.join(sorted(targets))}."
        )
    return next(iter(targets), None)


class VisualArtifactTarget:
    def __init__(
        self,
        root: Path,
        kind: ArtifactKind,
        message: str,
        requested_slug: str | None,
        intent_timestamp: str | None = None,
    ):
        self.root = root
        self.kind = kind
        self.message = message
        self.requested_slug = requested_slug
        self.slug: str | None = None
        self.mode: str | None = None
        self.intent_timestamp = intent_timestamp or datetime.now(timezone.utc).isoformat()
        self.lock = RLock()
        self._intent_recorded = False
        self._intent_error: str | None = None

    def create(self, slug: str, name: str, description: str, factory) -> FuncToolResult:
        with self.lock:
            if self.requested_slug or self.slug:
                return FuncToolResult(success=0, error="This turn already targets an artifact")
            if not ARTIFACT_SLUG_RE.fullmatch(slug):
                return FuncToolResult(success=0, error=f"slug must match {ARTIFACT_SLUG_RE.pattern}")
            try:
                if factory is not None:
                    slug = factory(self.kind, slug)
                manifest = ArtifactManifest(
                    kind=self.kind,
                    slug=slug,
                    name=name.strip(),
                    description=description.strip(),
                    created_at=utc_now_iso(),
                )
            except ValueError as exc:
                return FuncToolResult(success=0, error=str(exc))
            relative = f"{self.kind}s/{slug}"
            directory = self.root / relative
            try:
                directory.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".create-") as tmp:
                    staged = Path(tmp) / slug
                    staged.mkdir()
                    for child in ("render", "queries", "analysis"):
                        (staged / child).mkdir()
                    _atomic_write_text(staged / "manifest.json", manifest.model_dump_json(indent=2) + "\n")
                    with artifact_write_lock(directory):
                        if directory.exists():
                            return FuncToolResult(
                                success=0, error=f"{relative} already exists. Use its slug to edit it."
                            )
                        staged.rename(directory)
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Failed to create {relative}: {exc}")
            self.slug = slug
            self.mode = "new"
            return self.record_mutation(
                FuncToolResult(
                    result={
                        f"{self.kind}_slug": slug,
                        f"{self.kind}_dir": relative,
                        "mode": "new",
                        "queries_dir": f"{relative}/queries",
                        "render_dir": f"{relative}/render",
                        "analysis_dir": f"{relative}/analysis",
                        "manifest_path": f"{relative}/manifest.json",
                    }
                )
            )

    def _slug_guidance(self) -> str:
        if self.requested_slug:
            return f"Ask the user to confirm the selected {self.kind} in Studio."
        return (
            f"Use glob('{self.kind}s/*/manifest.json') and read the manifests to match the requested name. "
            f"If exactly one matches, retry with its slug as {self.kind}_slug. "
            "If the target is unclear, ask the user to identify it."
        )

    def configure_tools(self, tools: list[FunctionTool]) -> list[FunctionTool]:
        """A known edit target is supplied by the caller; otherwise ask for it explicitly."""
        field = f"{self.kind}_slug"
        for tool in tools:
            schema = tool.params_json_schema
            if field not in schema["properties"]:
                continue
            required = schema.setdefault("required", [])
            if self.requested_slug:
                schema["properties"].pop(field)
                if field in required:
                    required.remove(field)
            else:
                schema["properties"][field].pop("default", None)
                schema["properties"][field]["description"] = (
                    f"The target {self.kind} slug. For creation, use the slug returned by start_new_{self.kind}. "
                    f"To identify an existing {self.kind}: {self._slug_guidance()}"
                )
                if field not in required:
                    required.append(field)
        return [tool for tool in tools if not (self.requested_slug and tool.name.startswith("start_new_"))]

    def select(self, slug: str | None) -> FuncToolResult | None:
        with self.lock:
            # Programmatic creators may reuse the slug returned by start_new. A free-form
            # edit must identify its target on each call, even after another save succeeded.
            slug = slug or self.requested_slug or (self.slug if self.mode == "new" else None)
            if not slug:
                return FuncToolResult(success=0, error=f"{self.kind}_slug is required. {self._slug_guidance()}")
            if not ARTIFACT_SLUG_RE.fullmatch(slug):
                return FuncToolResult(
                    success=0, error=f"{self.kind}_slug must match {ARTIFACT_SLUG_RE.pattern}. {self._slug_guidance()}"
                )
            for selected in (self.requested_slug, self.slug):
                if selected and selected != slug:
                    return FuncToolResult(success=0, error=f"This turn targets {self.kind} {selected}, not {slug}")
            directory = self.root / f"{self.kind}s" / slug
            if not directory.is_dir():
                return FuncToolResult(
                    success=0, error=f"{self.kind} directory not found: {slug}. {self._slug_guidance()}"
                )
            if self.slug is None and not (directory / "render/app.jsx").is_file():
                # A prior creation may have stopped after saving queries but before writing app.jsx.
                try:
                    manifest = ArtifactManifest.model_validate_json((directory / "manifest.json").read_bytes())
                except (OSError, ValueError) as exc:
                    return FuncToolResult(success=0, error=f"Cannot resume {slug} without a valid manifest: {exc}")
                if manifest.slug != slug or manifest.kind != self.kind:
                    return FuncToolResult(success=0, error=f"Manifest does not identify {self.kind} {slug}")
            self.slug = slug
            self.mode = self.mode or "edit"
        return None

    def record_mutation(self, result: FuncToolResult) -> FuncToolResult:
        with self.lock:
            if result.success and self.slug and not self._intent_recorded:
                self._append_intent()
                if self._intent_error and isinstance(result.result, dict):
                    result.result["intent_warning"] = self._intent_error
        return result

    def _append_intent(self) -> None:
        self._intent_error = append_intent_section(
            self.directory / "analysis",
            user_message=self.message,
            mode=self.mode,
            timestamp=self.intent_timestamp,
            deduplicate=True,
        )
        self._intent_recorded = self._intent_error is None

    def finish_pending_intent(self) -> None:
        """A committed mutation cannot finish the turn with an unreported history failure."""
        with self.lock:
            if self._intent_error:
                self._append_intent()
                if self._intent_error:
                    raise OSError(self._intent_error)

    @property
    def directory(self) -> Path | None:
        return self.root / f"{self.kind}s" / self.slug if self.slug else None
