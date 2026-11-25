"""Resolve the target of visual authoring calls and record successful mutations."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from xml.etree.ElementTree import ParseError, fromstring

from agents import FunctionTool

from datus.schemas.artifact_manifest import ARTIFACT_SLUG_RE, ArtifactKind, ArtifactManifest
from datus.tools.func_tool._visual_artifact_helpers import _atomic_write_text, append_intent_section, utc_now_iso
from datus.tools.func_tool.base import FuncToolResult


def studio_edit_slug(message: str, kind: str) -> str | None:
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
    if len(targets) > 1:
        raise ValueError(f"Choose one {kind} to edit per turn")
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
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                return FuncToolResult(success=0, error=f"{relative} already exists. Use its slug to edit it.")
            try:
                for child in ("render", "queries", "analysis"):
                    (directory / child).mkdir()
                _atomic_write_text(directory / "manifest.json", manifest.model_dump_json(indent=2) + "\n")
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
                if field not in required:
                    required.append(field)
        return [tool for tool in tools if not (self.requested_slug and tool.name.startswith("start_new_"))]

    def select(self, slug: str | None) -> FuncToolResult | None:
        with self.lock:
            # Programmatic creators may reuse the slug returned by start_new. A free-form
            # edit must identify its target on each call, even after another save succeeded.
            slug = slug or self.requested_slug or (self.slug if self.mode == "new" else None)
            if not slug or not ARTIFACT_SLUG_RE.fullmatch(slug):
                return FuncToolResult(success=0, error=f"{self.kind}_slug must match {ARTIFACT_SLUG_RE.pattern}")
            for selected in (self.requested_slug, self.slug):
                if selected and selected != slug:
                    return FuncToolResult(success=0, error=f"This turn targets {self.kind} {selected}, not {slug}")
            directory = self.root / f"{self.kind}s" / slug
            if not directory.is_dir():
                return FuncToolResult(success=0, error=f"{self.kind} directory not found: {slug}")
            if self.slug is None and not (directory / "render/app.jsx").is_file():
                return FuncToolResult(success=0, error=f"{directory.name}/render/app.jsx is missing")
            self.slug = slug
            self.mode = self.mode or "edit"
        return None

    def record_mutation(self, result: FuncToolResult) -> FuncToolResult:
        with self.lock:
            if result.success and self.slug and not self._intent_recorded:
                warning = append_intent_section(
                    self.root / f"{self.kind}s" / self.slug / "analysis",
                    user_message=self.message,
                    mode=self.mode,
                    timestamp=self.intent_timestamp,
                    deduplicate=True,
                )
                if warning:
                    if isinstance(result.result, dict):
                        result.result["intent_warning"] = warning
                else:
                    self._intent_recorded = True
        return result

    @property
    def directory(self) -> Path | None:
        return self.root / f"{self.kind}s" / self.slug if self.slug else None
