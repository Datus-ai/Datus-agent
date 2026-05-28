# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers shared between the ``gen_visual_report`` /
``gen_visual_dashboard`` agentic nodes.

This sub-package collects everything that's *about* a visual artifact
but not itself an :class:`AgenticNode`:

* :mod:`._artifact_html_renderer` — slug-validated walker + template
  slotting + asset resolution; the canonical pipeline both kinds use.
* :mod:`.report_html_renderer` / :mod:`.dashboard_html_renderer` — thin
  kind-specific entrypoints (allowlists, template paths, init bootstrap
  flavour) that delegate to the shared pipeline.
* :mod:`._visual_artifact_finalize` — post-validate finalize stage
  (insights / intent / key_tables_schema bake) driven by
  ``BaseVisualArtifactAgenticNode._stream_post_build``.
* :mod:`.templates` — the two ``index.html`` skeletons the kind-specific
  renderers slot payloads into.

The ``BaseVisualArtifactAgenticNode`` itself, plus its
``gen_visual_*`` subclasses, intentionally stay in
``datus/agent/node/`` next to the other agentic-node peers — moving
them would split the node registry across two folders for no benefit.
"""
