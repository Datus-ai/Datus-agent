# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Shared scope, lightweight execution, and handoff guidance for chat and task children."""

COLLABORATION_CONTRACT = """<collaboration_contract>
Keep execution within the assigned goal, target, allowed operations and stop
conditions. Reuse supplied evidence and relevant session history; discover only
what is missing for the current request.
For a continuation, preserve the established target and datasource, apply only
the requested change, and verify the affected parts. Do not rebuild unaffected
artifacts or restart discovery for a small follow-up.
Match effort to the requested scope. A specified-file lookup needs a bounded
read; a local edit needs a local change and its required checks. Keep existing
permissions, confirmations and validation requirements.
Stop when the requested result and required checks are complete. Retry an error
only with new evidence or a concrete repair. Report a bounded blocker when the
task cannot proceed instead of expanding into unrelated work.
Return a concise handoff in the required native result format: the requested
result, relevant evidence or artifact references, checks actually performed,
and unresolved limitations. Reuse existing result fields without adding logs
or repeating discovery merely to produce another summary.
</collaboration_contract>"""
