# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Request-local version relationships for the existing Web file executor."""

import hashlib
from dataclasses import dataclass
from typing import Literal, Optional

from datus.utils.memory_loader import apply_single_replacement

FileSnapshot = Optional[tuple[bytes, int]]
MutationOwnership = Literal["owned", "pending", "external"]
# Bound proof work for ambiguous batches; an unproven revision is never rolled back.
_MAX_REPLAY_STATES = 256


@dataclass(eq=False)
class ClientMutationRecord:
    operation: str
    arguments: dict
    before: FileSnapshot
    completed: bool = False
    canceled: bool = False


class ClientMutationHistory:
    """Prove revisions by replaying known mutations, rather than frozen proposals.

    A batch keeps its original snapshot until every card settles. Pending cards
    may already have written their files before their tool_result arrives, so
    their effects are recognized without declaring that revision owned yet.
    """

    def __init__(self, before: FileSnapshot):
        self.anchor = before[0] if before is not None else None
        self.before = before
        self.records: list[ClientMutationRecord] = []
        self.external = False

    def prepare(self, before: FileSnapshot, operation: str, arguments: dict) -> ClientMutationRecord:
        current = before[0] if before is not None else None
        self.classify(current)
        if all(record.completed or record.canceled for record in self.records):
            self.anchor = current
            self.before = before
            self.records.clear()
        record = ClientMutationRecord(operation, dict(arguments), self.before)
        self.records.append(record)
        return record

    def complete(self, record: ClientMutationRecord, current: Optional[bytes]) -> MutationOwnership:
        record.completed = True
        ownership = self.classify(current)
        if all(item.completed or item.canceled for item in self.records):
            self.anchor = current
            self.records.clear()
        return ownership

    def cancel(self, record: ClientMutationRecord) -> None:
        if not record.completed:
            record.canceled = True

    def classify(self, current: Optional[bytes]) -> MutationOwnership:
        if self.external:
            return "external"
        if self._reachable(current, include_pending=False):
            return "owned"
        if self._reachable(current, include_pending=True):
            return "pending"
        # An observed gap is irreversible for this request. A later full-file
        # write must not hide user content that was already observed here.
        self.external = True
        return "external"

    def _reachable(self, target: Optional[bytes], *, include_pending: bool) -> bool:
        records = [r for r in self.records if not r.canceled and (include_pending or r.completed)]
        required = sum(1 << i for i, record in enumerate(records) if record.completed)
        seen = set()

        def visit(content: Optional[bytes], used: int) -> bool:
            if used & required == required and content == target:
                return True
            key = (hashlib.sha256(content).digest() if content is not None else None, used)
            if key in seen or len(seen) >= _MAX_REPLAY_STATES:
                return False
            seen.add(key)
            for i, record in enumerate(records):
                if used & (1 << i):
                    continue
                valid, updated = self._apply(record, content)
                if valid and visit(updated, used | (1 << i)):
                    return True
            return False

        return visit(self.anchor, 0)

    @staticmethod
    def _apply(record: ClientMutationRecord, content: Optional[bytes]) -> tuple[bool, Optional[bytes]]:
        args = record.arguments
        if record.operation == "native":
            return (True, args["after"]) if content == args["before"] else (False, None)
        if record.operation == "write_file":
            return True, args["content"].encode("utf-8")
        if content is None:
            return False, None
        if record.operation == "delete_file":
            return True, None
        try:
            updated, error = apply_single_replacement(content.decode("utf-8"), args["old_string"], args["new_string"])
        except UnicodeDecodeError:
            return False, None
        return (False, None) if error else (True, updated.encode("utf-8"))
