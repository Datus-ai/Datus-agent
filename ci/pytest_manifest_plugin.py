"""Pytest plugin used by nightly manifest collection."""

from __future__ import annotations


def pytest_collection_finish(session):
    print("DATUS_MANIFEST_NODEIDS_START")
    for item in session.items:
        print(f"DATUS_MANIFEST_NODEID {item.nodeid}")
    print("DATUS_MANIFEST_NODEIDS_END")
