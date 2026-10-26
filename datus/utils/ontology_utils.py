"""Recognize project ontology source files without depending on tool code."""

import posixpath
import re
from pathlib import Path

_ONTOLOGY_PATH = re.compile(r"(?:^|/)subject/semantic_models/ontology/[^/]+/.+\.ya?ml$", re.IGNORECASE)


def is_ontology_file_path(path: str | Path) -> bool:
    """Include new or temporarily invalid YAML documents, not just fixed filenames."""
    return bool(_ONTOLOGY_PATH.search(posixpath.normpath(str(path).replace("\\", "/"))))
