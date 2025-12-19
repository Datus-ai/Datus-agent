# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from datus.utils.text_utils import clean_text


def read_csv_and_clean_text(csv_path: str | Path) -> List[Dict[str, Any]]:
    if not csv_path:
        return []
    if isinstance(csv_path, str):
        csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        return []
    encoding = file_encoding(csv_path)
    df = pd.read_csv(csv_path, encoding=encoding, engine="python")
    rows = df.replace({np.nan: None}).to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, str):
                row[k] = clean_text(v)
    return rows


def file_encoding(file_path: Path) -> str:
    if not file_path.exists() or (not file_path.is_file()):
        return ""
    raw = file_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    from charset_normalizer import from_bytes

    best = from_bytes(raw).best()
    encoding = best.encoding.lower()
    if not encoding:
        return "utf-8"
    encoding = encoding.lower().replace("-", "_")
    if encoding in {"ascii", "utf_8", "utf8"}:
        return "utf-8"
    return encoding
